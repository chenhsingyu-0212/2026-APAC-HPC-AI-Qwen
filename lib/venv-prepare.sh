#!/bin/bash
# Make a set of pip venvs look enough like a container image to serve from.
#
#   run/venv-prepare.sh                                    # NYCU defaults
#   ENVS=/scratch/.../envs STAGE=/scratch/.../stage run/venv-prepare.sh
#
# Idempotent; safe to re-run after any pip install. Was nycu-prepare.sh, but
# nothing in it is NYCU-specific -- it is the recipe for running these engines
# without a container runtime, and NSCC uses it for a reason beyond convenience:
# the same wheels on both machines means an H100 row and an H200 row differ in
# the GPU and in nothing else, which is the entire point of having two machines.
#
# NYCU has no container runtime, so all three engines run from pip venvs. A wheel
# ships what is needed to IMPORT a library; these engines also COMPILE kernels on
# first use, and for that they need a toolkit laid out the way a toolkit is laid
# out. Four things are missing, and each one costs a checkpoint load to discover.
#
#   1. lib64. sglang's JIT links with -L$CUDA_HOME/lib64, and the wheels use lib/.
#   2. Unversioned .so names. `ld -lcudart` needs libcudart.so; the wheels ship
#      only libcudart.so.13, because the unversioned name lives in a -dev package.
#      Job 79389 got as far as loading 221 GB of weights and allocating a
#      1,523,487-token KV cache before dying on exactly this:
#        /usr/bin/ld: cannot find -lcudart: No such file or directory
#   3. Python headers. vLLM's triton compiles cuda_utils.c and needs Python.h,
#      which is on the LOGIN node and not on the compute nodes. And Debian's
#      /usr/include/python3.10/pyconfig.h is only a stub that includes
#      <x86_64-linux-gnu/python3.10/pyconfig.h>, so copying one directory yields a
#      header that is found and then fails to compile.
#   4. libpython, for the link step after that compile.
#
# Run this on the login node: it writes to $HOME and to the venvs, both on shared
# storage, so the compute nodes see the result.
set -uo pipefail

ENVS="${ENVS:-/home/$USER/envs}"
STAGE="${STAGE:-/home/$USER}"
PYVER="${PYVER:-3.10}"

say() { echo "  $*"; }

# ONE CUDA VERSION ACROSS THE WHOLE nvcc TOOLCHAIN. The wheels do not agree with
# themselves: sglang and vLLM pull nvcc/crt/nvrtc and nvidia-nvvm at 13.4.92 while
# nvidia-cuda-runtime stays at 13.0.96, and torch is built for cu130. Three
# separate failures come out of that one mismatch, in this order:
#
#   cccl   "CUDA compiler and CUDA toolkit headers are incompatible" -- nvcc 13.4
#          against CUDART_VERSION 13000.
#   ptxas  "Unsupported .version 9.4; current version is '9.0'" -- cicc from
#          nvidia-nvvm 13.4 emits PTX 9.4, ptxas from nvcc 13.0 only reads 9.0.
#          Note the package is nvidia-nvvm, NOT nvidia-cuda-nvvm, which does not
#          exist and whose absence silently aborts the whole pip install.
#
# The pin is PER VENV and derived from that venv's own nvidia-cuda-runtime,
# because the right answer differs: sglang and vLLM carry cudart 13.0.96 (torch
# cu130) and want an nvcc from the 13.0 series, while TensorRT-LLM carries cudart
# 13.4.92 and wants 13.4. A single global constant gets one of them wrong, which
# is how trtllm ended up with nvcc 13.0.88 against 13.4 headers and the cccl
# error came straight back.
#
# Matching major.minor is what cccl's check actually compares; the patch level is
# whatever that series ships.
echo "== 0/3  pin each venv's nvcc toolchain to its own CUDA runtime"
NV_INDEX="${NV_INDEX:-https://pypi.nvidia.com}"
for v in "$ENVS"/*; do
    [[ -x "$v/bin/pip" ]] || continue
    name="$(basename "$v")"
    rt="$("$v/bin/pip" list 2>/dev/null | awk '$1=="nvidia-cuda-runtime"{print $2}')"
    [[ -n "$rt" ]] || { say "$(printf '%-8s' "$name") no nvidia-cuda-runtime, nothing to pin"; continue; }
    series="${rt%.*}"          # 13.0.96 -> 13.0
    if "$v/bin/pip" install -q --extra-index-url "$NV_INDEX" \
            "nvidia-cuda-nvcc==$series.*" "nvidia-cuda-crt==$series.*" \
            "nvidia-cuda-nvrtc==$series.*" "nvidia-nvvm==$series.*" 2>/dev/null; then
        got="$("$v/bin/pip" list 2>/dev/null | awk '$1=="nvidia-cuda-nvcc"{print $2}')"
        say "$(printf '%-8s' "$name") cudart $rt -> nvcc ${got:-?}"
    else
        say "$(printf '%-8s' "$name") ! could not pin nvcc to the $series series (cudart $rt)"
    fi
done

# TensorRT-LLM against the cuda-python it was written for, not the newest one.
# Its metadata says `cuda-python>=13`, and cuda-bindings 13.4 removed
# cudaIpcMemHandle_t.reserved, which tensorrt_llm/_ipc_utils.py uses directly:
#
#   handles_reserved = dist.tp_allgather(local_handle.reserved)
#   AttributeError: 'cuda.bindings.runtime.cudaIpcMemHandle_t' object has no
#                   attribute 'reserved'
#
# That path is the custom all-reduce's IPC setup, so it only runs at TP>1 -- which
# is why every TP=1 smoke test passed and job 79484's TP4 cell did not. 13.0.3 is
# the newest release that still has the field.
if [[ -x "$ENVS/trtllm/bin/pip" ]]; then
    echo "== 0b/3  pin cuda-bindings for TensorRT-LLM"
    if "$ENVS/trtllm/bin/pip" install -q --extra-index-url "${NV_INDEX:-https://pypi.nvidia.com}" \
            "cuda-python==13.0.*" "cuda-bindings==13.0.3" 2>/dev/null \
       && "$ENVS/trtllm/bin/python3" -c \
            "from cuda.bindings import runtime; assert hasattr(runtime.cudaIpcMemHandle_t(), 'reserved')" 2>/dev/null; then
        say "trtllm  cuda-bindings $("$ENVS/trtllm/bin/pip" list 2>/dev/null | awk '$1=="cuda-bindings"{print $2}') -- cudaIpcMemHandle_t.reserved present"
    else
        say "trtllm  ! could not get a cuda-bindings with cudaIpcMemHandle_t.reserved"
    fi
fi

# CCCL headers, which nvcc's own bundled headers depend on. cuda_fp16.h ends with
#   #include <nv/target>
# and that file ships in nvidia-cuda-cccl, which TensorRT-LLM does not pull in. So
# flashinfer's JIT (flashinfer/data/csrc/norm.cu) failed on job 79514/79515 with
#   nvidia/cu13/include/cuda_fp16.h:4503:10: fatal error: nv/target: No such file
# even though nvcc itself was present and the version alignment was already right.
# THE HEADERS MUST BE IN THE CUDA_HOME TREE, not merely installed somewhere.
# A JIT passes -I$CUDA_HOME/include and nothing else, and the include path is kept
# single-flavour on purpose (see eng_include_dirs() in run/engine-lib.sh), so a
# header sitting in the cu12 tree is not reachable and must not be made reachable.
#
# Two were missing from nvidia/cu13/include, each costing an allocation to find:
#   nv/target        from nvidia-cuda-cccl -- cuda_fp16.h itself ends with
#                    #include <nv/target>, and no engine pulls that package in
#   curand_kernel.h  from nvidia-curand, the cu13 flavour; the venv had only
#                    nvidia-curand-cu12, whose headers land in the wrong tree
echo "== 0c/3  the cu13-flavour headers a JIT needs, inside the CUDA_HOME tree"
for v in "$ENVS"/*; do
    d="$v/lib/python$PYVER/site-packages/nvidia/cu13"
    [[ -x "$d/bin/nvcc" ]] || continue
    name="$(basename "$v")"
    for pair in "nv/target:nvidia-cuda-cccl" "curand_kernel.h:nvidia-curand"; do
        h="${pair%%:*}"; pkg="${pair##*:}"
        [[ -e "$d/include/$h" ]] && continue
        if "$v/bin/pip" install -q --extra-index-url "${NV_INDEX:-https://pypi.nvidia.com}" \
               "$pkg" 2>/dev/null && [[ -e "$d/include/$h" ]]; then
            say "$(printf %-8s "$name") installed $pkg for <$h>"
        else
            say "$(printf %-8s "$name") ! <$h> still not in cu13/include; an nvcc JIT will fail"
        fi
    done
    miss=""
    for h in nv/target curand_kernel.h cooperative_groups.h cuda_fp16.h cublas_v2.h; do
        [[ -e "$d/include/$h" ]] || miss="$miss $h"
    done
    if [[ -n "$miss" ]]; then
        say "$(printf %-8s "$name") ! missing from cu13/include:$miss"
    else
        say "$(printf %-8s "$name") every JIT header present in cu13/include"
    fi
done

echo "== 1/3  toolkit layout in each venv"
for v in "$ENVS"/*; do
    [[ -d "$v/lib/python$PYVER/site-packages/nvidia" ]] || continue
    name="$(basename "$v")"
    sp="$v/lib/python$PYVER/site-packages"
    n64=0 nso=0
    for d in "$sp"/nvidia/*/; do
        [[ -d "$d/lib" ]] || continue
        [[ -e "$d/lib64" ]] || { ln -s lib "$d/lib64" && n64=$(( n64 + 1 )); }
        for so in "$d"/lib/lib*.so.*; do
            [[ -e "$so" ]] || continue
            base="$(basename "$so")"; stem="${base%%.so.*}"
            [[ -e "$d/lib/$stem.so" ]] || { ln -s "$base" "$d/lib/$stem.so" && nso=$(( nso + 1 )); }
        done
    done
    say "$(printf '%-8s' "$name") +$n64 lib64 symlink(s), +$nso unversioned .so name(s)"
done

echo "== 2/3  Python headers, staged out of the login node"
inc="/usr/include/python$PYVER"
if [[ -r "$inc/Python.h" ]]; then
    mkdir -p "$STAGE/pyinclude"
    cp -a "$inc" "$STAGE/pyinclude/" 2>/dev/null
    # The multiarch pyconfig.h, reachable both as <x86_64-linux-gnu/...> from the
    # stub (so: under the package dir) and directly (so: under the parent).
    marc="/usr/include/x86_64-linux-gnu/python$PYVER"
    if [[ -d "$marc" ]]; then
        mkdir -p "$STAGE/pyinclude/python$PYVER/x86_64-linux-gnu/python$PYVER" \
                 "$STAGE/pyinclude/x86_64-linux-gnu/python$PYVER"
        cp -a "$marc/." "$STAGE/pyinclude/python$PYVER/x86_64-linux-gnu/python$PYVER/"
        cp -a "$marc/." "$STAGE/pyinclude/x86_64-linux-gnu/python$PYVER/"
    fi
    say "staged to $STAGE/pyinclude"
else
    say "! no $inc/Python.h on this host -- vLLM's triton JIT will fail on the compute nodes"
fi

echo "== 3/3  libpython, for the link step"
mkdir -p "$STAGE/pylib"
if cp -a /usr/lib/x86_64-linux-gnu/libpython$PYVER.so* "$STAGE/pylib/" 2>/dev/null; then
    say "staged $(ls "$STAGE/pylib" | wc -l) file(s) to $STAGE/pylib"
else
    say "! no libpython$PYVER.so on this host"
fi

echo
echo "== verify: compile and link the way the engines do"
t0="$(mktemp -d)"
# cuda_fp16.h is what flashinfer's JIT pulls in, and it is the one that needs
# <nv/target>; cuda/std/cstddef is the cccl-version canary. Compile both.
printf '#include <curand_kernel.h>\n#include <cooperative_groups.h>\n#include <cuda_fp16.h>\nint main(void){return 0;}\n' > "$t0/fp16.cu"
printf '#include <cuda/std/cstddef>\nint main(void){return 0;}\n' > "$t0/c.cu"
for v in "$ENVS"/*; do
    sp="$v/lib/python$PYVER/site-packages"; d="$sp/nvidia/cu13"
    [[ -x "$d/bin/nvcc" ]] || continue
    # Only the venvs that actually ship cccl get this check. TensorRT-LLM does
    # not install it, and "header not found" there is an absence, not the version
    # mismatch this test exists to catch.
    [[ -r "$d/include/cccl/cuda/std/cstddef" ]] || {
        say "--  $(basename "$v"): no cccl headers, version check not applicable"; continue; }
    if "$d/bin/nvcc" -arch=sm_90a -c "$t0/fp16.cu" -o "$t0/fp16.o" -I"$d/include" 2>"$t0/err"; then
        say "ok  $(basename "$v"): nvcc compiles the flashinfer header chain"
    else
        say "x   $(basename "$v"): nvcc cannot compile the flashinfer header chain:"; head -2 "$t0/err" | sed 's/^/      /'
    fi
    if "$d/bin/nvcc" -std=c++20 -arch=sm_90a -c "$t0/c.cu" -o "$t0/c.o" \
           -I"$d/include" -I"$d/include/cccl" 2>"$t0/err"; then
        say "ok  $(basename "$v"): nvcc compiles cccl"
    else
        say "x   $(basename "$v"): nvcc does not compile cccl:"; head -2 "$t0/err" | sed 's/^/      /'
    fi
done
rm -rf "$t0"

t="$(mktemp -d)"
printf '#include <Python.h>\nint main(void){return 0;}\n' > "$t/a.c"
if CPATH="$STAGE/pyinclude/python$PYVER:$STAGE/pyinclude" gcc -c "$t/a.c" -o "$t/a.o" 2>"$t/err"; then
    say "ok  Python.h compiles"
else
    say "x   Python.h does not compile:"; sed 's/^/      /' "$t/err" | head -3
fi
for v in "$ENVS"/*; do
    d="$v/lib/python$PYVER/site-packages/nvidia/cu13"
    [[ -d "$d/lib64" ]] || continue
    printf 'int main(void){return 0;}\n' > "$t/b.c"
    if gcc "$t/b.c" -L"$d/lib64" -lcudart -o "$t/b" 2>"$t/err2"; then
        say "ok  $(basename "$v"): ld finds -lcudart"
    else
        say "x   $(basename "$v"): ld cannot find -lcudart:"; sed 's/^/      /' "$t/err2" | head -2
    fi
done
rm -rf "$t"

echo
echo "A failed JIT build is cached. After fixing anything above, clear it:"
echo "  rm -rf ~/.cache/sglang/jit ~/.triton ~/.cache/flashinfer ~/.cache/tensorrt_llm"
