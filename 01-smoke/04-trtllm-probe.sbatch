#!/bin/bash
#SBATCH --job-name=trtllm-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --partition=hpcai
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#
# The hpcai partition caps a job at 17 minutes. Do not raise this.
#
# Does TensorRT-LLM run on this cluster at all, and what is its CLI actually?
#
#   sbatch run/04-trtllm-probe.sbatch
#   TRTLLM_VENV=/path/to/venv sbatch run/04-trtllm-probe.sbatch
#
# Answers Q2/Q3/Q4 of docs/trtllm-plan.md, which have been open since the start
# because every flag name in run/engines/trtllm.sh was written from documentation
# rather than from this binary.
#
# BARE METAL, no container. The venv carries its own MPI (pip openmpi) and its
# own CUDA wheels; what it cannot carry is the driver, and libcuda.so.1 only
# exists on a GPU node -- which is the entire reason this needs a job rather
# than a login shell. It is NOT the conda linker failure that killed ~/pgm/ on
# 2026-08-17; that was a missing dev symlink at JIT-link time, this is the
# runtime driver itself.
#
# No model is loaded and no server is started. This job reads help text and
# imports a library. If that is all it manages, that is still Q2 answered and a
# day of Phase A unblocked.

set -uo pipefail

TRTLLM_VENV="${TRTLLM_VENV:-/scratch/home/$USER/hsingyu/trtllm-venv}"
OUT="${OUT:-${SLURM_SUBMIT_DIR:-$PWD}/results/trtllm-help-${SLURM_JOB_ID:-manual}}"

say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT" || { echo "x cannot create $OUT" >&2; exit 1; }

echo "=============================================================="
echo " trtllm probe on $(hostname)   job ${SLURM_JOB_ID:-manual}"
echo " venv: $TRTLLM_VENV"
echo "=============================================================="

[[ -x "$TRTLLM_VENV/bin/trtllm-serve" ]] || {
    echo "x no trtllm-serve at $TRTLLM_VENV/bin/" >&2
    echo "  Install it first:" >&2
    echo "    <python3.12> -m venv $TRTLLM_VENV" >&2
    echo "    $TRTLLM_VENV/bin/pip install --extra-index-url https://pypi.nvidia.com tensorrt-llm openmpi" >&2
    exit 1; }

say "--- the driver this node has that the login node does not ---"
nvidia-smi -L || echo "  x no GPU visible"
find /usr/lib64 /usr/lib -maxdepth 3 -name 'libcuda.so*' 2>/dev/null | head -3

# A pip-installed TensorRT-LLM does not set up its own library path: the wheels
# scatter .so files across site-packages and nothing puts them on the loader's
# search path. `ldd` on the bindings, with nothing set, reports twelve missing
# libraries -- ten of which are sitting in the venv already.
#
# Measured on this venv 2026-09-16, what has to be on the path:
#   $VENV/lib                      libmpi.so (pip openmpi); tensorrt_llm
#                                  imports mpi4py at module scope
#   site-packages/nvidia/*/lib     the CUDA runtime wheels
#   site-packages/nvidia/cu13/lib  libcublas.so.13 -- tensorrt_llm 1.2.1 is
#                                  built against CUDA 13 while pip's default
#                                  torch is 2.9.1+cu128, so BOTH cublas major
#                                  versions have to be present. Different
#                                  sonames, so they coexist.
#   site-packages/torch/lib        libtorch*.so, libc10*.so
#   site-packages/tensorrt_libs    libnvinfer.so.10
#   <base python>/lib              libpython3.12.so.1.0
#
# What it cannot supply is the driver -- libcuda.so.1 and libnvidia-ml.so.1 come
# from /usr/lib64 on a GPU node and do not exist on the login node. That is the
# whole reason this needs a job.
trtllm_ldpath() {
    local sp d out
    sp="$(ls -d "$TRTLLM_VENV"/lib/python3.*/site-packages 2>/dev/null | head -1)"
    out="$TRTLLM_VENV/lib"
    # the base interpreter's lib dir, for libpython
    local basedir
    basedir="$(sed -n 's/^home = //p' "$TRTLLM_VENV/pyvenv.cfg" 2>/dev/null)"
    [[ -d "$basedir/../lib" ]] && out="$out:$(cd "$basedir/../lib" && pwd)"
    for d in "$sp"/nvidia/*/lib "$sp"/torch/lib "$sp"/tensorrt_libs "$sp"/nvidia/cu13/lib; do
        [[ -d "$d" ]] && out="$out:$d"
    done
    echo "$out"
}
export LD_LIBRARY_PATH="$(trtllm_ldpath)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
say "--- LD_LIBRARY_PATH has $(tr ':' '\n' <<< "$LD_LIBRARY_PATH" | grep -c .) entries ---"

# Say what is still unresolved before importing, so a failure names the library
# rather than arriving as a traceback thirty frames deep.
_b="$(ls "$TRTLLM_VENV"/lib/python3.*/site-packages/tensorrt_llm/bindings*.so 2>/dev/null | head -1)"
if [[ -n "$_b" ]]; then
    say "--- ldd on the bindings ---"
    ldd "$_b" 2>/dev/null | grep "not found" | sort -u | sed 's/^/  MISSING /' \
        || echo "  ok every shared library resolves"
fi

say "--- import (this is where mpi4py and libcuda.so.1 are settled) ---"
"$TRTLLM_VENV/bin/python" -c '
import tensorrt_llm, torch, sys
print("  tensorrt_llm", tensorrt_llm.__version__)
print("  torch       ", torch.__version__, "cuda", torch.version.cuda)
print("  cuda avail  ", torch.cuda.is_available(), torch.cuda.device_count(), "device(s)")
' 2>&1 | tail -25
IMPORT_RC=${PIPESTATUS[0]}
say "import exited $IMPORT_RC"

if (( IMPORT_RC != 0 )); then
    echo ""
    echo "x tensorrt_llm did not import."
    echo ""
    echo "  Read the MISSING lines above before concluding anything. Each error so"
    echo "  far has been one named library, and each has had a fix:"
    echo "    cannot load MPI library   -> pip install openmpi into the venv"
    echo "    libcuda.so.1              -> login node has no driver; use a GPU node"
    echo "    libcublasLt.so.13         -> pip install nvidia-cublas (the CUDA 13 one)"
    echo "  A missing library is not the same finding as 'bare metal will not work'."
    exit 1
fi

# ---------------------------------------------------------------- the help
# This is the deliverable. Every flag in run/engines/trtllm.sh is a guess until
# these files exist; with them the guesses become one edit.
say "--- capturing the CLI ---"
{
    echo "### captured $(date -Is) on $(hostname) from $TRTLLM_VENV"
    echo "### tensorrt_llm $("$TRTLLM_VENV/bin/python" -c 'import tensorrt_llm;print(tensorrt_llm.__version__)' 2>/dev/null)"
    echo ""
    echo "################################ trtllm-serve --help"
    "$TRTLLM_VENV/bin/trtllm-serve" --help 2>&1
    for sub in serve disaggregated disaggregated_mpi_worker mm_embedding_serve bench eval; do
        echo ""
        echo "################################ trtllm-serve $sub --help"
        "$TRTLLM_VENV/bin/trtllm-serve" "$sub" --help 2>&1 || echo "(no such subcommand)"
    done
} > "$OUT/trtllm-serve-help.txt" 2>&1

wc -l "$OUT/trtllm-serve-help.txt"

say "--- subcommands ---"
grep -E '^\s{2,}[a-z_-]+\s' "$OUT/trtllm-serve-help.txt" | head -20

say "--- the flags run/engines/trtllm.sh currently assumes ---"
for f in --tp_size --backend --host --port --ep_size --served_model_name --extra_llm_api_options; do
    grep -q -- "$f" "$OUT/trtllm-serve-help.txt" \
        && echo "  ok  $f" || echo "  --  $f  NOT PRESENT"
done

say "--- every long flag the top-level serve command takes ---"
sed -n '/trtllm-serve --help/,/^####/p' "$OUT/trtllm-serve-help.txt" \
    | grep -oE '\-\-[a-zA-Z0-9_-]+' | sort -u | tr '\n' ' '
echo ""

say "--- anything about disaggregation ---"
grep -inE 'disagg|context_serv|generation_serv|cache_transceiver|kv.?transfer' \
    "$OUT/trtllm-serve-help.txt" | head -20

echo ""
echo "=============================================================="
echo " full help: $OUT/trtllm-serve-help.txt"
echo " Copy the real flag names into run/engines/trtllm.sh, then smoke test 1"
echo " will verify them for every later run instead of trusting them."
echo "=============================================================="
