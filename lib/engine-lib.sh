#!/bin/bash
# Engine selection. Sourced by the smoke tests, and (from Phase A) by the PD job.
#
#   ENGINE=trtllm sbatch run/01-smoke-image.sbatch
#
# One file per engine under run/engines/. Each defines the contract below; this
# file loads one, checks it is complete, and supplies the defaults.
#
# ---------------------------------------------------------------- the contract
#
# REQUIRED (Phase B -- the smoke tests):
#
#   eng_default_image          echo the default container image path
#   eng_import_check           echo a SHELL SNIPPET, run inside the container.
#                              Must exit non-zero if the engine is not usable.
#                              It is emitted as text rather than run directly
#                              because the repo is not necessarily mounted in
#                              the container -- the snippet travels, the file
#                              does not.
#   eng_serve_cmd M N P T      echo the argv of ONE non-disaggregated server,
#                              one argument per line, for `mapfile -t`:
#                                  M model path or HF id
#                                  N served model name
#                                  P port
#                                  T tensor-parallel degree
#
# OPTIONAL (defaults below):
#
#   eng_health_path            default /health
#   eng_models_path            default /v1/models
#   eng_metrics_path           default /metrics
#   eng_notes                  echo caveats; printed at load time
#
# ADDED IN PHASE A (PD deployment -- see docs/trtllm-plan.md):
#
#   eng_pd_prepare, eng_pd_worker_cmd, eng_pd_gateway_cmd, eng_metrics_urls
#
# The split is deliberate: pd_parse()/pd_place() in pd-lib.sh compute a placement
# that is engine-neutral -- 2x4:4x2 means the same GPUs whoever serves them --
# and each engine serialises those arrays into its own form. SGLang serialises
# to CLI arguments; TensorRT-LLM serialises to a config file. Get that seam right
# and run/sweep-pd.sbatch sweeps any engine without an edit, because the sweep
# only ever speaks PD_SPEC.

# Resolve run/engines/ relative to this file. BASH_SOURCE is unset outside bash,
# and an unresolved ENG_DIR turns every later error message into nonsense
# ("unknown engine 'sglang' -- no /sglang.sh"), so fail on it explicitly.
if [[ -z "${ENG_DIR:-}" ]]; then
    _eng_self="${BASH_SOURCE[0]:-$0}"
    ENG_DIR="$(cd "$(dirname "$_eng_self")/engines" 2>/dev/null && pwd)"
fi

eng_available() { ls "$ENG_DIR" 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' '; }

# Defaults. An engine file that defines its own wins, because these are sourced
# first and bash lets a later definition replace an earlier one.
eng_health_path()  { echo "/health"; }
eng_models_path()  { echo "/v1/models"; }
eng_metrics_path() { echo "/metrics"; }
eng_notes()        { :; }

eng_load() {   # $1=engine name (default $ENGINE, default sglang)
    local name="${1:-${ENGINE:-sglang}}"

    if [[ -z "${ENG_DIR:-}" || ! -d "$ENG_DIR" ]]; then
        echo "x cannot locate the engines directory (ENG_DIR='${ENG_DIR:-}')" >&2
        echo "  run these scripts with bash, or set ENG_DIR=<repo>/run/engines" >&2
        return 1
    fi

    local file="$ENG_DIR/$name.sh"
    if [[ ! -r "$file" ]]; then
        echo "x unknown engine '$name' -- no $file" >&2
        echo "  available: $(eng_available)" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "$file" || { echo "x failed to source $file" >&2; return 1; }

    local f missing=()
    for f in eng_default_image eng_import_check eng_serve_cmd; do
        declare -F "$f" >/dev/null || missing+=("$f")
    done
    if (( ${#missing[@]} )); then
        echo "x $file does not implement: ${missing[*]}" >&2
        echo "  see the contract at the top of run/engine-lib.sh" >&2
        return 1
    fi

    ENGINE="$name"
    return 0
}

# The image, in precedence order: an explicit ENGINE_IMG, then the engine's own
# default. SGLANG_IMG is still honoured for that engine because it is the name
# the application notes and every existing script use, and breaking muscle
# memory is a poor trade for a tidier variable.
eng_image() {
    if [[ -n "${ENGINE_IMG:-}" ]]; then echo "$ENGINE_IMG"
    elif [[ "${ENGINE:-}" == sglang && -n "${SGLANG_IMG:-}" ]]; then echo "$SGLANG_IMG"
    else eng_default_image
    fi
}

# WHICH INTERPRETER. Inside a container, `python3` is the image's python and the
# engine's console scripts are on PATH. Bare metal they are not: srun starts the
# worker without a login shell, so nothing has put a venv on PATH, and `python3`
# is whatever the OS ships (3.10.6 on NYCU, 3.13 from conda on the Firmus login
# node -- job 17593). Every engine file therefore names its interpreter instead
# of assuming one.
#
# TensorRT-LLM has always worked this way; NYCU forced it on the other two,
# because that cluster has no container runtime at all.
eng_venv() {   # echoes the venv root for the loaded engine, or nothing
    local v=""
    case "${ENGINE:-}" in
        sglang) v="${SGLANG_VENV:-}" ;;
        vllm)   v="${VLLM_VENV:-}" ;;
        trtllm) v="${TRTLLM_VENV:-}" ;;
        lmdeploy) v="${LMDEPLOY_VENV:-}" ;;
    esac
    [[ -n "$v" && -x "$v/bin/python3" ]] && echo "$v"
    return 0
}

eng_python() {   # the python3 to run the engine with
    local v; v="$(eng_venv)"
    [[ -n "$v" ]] && echo "$v/bin/python3" || echo python3
}

eng_exe() {   # $1=console script name, e.g. vllm / trtllm-serve
    local v; v="$(eng_venv)"
    [[ -n "$v" ]] && echo "$v/bin/$1" || echo "$1"
}

# WHAT A BARE-METAL ENGINE NEEDS THAT A CONTAINER ALREADY HAD. An engine image
# ships a CUDA toolkit and Python headers; a pip venv does not, and both are
# needed at RUN time because these engines compile kernels on first use:
#
#   * sglang's deep_gemm calls find_cuda_home(), which checks CUDA_HOME, then
#     CUDA_PATH, then `which nvcc`, then /usr/local/cuda -- and asserts if all
#     four miss. NYCU has no /usr/local/cuda. Job 79370 died on that assertion
#     15 seconds in. The wheels do ship a toolkit, under
#     site-packages/nvidia/cu13 (bin/nvcc, include/, lib/), so point at it.
#
#   * vLLM's triton JIT compiles cuda_utils.c and fails with
#     "Python.h: No such file or directory" when python3-dev is absent. It is
#     present on NYCU's LOGIN node and absent on the compute nodes -- the same
#     login-versus-compute trap that produced two wrong TensorRT-LLM diagnoses
#     on Firmus. CPATH adds a search path without needing to write /usr/include.
#
# Both are emitted as KEY=VALUE lines for the worker wrapper, which prefixes each
# with `export`. Only KEY=VALUE belongs here: a shell command would get an
# `export` glued to the front of it.
eng_cuda_home() {   # a directory with bin/nvcc and include/, or nothing
    local v d; v="$(eng_venv)"; [[ -n "$v" ]] || return 0
    for d in "$v"/lib/python3.*/site-packages/nvidia/*/; do
        [[ -x "$d/bin/nvcc" && -r "$d/include/cuda_runtime.h" ]] && { echo "${d%/}"; return 0; }
    done
    return 0
}

# The include directories of the CUDA tree we are compiling against -- and ONLY
# that tree.
#
# It is tempting to put every nvidia/*/include on the path, and jobs 79568/79569
# show why that is wrong. These venvs carry two CUDA flavours at once: the cu13
# packages install into nvidia/cu13/ while the cu12 ones (pulled by TensorRT-LLM's
# torch, which is built for cu128) install into nvidia/cuda_runtime/,
# nvidia/curand/ and so on. With all of them on CPATH, nvcc 13.4 reached
# nvidia/cuda_runtime/include/cooperative_groups.h -- the CUDA 12.8 copy -- and
# cccl stopped the build:
#
#   cuda_toolkit.h:41: error: "CUDA compiler and CUDA toolkit headers are
#                              incompatible, please check your include paths"
#
# A MIXED include path is worse than a missing header: the missing header names
# itself, while the mixed one produces a version error that points at neither
# culprit. So the path stays single-flavour, and a header that is absent gets fixed
# by installing the matching flavour of its package -- see run/venv-prepare.sh --
# not by widening the search.
eng_include_dirs() {   # colon-separated, or nothing
    local cu out=""
    cu="$(eng_cuda_home)"; [[ -n "$cu" ]] || return 0
    [[ -d "$cu/include" ]]      && out="$cu/include"
    [[ -d "$cu/include/cccl" ]] && out="${out:+$out:}$cu/include/cccl"
    echo "$out"
}

eng_venv_env() {   # KEY=VALUE lines every bare-metal engine wants
    local v cu inc nvinc
    v="$(eng_venv)"; [[ -n "$v" ]] || return 0
    cu="$(eng_cuda_home)"
    if [[ -n "$cu" ]]; then
        echo "CUDA_HOME=$cu"
        echo "CUDA_PATH=$cu"
        echo "PATH=$cu/bin:$v/bin:\${PATH}"
    else
        echo "PATH=$v/bin:\${PATH}"
    fi
    # Where we staged the Python headers, if anyone did. A COLON-SEPARATED LIST,
    # because Debian's /usr/include/python3.10/pyconfig.h is only a stub that
    # does `#include <x86_64-linux-gnu/python3.10/pyconfig.h>` -- staging one
    # directory yields a Python.h that is found and then fails to compile, which
    # is what job 79389's first cell printed four times. Both the package
    # directory and its parent have to be reachable.
    nvinc="$(eng_include_dirs)"
    inc="${ENG_PY_INCLUDE:-}"
    [[ -n "$nvinc" ]] && inc="${inc:+$inc:}$nvinc"
    if [[ -n "$inc" ]]; then
        echo "CPATH=$inc\${CPATH:+:\${CPATH}}"
        local first="${ENG_PY_INCLUDE%%:*}"
        [[ -z "$first" || -r "$first/Python.h" ]] \
            || echo "# ENG_PY_INCLUDE=$ENG_PY_INCLUDE has no Python.h -- triton JIT will fail" >&2
    fi
    # Staged libpython, needed at BOTH link time and run time. TensorRT-LLM's
    # __init__.py calls _preload_python_lib(), which dlopens libpython3.10.so --
    # absent from NYCU's compute nodes, so job 79389's TRT-LLM cell died in five
    # seconds with OSError before any of its own code ran.
    if [[ -n "${ENG_PY_LIB:-}" ]]; then
        echo "LIBRARY_PATH=$ENG_PY_LIB\${LIBRARY_PATH:+:\${LIBRARY_PATH}}"
        echo "LD_LIBRARY_PATH=$ENG_PY_LIB\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
    fi
    return 0
}

# A container that mounts the real $HOME inherits its .bashrc, and on this
# cluster that means conda's init prepends ~/miniforge3/bin to PATH -- so
# `python3` inside the container is the USER'S python, not the image's, and
# nothing the image ships imports. Job 17593 burned eleven minutes proving it.
#
# So every run gets an empty directory as $HOME, exactly as the reference scripts'
# mounts() does. The work directory is bound at its own path so logs and
# artefacts written from inside still land where the caller expects.
eng_mounts() {   # $1=a name for this run's private home
    local base="${ENG_HOME_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/.homes}/$1"
    mkdir -p "$base" 2>/dev/null
    local m="${MOUNTS_BASE:-/scratch:/scratch},${base}:${HOME}"
    local w="${SLURM_SUBMIT_DIR:-}"
    [[ -n "$w" && "$w" != /scratch* ]] && m="${m},${w}:${w}"
    echo "$m"
}

eng_banner() {
    echo "=========================================="
    echo " engine: ${ENGINE:-?}"
    echo " image:  $(eng_image)"
    echo " mounts: ${MOUNTS:-<unset>}"
    local n; n="$(eng_notes)"
    [[ -n "$n" ]] && { echo " ---"; sed 's/^/ ! /' <<< "$n"; }
    echo "=========================================="
}

# Poll until healthy, but give up the moment the server process dies -- a server
# that exited in the first ten seconds otherwise costs you the full timeout, and
# the log is what you want to see, not ten minutes of silence.
eng_wait_healthy() {   # $1=base url  $2=server pid  $3=timeout  $4=log path
    local url="$1" pid="$2" timeout="$3" log="${4:-}" waited=0 hp
    hp="$(eng_health_path)"
    while (( waited < timeout )); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "x the server exited after ${waited}s, before becoming healthy" >&2
            [[ -r "$log" ]] && tail -30 "$log" >&2
            return 1
        fi
        if curl -sf --max-time 5 "${url}${hp}" >/dev/null 2>&1; then
            echo "  ok healthy after ${waited}s"; return 0
        fi
        sleep 5; waited=$(( waited + 5 ))
    done
    echo "x never became healthy in ${timeout}s" >&2
    [[ -r "$log" ]] && tail -30 "$log" >&2
    return 1
}

# /health alone only says the process is up. What it thinks it is serving is a
# separate question, and a wrong served-model-name here becomes a wrong --model
# at benchmark time.
eng_model_id() {   # $1=base url
    curl -sf --max-time 10 "$1$(eng_models_path)" 2>/dev/null \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}
