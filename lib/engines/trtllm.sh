#!/bin/bash
# TensorRT-LLM. The intended scored stack.
#
# VERIFIED 2026-09-16 against tensorrt_llm 1.2.1 on this cluster (job 17604).
# The captured CLI is in results/trtllm-help-17604/trtllm-serve-help.txt; re-capture
# with run/04-trtllm-probe.sbatch after any version change.
#
# Not a container. TensorRT-LLM is pip-installed into a venv on shared storage --
# nvcr.io is unreachable from this cluster and /scratch/enroot-images is root-owned,
# so an image was never an option. The rules allow this explicitly: "container image
# plus build files, OR a self-built setup on shared storage".
#
# Four things the venv needs that pip does not arrange, all found the hard way:
#
#   1. Python 3.12. tensorrt-llm 1.2.1 ships cp310 and cp312 wheels only; the
#      cluster's default python3 is conda's 3.13 and the install simply fails.
#   2. pip install openmpi INTO the venv. tensorrt_llm imports mpi4py at module
#      scope and this cluster has no system MPI at all.
#   3. pip install nvidia-cublas -- the UNSUFFIXED name, which is the CUDA 13
#      build. tensorrt_llm 1.2.1 is compiled against CUDA 13 while pip's default
#      torch is 2.9.1+cu128, so libcublas.so.12 and .so.13 must BOTH be present.
#      Different sonames, so they coexist.
#   4. LD_LIBRARY_PATH covering ~20 directories -- see trtllm_ldpath() in
#      run/04-trtllm-probe.sbatch. With nothing set, ldd on the bindings reports
#      twelve missing libraries, ten of which are already in the venv.
#   5. ray[default]==2.53.0, PINNED. TLLM_DISABLE_MPI does not mean "no distributed
#      runtime" -- it means "use Ray instead of MPI", and 1.2.1 imports
#          from ray.util.placement_group import PlacementGroupSchedulingStrategy
#      which Ray moved to ray.util.scheduling_strategies somewhere between
#      2.53.0 and 2.58.0. Installing plain `ray` gets the latest and fails at
#      import (job 18316). 2.40/2.46/2.49/2.53 all still carry the old path;
#      2.53.0 is the newest that does.
#
# The whole install, reproducibly:
#
#   <python3.12> -m venv $TRTLLM_VENV
#   $TRTLLM_VENV/bin/pip install --extra-index-url https://pypi.nvidia.com \
#       tensorrt-llm openmpi nvidia-cublas "ray[default]==2.53.0"
#
#   ray[default], NOT plain ray. Plain ray pulls only click/filelock/jsonschema/
#   msgpack/packaging/protobuf/pyyaml/requests, leaving out virtualenv,
#   opencensus and smart_open -- the Runtime Env Agent's own dependencies.
#   Without them the agent never starts, the raylet times out connecting to it
#   after 30 s and exits, and every worker then dies with
#       Failed to register worker to Raylet: IOError: ... End of file
#   a symptom five jobs were spent chasing. The raylet said so in raylet.err
#   from the very first attempt:
#       The raylet exited immediately because the runtime env agent timed out
#   Nobody read it until job 19386 copied the file off the compute node.
#
# What the venv cannot supply is the driver: libcuda.so.1 and libnvidia-ml.so.1
# come from /usr/lib64 on a GPU node. Nothing here runs on the login node.

# Left EMPTY on purpose, and resolved at use time by trtllm_bin(). This file is
# sourced by eng_load(), which sets ENGINE only AFTER sourcing -- so eng_exe()
# called here cannot yet know which engine it is being asked about and returns the
# bare name. Job 79370's TRT-LLM cell died in five seconds with
# "trtllm-serve: command not found" for exactly that reason: srun starts the
# worker with no login shell, so $TRTLLM_VENV/bin is not on PATH.
TRTLLM_BIN="${TRTLLM_BIN:-}"
trtllm_bin() { echo "${TRTLLM_BIN:-$(eng_exe trtllm-serve)}"; }
TRTLLM_TP_FLAG="${TRTLLM_TP_FLAG:---tp_size}"
TRTLLM_BACKEND_FLAG="${TRTLLM_BACKEND_FLAG:---backend}"
TRTLLM_BACKEND="${TRTLLM_BACKEND:-pytorch}"
# CONFIRMED ABSENT in 1.2.1: there is no --served_model_name. The served id comes
# from the model path, and the smoke tests read it back from /v1/models, which is
# what --model must be at benchmark time. Left configurable in case a later
# version adds one.
TRTLLM_NAME_FLAG="${TRTLLM_NAME_FLAG:-}"
TRTLLM_EXTRA="${TRTLLM_EXTRA:-}"                # whitespace-split, appended verbatim

# EMPTY ON PURPOSE. There is no TensorRT-LLM container on this cluster and no
# way to get one: nvcr.io is unreachable and /scratch/enroot-images is
# root-owned. This engine is a pip venv on shared storage, which the rules
# explicitly permit, and pd_launch runs it without --container-image. Set
# TRTLLM_IMG if a container ever appears.
eng_default_image() { echo "${TRTLLM_IMG:-}"; }

# THE TRAP. TensorRT-LLM serves three different things and only one of them is
# Prometheus (tensorrt_llm/serve/openai_server.py, 1.2.1):
#
#   /metrics             iteration stats, as JSON -- NOT Prometheus text
#   /perf_metrics        per-request perf records
#   /prometheus/metrics  the actual Prometheus endpoint
#
# observe.sh scrapes Prometheus text. Pointed at /metrics it would parse JSON as
# metrics, find nothing, and write an empty engine_metrics.csv -- with no error,
# which is precisely the failure checklist items 7 and 8 exist to catch.
eng_metrics_path() { echo "/prometheus/metrics"; }

eng_notes() {
    cat <<'N'
pip venv, not a container -- needs LD_LIBRARY_PATH set (see 04-trtllm-probe.sbatch)
and a GPU node for libcuda.so.1. Metrics are at /prometheus/metrics, not /metrics.
N
}

eng_import_check() {
    # Interpolate the flag names into the snippet rather than relying on the
    # environment surviving into the container.
    printf 'BIN=%q\nTP_FLAG=%q\nBACKEND_FLAG=%q\n' \
        "$(trtllm_bin)" "$TRTLLM_TP_FLAG" "$TRTLLM_BACKEND_FLAG"
cat <<'SNIP'
python3 -c "
import tensorrt_llm, sys
print('  python      ', sys.version.split()[0])
print('  tensorrt_llm', getattr(tensorrt_llm, '__version__', 'imported'))
" || { echo "x tensorrt_llm did not import" >&2; exit 1; }

command -v "$BIN" >/dev/null 2>&1 || { echo "x no '$BIN' on PATH" >&2; exit 1; }
HELP="$("$BIN" --help 2>&1)" || { echo "x '$BIN --help' failed" >&2; exit 1; }
echo "  ok $BIN runs"

# The point of this block: every flag eng_serve_cmd is about to use, checked
# against the help text that this image actually prints.
rc=0
for f in "$TP_FLAG" "$BACKEND_FLAG" --host --port; do
    if grep -q -- "$f" <<< "$HELP"; then
        echo "  ok $f"
    else
        echo "  x  $f is NOT in this build's --help" >&2
        rc=1
    fi
done
if (( rc )); then
    echo "" >&2
    echo "  The flag names in run/engines/trtllm.sh do not match this image." >&2
    echo "  Nearest candidates in its --help:" >&2
    grep -oE '^\s+--[a-z0-9_-]+' <<< "$HELP" | tr -d ' ' | sort -u | head -40 | sed 's/^/    /' >&2
    echo "  Fix the TRTLLM_*_FLAG variables at the top of that file." >&2
    exit 1
fi

# Phase A needs these; report now rather than in a PD job.
echo "  --- disaggregation (Phase A) ---"
# The || has to hang off grep itself: through a pipe it would attach to sed,
# which succeeds on empty input, and the warning would never fire.
if grep -qi disagg <<< "$HELP"; then
    grep -i disagg <<< "$HELP" | head -5 | sed 's/^/  /'
else
    echo "  !  nothing in this --help mentions disaggregation."
    echo "     It may be a subcommand: check 'trtllm-serve disaggregated --help'."
fi
python3 -c "import nixl" 2>/dev/null \
    && echo "  ok nixl importable" || echo "  !  nixl not importable"
command -v ucx_info >/dev/null 2>&1 \
    && echo "  ok ucx present" || echo "  !  no ucx_info -- check the cache transceiver backend"
SNIP
}

eng_serve_cmd() {   # model name port tp
    local cmd=("$(trtllm_bin)" "$1")
    [[ -n "$TRTLLM_BACKEND" ]] && cmd+=("$TRTLLM_BACKEND_FLAG" "$TRTLLM_BACKEND")
    [[ -n "$TRTLLM_NAME_FLAG" ]] && cmd+=("$TRTLLM_NAME_FLAG" "$2")
    cmd+=("$TRTLLM_TP_FLAG" "$4")
    # Written by eng_pd_prepare from the variant knobs; absent means "defaults".
    [[ -n "${PD_WORK:-}" && -r "$PD_WORK/llm-solo.yaml" ]] \
        && cmd+=(--extra_llm_api_options "$PD_WORK/llm-solo.yaml")
    cmd+=(--host 0.0.0.0 --port "$3")
    # shellcheck disable=SC2206
    [[ -n "$TRTLLM_EXTRA" ]] && cmd+=($TRTLLM_EXTRA)
    printf '%s\n' "${cmd[@]}"
}

# ---------------------------------------------------------------- Phase A notes
# The disaggregated architecture, read out of tensorrt_llm 1.2.1's own source on
# 2026-09-16 (commands/serve.py, llmapi/disagg_utils.py, llmapi/llm_args.py) and
# cross-checked against the help captured by job 17604.
#
# THE GATEWAY DOES NOT START WORKERS. `trtllm-serve disaggregated -c cfg.yaml`
# parses the config, binds the gateway socket, and runs OpenAIDisaggServer, which
# proxies to the addresses in `urls:`. The workers must already be listening.
# That is the same shape as sglang_router, so pd_launch() + pd_launch_router()
# carry over unchanged.
#
# WORKERS are ordinary `trtllm-serve` processes that declare their role:
#
#   trtllm-serve "$MODEL" --backend pytorch \
#       --server_role CONTEXT|GENERATION \      # ServerRole(IntEnum) in disagg_utils
#       --tp_size <tp> --ep_size <ep> \
#       --max_seq_len "$CONTEXT_LEN" \
#       --kv_cache_free_gpu_memory_fraction "$MEM_FRACTION" \
#       --extra_llm_api_options <per-role yaml> \
#       --host 0.0.0.0 --port <port>
#
# CONTEXT is prefill, GENERATION is decode. Note the memory fraction is a
# fraction of what is LEFT AFTER the weights load -- not of total memory, the way
# SGLang's --mem-fraction-static is. Same number, different KV cache.
#
# THE KV TRANSCEIVER is a CacheTransceiverConfig (llm_args.py:1807), a field of
# the LLM args, not of the disagg config:
#
#   cache_transceiver_config:
#     backend: UCX            # DEFAULT | UCX | NIXL | MOONCAKE | MPI
#     max_tokens_in_buffer: <n>
#     kv_transfer_timeout_ms: <ms>
#
# Two places it can be supplied, and which one depends on how workers start:
#   * workers launched independently (what we do)  -> each worker's
#     --extra_llm_api_options yaml
#   * workers launched by `disaggregated_mpi_worker -c <disagg.yaml>` -> the
#     per-server block of the disagg config, because extract_ctx_gen_cfgs()
#     sweeps every unrecognised key into other_args and serve.py:773 feeds that
#     straight into get_llm_args()
# TO CONFIRM ON HARDWARE: that an independently launched worker with
# --server_role and a cache_transceiver_config actually pairs with the gateway.
#
# THE GATEWAY CONFIG that eng_pd_prepare has to write:
#
#   hostname: <gateway ip>
#   port: 8000
#   context_servers:
#     num_instances: 4
#     urls: ["10.0.0.1:30100", "10.0.0.1:30101", ...]
#     tensor_parallel_size: 2
#   generation_servers:
#     num_instances: 2
#     urls: ["10.0.0.4:30200", ...]
#     tensor_parallel_size: 4
#
# extract_ctx_gen_cfgs() requires len(urls) == num_instances and computes
# instance_num_ranks = tp * pp * cp -- the same arithmetic pd_place() does. So
# eng_pd_prepare is a pure serialisation of PD_ROLE/PD_IP/PD_PORT/PD_TP:
# prefill entries become context_servers, decode entries generation_servers.

# ---------------------------------------------------------------- PD (Phase A)
# The same placement arrays, serialised into TensorRT-LLM's form -- which is a
# YAML file rather than command-line arguments. That difference is the reason
# eng_pd_prepare exists in the contract at all.

TRTLLM_VENV="${TRTLLM_VENV:-/scratch/home/$USER/hsingyu/trtllm-venv}"
TRTLLM_KV_BACKEND="${TRTLLM_KV_BACKEND:-UCX}"       # DEFAULT|UCX|NIXL|MOONCAKE|MPI
TRTLLM_KV_BUFFER_TOKENS="${TRTLLM_KV_BUFFER_TOKENS:-}"
TRTLLM_KV_TIMEOUT_MS="${TRTLLM_KV_TIMEOUT_MS:-}"

# A pip-installed TensorRT-LLM sets up no library path of its own: ldd on the
# bindings reports twelve missing libraries, ten of which are already in the
# venv. Measured 2026-09-16; see run/04-trtllm-probe.sbatch for the same logic.
eng_pd_env() {
    # The toolkit and Python headers a container would have shipped -- see
    # eng_venv_env() in engine-lib.sh. Emitted first so the LD_LIBRARY_PATH built
    # below still wins on library resolution.
    eng_venv_env
    local sp base out
    sp="$(ls -d "$TRTLLM_VENV"/lib/python3.*/site-packages 2>/dev/null | head -1)"
    out="$TRTLLM_VENV/lib"
    base="$(sed -n 's/^home = //p' "$TRTLLM_VENV/pyvenv.cfg" 2>/dev/null)"
    [[ -d "$base/../lib" ]] && out="$out:$(cd "$base/../lib" && pwd)"
    local d
    for d in "$sp"/nvidia/*/lib "$sp"/torch/lib "$sp"/tensorrt_libs; do
        [[ -d "$d" ]] && out="$out:$d"
    done
    echo "LD_LIBRARY_PATH=\"$out\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""

    # Job 17817: TensorRT-LLM's MpiPoolSession creates its workers with
    # MPI_Comm_spawn, and a pip-installed OpenMPI cannot spawn under Slurm --
    #   OPAL ERROR: dpm.c line 1997
    #   MPI.COMM_SELF.Spawn(...) -> MPI_ERR_UNKNOWN
    # Dynamic process management needs a PMI-enabled launcher, which `pip
    # install openmpi` does not provide.
    #
    # TLLM_DISABLE_MPI=1 (tensorrt_llm/_utils.py:535) takes the no-MPI path.
    #
    # LIMIT, not yet tested: this can only be right while each worker is a
    # single rank. TP>1 needs real ranks, which means launching the worker under
    # `srun --mpi=pmix -n <tp>` or using `trtllm-serve disaggregated_mpi_worker`
    # instead. The smoke test runs TP=1 precisely to separate these two problems.
    [[ "${TRTLLM_DISABLE_MPI:-1}" == 1 ]] && echo 'TLLM_DISABLE_MPI=1'

    # Job 19056: Ray now starts (2.53.0 fixed the import) but the worker cannot
    # register --
    #   Ray cluster not found, starting a new one.
    #   Failed to register worker to Raylet: IOError: ... End of file
    #
    # Each of our workers is its own process on a shared node, so each one finds
    # no cluster and starts "a new one" -- four local Ray instances colliding on
    # Ray's default temp dir and port range. Give every worker its own, the same
    # way TRITON_CACHE_DIR is already per worker.
    #
    # $role and $port are set by the wrapper before this line runs, so these
    # expand per worker at launch rather than being baked in here.
    # RAY_TMPDIR also becomes the object store's location: /dev/shm here is 63 MB
    # and Ray falls back to the temp dir on its own, with a warning. That warning
    # is expected and is NOT the failure -- the failure was the descriptor limit,
    # raised in the wrapper.
    echo 'RAY_TMPDIR="/tmp/ray-$role-$port"'
    echo 'RAY_ADDRESS="local"'
    # Make Ray say why, rather than costing another queue slot to guess.
    echo 'RAY_BACKEND_LOG_LEVEL=debug'
    echo 'RAY_DEDUP_LOGS=0'
    # Ray picks its ports from a range; keep the ranges disjoint per worker so
    # two instances on one node cannot claim the same one.
    #
    # NOT `port % 100`: the placement uses 30100-30103 for prefill and
    # 30200-30201 for decode, so modulo 100 maps 30100 and 30200 to the same 0
    # and the two would collide -- which is the bug this is meant to prevent.
    # port-30000 keeps them distinct (100..103, 200..201).
    # BOUNDED, not linear in the port. The old form was 20000 + (port-30000)*60,
    # which is fine while ports start at 30000/30100/30200 and overflows past
    # 65535 the moment a second cell moves its base to 31000 or 32000 -- port
    # 32000 would have asked Ray for 140000. Take a slot from the port's hundreds
    # digit instead: 30000 -> 20000, 30100 -> 20060, 31000 -> 20600, and the
    # whole range stays inside 20000-25999.
    echo 'RAY_min_worker_port=$(( 20000 + ((port / 100) % 100) * 60 ))'
    echo 'RAY_max_worker_port=$(( 20000 + ((port / 100) % 100) * 60 + 49 ))'
    # No mkdir here: pd_write_launcher prefixes every line of this output with
    # `export`, so only KEY=VALUE belongs in it. Ray creates RAY_TMPDIR itself.
}

# Written once per deployment into $PD_WORK:
#
#   disagg.yaml   what the gateway reads. context_servers / generation_servers,
#                 each with num_instances and the urls the workers listen on.
#                 extract_ctx_gen_cfgs() requires len(urls) == num_instances.
#   llm-ctx.yaml  --extra_llm_api_options for the prefill workers
#   llm-gen.yaml  the same for decode
#
# cache_transceiver_config is a field of the LLM args (llm_args.py:1807), not of
# the disagg config. Because we launch each worker ourselves rather than through
# `disaggregated_mpi_worker -c disagg.yaml`, it belongs in the per-role file.

# The tuning knobs TensorRT-LLM takes through --extra_llm_api_options rather than
# on the command line. Written to a file because that is the only route the LLM API
# offers; the campaign calls eng_pd_prepare before every cell, solo included, so
# $PD_WORK exists by the time this runs.
#
# CUDA GRAPHS ARE THE REASON THIS EXISTS. Job 79619/79620 served every request and
# cleared nothing: ITL p90 24-32 ms against a 20 ms gate, while SGLang on the same
# four GPUs held 15.5 ms. The engine log said why, four times over:
#
#   CUDA Graph is empty
#
# SGLang captures 52 decode batch sizes by default and TensorRT-LLM captures none,
# so every decode step pays 94 layers of kernel launches. That is the right order
# of magnitude for the 9 ms difference, which makes it the first thing to try --
# and it means the untuned number is not a verdict on the engine.
trtllm_write_opts() {   # $1=path
    local f="$1" wrote=0
    {
        echo "# generated from the campaign's variant knobs -- do not edit"
        # OPT-IN, deliberately. TensorRT-LLM serves nothing at /prometheus/metrics
        # unless return_perf_metrics is set -- which is why every trtllm run so far
        # produced an empty engine_metrics.csv, and aiperf's own warning named the
        # fix. But it makes the engine collect per-request perf metrics, which is
        # work, and this engine currently sits 0.89 ms from a 20 ms ITL gate. Paying
        # unknown latency for diagnostics we do not need in order to rank engines is
        # the wrong trade, and turning it on unconditionally would also give `base`
        # an options file it never had in jobs 79619/79620 -- the control would stop
        # being the control. So: RETURN_PERF_METRICS=1 when the metrics are the
        # point, off when the number is.
        [[ "${RETURN_PERF_METRICS:-0}" == 1 ]] && { echo "return_perf_metrics: true"; wrote=1; }
        if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
            echo "cuda_graph_config:"
            echo "  max_batch_size: $CUDA_GRAPH_MAX_BS"
            echo "  enable_padding: ${CUDA_GRAPH_PADDING:-true}"
            wrote=1
        fi
        if [[ -n "${EP_SIZE:-}" ]]; then
            echo "moe_expert_parallel_size: $EP_SIZE"; wrote=1
        fi
        # MEM_FRACTION is emitted only when it DIFFERS from TensorRT-LLM's own
        # default of 0.90. Otherwise `base` would acquire an options file it did not
        # have in jobs 79619/79620, and the control would stop being the control.
        local want_kv=0
        [[ -n "${KV_CACHE_DTYPE:-}" ]] && want_kv=1
        [[ -n "${MEM_FRACTION:-}" && "${MEM_FRACTION}" != "0.90" && "${MEM_FRACTION}" != "0.9" ]] && want_kv=1
        if (( want_kv )); then
            echo "kv_cache_config:"
            [[ -n "${MEM_FRACTION:-}" && "${MEM_FRACTION}" != "0.90" && "${MEM_FRACTION}" != "0.9" ]] \
                && echo "  free_gpu_memory_fraction: $MEM_FRACTION"
            # TensorRT-LLM spells the fp8 KV dtype fp8, not fp8_e4m3.
            [[ -n "${KV_CACHE_DTYPE:-}" ]] && echo "  dtype: ${KV_CACHE_DTYPE/fp8_e4m3/fp8}"
            wrote=1
        fi
        if [[ -n "${MAX_RUNNING_SOLO:-}" ]]; then
            echo "max_batch_size: $MAX_RUNNING_SOLO"; wrote=1
        fi

        # ---- the decode-side knobs, read out of llm_args.py rather than guessed
        #
        # TensorRT-LLM's whole deficit on this checkpoint is ITL -- 23.8-30.5 ms
        # against a 20 ms gate, while its TTFT BEATS SGLang's. So the knobs worth an
        # allocation are the ones on the decode path, and two of them are chosen by
        # what create_moe.py actually gates on:
        #
        #   moe_config.backend defaults to CUTLASS, the generic path. TRTLLM
        #   (TRTLLMGenFusedMoE) and CUTEDSL (CuteDslFusedMoE) are each guarded by
        #   `quant_config.quant_mode.has_fp8_block_scales()` -- which is exactly this
        #   checkpoint's quantisation, weight_block_size [128,128] / fmt e4m3. Those
        #   two are the specialised paths for what we are running.
        #
        #   BEWARE THE SILENT FALLBACK. If the guard fails, create_moe.py logs
        #   "Using CutlassFusedMoE instead" and carries on. A mislabelled backend does
        #   not error, it reports Cutlass numbers under another name -- so
        #   trtllm_check_moe() below greps for that line and the campaign says so.
        [[ -n "${MOE_BACKEND:-}" ]] && { echo "moe_config:"; echo "  backend: $MOE_BACKEND"; wrote=1; }
        # enable_attention_dp defaults to false. At TP4 the attention work is split
        # across the group; data-parallel attention replicates it instead, which is
        # the trade that sets ITL rather than throughput.
        [[ -n "${ATTENTION_DP:-}" ]] && { echo "enable_attention_dp: $ATTENTION_DP"; wrote=1; }
        # attn_backend is ALREADY 'TRTLLM' by default -- an earlier note of mine
        # assumed otherwise. The untested value is FLASHINFER, which is the library
        # SGLang's fa3 path comes from.
        [[ -n "${ATTN_BACKEND:-}" ]] && { echo "attn_backend: $ATTN_BACKEND"; wrote=1; }
    } > "$f"
    (( wrote )) || { rm -f "$f"; return 1; }
    return 0
}

eng_pd_prepare() {   # $1=workdir
    local w="${1:?eng_pd_prepare needs a work directory}" i
    mkdir -p "$w" || return 1

    # The solo path calls this too, and then there is no PD placement to serialise
    # -- only the options file. Returning success is correct: there is nothing else
    # for this engine to prepare.
    trtllm_write_opts "$w/llm-solo.yaml" \
        && echo "  wrote $w/llm-solo.yaml" >&2 \
        || true
    if [[ "${PD_ROLE[0]:-}" == solo || "${#PD_ROLE[@]}" -eq 0 ]]; then
        return 0
    fi

    # The disagg gateway is configured by address, so the placement must already be
    # resolved. Say so, rather than letting `set -u` report PD_IP as an unbound
    # variable from inside a loop (job 215874).
    if (( ${#PD_IP[@]} == 0 )); then
        echo "x eng_pd_prepare: PD_IP is empty -- call pd_resolve before this" >&2
        return 1
    fi
    local ctx_urls=() gen_urls=() ctx_tp=0 gen_tp=0 ctx_n=0 gen_n=0
    for i in "${!PD_ROLE[@]}"; do
        if [[ "${PD_ROLE[$i]}" == prefill ]]; then
            ctx_urls+=("${PD_IP[$i]}:${PD_PORT[$i]}"); ctx_tp="${PD_TP[$i]}"; (( ctx_n++ ))
        else
            gen_urls+=("${PD_IP[$i]}:${PD_PORT[$i]}"); gen_tp="${PD_TP[$i]}"; (( gen_n++ ))
        fi
    done
    (( ctx_n && gen_n )) || { echo "x eng_pd_prepare: need both roles placed" >&2; return 1; }

    _yaml_urls() { local u; for u in "$@"; do echo "    - \"$u\""; done; }

    {
        echo "# generated by eng_pd_prepare from the placement -- do not edit"
        echo "hostname: ${PD_IP[0]}"
        echo "port: ${ROUTER_PORT:-8000}"
        echo "max_retries: 1"
        echo "context_servers:"
        echo "  num_instances: $ctx_n"
        echo "  tensor_parallel_size: $ctx_tp"
        echo "  urls:"; _yaml_urls "${ctx_urls[@]}"
        echo "generation_servers:"
        echo "  num_instances: $gen_n"
        echo "  tensor_parallel_size: $gen_tp"
        echo "  urls:"; _yaml_urls "${gen_urls[@]}"
    } > "$w/disagg.yaml"

    local role
    for role in ctx gen; do
        {
            echo "# --extra_llm_api_options for the $role workers"
            echo "cache_transceiver_config:"
            echo "  backend: $TRTLLM_KV_BACKEND"
            [[ -n "$TRTLLM_KV_BUFFER_TOKENS" ]] && echo "  max_tokens_in_buffer: $TRTLLM_KV_BUFFER_TOKENS"
            [[ -n "$TRTLLM_KV_TIMEOUT_MS" ]]    && echo "  kv_transfer_timeout_ms: $TRTLLM_KV_TIMEOUT_MS"
        } > "$w/llm-$role.yaml"
    done

    unset -f _yaml_urls
    echo "  wrote $w/disagg.yaml (ctx ${ctx_n}x tp$ctx_tp, gen ${gen_n}x tp$gen_tp)" >&2
    return 0
}

eng_pd_worker_cmd() {   # $1=index
    local i="$1" role opts
    if [[ "${PD_ROLE[$i]}" == prefill ]]; then role=CONTEXT; opts="$PD_WORK/llm-ctx.yaml"
    else                                        role=GENERATION; opts="$PD_WORK/llm-gen.yaml"; fi

    local -a cmd=(
        "$TRTLLM_VENV/bin/trtllm-serve" "$MODEL_PATH"
        "$TRTLLM_BACKEND_FLAG" "$TRTLLM_BACKEND"
        --server_role "$role"
        "$TRTLLM_TP_FLAG" "${PD_TP[$i]}"
        --max_seq_len "$CONTEXT_LEN"
        # NOT the same meaning as SGLang's --mem-fraction-static: this is a
        # fraction of what is LEFT AFTER the weights load, not of total memory.
        --kv_cache_free_gpu_memory_fraction "$MEM_FRACTION"
        --extra_llm_api_options "$opts"
    )
    # --ep_size IS NOT DEFAULTED TO THE TP DEGREE. This line used to read
    # ${DECODE_EP:-${PD_TP[$i]}}, which is the same mistake SGLang makes for a PD
    # decode worker and which cost 2.27x there: 986 -> 2,240 good tok/s on removing
    # it (jobs 214760 -> 214830). Expert parallelism replaces the MoE layer's
    # TP-style reduction with dispatch/combine all-to-alls, and at these batch sizes
    # those measured dearer -- 188 collectives per step at a ~25 us floor is 4.3-5.2
    # ms on every decode step. It is opt-in here now, as in sglang.sh and vllm.sh,
    # so an untested engine does not start out carrying a defect we already paid for.
    [[ "${PD_ROLE[$i]}" == decode && -n "${DECODE_EP:-}" ]] && cmd+=(--ep_size "$DECODE_EP")
    [[ -n "$TRTLLM_EXTRA" ]] && cmd+=($TRTLLM_EXTRA)
    cmd+=(--host 0.0.0.0 --port "${PD_PORT[$i]}")
    printf '%s\n' "${cmd[@]}"
}

# The gateway is a pure proxy: it parses the config, binds the socket and
# forwards to the urls. The workers must already be listening -- the same shape
# as sglang_router, which is why pd_launch_gateway serves both unchanged.
eng_pd_gateway_cmd() {   # $1=port
    printf '%s\n' "$TRTLLM_VENV/bin/trtllm-serve" disaggregated \
        -c "$PD_WORK/disagg.yaml" -t "${HEALTH_TIMEOUT:-600}"
}

# Did the MoE backend we asked for actually get used? create_moe.py falls back to
# CutlassFusedMoE with a warning rather than failing, so without this a run labelled
# "moetrtllm" can quietly be another copy of base.
trtllm_check_moe() {   # $1=engine log dir
    local d="$1" hit
    [[ -n "${MOE_BACKEND:-}" ]] || return 0
    hit="$(grep -rhoE 'Using CutlassFusedMoE instead' "$d" 2>/dev/null | head -1)"
    if [[ -n "$hit" ]]; then
        echo "  ! MOE_BACKEND=$MOE_BACKEND was REFUSED -- the engine fell back to" >&2
        echo "    CutlassFusedMoE, so this cell is a duplicate of base under another name." >&2
        return 1
    fi
    echo "  ok moe backend $MOE_BACKEND accepted (no Cutlass fallback in the log)" >&2
    return 0
}
