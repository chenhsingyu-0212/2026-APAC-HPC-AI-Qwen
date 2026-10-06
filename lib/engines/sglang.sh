#!/bin/bash
# SGLang. The stack the Firmus application notes walk through, and the one every
# script in this repo was written against.
#
# This file is a straight port of what 01/02/03-smoke-*.sbatch did before the
# engine split, and it is deliberately not an improvement on it: it is the
# regression test for the refactor. If `ENGINE=sglang` ever stops behaving as it
# did, the abstraction is wrong, and you want to learn that on a 0.6B model
# rather than during a 235B PD run.

# ${SGLANG_IMG-...} without the colon: an explicitly EMPTY SGLANG_IMG means bare
# metal (NYCU has no container runtime) and must not fall back to the default.
eng_default_image() { echo "${SGLANG_IMG-/scratch/enroot-images/sglang-harbor.sqsh}"; }

eng_import_check() {
cat <<SNIP
PY=$(eng_python)
"\$PY" -c "
import sglang, sys
print('  python', sys.version.split()[0])
print('  sglang', getattr(sglang, '__version__', 'imported'))
" || { echo "x sglang did not import" >&2; exit 1; }

# The Dockerfile's VERIFY stage checks this too. Repeat it here: the image you
# run on the cluster is not necessarily the one you built.
"\$PY" -m sglang.launch_server --help > /dev/null 2>&1 \
    || { echo "x sglang.launch_server does not run" >&2; exit 1; }
echo "  ok sglang.launch_server runs"

# Not fatal -- a single-node smoke test never touches it -- but PD does, and
# finding out here costs nothing.
"\$PY" -c "import nixl" 2>/dev/null \
    && echo "  ok nixl importable (PD KV transfer)" \
    || echo "  !  nixl not importable -- fine for these smoke tests, fatal for PD"
SNIP
}

eng_serve_cmd() {   # model name port tp
    local -a cmd=(
        "$(eng_python)" -m sglang.launch_server
        --model-path "$1" --served-model-name "$2" --trust-remote-code
        --tp-size "$4"
        # Same reason as the PD path: without --enable-metrics there is no
        # /metrics, engine_metrics.csv is empty, and every verdict that needs
        # queue depth, KV usage or batch size quietly degrades.
        --enable-metrics
    )
    # Match the PD workers' context length when one is set. Left unset, SGLang
    # takes the checkpoint's 262144, which is a different configuration from the
    # one PD is measured at -- and comparing them would be comparing two things.
    [[ -n "${CONTEXT_LEN:-}" ]] && cmd+=(--context-length "$CONTEXT_LEN")
    [[ -n "${MEM_FRACTION:-}" ]] && cmd+=(--mem-fraction-static "$MEM_FRACTION")
    # THE TUNING KNOBS, in SGLang's vocabulary. Each is emitted only when its
    # variable is set, so a variant that names one changes exactly one thing and
    # `base` stays the engine's own defaults.
    #
    # Why these four, given jobs 79456/79457: we are ITL-bound with KV to spare,
    # so the levers are the ones that touch decode latency rather than capacity.
    #   EP_SIZE            experts across ranks instead of sharded within them --
    #                      the MoE-specific lever, and 8 of 128 experts per token
    #                      means the all-to-all is small
    #   KV_CACHE_DTYPE     fp8 KV halves what decode must read each step
    #   CHUNKED_PREFILL    smaller prefill chunks interleave less violently with
    #                      decode, which is what ITL p90 is measuring
    #   MAX_RUNNING        a ceiling on the decode batch, so ITL degrades by
    #                      queueing rather than by slowing every request
    # THE MoE KERNEL, and it is not a small knob. SGLang's --moe-runner-backend
    # defaults to `auto`, and auto does NOT mean "pick the best one":
    #
    #     self.use_deep_gemm = get_moe_runner_backend().is_deep_gemm()
    #
    # and is_deep_gemm() is a literal string comparison against "deep_gemm". So every
    # SGLang number this project has produced ran WITHOUT DeepGEMM, while
    # sgl-deep-gemm 0.2.0 sat installed in the venv the whole time.
    #
    # It should matter here specifically. DeepGEMM exists for FP8 BLOCK-SCALED GEMMs,
    # which is exactly this checkpoint (weight_block_size [128,128], fmt e4m3), and
    # the measured prefill on NSCC was 290 TFLOPS/GPU -- 14.7% of H100 FP8 peak. That
    # 15% is the MoE GEMM. A teammate reached 3190 good tok/s on vLLM by switching
    # this same thing to DeepGEMM and could then raise N without losing the gate.
    [[ -n "${MOE_RUNNER_BACKEND:-}" ]] && cmd+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
    [[ -n "${EP_SIZE:-}" ]] && cmd+=(--ep-size "$EP_SIZE")
    [[ -n "${KV_CACHE_DTYPE:-}" ]] && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "${CHUNKED_PREFILL:-}" ]] && cmd+=(--chunked-prefill-size "$CHUNKED_PREFILL")
    [[ -n "${MAX_RUNNING_SOLO:-}" ]] && cmd+=(--max-running-requests "$MAX_RUNNING_SOLO")
    [[ -n "${SCHEDULE_CONSERVATIVENESS:-}" ]] && cmd+=(--schedule-conservativeness "$SCHEDULE_CONSERVATIVENESS")
    [[ "${TORCH_COMPILE:-0}" == 1 ]] && cmd+=(--enable-torch-compile)
    # Stands in for the client's --cache-bust first_turn_prefix where the
    # installed AIPerf has no such flag (see run/client-lib.sh). Same intent --
    # no part of a measured request may be served from a previous one's prefix --
    # reached from the server side, where the log proves it happened.
    [[ "${DISABLE_PREFIX_CACHE:-0}" == 1 ]] && cmd+=(--disable-radix-cache)
    cmd+=(--host 0.0.0.0 --port "$3")
    printf '%s\n' "${cmd[@]}"
}

# ---------------------------------------------------------------- PD (Phase A)
# Serialise the engine-neutral placement (PD_ROLE / PD_IP / PD_PORT / PD_TP /
# PD_BOOT, filled by pd_place and pd_resolve) into SGLang's own form, which is
# command-line arguments.

eng_pd_prepare() { :; }   # nothing to write; SGLang takes it all on the CLI

# Bare metal needs CUDA_HOME for deep_gemm and CPATH for triton -- see
# eng_venv_env() in engine-lib.sh. Inside a container both already exist and
# eng_venv returns nothing, so this emits nothing there.
# SGLang reads several knobs from the environment rather than argv, and the launcher
# is the only place a worker's environment is set. Each is forwarded only when the
# caller asks, so an unset variable changes nothing.
#
#   SGLANG_MOE_CONFIG_DIR                      a directory of tuned Triton MoE configs,
#       checked before the package's own. The 3,591 H200 track measured a tuned set at
#       decode ITL -10.6% without speculation and -22% with EAGLE-3. Ours is NOT tuned
#       for this shape: the file for E=128,N=384 fp8_w8a8 block[128,128] exists only
#       under configs/triton_3_3_1 while the installed Triton is 3.7.1, and the
#       down-projection file exists in none of the eight version directories.
#   SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB    the size above which custom all-reduce
#       hands off to NCCL. Prefill's large batches exceed the default, so they were
#       paying NCCL even after we fixed custom all-reduce being disabled outright.
#       Measured at +2.5% prefill capacity when raised to 131072.
#   SGLANG_DISAGGREGATION_WAITING_TIMEOUT      how long a decode worker waits for KV
#       before giving up on a request.
eng_pd_env() {
    eng_venv_env
    local v
    for v in SGLANG_MOE_CONFIG_DIR SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB \
             SGLANG_DISAGGREGATION_WAITING_TIMEOUT SGLANG_DISAGG_STAGING_BUFFER; do
        [[ -n "${!v:-}" ]] && echo "$v=${!v}"
    done
    # THE NIXL BACKEND PARAMS ARE JSON, AND JSON HAS COMMAS. The campaign splits a
    # variant's assignments on commas, so
    #   SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS={"num_threads":"16","split_batch_size":"256"}
    # cannot be written in a VARIANTS token at all -- it would be torn into three
    # fragments, two of which are not KEY=VALUE and would abort the cell. Take the two
    # fields as separate comma-free knobs and assemble the JSON here.
    local -a _nixl=()
    [[ -n "${NIXL_NUM_THREADS:-}" ]]      && _nixl+=("\"num_threads\":\"$NIXL_NUM_THREADS\"")
    [[ -n "${NIXL_SPLIT_BATCH_SIZE:-}" ]] && _nixl+=("\"split_batch_size\":\"$NIXL_SPLIT_BATCH_SIZE\"")
    if (( ${#_nixl[@]} )); then
        local IFS=,
        echo "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS={${_nixl[*]}}"
    fi
    return 0
}

# MEM-FRACTION, PER ROLE, BECAUSE THE TWO ROLES NEED OPPOSITE THINGS FROM IT.
# mem_fraction_static sizes the KV pool, and what is left over is the activation and
# kernel workspace. A decode worker holds 200 output tokens for every request it is
# serving and wants the pool as large as it can get. A disaggregated prefill worker
# holds each request's KV only until the decode side pulls it, and job 215960 measured
# what that actually costs: over ~1,250 Prefill-batch steps on each worker the pool's
# `token usage` sat at 0.11-0.13 and peaked at 0.19 -- including the first-10s burst
# where every one of our TTFT failures happens. The pool is five times larger than
# the job needs.
#
# What it is starving instead, from the same logs:
#
#   Available memory 9.70GB is less than required memory 16.94GB for warmup,
#   reducing max_m to 8192 to avoid out of memory
#
# DeepGEMM asked for 16.94 GB of workspace and got 9.70, so it halved its max_m. That
# is the ceiling on --chunked-prefill-size, and it is why the 12288-token prefill
# batch of job 215921 died of OOM at a single global mem-fraction of 0.85: the memory
# it needed was sitting unused in prefill's KV pool.
#
# 0.85 -> 0.76 moves 7.2 GB of an 80 GB card from the pool to the workspace, which
# covers the 16.94 GB DeepGEMM wants and still leaves the pool 2.5x the measured peak.
eng_mem_fraction() {   # $1=role
    local v=""
    case "$1" in
        prefill) v="${MEM_FRACTION_PREFILL:-}" ;;
        decode)  v="${MEM_FRACTION_DECODE:-}" ;;
    esac
    echo "${v:-$MEM_FRACTION}"
}

eng_pd_worker_cmd() {   # $1=index into the placement arrays
    local i="$1" role="${PD_ROLE[$1]}" tp="${PD_TP[$1]}"
    local -a cmd=(
        "$(eng_python)" -m sglang.launch_server
        --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" --trust-remote-code
        --context-length "$CONTEXT_LEN" --tp-size "$tp"
        --mem-fraction-static "$(eng_mem_fraction "$role")"
        --disaggregation-mode "$role"
        --disaggregation-transfer-backend "${TRANSFER_BACKEND:-nixl}"
        # --enable-metrics is not optional: without it SGLang serves no /metrics,
        # engine_metrics.csv comes out empty, and every verdict needing queue
        # depth, KV usage, preemptions or batch size silently degrades.
        --enable-metrics
    )
    # The same knobs as the solo path, so a variant that wins at four GPUs can be
    # carried to the 16-GPU PD run without re-deriving it.
    # MoE kernel, PER ROLE. The two sides want different things and a single setting
    # gets one of them wrong: prefill runs 8192-token batches and is GEMM-bound, so it
    # wants DeepGEMM's FP8 block-scaled path; decode runs batches of tens and is
    # launch- and latency-bound, where a simpler kernel can win. The reference
    # configuration that reached 3190 good tok/s splits it exactly that way --
    # DeepGEMM on four TP2 prefill workers, Triton on two TP4 decode engines.
    #
    # MOE_RUNNER_BACKEND still applies to both when the split is not wanted.
    local moe="${MOE_RUNNER_BACKEND:-}"
    if [[ "$role" == prefill && -n "${MOE_RUNNER_BACKEND_PREFILL:-}" ]]; then
        moe="$MOE_RUNNER_BACKEND_PREFILL"
    elif [[ "$role" == decode && -n "${MOE_RUNNER_BACKEND_DECODE:-}" ]]; then
        moe="$MOE_RUNNER_BACKEND_DECODE"
    fi
    [[ -n "$moe" ]] && cmd+=(--moe-runner-backend "$moe")
    [[ -n "${KV_CACHE_DTYPE:-}" ]] && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "${CHUNKED_PREFILL:-}" ]] && cmd+=(--chunked-prefill-size "$CHUNKED_PREFILL")
    # A BIGGER PREFILL BATCH, ON THE PREFILL ROLE. --chunked-prefill-size alone is not
    # enough: --max-prefill-tokens caps the batch independently, so raising one and
    # not the other leaves the batch where it was. Measured by the 3,591 H200 track at
    # +3.5% prefill capacity for 12288/12288 with mem-fraction dropped to 0.88, the
    # drop being needed because the larger batch wants more activation memory.
    #
    # Prefill capacity is the single thing binding our H100 runs, which is why this is
    # per-role: a decode worker has no use for it.
    [[ -n "${MAX_PREFILL_TOKENS:-}" && "$role" == prefill ]] \
        && cmd+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
    # AN ESCAPE HATCH, PER ROLE. Every knob above needed a code change, a test and a sync
    # before it could be measured once, and most were measured once and never again. The
    # reference H200 harness has had P_EXTRA / D_EXTRA from the start, which is why its
    # matrix could sweep nine all-reduce and KV-transfer options in one job while ours
    # needed an edit per option.
    #
    # Word-split on purpose: these carry whole flag strings, e.g.
    #   PD_EXTRA="--enable-torch-symm-mem"
    #   DECODE_EXTRA="--enable-fused-moe-sum-all-reduce --max-running-requests=128"
    #
    # A FLAG WITH A VALUE IS WRITTEN --flag=value, NOT --flag value, when it comes from a
    # VARIANTS token. The campaign's variant parser splits on whitespace, so a space
    # inside an assignment tears the variant in half -- job 216204 lost its control cell
    # and 2,576 SU to exactly that. argparse takes --flag=value as one word. The campaign
    # now refuses a variant with whitespace in an assignment, so this is enforced rather
    # than remembered.
    # A single flag VALUE containing a space is not supported; add a named knob for that.
    # PD_EXTRA applies to both roles and the role-specific one is appended after it, so a
    # cell can set a common flag and still override on one side.
    # shellcheck disable=SC2206
    [[ -n "${PD_EXTRA:-}" ]] && cmd+=(${PD_EXTRA})
    if [[ "$role" == prefill ]]; then
        # shellcheck disable=SC2206
        [[ -n "${PREFILL_EXTRA:-}" ]] && cmd+=(${PREFILL_EXTRA})
    else
        # shellcheck disable=SC2206
        [[ -n "${DECODE_EXTRA:-}" ]] && cmd+=(${DECODE_EXTRA})
    fi
    [[ "${TORCH_COMPILE:-0}" == 1 ]] && cmd+=(--enable-torch-compile)
    [[ "${DISABLE_PREFIX_CACHE:-0}" == 1 ]] && cmd+=(--disable-radix-cache)
    # DEEPGEMM AND CUDA-GRAPH CAPTURE CANNOT SHARE A STARTUP. Jobs 215267 and 215459
    # both lost their DeepGEMM prefill cell to
    #
    #   RuntimeError: num_active_captures_ > 0 INTERNAL ASSERT FAILED ...
    #   markCaptureEnd called with no captures in progress
    #
    # inside PrefillCudaGraphRunner. 215459 had a fully warm cache -- 29 grouped
    # kernels, 60/60 committed, and not one new file after the run -- and it still
    # printed "Entering DeepGEMM JIT" and still broke the capture, so the trigger is
    # first USE of those cubins (a module load) rather than compiling them, and a
    # module load inside a capture is illegal either way.
    #
    # SGLang's own compile_deep_gemm resolves this by setting both phases' graph
    # backend to DISABLED. Prefill batches here are 4,008 tokens, so launch overhead
    # is amortised and the graphs are worth little; decode graphs are untouched.
    if [[ "${DISABLE_PREFILL_GRAPH:-0}" == 1 && "$role" == prefill ]]; then
        cmd+=(--disable-prefill-cuda-graph)
    fi
    if [[ "$role" == prefill ]]; then
        cmd+=(--disaggregation-bootstrap-port "${PD_BOOT[$i]}")
    else
        # --ep-size IS NOT A FREEBIE AND IS NO LONGER DEFAULTED ON. It used to fall back
        # to $tp, so every PD decode worker this project ever launched ran with expert
        # parallelism. Two measurements say that was wrong:
        #
        #   ours   EP4 was the WORST of six knobs at 4 GPUs solo, -26.3%.
        #   theirs a standalone DP8+EP8 group is slower than a standalone TP4 group at
        #          every batch below 256, with a fixed per-step cost 4.3-5.2 ms higher --
        #          22.8-27.9 us across the 188 dispatch/combine collectives a step
        #          issues. Two independent TP4 engines are the only 8-GPU decode
        #          arrangement that escapes it.
        #
        # EP trades an all-reduce for an all-to-all, and at this scale the all-to-all is
        # dearer. Our NSCC decode ITL of 15.3 ms was carrying that 4-5 ms.
        [[ -n "${DECODE_EP:-}" ]] && cmd+=(--ep-size "$DECODE_EP")
        cmd+=(--max-running-requests "${MAX_RUNNING:-256}")
    fi
    cmd+=(--host 0.0.0.0 --port "${PD_PORT[$i]}")
    printf '%s\n' "${cmd[@]}"
}

# ROUTING POLICY. sglang_router defaults to --policy cache_aware, and on this
# benchmark that is the worst available choice. Job 17682 measured it: of four
# prefill workers, one took 126 requests and the other three took one apiece,
# decode never exceeded #running-req: 1, forty concurrent requests serialised,
# the 60s warmup phase ran 503s, and the job hit the 17-minute wall.
#
# Cache-aware routing trades load balance for prefix-cache affinity. The official
# command passes --cache-bust first_turn_prefix, so every request carries a
# DIFFERENT prefix and that affinity is worth nothing -- three quarters of the
# prefill capacity was being paid for a cache hit that cannot happen.
#
# Routing is on the rules' allowed list ("Prefill/Decode split ratio and
# routing"), so this is an optimization, not a workaround.
#
#   round_robin   for prefill: identical request shapes, deliberately unique
#                 prefixes, so there is nothing to be aware of
#
# DECODE was power_of_two -- the router's own help suggests it for PD mode --
# until job 17914. There, all eight warm-up requests reached the gateway, none
# was ever dispatched to a worker, and none returned in 300s, alongside:
#
#   WARN smg::core::worker_manager: No loads fetched from workers
#
# power_of_two picks two workers at random and sends to the less loaded, so it
# needs load data (load_check_interval_secs: 5). Without it, request dispatch
# appears to block. It also refuses to start at all with a single decode worker
# (job 17816), which rules out the cheap one-node topologies.
#
# NOT PROVEN: 17912 used power_of_two on one node with TP=1 everywhere and
# routed 13/13 correctly. The difference may be the load fetch, or the
# heterogeneous TP that 17914 also warned about ("conflicting tp_size:
# prefill=2, decode=4"). round_robin needs no load data either way, and this
# benchmark's requests are identical in shape, so there is nothing for a
# load-aware policy to exploit.
# Setting either to the empty string omits the flag, which hands the router its own
# default (cache_aware). That is not a tuning option -- it is the only way to run the
# organisers' reference deployment faithfully, because reference-scripts/07 passes
# neither flag. Our default stays round_robin.
PD_PREFILL_POLICY="${PD_PREFILL_POLICY-round_robin}"
PD_DECODE_POLICY="${PD_DECODE_POLICY-round_robin}"

eng_pd_gateway_cmd() {   # $1=port
    local i
    local -a cmd=("$(eng_python)" -m sglang_router.launch_router --pd-disaggregation)
    for i in "${!PD_ROLE[@]}"; do
        if [[ "${PD_ROLE[$i]}" == prefill ]]; then
            cmd+=(--prefill "$(pd_worker_url "$i")" "${PD_BOOT[$i]}")
        else
            cmd+=(--decode "$(pd_worker_url "$i")")
        fi
    done
    [[ -n "$PD_PREFILL_POLICY" ]] && cmd+=(--prefill-policy "$PD_PREFILL_POLICY")
    [[ -n "$PD_DECODE_POLICY" ]]  && cmd+=(--decode-policy "$PD_DECODE_POLICY")
    # THE METRICS PORT HAS TO MOVE WITH THE CELL. sglang_router binds a second
    # listener for Prometheus and hardcodes it to 29000, which the campaign's port
    # stride does not touch. The second cell of job 215941 therefore died with
    #
    #   failed to install Prometheus metrics exporter:
    #   FailedToCreateHTTPListener("Address already in use (os error 98)")
    #
    # 45 s after launch, and the cell produced nothing -- the fourth time a port we
    # had not enumerated outlived a teardown. Deriving it from the router port makes
    # it shift by cell exactly as every other port does; 21000 is the offset that
    # puts cell 0 on the router's own default, so nothing changes for a single cell.
    cmd+=(--prometheus-port "$(( $1 + 21000 ))")
    cmd+=(--host 0.0.0.0 --port "$1")
    printf '%s\n' "${cmd[@]}"
}
