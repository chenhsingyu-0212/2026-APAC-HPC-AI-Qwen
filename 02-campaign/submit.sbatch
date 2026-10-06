#!/bin/bash
#SBATCH --job-name=qwen3-pd-8plus8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:17:00
#SBATCH --partition=hpcai
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#
# The hpcai partition caps a job at 17 minutes (sinfo TIMELIMIT 17:00), and
# Slurm's kill at the cap is a SIGKILL that leaves no artefacts. Every
# reference script under /scratch/reference-scripts uses -t 00:17:00 or less.
# Do not raise this: the job will simply be rejected at submission.
#
# Qwen3-235B-A22B PD disaggregated inference -- deploy and score in one job.
#
# The task rules require that deploying the serving stack and invoking the
# official AIPerf command are consecutive steps of the same job. They are:
# sections 4-7 bring the stack up, section 8 runs the scored command, and
# nothing in between touches that command.
#
# Topology (the verified 8+8 split from the Firmus application notes):
#   node 0 (head)  4x prefill  --tp-size 2               ports 30100-30103
#   node 1 (tail)  2x decode   --tp-size 4 --ep-size 4   ports 30200-30201
#   node 0         router on port 8000, the only endpoint the client talks to
#
# Each worker is its own container and its own Slurm step, with --gres=gpu:<tp>,
# matching /scratch/reference-scripts/07-pd-235b-8plus8.sbatch. Slurm chooses the
# devices; nothing here sets CUDA_VISIBLE_DEVICES.
#
# Everything the benchmark client sees goes through the router. Pointing it at a
# worker is a rule violation, so the gateway is the router and nothing else.
#
#   sbatch run/submit.sbatch
#   CONCURRENCY=128 sbatch run/submit.sbatch     # override any config value
#
# The paths and flag values below are confirmed against the cluster (recon
# 2026-09-16) but this script itself has not yet been run there. Work through the
# three smoke tests first.

set -uo pipefail

# ============================================================ 0. where am I
# Slurm copies the batch script to /var/spool/slurm/job<id>/, so ${BASH_SOURCE[0]}
# points at that copy and not into the repo -- job 17590 died in one second on
# exactly this. SLURM_SUBMIT_DIR is where sbatch was invoked from; fall back to
# BASH_SOURCE so the script still works when run directly.
HERE=""
# PBS names its variables differently and starts the job in $HOME rather than the
# directory it was submitted from, so every "${SLURM_SUBMIT_DIR:-$PWD}" below would
# resolve to the wrong place. Map the two that matter onto the names this repo
# already reads, once and here, so nothing downstream has to know which scheduler
# it is running under.
_pbsid="${PBS_JOBID:-}"; _pbsid="${_pbsid%%.*}"
: "${SLURM_SUBMIT_DIR:=${PBS_O_WORKDIR:-}}"
: "${SLURM_JOB_ID:=$_pbsid}"
export SLURM_SUBMIT_DIR SLURM_JOB_ID

for _d in "${SLURM_SUBMIT_DIR:-}/run" "${SLURM_SUBMIT_DIR:-}" \
          "$(dirname "${BASH_SOURCE[0]}")" "$PWD/run" "$PWD"; do
    [[ -n "$_d" && -r "$_d/pd-lib.sh" ]] && { HERE="$(cd "$_d" && pwd)"; break; }
done
[[ -n "$HERE" ]] || { echo "x cannot find pd-lib.sh -- run this from the repo, or set SLURM_SUBMIT_DIR" >&2; exit 1; }

# The site supplies the machine-shaped defaults -- image paths or the absence of
# them, venv locations, the model path, the client binary. It is loaded first so
# section 1 can still override everything from the environment.
# shellcheck source=run/site-lib.sh
source "$HERE/site-lib.sh" || { echo "x cannot source site-lib.sh" >&2; exit 1; }
site_load || exit 1

# ============================================================ 1. configuration
# Everything you would normally edit lives here. Each can be overridden from the
# environment, so a sweep is `CONCURRENCY=96 sbatch ...` rather than an edit.
SGLANG_IMG="${SGLANG_IMG-/scratch/enroot-images/sglang-harbor.sqsh}"
AIPERF_IMG="${AIPERF_IMG-/scratch/enroot-images/aiperf.sqsh}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"      # the client binary, container or venv
MODEL_PATH="${MODEL_PATH:-/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
KIT_DIR="${KIT_DIR:-$SLURM_SUBMIT_DIR/AIPerf}"      # observe.sh / official.sh / gpu_sample.sh
MOUNTS="${MOUNTS:-/scratch:/scratch,$HOME:$HOME}"

CONCURRENCY="${CONCURRENCY:-64}"        # the team-declared N
SERVED_NAME="${SERVED_NAME:-qwen}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"      # template + 4000 ISL + 200 OSL must fit
MEM_FRACTION="${MEM_FRACTION:-0.90}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-nixl}"
MAX_RUNNING="${MAX_RUNNING:-256}"       # decode workers only

# The topology, as <prefill-tp>x<count>:<decode-tp>x<count>. The default is the
# split the application notes verified; run/sweep-pd.sbatch searches others and
# prints the PD_SPEC of whichever wins, so adopting it is one variable.
PD_SPEC="${PD_SPEC:-2x4:4x2}"
# EMPTY BY DEFAULT, deliberately. This used to be 4, so every PD decode worker ran
# with expert parallelism -- and EP4 was the worst of the six knobs we swept at 4 GPUs
# (-26.3%), while the reference 3190 configuration measured a DP8+EP8 decode group as
# 4.3-5.2 ms per step SLOWER than plain TP4. Set it only to measure it.
DECODE_EP="${DECODE_EP:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}" # per endpoint, per the application notes
# One container per worker, so CPUs are requested per step. 4x16 on the prefill
# node and 2x16 on the decode node, out of 128 -- the reference's numbers.
PD_CPUS_PER_WORKER="${PD_CPUS_PER_WORKER:-16}"
PD_ROUTER_CPUS="${PD_ROUTER_CPUS:-2}"
# Routing policy is the ENGINE's default, set in run/engines/<engine>.sh, not
# this script's. Duplicating the defaults here is how job 18465 ended up with
# decode_policy=power_of_two after sglang.sh had been changed to round_robin:
# this file ran first, the variable was already set, and the engine's default
# never applied. Set PD_PREFILL_POLICY / PD_DECODE_POLICY in the environment to
# override; leave them unset to get what the engine chose.
OUT_ROOT="${OUT_ROOT:-$SLURM_SUBMIT_DIR/results/${SLURM_JOB_ID:-manual}}"

# ============================================================ 2. preconditions
die() { echo "x $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

# An empty image is not a missing image: it means this engine runs bare metal,
# which is the only mode NYCU has (no container runtime is installed at all).
for f in "$SGLANG_IMG" "$AIPERF_IMG"; do
    [[ -z "$f" || -e "$f" ]] || die "container image not found: $f"
done
[[ -n "$AIPERF_IMG" ]] || command -v "$AIPERF_BIN" >/dev/null 2>&1 \
    || die "no container for the client and no aiperf at: $AIPERF_BIN"
[[ -d "$MODEL_PATH" ]] || die "model path not found: $MODEL_PATH"
for s in official.sh observe.sh gpu_sample.sh diagnose.sh; do
    [[ -x "$KIT_DIR/$s" ]] || die "$KIT_DIR/$s missing or not executable"
done

# Confirm the command we are about to be scored on still matches the rules,
# before spending an allocation on it.
RULES="${RULES:-$SLURM_SUBMIT_DIR/2026-APAC-HPC-AI/1. Qwen3 AI Task Rules.md}"
if [[ -f "$RULES" ]]; then
    "$KIT_DIR/official.sh" --verify "$RULES" || die "the official command no longer matches the rules"
else
    echo "! rules file not found at $RULES -- skipping the command check"
fi

mkdir -p "$OUT_ROOT" || die "cannot create $OUT_ROOT"
WORKDIR="$OUT_ROOT/deploy"; mkdir -p "$WORKDIR"

# ============================================================ 3. node discovery
# HERE was resolved in section 0, before the site was loaded.
# shellcheck source=run/pd-lib.sh
source "$HERE/pd-lib.sh" || die "cannot source pd-lib.sh"
# shellcheck source=run/engine-lib.sh
source "$HERE/engine-lib.sh" || die "cannot source engine-lib.sh"
eng_load || exit 1
SGLANG_IMG="$(eng_image)"     # whatever container this engine runs in
pd_parse "$PD_SPEC" || exit 1

# Slurm expands a range like "DGX-H200-CN[05-06]"; PBS lists one line per chunk
# in $PBS_NODEFILE. Same answer, two mechanisms -- see run/sched-lib.sh.
mapfile -t NODES < <(sched_nodes)
(( ${#NODES[@]} >= 2 )) || die "need 2 nodes, got ${#NODES[@]}: ${NODES[*]}"
# Before anything is deployed: is every node actually able to run work? PBS hands
# out nodes that accept a job, report R and then answer no launcher at all, and a
# worker that never starts still bills for the walltime it holds. A no-op on Slurm.
sched_preflight "${NODES[@]}" || die "one of the allocated nodes cannot run work"
sched_reset_pools "${NODES[@]}"
HEAD="${NODES[0]}"; TAIL="${NODES[1]}"
pd_place "$GPUS_PER_NODE" "${NODES[@]}" || exit 1
pd_resolve || exit 1
GATEWAY="$(pd_gateway "$ROUTER_PORT")"   # an IP, not a name: see pd_ip_of()
say "topology $PD_SPEC -> $(pd_split) GPUs across ${#NODES[@]} nodes"
pd_print_placement

# ============================================================ 4. deploy
# The engine writes whatever config it needs from the placement; SGLang needs
# none, TensorRT-LLM writes disagg.yaml plus a per-role llm options file.
export PD_WORK="$WORKDIR" ROUTER_PORT HEALTH_TIMEOUT
declare -F eng_pd_prepare >/dev/null && { eng_pd_prepare "$WORKDIR" || die "eng_pd_prepare failed"; }
pd_write_launcher "$WORKDIR/worker.sh"
export MODEL_PATH SERVED_NAME CONTEXT_LEN MEM_FRACTION TRANSFER_BACKEND \
       DECODE_EP MAX_RUNNING

# What pd_launch needs. PD_WORK holds the launcher, the per-worker $HOME
# directories and the worker logs, and is bind-mounted into every container.
export PD_IMAGE="$SGLANG_IMG" PD_WORK="$WORKDIR" PD_MOUNTS="$MOUNTS" \
       PD_CPUS_PER_WORKER PD_ROUTER_CPUS PD_PREFILL_POLICY PD_DECODE_POLICY

TEL_PIDS=()
cleanup() {
    say "tearing down"
    pd_teardown
    local p
    for p in "${TEL_PIDS[@]:-}"; do
        [[ -n "$p" ]] || continue
        kill "$p" 2>/dev/null; wait "$p" 2>/dev/null
    done
}
trap cleanup EXIT INT TERM

pd_launch "$WORKDIR/worker.sh" || exit 1

# ============================================================ 5. health
# Workers first, all of them, before the router: a router pointed at a worker
# that is still loading weights fails in a way that looks like a routing bug.
say "waiting for workers (up to ${HEALTH_TIMEOUT}s each)"
pd_wait_all "$HEALTH_TIMEOUT" || exit 1

# ============================================================ 6. router
say "starting the $ENGINE gateway on $GATEWAY"
pd_launch_gateway "$ROUTER_PORT" || exit 1
pd_wait_healthy "http://$GATEWAY" gateway "$HEALTH_TIMEOUT" \
                "${PD_PIDS[-1]}" "$WORKDIR/gateway.log" || exit 1

# Not a single shot: the gateway reports healthy before its workers register.
MODEL_ID="$(pd_wait_models "http://$GATEWAY" "${MODELS_TIMEOUT:-180}" \
                           "${PD_PIDS[-1]}" "$WORKDIR/gateway.log")" \
    || die "the gateway never reported a servable model"

{
    echo "job=${SLURM_JOB_ID:-manual}  nodes=${NODES[*]}"
    echo "pd_spec=$PD_SPEC  split=$(pd_split)  gpus=$(pd_gpus_used)"
    echo "context_len=$CONTEXT_LEN mem_fraction=$MEM_FRACTION backend=$TRANSFER_BACKEND"
    echo "decode_ep=$DECODE_EP max_running_requests=$MAX_RUNNING concurrency=$CONCURRENCY"
    echo "model_id=$MODEL_ID gateway=$GATEWAY"
    echo "container=one-per-worker cpus_per_worker=$PD_CPUS_PER_WORKER router_cpus=$PD_ROUTER_CPUS"
    echo "engine=$ENGINE image=$SGLANG_IMG"
    echo "prefill_policy=${PD_PREFILL_POLICY:-n/a} decode_policy=${PD_DECODE_POLICY:-n/a}"
    echo "--- placement ---"
    pd_print_placement
} > "$OUT_ROOT/deployment.txt"

# ============================================================ 6b. prewarm
# Before the clock starts, not during it. PREWARM_REQUESTS defaults to twice the
# prefill worker count so round-robin reaches every one of them.
# WARM WITH THE CLIENT YOU MEASURE WITH.
#
# The first three attempts at this used curl, and all three hung: eight requests
# reached the gateway, the router logged "started processing" for each, none was
# ever dispatched to a worker, and none finished in 300 s (jobs 17914, 18315,
# 18355 -- the last of those streaming, which ruled that out as the cause).
#
# Meanwhile job 17682, which ran the official command with no warm-up at all,
# served 129 requests through the same deployment and logged 241 decode batches.
# So the deployment was never the problem: the difference was the client.
#
# Rather than keep bisecting what AIPerf sends that curl does not, warm with
# AIPerf. It is the same binary, the same container, the same request shape and
# the same tokenizer as the scored run -- and unlike curl it is known to work
# here. This is also the honest thing to warm with: a warm-up that exercises a
# path the benchmark never takes proves nothing about the benchmark.
#
# Not the official command, and not pretending to be: a handful of requests at
# the right shape, with no bearing on the score. The rules constrain the scored
# re-run, not what precedes it.
PREWARM_REQUESTS="${PREWARM_REQUESTS:-$(( PD_P_N * 2 ))}"
if (( PREWARM_REQUESTS > 0 )); then
    pd_need_time $(( 240 + 60 )) "prewarming" || die "no budget left to prewarm"
    say "prewarming with aiperf: $PREWARM_REQUESTS requests at ISL ${ISL:-4000}"
    PREWARM_START=$SECONDS
    CLIENT_ARGS=(); mapfile -t CLIENT_ARGS < <(site_client_args)
    [[ -n "$AIPERF_IMG" ]] && CLIENT_ARGS+=("--container-workdir=$WORKDIR")
    srun --overlap -N1 -n1 -w "$HEAD" ${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"} \
         "$AIPERF_BIN" profile \
            --artifact-dir "$WORKDIR/prewarm" \
            --url "http://$GATEWAY" \
            --endpoint-type chat --endpoint /v1/chat/completions \
            --model "$MODEL_ID" --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 \
            --streaming -H 'Accept: text/event-stream' \
            --concurrency "$PREWARM_REQUESTS" --request-count "$PREWARM_REQUESTS" \
            --api-key NOT_USED \
            --synthetic-input-tokens-mean "${ISL:-4000}" --synthetic-input-tokens-stddev 0 \
            --output-tokens-mean 16 --output-tokens-stddev 0 \
            --extra-inputs min_tokens:16 --extra-inputs max_tokens:16 \
            --extra-inputs ignore_eos:true \
            --ui simple > "$WORKDIR/prewarm.log" 2>&1
    PREWARM_RC=$?
    PD_PREWARM_SECONDS=$(( SECONDS - PREWARM_START ))
    say "  prewarm exited $PREWARM_RC after ${PD_PREWARM_SECONDS}s"
    if (( PREWARM_RC != 0 )); then
        say "  ! prewarm did not succeed -- continuing to the scored command anyway."
        say "    A failed warm-up is worth less than the allocation it would waste."
        tail -15 "$WORKDIR/prewarm.log" 2>/dev/null
    fi
fi
# Appended here, not written with the rest of deployment.txt above: that block
# runs before the prewarm, so the duration would always have been blank.
echo "prewarm_requests=${PREWARM_REQUESTS:-0} prewarm_seconds=${PD_PREWARM_SECONDS:-n/a}" \
    >> "$OUT_ROOT/deployment.txt"

# ============================================================ 7. time budget
# Deployment is done; what is left has to cover the official command exactly as
# the rules fix it -- 60s warmup + 120s benchmark + 30s grace -- plus the client
# starting up and the diagnosis afterwards. If the checkpoint load ate the
# budget, stop here: a truncated official run is not a lower score, it is no
# measurement, and the artefacts a SIGKILL leaves behind are worth nothing.
# 60s warmup + 120s benchmark + 30s grace, plus the client's own startup. The
# margin was 90s and job 18466 died on it: 248s remained, the guard demanded
# 300s, and the run was refused with everything else in place -- a loaded 235B,
# six healthy workers, and a warm deployment that had just cost 400s to reach.
#
# 25s is what the AIPerf container actually takes to start here. Keep the margin
# honest rather than comfortable: the guard exists to avoid a SIGKILL mid
# measurement, not to reserve slack nobody needs.
OFFICIAL_SECONDS=$(( 60 + 120 + 30 ))
pd_need_time $(( OFFICIAL_SECONDS + 25 )) "the official command" || exit 1

# Sample for what is actually left, minus a margin to write the CSV out.
TEL_SECONDS="${TEL_SECONDS:-$(( $(pd_time_left) - 45 ))}"
(( TEL_SECONDS > 60 )) || TEL_SECONDS=60

# ============================================================ 8. telemetry
# Both nodes, host-side (not in a container) so nvidia-smi and the IB counters
# under /sys are the real ones. Started before the scored command, stopped after.
for n in "${NODES[@]}"; do
    srun --overlap -N1 -n1 -w "$n" \
         "$KIT_DIR/gpu_sample.sh" --out "$OUT_ROOT/tel-$n" --label "$n" \
         --interval 1 --duration "$TEL_SECONDS" > "$WORKDIR/tel-$n.log" 2>&1 &
    TEL_PIDS+=($!)
done
say "telemetry sampling on both nodes for ${TEL_SECONDS}s"

# ============================================================ 9. the scored run
# observe.sh monitors from the outside; official.sh is the fixed command and
# takes no monitoring flags. Nothing between here and there alters it.
say "running the official command at concurrency $CONCURRENCY"
# Poll every worker, not the gateway. Queue depth, KV usage and preemptions are
# engine-internal: the router does not have them, and in a PD split they differ
# between the prefill and decode halves, which is most of what you want to know.
mapfile -t METRICS_ARGS < <(pd_metrics_args)

CLIENT_ARGS=(); mapfile -t CLIENT_ARGS < <(site_client_args)
[[ -n "$AIPERF_IMG" ]] && CLIENT_ARGS+=("--container-workdir=$OUT_ROOT")
srun --overlap -N1 -n1 -w "$HEAD" ${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"} \
     "$KIT_DIR/observe.sh" --out "$OUT_ROOT/scored" --gateway "$GATEWAY" \
        --concurrency "$CONCURRENCY" "${METRICS_ARGS[@]}" -- \
        "$KIT_DIR/official.sh" --aiperf "$AIPERF_BIN" --host "$GATEWAY" \
            --model "$MODEL_ID" --concurrency "$CONCURRENCY"
RC=$?
say "official command exited $RC"

# ============================================================ 10. diagnose
for n in "${NODES[@]}"; do
    pkill -f "gpu_sample.sh --out $OUT_ROOT/tel-$n" 2>/dev/null
done
sleep 3

for n in "${NODES[@]}"; do
    [[ -d "$OUT_ROOT/tel-$n" ]] || continue
    echo ""; say "bottleneck report against $n"
    # --weights is the ACTIVE parameter bytes: summing this MoE checkpoint would
    # overstate the per-step traffic by roughly an order of magnitude. 20.5 GiB
    # is the low-batch bound (22B active at FP8); re-run with --weights 219 for
    # the high-batch bound, where a large decode batch touches nearly every expert.
    # --model-dir supplies config.json for the KV term. Without it the roofline
    # counts weights only, MBU comes out a lower bound, and the KV share reads 0%
    # -- which makes every point look WEIGHT BOUND whether it is or not.
    "$KIT_DIR/diagnose.sh" "$OUT_ROOT/scored" --telemetry "$OUT_ROOT/tel-$n" \
        --model-dir "$MODEL_PATH" --weights "${WEIGHTS_GIB:-20.5}" --tp "$PD_D_TP" \
        | tee "$OUT_ROOT/diagnose-$n.txt"
done

say "artifacts in $OUT_ROOT"
exit $RC
