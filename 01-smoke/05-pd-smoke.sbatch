#!/bin/bash
#SBATCH --job-name=pd-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --partition=hpcai
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#
# The hpcai partition caps a job at 17 minutes. Do not raise this.
#
# PD disaggregation on ONE node with a small model. Everything submit.sbatch
# does, for a fraction of the queue cost.
#
#   sbatch run/05-pd-smoke.sbatch
#   ENGINE=trtllm sbatch run/05-pd-smoke.sbatch
#   PD_SPEC=1x2:1x1 REQUESTS=24 sbatch run/05-pd-smoke.sbatch
#
# It exists because the 235B path costs an hour of queue per attempt and the
# first three attempts died on things a 0.6B model would have shown in minutes:
#   17682  the router sent every request to one prefill worker
#   17743  /health returned 200 before the gateway's workers had registered
#   ...and the TensorRT-LLM PD backend has never run at all.
#
# Two prefill workers is the minimum that makes ROUTING DISTRIBUTION visible,
# which is the whole point: with one of each you cannot tell a working router
# from a broken one.

set -uo pipefail

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
[[ -n "$HERE" ]] || { echo "x cannot find pd-lib.sh" >&2; exit 1; }

die() { echo "x $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

# shellcheck source=run/pd-lib.sh
source "$HERE/pd-lib.sh"    || die "cannot source pd-lib.sh"
# shellcheck source=run/engine-lib.sh
source "$HERE/engine-lib.sh" || die "cannot source engine-lib.sh"
eng_load || exit 1

MODEL_PATH="${MODEL_PATH:-/scratch/models/Qwen3-0.6B}"
SERVED_NAME="${SERVED_NAME:-qwen}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.60}"   # small model, several workers on one GPU set
TRANSFER_BACKEND="${TRANSFER_BACKEND:-nixl}"
MAX_RUNNING="${MAX_RUNNING:-64}"
DECODE_EP="${DECODE_EP:-1}"
# 2 prefill + 2 decode. Two of EACH is the minimum that makes routing visible,
# and job 17816 found the harder constraint: sglang_router refuses to start with
# --decode-policy power_of_two and a single decode worker --
#   "Power-of-two policy for decode requires at least 2 decode workers"
PD_SPEC="${PD_SPEC:-1x2:1x2}"          # 2 prefill TP1 + 2 decode TP1 = 4 GPUs
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
REQUESTS="${REQUESTS:-24}"
PD_CPUS_PER_WORKER="${PD_CPUS_PER_WORKER:-8}"
PD_ROUTER_CPUS="${PD_ROUTER_CPUS:-2}"
OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/results/pdsmoke-${SLURM_JOB_ID:-manual}}"

[[ -d "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
mkdir -p "$OUT_ROOT" || die "cannot create $OUT_ROOT"
WORKDIR="$OUT_ROOT/deploy"; mkdir -p "$WORKDIR"

eng_banner
say "spec=$PD_SPEC model=$MODEL_PATH requests=$REQUESTS"

pd_parse "$PD_SPEC" || exit 1
# Slurm expands a range like "DGX-H200-CN[05-06]"; PBS lists one line per chunk
# in $PBS_NODEFILE. Same answer, two mechanisms -- see run/sched-lib.sh.
mapfile -t NODES < <(sched_nodes)
(( ${#NODES[@]} >= 1 )) || { echo "x no nodes" >&2; exit 1; }
# Before anything is deployed: is every node actually able to run work? PBS hands
# out nodes that accept a job, report R and then answer no launcher at all, and a
# worker that never starts still bills for the walltime it holds. A no-op on Slurm.
sched_preflight "${NODES[@]}" || { echo "x an allocated node cannot run work" >&2; exit 1; }
sched_reset_pools "${NODES[@]}"
pd_place "$GPUS_PER_NODE" "${NODES[@]}" || exit 1
pd_resolve || exit 1
GATEWAY="$(pd_gateway "$ROUTER_PORT")"
pd_print_placement

export MODEL_PATH SERVED_NAME CONTEXT_LEN MEM_FRACTION TRANSFER_BACKEND \
       DECODE_EP MAX_RUNNING ROUTER_PORT HEALTH_TIMEOUT
export PD_IMAGE="$(eng_image)" PD_WORK="$WORKDIR" \
       PD_MOUNTS="${MOUNTS_BASE:-/scratch:/scratch}" \
       PD_CPUS_PER_WORKER PD_ROUTER_CPUS

# Pull the engine's own diagnostics off the compute node BEFORE teardown.
#
# Four jobs were spent on this: MPI spawn, then the Ray version, then colliding
# Ray instances, then the descriptor limit -- one hypothesis per queue slot, and
# all four wrong. Ray writes why it refused the connection into raylet.err and
# python-core-worker-*.log, in RAY_TMPDIR on the compute node, where teardown
# then loses it.
#
# Guessing is more expensive than looking. Collect first, then conclude.
collect_engine_logs() {
    local n dest
    dest="$OUT_ROOT/engine-logs"; mkdir -p "$dest" 2>/dev/null || return 0
    for n in "${NODES[@]}"; do
        srun --overlap -N1 -n1 -w "$n" bash -c '
            for d in /tmp/ray-*/session_latest/logs /tmp/ray-*/session_*/logs; do
                [ -d "$d" ] || continue
                for f in "$d"/raylet.err "$d"/raylet.out "$d"/gcs_server.err                          "$d"/python-core-worker-*.log "$d"/python-core-driver-*.log; do
                    [ -r "$f" ] || continue
                    echo "######## $(hostname) $f"
                    tail -60 "$f"
                done
            done
            echo "######## $(hostname) ulimit -n = $(ulimit -n)  (hard $(ulimit -Hn))"
            echo "######## $(hostname) /dev/shm"; df -h /dev/shm
        ' 2>/dev/null > "$dest/$n.txt" || true
        [ -s "$dest/$n.txt" ] && say "  engine logs -> $dest/$n.txt ($(wc -l < "$dest/$n.txt") lines)"
    done
}

cleanup() {
    say "collecting engine logs before teardown"
    collect_engine_logs || true
    say "tearing down"
    pd_teardown
}
trap cleanup EXIT INT TERM

declare -F eng_pd_prepare >/dev/null && { eng_pd_prepare "$WORKDIR" || die "eng_pd_prepare failed"; }
pd_write_launcher "$WORKDIR/worker.sh"
pd_launch "$WORKDIR/worker.sh" || exit 1
pd_wait_all "$HEALTH_TIMEOUT"  || exit 1

say "starting the $ENGINE gateway"
pd_launch_gateway "$ROUTER_PORT" || exit 1
pd_wait_healthy "http://$GATEWAY" gateway "$HEALTH_TIMEOUT" \
                "${PD_PIDS[-1]}" "$WORKDIR/gateway.log" || exit 1

# The race 17743 died on: /health is true before the workers have registered.
MODEL_ID="$(pd_wait_models "http://$GATEWAY" 120 "${PD_PIDS[-1]}" "$WORKDIR/gateway.log")" \
    || die "the gateway never reported a servable model"

# ---------------------------------------------------------------- traffic
# Sequential on purpose. Round-robin over two workers is visible at any rate;
# what we are testing is the router's choice, not the engine's throughput.
say "sending $REQUESTS requests through the gateway"
OK=0; ERR=0
for i in $(seq 1 "$REQUESTS"); do
    # Streaming, with the same Accept header the official command sends. A
    # non-streaming POST hangs on this PD router (jobs 17914, 18315), so a
    # smoke test that used one would be validating a path nobody measures.
    if curl -sf --max-time 60 "http://$GATEWAY/v1/chat/completions" \
         -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
         -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"count to $i\"}],\"max_tokens\":16,\"stream\":true}" \
         -o /dev/null 2>/dev/null; then OK=$(( OK + 1 )); else ERR=$(( ERR + 1 )); fi
done
say "  $OK ok, $ERR failed"

# ---------------------------------------------------------------- verdict
echo ""
echo "=============================================================="
echo " routing distribution"
echo "=============================================================="
RC=0
declare -a COUNTS=()
for i in "${!PD_ROLE[@]}"; do
    [[ "${PD_ROLE[$i]}" == prefill ]] || continue
    n="$(grep -c 'POST /generate\|/v1/chat' "$WORKDIR/$(pd_worker_name "$i").log" 2>/dev/null)"
    printf '  %-16s %s requests\n' "$(pd_worker_name "$i")" "${n:-0}"
    COUNTS+=("${n:-0}")
done

# The test: did every prefill worker get a share? Job 17682 measured 126/1/1/1
# with the router's default cache_aware policy. Anything that lopsided again
# means the per-model policy registry overrides --prefill-policy, which is the
# open question this job exists to close.
MIN=999999; MAX=0
for c in "${COUNTS[@]}"; do (( c < MIN )) && MIN=$c; (( c > MAX )) && MAX=$c; done
echo ""
echo "  spread: min=$MIN max=$MAX over ${#COUNTS[@]} prefill worker(s)"
if (( ${#COUNTS[@]} < 2 )); then
    echo "  !  only one prefill worker -- this job cannot see a routing problem."
    echo "     Use PD_SPEC with at least two, e.g. 1x2:1x1."
elif (( MIN == 0 )); then
    echo "  x  a prefill worker received NOTHING. The router is not distributing."
    echo "     With --prefill-policy $( [[ -n "${PD_PREFILL_POLICY:-}" ]] && echo "$PD_PREFILL_POLICY" || echo "unset" ) this means the"
    echo "     mode-level policy is being overridden -- check the per-model registry."
    RC=1
elif (( MAX > MIN * 3 )); then
    echo "  !  lopsided ($MIN vs $MAX). Not fatal at this request count, but watch it."
else
    echo "  ok every prefill worker took a share"
fi

echo ""
echo "  decode batch reached:"
for i in "${!PD_ROLE[@]}"; do
    [[ "${PD_ROLE[$i]}" == decode ]] || continue
    grep -oE '#running-req: [0-9]+' "$WORKDIR/$(pd_worker_name "$i").log" 2>/dev/null \
        | sort -t' ' -k2 -n | tail -1 | sed "s/^/    $(pd_worker_name "$i") max /" \
        || echo "    $(pd_worker_name "$i") no decode batches logged"
done

echo ""
echo "=============================================================="
(( ERR == 0 && RC == 0 )) \
    && echo " PASS -- $ENGINE PD works on one node, and the router distributes" \
    || echo " FAIL -- $OK ok / $ERR failed, routing rc=$RC"
echo " artefacts: $OUT_ROOT"
echo "=============================================================="
exit $(( ERR > 0 ? 1 : RC ))
