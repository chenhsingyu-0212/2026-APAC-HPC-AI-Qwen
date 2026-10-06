#!/bin/bash
#SBATCH --job-name=qwen3-pd-sweep
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
# Search the Prefill:Decode ratio. Exploratory, not scored.
#
#   sbatch run/sweep-pd.sbatch                       # one topology, many N
#   PD_SPEC=2x6:4x1 sbatch run/sweep-pd.sbatch
#   ./run/submit-sweep.sh "2x4:4x2 2x6:4x1 2x2:4x3"  # one chained job per topology
#
# ONE TOPOLOGY PER JOB. The partition caps a job at 17 minutes and loading the
# 235B checkpoint spends 2-4 of them, so a second topology in the same job would
# pay that load twice out of a budget that cannot afford it once more. Concurrency
# is swept inside the job, because that is the axis that is free once the weights
# are resident. Use run/submit-sweep.sh to chain a job per topology through the
# queue -- with a dozen jobs queued for two nodes, the queue is the real cost.
#
# The rules leave your own testing unconstrained and say so explicitly: any
# duration, any seed, any sweep. Only the scored re-run is fixed. So this uses a
# short benchmark window -- long enough to separate two configurations, far
# short of the official 120s -- and keeps everything that decides *comparability*
# identical to the official command: the same 4000/200 shape and the same
# goodput thresholds, so a pass rate here means what it will mean there.
#
# Redeploying is what costs: loading a 235B checkpoint dwarfs a 15-second
# measurement. So each topology is brought up once and every concurrency is
# measured before it comes down. Comparing ratios at a single fixed N would be
# misleading anyway -- the best N moves with the ratio.
#
# What comes out: results/<job>/sweep/summary.txt, ranked by good output tok/s
# among the points that clear the 0.90 gate, and the PD_SPEC of the winner ready
# to hand to run/submit.sbatch.

set -uo pipefail

# ============================================================ configuration
SGLANG_IMG="${SGLANG_IMG:-/scratch/enroot-images/sglang-harbor.sqsh}"
AIPERF_IMG="${AIPERF_IMG:-/scratch/enroot-images/aiperf.sqsh}"
MODEL_PATH="${MODEL_PATH:-/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
KIT_DIR="${KIT_DIR:-$SLURM_SUBMIT_DIR/AIPerf}"
MOUNTS="${MOUNTS:-/scratch:/scratch,$HOME:$HOME}"

# One topology. PD_SPEC is the name submit.sbatch uses, accepted here too so a
# winner moves between them without an edit. SPECS still takes a list, and is
# still honoured, but a list will run out of time -- submit-sweep.sh is the
# supported way to cover several.
SPECS="${SPECS:-${PD_SPEC:-2x4:4x2}}"
# The concurrencies the organisers' own 09-pd-235b-fatp-peak-bench.sbatch sweeps
# on this hardware at ISL 4000 -- an informed starting point, not a rule.
# Jobs 18964/18965 measured ITL p90 18.74 ms at N=40 against a 20 ms gate -- so
# the cliff is within a few steps. Bracket it tightly and keep two points below
# 40 for the slope: a cliff you cannot see the approach to is a guess.
CONCURRENCIES="${CONCURRENCIES:-24 32 40 44 48 56}"

SERVED_NAME="${SERVED_NAME:-qwen}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-nixl}"
MAX_RUNNING="${MAX_RUNNING:-256}"
DECODE_EP="${DECODE_EP:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
PD_CPUS_PER_WORKER="${PD_CPUS_PER_WORKER:-16}"
PD_ROUTER_CPUS="${PD_ROUTER_CPUS:-2}"

ISL="${ISL:-4000}"; OSL="${OSL:-200}"
SLA="${SLA:-time_to_first_token:3000 inter_token_latency:20}"
BENCH_SECONDS="${BENCH_SECONDS:-15}"     # 10-20s separates configurations
WARMUP_SECONDS="${WARMUP_SECONDS:-20}"
GRACE_SECONDS="${GRACE_SECONDS:-15}"     # exploratory; the official 30 is for the scored run
# What one point costs end to end, for the budget check: the benchmark window
# plus the client's own startup. Deliberately generous -- overrunning loses the
# whole job to a SIGKILL, and with it every point already measured.
POINT_SECONDS="${POINT_SECONDS:-$(( WARMUP_SECONDS + BENCH_SECONDS + GRACE_SECONDS + 25 ))}"
OUT_ROOT="${OUT_ROOT:-$SLURM_SUBMIT_DIR/results/${SLURM_JOB_ID:-manual}}"

# ============================================================ preconditions
die() { echo "x $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

for f in "$SGLANG_IMG" "$AIPERF_IMG"; do [[ -e "$f" ]] || die "container image not found: $f"; done
[[ -d "$MODEL_PATH" ]] || die "model path not found: $MODEL_PATH"
for s in observe.sh diagnose.sh gpu_sample.sh; do
    [[ -x "$KIT_DIR/$s" ]] || die "$KIT_DIR/$s missing or not executable"
done

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
# shellcheck source=run/pd-lib.sh
source "$HERE/pd-lib.sh" || die "cannot source pd-lib.sh"
# shellcheck source=run/engine-lib.sh
source "$HERE/engine-lib.sh" || die "cannot source engine-lib.sh"
eng_load || exit 1
SGLANG_IMG="$(eng_image)"

# Slurm expands a range like "DGX-H200-CN[05-06]"; PBS lists one line per chunk
# in $PBS_NODEFILE. Same answer, two mechanisms -- see run/sched-lib.sh.
mapfile -t NODES < <(sched_nodes)
(( ${#NODES[@]} >= 2 )) || die "need 2 nodes, got ${#NODES[@]}"
# Before anything is deployed: is every node actually able to run work? PBS hands
# out nodes that accept a job, report R and then answer no launcher at all, and a
# worker that never starts still bills for the walltime it holds. A no-op on Slurm.
sched_preflight "${NODES[@]}" || die "one of the allocated nodes cannot run work"
sched_reset_pools "${NODES[@]}"
HEAD="${NODES[0]}"

# Reject an unplaceable spec now rather than after an hour of the sweep.
for spec in $SPECS; do
    ( pd_parse "$spec" && pd_place "$GPUS_PER_NODE" "${NODES[@]}" ) >/dev/null 2>&1 \
        || die "spec '$spec' cannot be placed on ${#NODES[@]} nodes of $GPUS_PER_NODE GPUs"
done
say "$(wc -w <<<"$SPECS") topologies x $(wc -w <<<"$CONCURRENCIES") concurrencies"

SWEEP="$OUT_ROOT/sweep"; mkdir -p "$SWEEP" || die "cannot create $SWEEP"

# ============================================================ telemetry
# One sampler for the whole sweep. Every point records its own window, so
# diagnose.sh slices the right slice out of a shared time series.
TEL_PIDS=()
for n in "${NODES[@]}"; do
    srun --overlap -N1 -n1 -w "$n" \
         "$KIT_DIR/gpu_sample.sh" --out "$OUT_ROOT/tel-$n" --label "$n" --interval 2 \
         > "$SWEEP/tel-$n.log" 2>&1 &
    TEL_PIDS+=($!)
done

cleanup() {
    pd_teardown
    local p
    for p in "${TEL_PIDS[@]:-}"; do
        [[ -n "$p" ]] || continue
        kill "$p" 2>/dev/null
        wait "$p" 2>/dev/null
    done
    TEL_PIDS=()
}
trap cleanup EXIT INT TERM

export MODEL_PATH SERVED_NAME CONTEXT_LEN MEM_FRACTION \
       MEM_FRACTION_PREFILL MEM_FRACTION_DECODE TRANSFER_BACKEND \
       PD_EXTRA PREFILL_EXTRA DECODE_EXTRA \
       DECODE_EP MAX_RUNNING
export PD_IMAGE="$SGLANG_IMG" PD_MOUNTS="$MOUNTS" PD_CPUS_PER_WORKER PD_ROUTER_CPUS

# ============================================================ the sweep
FAILED=()
for spec in $SPECS; do
    TAG="${spec//:/_}"
    DIR="$SWEEP/$TAG"; mkdir -p "$DIR"
    export PD_WORK="$DIR/deploy"; mkdir -p "$PD_WORK"

    echo ""
    echo "=========================================="
    pd_parse "$spec" || { FAILED+=("$spec:parse"); continue; }
    pd_place "$GPUS_PER_NODE" "${NODES[@]}" || { FAILED+=("$spec:place"); continue; }
    say "topology $spec -> $(pd_split) GPUs"
    pd_print_placement
    echo "=========================================="

    # A topology that cannot reach even one measured point is not worth the
    # weights load: it would burn the rest of the allocation and report nothing.
    pd_need_time $(( HEALTH_TIMEOUT / 4 + POINT_SECONDS + 60 )) "topology $spec" \
        || { FAILED+=("$spec:no-time"); break; }

    # Resolve first: an engine whose prepare step writes a gateway config by
    # address needs PD_IP, and that is what pd_resolve fills. See the note in
    # 10-campaign.sbatch -- the same order was wrong there and cost job 215874.
    pd_resolve || { FAILED+=("$spec:resolve"); continue; }
    declare -F eng_pd_prepare >/dev/null && \
        { eng_pd_prepare "$PD_WORK" || { FAILED+=("$spec:prepare"); continue; }; }
    pd_write_launcher "$PD_WORK/worker.sh"
    GATEWAY="$(pd_gateway "$ROUTER_PORT")"
    pd_launch "$PD_WORK/worker.sh" || { FAILED+=("$spec:launch"); pd_teardown; continue; }

    if ! pd_wait_all "$HEALTH_TIMEOUT"; then
        say "! $spec never came up -- skipping it"
        FAILED+=("$spec:health"); pd_teardown; continue
    fi

    pd_launch_router "$ROUTER_PORT" || { FAILED+=("$spec:router"); pd_teardown; continue; }
    if ! pd_wait_healthy "http://$GATEWAY" "router($spec)" "$HEALTH_TIMEOUT" \
                         "${PD_PIDS[-1]}" "$PD_WORK/router.log"; then
        FAILED+=("$spec:router"); pd_teardown; continue
    fi

    # Not a single shot: /health is true before the gateway has its workers.
    MODEL_ID="$(pd_wait_models "http://$GATEWAY" 120 "${PD_PIDS[-1]}" "$PD_WORK/gateway.log")" \
        || { FAILED+=("$spec:model-id"); pd_teardown; continue; }

    # Pay the first-request cost once, before the first measured point, so it is
    # not charged to whichever concurrency happens to be swept first. Jobs 18964
    # and 18965 measured this at 38-61s with round-robin routing.
    PREWARM_REQUESTS="${PREWARM_REQUESTS:-$(( PD_P_N * 2 ))}"
    if (( PREWARM_REQUESTS > 0 )); then
        say "  prewarming with aiperf: $PREWARM_REQUESTS requests"
        _pw=$SECONDS
        srun --overlap -N1 -n1 -w "$HEAD" \
             --container-image="$AIPERF_IMG" --container-mounts="$MOUNTS" \
             --container-workdir="$PD_WORK" \
             aiperf profile --artifact-dir "$PD_WORK/prewarm" \
                --url "http://$GATEWAY" --endpoint-type chat --endpoint /v1/chat/completions \
                --model "$MODEL_ID" --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 \
                --streaming -H 'Accept: text/event-stream' --api-key NOT_USED \
                --concurrency "$PREWARM_REQUESTS" --request-count "$PREWARM_REQUESTS" \
                --synthetic-input-tokens-mean "$ISL" --synthetic-input-tokens-stddev 0 \
                --output-tokens-mean 16 --output-tokens-stddev 0 \
                --extra-inputs min_tokens:16 --extra-inputs max_tokens:16 \
                --extra-inputs ignore_eos:true --ui simple \
                > "$PD_WORK/prewarm.log" 2>&1
        say "  prewarm done in $(( SECONDS - _pw ))s"
    fi
    { echo "pd_spec=$spec split=$(pd_split)"; pd_print_placement; } > "$DIR/deployment.txt"

    mapfile -t METRICS_ARGS < <(pd_metrics_args)

    # Deployed once, measured at every concurrency: the model load dwarfs the
    # measurement, and the best N moves with the ratio.
    for N in $CONCURRENCIES; do
        # Stop cleanly with the points already measured rather than being killed
        # part-way through writing the next one.
        if ! pd_need_time "$POINT_SECONDS" "c$N"; then
            say "  out of allocation -- stopping after the points already measured"
            break
        fi
        say "  $spec @ c$N (${BENCH_SECONDS}s benchmark)"
        srun --overlap -N1 -n1 -w "$HEAD" \
             --container-image="$AIPERF_IMG" --container-mounts="$MOUNTS" \
             "$KIT_DIR/observe.sh" --out "$DIR" --concurrency "$N" \
                --isl "$ISL" --osl "$OSL" --sla "$SLA" --model "$MODEL_ID" \
                "${METRICS_ARGS[@]}" -- \
             aiperf profile \
                --artifact-dir "$DIR/c$N/artifacts" \
                --url "http://$GATEWAY" \
                --endpoint-type chat --endpoint /v1/chat/completions \
                --model "$MODEL_ID" --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 \
                --streaming -H 'Accept: text/event-stream' \
                --concurrency "$N" --workers-max "$N" --api-key NOT_USED \
                --warmup-duration "$WARMUP_SECONDS" --benchmark-duration "$BENCH_SECONDS" \
                --benchmark-grace-period "$GRACE_SECONDS" \
                --synthetic-input-tokens-mean "$ISL" --synthetic-input-tokens-stddev 0 \
                --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
                --extra-inputs min_tokens:"$OSL" --extra-inputs max_tokens:"$OSL" \
                --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 \
                --num-dataset-entries 2000 --random-seed 2026 \
                --cache-bust first_turn_prefix \
                --goodput "$SLA" --ui simple \
             > "$DIR/c$N.log" 2>&1 \
            || { say "  ! $spec @ c$N failed (see $DIR/c$N.log)"; FAILED+=("$spec@c$N"); }
    done

    pd_teardown
    say "  $spec done, torn down"
done

# ============================================================ summary
python3 - "$SWEEP" "$OUT_ROOT" <<'PYEOF'
import csv, os, sys
sweep, root = sys.argv[1], sys.argv[2]

rows = []
for tag in sorted(os.listdir(sweep)):
    f = os.path.join(sweep, tag, "metrics.csv")
    if not os.path.isfile(f):
        continue
    spec = tag.replace("_", ":")
    for r in csv.DictReader(f and open(f)):
        def num(k, d=None):
            try:    return float(r.get(k) or "")
            except ValueError: return d
        g, p, o = num("goodput"), num("pass_rate"), num("osl") or 200
        if g is None:
            continue
        rows.append({"spec": spec, "point": r["point"], "goodput": g,
                     "pass": p, "toks": g * o, "ttft": num("ttft_p90"),
                     "itl": num("itl_p90") or num("itl_avg")})

if not rows:
    print("\nno parsable results -- every topology failed"); sys.exit(1)

gate = [r for r in rows if (r["pass"] or 0) >= 0.90]
gate.sort(key=lambda r: -r["toks"])
out = []
out.append("")
out.append("=" * 76)
out.append(" P:D ratio sweep -- ranked by good output tok/s among points clearing 0.90")
out.append("=" * 76)
out.append(" %-10s %-7s %10s %7s %9s %9s %8s" %
           ("spec", "point", "good tok/s", "pass", "goodput", "ttft p90", "itl p90"))
for r in (gate or rows):
    out.append(" %-10s %-7s %10.0f %7.3f %9.2f %9.0f %8.1f" %
               (r["spec"], r["point"], r["toks"], r["pass"] or 0, r["goodput"],
                r["ttft"] or 0, r["itl"] or 0))

if not gate:
    out.append("")
    out.append(" ! nothing cleared good_request_fraction >= 0.90 -- the rows above are")
    out.append("   unscoreable. Lower the concurrencies before reading anything into them.")
else:
    best = gate[0]
    out.append("")
    out.append(" winner   %s at %s -- %.0f good output tok/s (pass %.3f)"
               % (best["spec"], best["point"], best["toks"], best["pass"]))
    out.append("")
    out.append(" Take it to a scored run with:")
    out.append("   PD_SPEC=%s CONCURRENCY=%s sbatch run/submit.sbatch"
               % (best["spec"], best["point"].lstrip("c")))
    # a ratio that wins by less than the measurement's own noise has not won
    same = [r for r in gate if r["spec"] == best["spec"]]
    other = [r for r in gate if r["spec"] != best["spec"]]
    if other:
        margin = 100 * (best["toks"] - max(o["toks"] for o in other)) / best["toks"]
        out.append("")
        out.append(" margin over the next topology: %.1f%%" % margin)
        if margin < 5:
            out.append(" ! that is inside the range a short run can produce by itself.")
            out.append("   Re-measure both at the full duration before choosing.")

txt = "\n".join(out) + "\n"
print(txt)
open(os.path.join(sweep, "summary.txt"), "w").write(txt)
PYEOF

echo ""
if (( ${#FAILED[@]} > 0 )); then
    echo "! ${#FAILED[@]} point(s) or topologies failed: ${FAILED[*]}"
fi
echo "artifacts in $OUT_ROOT   summary: $SWEEP/summary.txt"
echo "diagnose any single topology with:"
echo "  $KIT_DIR/diagnose.sh $SWEEP/<spec> --telemetry $OUT_ROOT/tel-<node> \\"
echo "      --model-dir $MODEL_PATH --weights \${WEIGHTS_GIB:-20.5} --tp <decode tp>"
