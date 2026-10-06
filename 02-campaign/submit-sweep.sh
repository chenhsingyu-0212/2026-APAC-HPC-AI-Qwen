#!/bin/bash
# Submit one sweep job per PD topology, chained through the queue.
#
#   ./run/submit-sweep.sh                            # the default three topologies
#   ./run/submit-sweep.sh "2x4:4x2 2x6:4x1"
#   CONCURRENCIES="16 40 56" ./run/submit-sweep.sh "2x4:4x2"
#   ./run/submit-sweep.sh --dry-run "2x4:4x2 2x2:4x3"
#
# Why this exists: the partition caps a job at 17 minutes, so one job holds one
# topology (see run/sweep-pd.sbatch). But the cost of a topology is not its 17
# minutes -- it is the wait for a slot. At the time of writing there were a dozen
# jobs queued by seven teams for two nodes, so a topology you submit only after
# the previous one finishes costs you another trip through the queue.
#
# So submit them all now, chained with afterany, and let Slurm run them back to
# back as slots come free. afterany and not afterok on purpose: a topology that
# fails to come up should not cancel the ones behind it. That is a data point,
# not a reason to stop.
#
# Each job writes results/<jobid>/sweep/<topology>/; compare them afterwards.

set -uo pipefail

# Slurm copies the batch script to /var/spool/slurm/job<id>/, so ${BASH_SOURCE[0]}
# points at that copy and not into the repo -- job 17590 died in one second on
# exactly this. SLURM_SUBMIT_DIR is where sbatch was invoked from; fall back to
# BASH_SOURCE so the script still works when run directly.
HERE=""
for _d in "${SLURM_SUBMIT_DIR:-}/run" "${SLURM_SUBMIT_DIR:-}" \
          "$(dirname "${BASH_SOURCE[0]}")" "$PWD/run" "$PWD"; do
    [[ -n "$_d" && -r "$_d/pd-lib.sh" ]] && { HERE="$(cd "$_d" && pwd)"; break; }
done
[[ -n "$HERE" ]] || { echo "x cannot find pd-lib.sh -- run this from the repo, or set SLURM_SUBMIT_DIR" >&2; exit 1; }
DRY=0
[[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }

# 8+8, 12+4 and 4+12: the split the application notes verified, and one step
# either side of it.
SPECS="${1:-2x4:4x2 2x6:4x1 2x2:4x3}"

[[ -f "$HERE/sweep-pd.sbatch" ]] || { echo "x $HERE/sweep-pd.sbatch not found" >&2; exit 1; }

# Reject a bad spec before anything reaches the queue.
# shellcheck source=run/pd-lib.sh
source "$HERE/pd-lib.sh" || { echo "x cannot source pd-lib.sh" >&2; exit 1; }
for spec in $SPECS; do
    pd_parse "$spec" >/dev/null || exit 1
    pd_place "${GPUS_PER_NODE:-8}" node0 node1 >/dev/null 2>&1 \
        || { echo "x '$spec' does not fit on 2 nodes of ${GPUS_PER_NODE:-8} GPUs" >&2; exit 1; }
done

echo "chaining $(wc -w <<< "$SPECS") topologies, one job each:"
PREV=""
for spec in $SPECS; do
    args=(--job-name="sweep-${spec//:/_}" --export="ALL,PD_SPEC=$spec")
    [[ -n "$PREV" ]] && args+=(--dependency="afterany:$PREV")

    if (( DRY )); then
        echo "  sbatch ${args[*]} $HERE/sweep-pd.sbatch"
        PREV="<job$(( ${#PREV} + 1 ))>"
        continue
    fi

    OUT="$(sbatch "${args[@]}" "$HERE/sweep-pd.sbatch" 2>&1)" || {
        echo "x sbatch refused '$spec': $OUT" >&2
        echo "  Nothing after it was submitted." >&2
        exit 1
    }
    JOB="${OUT##* }"
    printf '  %-12s job %-8s %s\n' "$spec" "$JOB" "${PREV:+after $PREV}"
    PREV="$JOB"
done

(( DRY )) && exit 0
echo ""
echo "watch:   squeue -u \$USER"
echo "cancel:  scancel -u \$USER --name=sweep-*"
echo ""
echo "Each job stops early and reports what it measured if the allocation runs"
echo "short, so a slow checkpoint load costs you points, not the whole job."
