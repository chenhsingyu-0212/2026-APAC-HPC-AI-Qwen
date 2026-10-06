#!/bin/bash
#SBATCH --job-name=firstrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --partition=hpcai
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#
# The hpcai partition caps a job at 17 minutes (sinfo TIMELIMIT 17:00), and
# Slurm's kill at the cap is a SIGKILL that leaves no artefacts. Every
# reference script under /scratch/reference-scripts uses -t 00:17:00 or less.
# Do not raise this: the job will simply be rejected at submission.
#
# Everything that can be learned from one GPU, in one trip through the queue.
#
#   sbatch run/00-firstrun.sbatch
#   ENGINE=trtllm sbatch run/00-firstrun.sbatch     # once such an image exists
#
# Four stages that were four jobs. With a dozen jobs queued by seven teams for
# two nodes, the queue is what a job costs -- not its 15 minutes. Four
# submissions is four waits; this is one. On a 0.6B model the whole thing runs
# in six to eight minutes, so the cap is a safety net rather than a budget.
#
#   0  recon        what is in the container, what the CLI looks like, how many
#                   InfiniBand ports this node actually has, what /metrics holds
#   1  smoke image  container starts, GPU visible, engine importable
#   2  smoke serve  a single non-PD server loads and answers /health
#   3  smoke aiperf the benchmark client completes a round trip against it
#
# Each stage is the standalone script, called as-is -- the #SBATCH lines in them
# are comments once we are already inside an allocation, and their srun calls
# become steps of this job. Nothing is duplicated here, so fixing a stage fixes
# it for the standalone path too.
#
# Stages run in order and stop at the first failure: 2 cannot mean anything if 1
# failed, and 3 cannot mean anything if 2 did. Stage 0 always runs first, because
# its answers are worth having even when everything after it fails.

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
# shellcheck source=run/pd-lib.sh
source "$HERE/pd-lib.sh" || { echo "x cannot source pd-lib.sh" >&2; exit 1; }

export ENGINE="${ENGINE:-sglang}"
export SMOKE_MODEL="${SMOKE_MODEL:-/scratch/models/Qwen3-0.6B}"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$HERE/..}"

# What each stage needs, generously. Overrunning loses the whole job, including
# the stages that already passed.
declare -A NEED=( [0]=240 [1]=120 [2]=180 [3]=240 )

STAGES=(
    "0|recon|$HERE/00-recon.sbatch"
    "1|smoke-image|$HERE/01-smoke-image.sbatch"
    "2|smoke-serve|$HERE/02-smoke-serve.sbatch"
    "3|smoke-aiperf|$HERE/03-smoke-serve-aiperf.sbatch"
)

echo "=============================================================="
echo " first run on $(hostname)   job ${SLURM_JOB_ID:-manual}"
echo " engine=$ENGINE  model=$SMOKE_MODEL"
echo " budget: $(pd_time_left)s"
echo "=============================================================="

declare -a RESULT=()
FAILED_AT=""

for entry in "${STAGES[@]}"; do
    IFS='|' read -r n name script <<< "$entry"

    if [[ ! -r "$script" ]]; then
        RESULT+=("$n $name SKIP (no $script)"); continue
    fi
    if ! pd_need_time "${NEED[$n]}" "stage $n ($name)"; then
        RESULT+=("$n $name SKIP (out of allocation)")
        continue
    fi

    echo ""
    echo "##############################################################"
    echo "# stage $n: $name"
    echo "##############################################################"
    START=$SECONDS
    bash "$script"
    RC=$?
    TOOK=$(( SECONDS - START ))

    if (( RC == 0 )); then
        RESULT+=("$n $name PASS (${TOOK}s)")
    else
        RESULT+=("$n $name FAIL rc=$RC (${TOOK}s)")
        # Stage 0 is diagnostic, not a gate -- a container it could not inspect
        # is exactly what stage 1 is for. Anything after 0 gates what follows.
        if (( n > 0 )); then FAILED_AT="$n ($name)"; break; fi
    fi
done

echo ""
echo "=============================================================="
echo " summary"
echo "=============================================================="
printf '  %s\n' "${RESULT[@]}"
echo ""
echo "  time left: $(pd_time_left)s"

if [[ -n "$FAILED_AT" ]]; then
    echo ""
    echo "  Stopped at stage $FAILED_AT. Later stages would only have"
    echo "  repeated the same failure and spent the allocation doing it."
    echo "  Fix it, then re-run this whole script -- one queue trip, not four."
    exit 1
fi

echo ""
echo "  All stages passed. That is checklist item 2, and the engine backend"
echo "  works end to end on this cluster."
echo ""
echo "  Read stage 0's InfiniBand section before submitting anything on two"
echo "  nodes: the rules' platform has 8x NDR, and a PD job that cannot see"
echo "  eight ACTIVE ports will move the KV cache over the wrong path without"
echo "  ever erroring."
echo ""
echo "  Next: run/submit.sbatch -- the first two-node slot."
