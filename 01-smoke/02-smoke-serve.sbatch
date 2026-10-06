#!/bin/bash
#SBATCH --job-name=smoke-serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=hpcai
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#
# The hpcai partition caps a job at 17 minutes (sinfo TIMELIMIT 17:00), and
# Slurm's kill at the cap is a SIGKILL that leaves no artefacts. Every
# reference script under /scratch/reference-scripts uses -t 00:17:00 or less.
# Do not raise this: the job will simply be rejected at submission.
#
# Smoke test 2/3: a single non-disaggregated server loads a model, serves, and
# reports healthy.
#
#   sbatch run/02-smoke-serve.sbatch                        # ENGINE=sglang
#   ENGINE=trtllm sbatch run/02-smoke-serve.sbatch
#   SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/02-smoke-serve.sbatch
#   sbatch --gres=gpu:8 run/02-smoke-serve.sbatch    # only if you must use the 235B
#
# Use a small model. The point of this step is to prove that model loading,
# sharding and the HTTP serving path work at all, and a 0.6B answers that in a
# minute where the 235B costs you a large slice of your allocation to learn the
# same thing. Weight loading time is not what you are testing here.
#
# Exit code is the verdict: 0 means the server became healthy and shut down
# cleanly. No PD flags are involved yet -- that complexity comes in submit.sbatch.
#
# Run it for each engine you intend to use. Two engines that both reach PASS here
# are two engines you can compare later; one that does not is a Phase E you have
# just saved yourself.

set -uo pipefail

# Slurm copies the batch script to /var/spool/slurm/job<id>/, so ${BASH_SOURCE[0]}
# points at that copy and not into the repo -- job 17590 died in one second on
# exactly this. SLURM_SUBMIT_DIR is where sbatch was invoked from; fall back to
# BASH_SOURCE so the script still works when run directly.
HERE=""
for _d in "${SLURM_SUBMIT_DIR:-}/run" "${SLURM_SUBMIT_DIR:-}" \
          "$(dirname "${BASH_SOURCE[0]}")" "$PWD/run" "$PWD"; do
    [[ -n "$_d" && -r "$_d/engine-lib.sh" ]] && { HERE="$(cd "$_d" && pwd)"; break; }
done
[[ -n "$HERE" ]] || { echo "x cannot find engine-lib.sh -- run this from the repo, or set SLURM_SUBMIT_DIR" >&2; exit 1; }
# shellcheck source=run/engine-lib.sh
source "$HERE/engine-lib.sh" || { echo "x cannot source engine-lib.sh" >&2; exit 1; }

die() { echo "x $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

eng_load || exit 1
IMG="$(eng_image)"
# NOT $HOME -- see eng_mounts() in engine-lib.sh
MOUNTS="${MOUNTS:-$(eng_mounts smoke2-${SLURM_JOB_ID:-manual})}"
SMOKE_MODEL="${SMOKE_MODEL:-/scratch/models/Qwen3-0.6B}"
SERVED_NAME="${SERVED_NAME:-qwen}"
PORT="${PORT:-31000}"
TP="${TP:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
LOG="${LOG:-${SLURM_SUBMIT_DIR:-$PWD}/smoke-serve-$ENGINE-${SLURM_JOB_ID:-manual}.log}"

[[ -e "$IMG" ]] || die "container image not found: $IMG"
if [[ ! -d "$SMOKE_MODEL" && "$SMOKE_MODEL" != */* ]]; then
    die "SMOKE_MODEL '$SMOKE_MODEL' is neither a directory nor an HF id"
fi
if [[ "$SMOKE_MODEL" == /* && ! -d "$SMOKE_MODEL" ]]; then
    echo "x model path not found: $SMOKE_MODEL" >&2
    echo "  what is staged on shared storage:" >&2
    ls -1 "$(dirname "$SMOKE_MODEL")" 2>/dev/null | head -20 >&2
    echo "  set SMOKE_MODEL to one of those, or to a small HF id." >&2
    exit 1
fi

HOSTNAME_="$(hostname)"
eng_banner
say "serving $SMOKE_MODEL on $HOSTNAME_:$PORT (tp=$TP)"

# One argument per line, so a path or a flag value containing a space survives.
mapfile -t SERVE_CMD < <(eng_serve_cmd "$SMOKE_MODEL" "$SERVED_NAME" "$PORT" "$TP")
(( ${#SERVE_CMD[@]} )) || die "$ENGINE's eng_serve_cmd produced nothing"
say "  ${SERVE_CMD[*]}"

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && { say "stopping the server"; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
}
trap cleanup EXIT INT TERM

srun --container-image="$IMG" --container-mounts="$MOUNTS" \
     "${SERVE_CMD[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

say "waiting for $(eng_health_path) (up to ${HEALTH_TIMEOUT}s)"
eng_wait_healthy "http://$HOSTNAME_:$PORT" "$SERVER_PID" "$HEALTH_TIMEOUT" "$LOG" || exit 1

# /health alone only says the process is up. Ask what it thinks it is serving:
# a wrong served model name here becomes a wrong --model at benchmark time.
MODEL_ID="$(eng_model_id "http://$HOSTNAME_:$PORT")"
[[ -n "$MODEL_ID" ]] || { say "healthy, but $(eng_models_path) returned nothing"; exit 1; }
say "  ok $(eng_models_path) reports '$MODEL_ID'"
if [[ "$MODEL_ID" != "$SERVED_NAME" ]]; then
    say "  !  that is not SERVED_NAME ('$SERVED_NAME'). Not an error -- not every"
    say "     engine takes a naming flag -- but '$MODEL_ID' is what --model must be."
fi

echo "=========================================="
echo "PASS -- $ENGINE loaded the model, served it, and reported healthy"
echo " model id for the next step: $MODEL_ID"
echo " server log: $LOG"
echo "=========================================="
