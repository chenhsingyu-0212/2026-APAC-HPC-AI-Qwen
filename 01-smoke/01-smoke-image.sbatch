#!/bin/bash
#SBATCH --job-name=smoke-image
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
# Smoke test 1/3: the container starts, the GPU is visible inside it, and the
# serving engine is usable.
#
#   sbatch run/01-smoke-image.sbatch                        # ENGINE=sglang
#   ENGINE=trtllm sbatch run/01-smoke-image.sbatch
#   ENGINE=vllm   sbatch run/01-smoke-image.sbatch
#   ENGINE=trtllm ENGINE_IMG=/scratch/enroot-images/x.sqsh sbatch run/01-smoke-image.sbatch
#
# What "usable" means is the engine's business: run/engines/<name>.sh supplies
# the check. For TensorRT-LLM that includes verifying every CLI flag this repo
# is about to use against the image's own --help, so a renamed flag fails here,
# in fifteen minutes, rather than inside a PD job where it reads as a topology
# problem.
#
# The exit code is the whole verdict: 0 means the image, the GPU allocation and
# the engine environment inside the container are all sound. Anything else and
# there is no point looking at PD topologies yet.
#
# Submit it. Do not reach for `srun --pty` to poke at this interactively: an
# interactive session holds the GPU allocation for as long as it stays open, and
# most schedulers lower your priority the more time you have recently used, so
# idle exploration pushes your own later runs down the queue.

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

eng_load || exit 1
IMG="$(eng_image)"
# NOT $HOME -- see eng_mounts() in engine-lib.sh
MOUNTS="${MOUNTS:-$(eng_mounts smoke1-${SLURM_JOB_ID:-manual})}"

[[ -e "$IMG" ]] || die "container image not found: $IMG
  set ENGINE_IMG, or see what is staged:  ls /scratch/enroot-images/"

eng_banner
echo " mounts: $MOUNTS"
echo "=========================================="

# The engine's check is emitted as text and spliced into the middle of this
# script, so it travels into the container without the repo needing to be
# mounted there.
check_script() {
cat <<'PRE'
set -uo pipefail
echo "--- GPUs visible inside the container ---"
nvidia-smi -L || { echo "x nvidia-smi found no device" >&2; exit 1; }
[[ -n "$(nvidia-smi -L 2>/dev/null)" ]] || { echo "x empty device list" >&2; exit 1; }

echo "--- engine ---"
PRE
eng_import_check
cat <<'POST'

echo "--- writable HOME ---"
t="$HOME/.smoke-$$"
touch "$t" 2>/dev/null && rm -f "$t" || { echo "x \$HOME is not writable inside the container" >&2; exit 1; }
POST
}

# --container-mounts binds /scratch (images and weights) and a writable $HOME:
# the container filesystem itself is read-only and ephemeral, and enough tooling
# writes to $HOME on import that leaving it unbound fails in confusing ways.
srun --container-image="$IMG" --container-mounts="$MOUNTS" \
     bash -c "$(check_script)"
RC=$?

echo "=========================================="
(( RC == 0 )) && echo "PASS -- image, GPU and $ENGINE are all sound" \
              || echo "FAIL -- exit $RC; fix this before smoke test 2"
echo "=========================================="
exit $RC
