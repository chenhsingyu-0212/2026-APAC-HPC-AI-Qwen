#!/bin/bash
# Submit a job with this site's spelling of the resource request.
#
#   run/submit-site.sh run/submit.sbatch
#   PD_SITE=nycu NODES=2 GPUS=8 run/submit-site.sh run/submit.sbatch
#   PD_SITE=nycu SITE_PARTITION=defq NODES=1 run/submit-site.sh run/05-pd-smoke.sbatch
#   PD_SITE=nscc NODES=2 GPUS=8 run/submit-site.sh run/10-campaign.sbatch
#
# Two schedulers now. The batch scripts carry BOTH sets of directives, which works
# because each scheduler treats the other's as an ordinary comment: `#PBS` is a
# comment to sbatch, `#SBATCH` is a comment to qsub. So one script serves both and
# this wrapper supplies the resource request in the right dialect.
#
# The #SBATCH lines inside a batch script are static text, and the two clusters
# want different text for the same request:
#
#   Firmus   --gres=gpu:8   --partition=hpcai  --time=00:17:00 (hard cap)
#   NYCU     --gres=gpu:gpu:8  --partition=h200q  no cap
#
# NYCU's form is enforced by a job_submit plugin, not a convention -- `--gres=gpu:8`
# is rejected at submission with "You must specify GPU type explicitly", so the
# scripts cannot simply carry one spelling and be portable. Command-line options
# override #SBATCH directives, so this wrapper supplies them and the scripts keep
# their Firmus defaults for anyone submitting them directly.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$HERE/site-lib.sh" || exit 1
site_load || exit 1

SCRIPT="${1:-}"
[[ -n "$SCRIPT" ]] || { echo "usage: ${0##*/} <sbatch script> [extra sbatch args...]" >&2; exit 2; }
shift

NODES="${NODES:-${SITE_NODES:-2}}"
GPUS="${GPUS:-${SITE_GPUS_PER_NODE:-8}}"        # per node

# Let a site object to a shape that does not match what it is for.
declare -F nscc_check_shape >/dev/null 2>&1 && nscc_check_shape "$NODES" "$GPUS"

# The batch script reads GPUS_PER_NODE; this wrapper reads GPUS. They name the same
# quantity and nothing kept them in step, so they drifted: job 81755 was submitted
# with GPUS=4, the script fell back to its hardcoded GPUS_PER_NODE=8, and every
# per-GPU figure in that run was divided by twice the GPUs that existed -- reporting
# LMDeploy on H200 at 44.0 tok/s/GPU when it was 88.0. On a PD shape the same drift
# would have had pd_place assigning workers to GPUs 4-7 of a 4-GPU allocation.
export GPUS_PER_NODE="$GPUS"

# --time: the site cap when there is one, otherwise be explicit rather than
# inheriting UNLIMITED. An unlimited job that hangs holds the nodes until someone
# notices, and on NYCU two H200 nodes sat DRAINed for three days after exactly
# that kind of stuck job ("Kill task failed : Not responding").
if (( ${SITE_DEFAULT_MINUTES:-0} > 0 )); then
    TIME_LIMIT="${TIME_LIMIT:-$(printf '%02d:%02d:00' \
        $(( SITE_DEFAULT_MINUTES / 60 )) $(( SITE_DEFAULT_MINUTES % 60 )))}"
elif (( ${SITE_MAX_MINUTES:-0} > 0 )); then
    # HH:MM:SS, not 00:<minutes>:00. Firmus caps a job at 17 minutes so the old
    # printf '00:%02d:00' happened to be right there; NSCC's aiq1 allows 1440, which
    # came out as the malformed "00:1440:00" -- and priced at 384 GPU-hours, reserving
    # 49,152 SU of a 64,000 grant for a job nobody meant to run that long.
    TIME_LIMIT="${TIME_LIMIT:-$(printf '%02d:%02d:00' \
        $(( SITE_MAX_MINUTES / 60 )) $(( SITE_MAX_MINUTES % 60 )))}"
else
    TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
fi

# ---------------------------------------------------------------- PBS
if [[ "${SITE_SCHEDULER:-slurm}" == pbs ]]; then
    # -P is mandatory and there is exactly one project that works; see
    # run/sites/nscc.sh. Without it qsub refuses outright.
    [[ -n "${SITE_PROJECT:-}" ]] || { echo "x SITE_PROJECT is required on PBS" >&2; exit 2; }

    # ncpus is not a free choice here: the site enforces 14 CPUs per GPU and a
    # request for fewer comes back rounded up, so asking for the ratio directly is
    # the only way to predict what will be charged.
    NCPUS=$(( GPUS * ${SITE_CPUS_PER_GPU:-14} ))

    # place=scatter, and this is load-bearing. `-l select=2:...` means two CHUNKS,
    # not two nodes: job 213470 asked for select=2 and PBS packed both chunks onto
    # a2ap-dgx037, so a two-node PD deployment would have quietly become a
    # one-node one with half the GPUs it asked for.
    PLACE="${PBS_PLACE:-scatter}"

    declare -a QARGS=(
        -P "$SITE_PROJECT"
        # -V exports the submitting environment. Slurm needs --export=ALL for this
        # and PBS needs -V, and WITHOUT IT NOTHING IS PASSED: job 214488 reached the
        # compute node with no PD_SITE, no ENGINES, no SPECS and no CONCURRENCIES,
        # fell back to site_detect(), and was handed the wrong cluster's model path.
        -V
        -l "select=${NODES}:ncpus=${NCPUS}:ngpus=${GPUS}"
        -l "place=${PLACE}"
        -l "walltime=${TIME_LIMIT}"
        -j oe
    )
    # NO -q. aiq1, aiq2 and aidev all reject a direct submission ("Route queue
    # only"); submitting with no queue lets the `normal` routing queue place it.
    [[ -n "${JOB_NAME:-}" ]] && QARGS+=(-N "$JOB_NAME")
    [[ -n "${SITE_SCRATCH:-}" ]] && QARGS+=(-o "$SITE_SCRATCH/logs/")

    # The price, before the decision rather than after it. This is a prepaid
    # allocation: the SU come out of the balance the moment the job starts and sit
    # in "In Doubt" for its whole walltime, whether or not it uses it.
    if declare -F nscc_su >/dev/null 2>&1; then
        HOURS="$(awk -F: '{printf "%.4f", $1 + $2/60 + $3/3600}' <<< "$TIME_LIMIT")"
        echo "site      ${SITE_NAME:-$PD_SITE}"
        echo "price     $(nscc_su "$(( NODES * GPUS ))" "$HOURS")  for ${NODES}x${GPUS} GPUs over $TIME_LIMIT"
    fi
    echo "submit    qsub ${QARGS[*]} $* $SCRIPT"
    exec qsub "${QARGS[@]}" "$@" "$SCRIPT"
fi

# ---------------------------------------------------------------- Slurm
declare -a ARGS=(
    --partition="${SITE_PARTITION}"
    --nodes="$NODES"
    "$(site_gpu_arg "$GPUS")"
    --time="$TIME_LIMIT"
)
# NCHC's plugin refuses a job with no account and prints the wallet balance while
# doing it. SITE_ACCOUNT is empty everywhere else, so nothing changes there.
[[ -n "${SITE_ACCOUNT:-}" ]] && ARGS+=(--account="$SITE_ACCOUNT")
# Memory and CPUs: explicit where the site caps them per job, --exclusive where
# it does not. NYCU's association allows cpu=96, gres/gpu=4, mem=1000G per job,
# and an --exclusive request for the node's whole 2 TB is simply refused.
[[ -n "${SITE_JOB_MEM:-}"  ]] && ARGS+=(--mem="${JOB_MEM:-$SITE_JOB_MEM}")
[[ -n "${SITE_JOB_CPUS:-}" ]] && ARGS+=(--cpus-per-task="${JOB_CPUS:-$SITE_JOB_CPUS}")
[[ -n "${NODELIST:-}" ]] && ARGS+=(--nodelist="$NODELIST")
[[ -n "${EXCLUSIVE:-}" ]] && ARGS+=(--exclusive)

echo "site      ${SITE_NAME:-$PD_SITE}"
# The PBS branch above prints a price because NSCC's allowance cannot be refilled.
# Nano5's can, but it is SHARED with the project's other users and holds about
# 3,000 SU, so a 16-GPU job is a visible fraction of it and the number belongs on
# screen before the job goes out rather than in a comment somewhere.
if declare -F "${PD_SITE}_su" >/dev/null 2>&1; then
    HOURS="$(awk -F: '{printf "%.4f", $1 + $2/60 + $3/3600}' <<< "$TIME_LIMIT")"
    echo "price     $("${PD_SITE}_su" "$(( NODES * GPUS ))" "$HOURS")  for ${NODES}x${GPUS} GPUs over $TIME_LIMIT"
fi
echo "submit    sbatch ${ARGS[*]} $* $SCRIPT"
exec sbatch "${ARGS[@]}" "$@" "$SCRIPT"
