#!/bin/bash
# Site selection. The engine layer answers "which server do I start"; this one
# answers "on which machine, and how does that machine spell things".
#
#   PD_SITE=nycu sbatch run/submit.sbatch
#
# It exists because the two clusters we can reach disagree on almost every
# mechanical detail while agreeing on the benchmark:
#
#                        Firmus                      NYCU
#   containers           enroot, .sqsh images        NONE AT ALL
#   engines              container / pip venv        pip venv only
#   --gres spelling      gpu:8                       gpu:gpu:8   (site plugin)
#   partition            hpcai                       defq (H100) / h200q (H200)
#   wall clock           17 min hard cap              UNLIMITED
#   model                /scratch/models/...         /home/hsingyu/models/...
#
# The NYCU gres spelling is not a style preference. A job_submit plugin rejects
# the ordinary form outright:
#
#   sbatch: error: Error: You must specify GPU type explicitly.
#   sbatch: error:   - For full GPU: --gres=gpu:gpu:4
#
# so a script that hardcodes `--gres=gpu:8` cannot be submitted there at all.
#
# ------------------------------------------------------------- what a site sets
#
#   SITE_NAME            for logs
#   SITE_PARTITION       default partition
#   SITE_GRES_FMT        printf format taking the GPU count, e.g. 'gpu:%d'
#   SITE_MAX_MINUTES     partition wall-clock cap, 0 for none. The budget guards
#                        in pd-lib.sh read the real limit from Slurm; this is for
#                        choosing a --time at submission.
#   SITE_GPUS_PER_NODE
#   MODEL_PATH           default checkpoint
#   SGLANG_IMG / VLLM_IMG / TRTLLM_IMG
#                        EMPTY STRING means bare metal, deliberately. The engine
#                        files use ${VAR-default} (no colon) so an explicit empty
#                        value survives and is not replaced by the default.
#   SGLANG_VENV / VLLM_VENV / TRTLLM_VENV
#                        when set, the engine runs from that venv's bin/
#   AIPERF_IMG           client container, or empty
#   AIPERF_BIN           client binary when there is no container
#   SITE_PROLOGUE        shell snippet to eval before touching Slurm commands
#
# Every one of these is only a default: the environment still wins, so
# `MODEL_PATH=/tmp/tiny PD_SITE=nycu sbatch ...` does what it says.

if [[ -z "${SITE_DIR:-}" ]]; then
    _site_self="${BASH_SOURCE[0]:-$0}"
    SITE_DIR="$(cd "$(dirname "$_site_self")/sites" 2>/dev/null && pwd)"
fi

site_available() { ls "$SITE_DIR" 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' '; }

# Guess from the machine we are on, by looking for something only that machine
# has rather than by matching hostnames. NYCU's login node is called `login`
# (`login.cm.cluster`) and its compute nodes `DGX-CN01`, `DGX-H200-CN03` --
# nothing in any of those names says NYCU, and a pattern list of them is one
# renamed node away from being wrong. /cm is Bright Cluster Manager's root and
# is present on every node of that cluster; /scratch/reference-scripts is the
# organisers' kit and exists only on Firmus.
#
# An unrecognised machine is an error, not a default: guessing wrong here means
# either submitting a job Slurm will reject outright (the --gres spelling) or
# mounting a container image that does not exist.
# ORDER MATTERS, and getting it wrong is not a crash. NSCC ALSO has Bright Cluster
# Manager installed -- /cm/shared/apps/slurm and /cm/local both exist there -- so the
# NYCU test matched on NSCC and job 214488 was handed /home/hsingyu/models/... on a
# machine where that path does not exist. It failed for the right reason by luck; a
# marker collision like this can equally pick up the wrong venv or the wrong GPU
# format and produce a plausible number.
#
# So the discriminating marker is tested first. /opt/pbs exists on NSCC and on
# neither of the Slurm clusters (verified on both), and it is the right thing to key
# on anyway, being the scheduler rather than a directory layout.
site_detect() {
    if [[ -d /opt/pbs && -d /scratch/users/industry ]]; then
        echo nscc
    elif [[ -d /scratch/enroot-images || -d /scratch/reference-scripts ]]; then
        echo firmus
    elif [[ -d /cm/shared/apps/slurm || -d /cm/local ]]; then
        echo nycu
    else
        echo ""
    fi
}

site_load() {   # $1=site name (default $PD_SITE, else detected)
    local name="${1:-${PD_SITE:-}}"
    [[ -n "$name" ]] || name="$(site_detect)"

    if [[ -z "$name" ]]; then
        echo "x cannot tell which site this is -- set PD_SITE=<name>" >&2
        echo "  available: $(site_available)" >&2
        return 1
    fi
    local file="$SITE_DIR/$name.sh"
    if [[ ! -r "$file" ]]; then
        echo "x unknown site '$name' -- no $file" >&2
        echo "  available: $(site_available)" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "$file" || { echo "x failed to source $file" >&2; return 1; }
    PD_SITE="$name"
    [[ -n "${SITE_PROLOGUE:-}" ]] && eval "$SITE_PROLOGUE"
    return 0
}

# The --gres string for N GPUs, in this site's spelling.
site_gres() { printf "${SITE_GRES_FMT:-gpu:%d}" "${1:-8}"; }

# THE WHOLE GPU ARGUMENT, not just the value after --gres=. Nano5's job_submit
# plugin rejects --gres=gpu:N outright and demands --gpus-per-node=N:
#
#   ERROR: The number of GPUs is not specified. Please try using --gpus-per-node=<num>.
#
# so a site needs to replace the flag and not only its argument. Sites that do not
# set SITE_GPU_ARG_FMT keep the --gres form they already had.
# `printf --` is required, not stylistic: the format string starts with "--" and
# bash's printf otherwise reads it as its own option and dies with
# "printf: --: invalid option".
site_gpu_arg() {
    if [[ -n "${SITE_GPU_ARG_FMT:-}" ]]; then printf -- "$SITE_GPU_ARG_FMT" "${1:-8}"
    else echo "--gres=$(site_gres "${1:-8}")"   # keeps SITE_GRES_FMT, e.g. NYCU's gpu:gpu:N
    fi
}

# srun arguments for running the benchmark client: container arguments where
# there is a container, nothing where the client is a venv binary. The caller
# always writes `srun $(site_client_args) "$AIPERF_BIN" profile ...`, so the
# same line works on both sites.
site_client_args() {
    [[ -n "${AIPERF_IMG:-}" ]] || return 0
    printf '%s\n' "--container-image=$AIPERF_IMG" "--container-mounts=${MOUNTS:-/scratch:/scratch}"
}

site_banner() {
    echo "=========================================="
    echo " site:      ${SITE_NAME:-${PD_SITE:-?}}"
    echo " partition: ${SITE_PARTITION:-?}   gpus: $(site_gpu_arg "${SITE_GPUS_PER_NODE:-8}")"
    echo " wall cap:  ${SITE_MAX_MINUTES:-0} min (0 = none)"
    echo " model:     ${MODEL_PATH:-<unset>}"
    echo " client:    ${AIPERF_IMG:+container $AIPERF_IMG}${AIPERF_IMG:-${AIPERF_BIN:-aiperf}}"
    echo "=========================================="
}
