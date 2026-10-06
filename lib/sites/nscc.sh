#!/bin/bash
# NSCC Singapore ASPIRE 2A+ -- the OTHER official competition platform.
#
# Measured on 2026-09-28 by logging in and reading, not by assuming. What is still
# unknown is marked as such; nscc_ready() refuses to run while anything load-bearing
# is missing, because a guess here costs budget rather than just time.
#
# WHY THIS PLATFORM IS DIFFERENT FROM NYCU
#   * The rules name Firmus and NSCC ASPIRE 2A+ as the two platforms, so numbers
#     measured here are SCORING CANDIDATES rather than supporting data.
#   * enroot IS available (container_platform=enroot is a node resource), so the
#     nine bare-metal failures NYCU cost us do not have to be repeated.
#   * IT IS METERED. 64000 SU granted, 128 SU per GPU-hour -- 500 GPU-hours for the
#     whole competition. A 16-GPU deployment plus the official 120-second run is
#     about 5.3 GPU-hours, so roughly 94 of those in total. THIS IS NOT THE PLACE
#     TO SWEEP PARAMETERS. The search was done on NYCU for free; here we confirm.
#
# PBS Pro, not Slurm. That is a real port, not a setting -- see the notes at the
# bottom of this file.

SITE_NAME="NSCC ASPIRE 2A+"
SITE_SCHEDULER="${SITE_SCHEDULER:-pbs}"

# -P is mandatory and there is exactly one project that works. 50000127 is last
# year's ("APAC HPC-AI Competition 2A+ 33", expired 2025-10-31) and qsub rejects it
# with "Project has already ended"; 59000033 does not exist. 59010033 is
# "APAC HPC-AI Competition 2026 2A+ 33", valid to 2026-11-15.
SITE_PROJECT="${SITE_PROJECT:-59010033}"

# aiq1 24h, aiq2 120h, aidev 12h. aiq1 is where a submission lands by default.
SITE_PARTITION="${SITE_PARTITION:-aiq1}"
SITE_MAX_MINUTES="${SITE_MAX_MINUTES:-1440}"      # aiq1 = 24h -- the CAP, not a default
# What a run is actually given unless the caller says otherwise. The cap must not be
# the default on a prepaid allocation: the requested walltime is reserved whether or
# not it is used, and 24 h of 16 GPUs is 49,152 SU of the 64,000 grant.
#
# Read the real figure with /app/apps/local/bin/myprojects, and read the PROJECT
# NUMBER off the table: that command prints every project the account can see, and
# ours is the one submit-site.sh passes to qsub -P, 59010033. On 2026-10-02 it stood
# at 9,452.94 SU used of 64,000, balance 54,453.19 -- i.e. 73.851 GPU-hours, all of
# it ours. (50000127 also appears in that output and is a different project; reading
# its row by mistake produced a report of 78,106 SU used that had nothing to do with
# us.) The table's In Doubt column is SU deducted for jobs currently running.
#
# Charging follows walltime USED, not reserved: job 215267 reserved 02:30, used
# 00:56:42, and once it ended In Doubt returned to ~0. So a generous walltime costs
# nothing beyond holding the reservation while the job runs.
SITE_DEFAULT_MINUTES="${SITE_DEFAULT_MINUTES:-120}"
SITE_GPUS_PER_NODE="${SITE_GPUS_PER_NODE:-8}"

# The site enforces 14 CPUs per GPU: a request for ncpus=8:ngpus=1 came back as
# ncpus=14. Ask for the ratio directly so the accounting is predictable.
SITE_CPUS_PER_GPU="${SITE_CPUS_PER_GPU:-14}"

# ---------------------------------------------------------------- what runs here
# POLICY, not a default to be overridden casually: on NSCC every run is 16 GPUs at the
# OFFICIAL WINDOW. This is one of the two platforms the rules name, so a number
# measured here is a scoring candidate rather than supporting data -- and the scored
# shape is "all 16 GPUs provided on the chosen platform" (Task Rules) measured over
# --benchmark-duration 120.
#
# Exploration belongs on NYCU, which is free. The rules themselves say a 10-20 second
# window is enough to tell two configurations apart and to save the full duration for
# the scored run; the corollary is that a 20-second number on a SCORING platform is
# neither one thing nor the other. Every 16-GPU number this project has from NSCC is a
# 20-second one, which is exactly the gap this closes.
#
# These are set rather than defaulted (no :- on the window) because the campaign reads
# them after site_load, so a site value wins. Override deliberately on the command
# line if a cheap probe is genuinely wanted.
SITE_NODES="${SITE_NODES:-2}"                  # 2 x 8 = the 16 the rules require
SITE_GPUS_PER_NODE="${SITE_GPUS_PER_NODE:-8}"

# The locked command's own durations and dataset size.
WARMUP_SECONDS="${WARMUP_SECONDS:-60}"
BENCH_SECONDS="${BENCH_SECONDS:-120}"
GRACE_SECONDS="${GRACE_SECONDS:-30}"
DATASET_ENTRIES="${DATASET_ENTRIES:-12800}"
SEED="${SEED:-2026}"
# 210 s of measurement per point, and the campaign's budget guard needs to know.
POINT_SECONDS="${POINT_SECONDS:-300}"
# A 235B checkpoint across four workers, plus DeepGEMM's startup if it is on.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"

# Say so if a submission is smaller than the scored shape. Not an error -- an 8-GPU PD
# shakedown before a 16-GPU run is worth 340 SU -- but it should be a choice that was
# made rather than one that happened.
nscc_check_shape() {   # $1=nodes $2=gpus-per-node
    local total=$(( ${1:-0} * ${2:-0} ))
    (( total == 16 )) && return 0
    echo "  ! NSCC is a SCORING platform and the scored shape is 16 GPUs; this asks for ${total}." >&2
    echo "    Fine for a deliberate shakedown, wrong for anything meant to be reported." >&2
    return 0
}

# PBS selects with -l select=<chunks>:ncpus=..:ngpus=.. rather than --gres, so the
# Slurm gres format is meaningless here and left empty on purpose.
SITE_GRES_FMT="${SITE_GRES_FMT-}"

# /home is 50 GB -- far too small for a 221 GB checkpoint. /scratch has 2.1 PB free.
SITE_SCRATCH="${SITE_SCRATCH:-/scratch/users/industry/ai-hpc/apacsc33}"
MODEL_PATH="${MODEL_PATH:-$SITE_SCRATCH/models/Qwen3-235B-A22B-Instruct-2507-FP8}"

# NO CONTAINERS, deliberately, even though singularity/4.1.5 is a module here and
# both docker.io and nvcr.io are reachable. Picking an image tag means guessing which
# CUDA build pairs with driver 595.71.05, and -- the real reason -- a container would
# run DIFFERENT wheels from the ones NYCU measured with. These venvs carry exactly
# what NYCU carries (sglang 0.5.20, vllm 0.30.0, tensorrt_llm 1.2.1, torch
# 2.13.0+cu130, python 3.10), so an H100 row and an H200 row differ in the GPU and in
# nothing else. That is the entire reason for having two machines.
SGLANG_IMG="${SGLANG_IMG-}"
VLLM_IMG="${VLLM_IMG-}"
TRTLLM_IMG="${TRTLLM_IMG-}"
LMDEPLOY_IMG="${LMDEPLOY_IMG-}"
AIPERF_IMG="${AIPERF_IMG-}"

SGLANG_VENV="${SGLANG_VENV:-$SITE_SCRATCH/envs/sglang}"
VLLM_VENV="${VLLM_VENV:-$SITE_SCRATCH/envs/vllm}"
TRTLLM_VENV="${TRTLLM_VENV:-$SITE_SCRATCH/envs/trtllm}"
AIPERF_BIN="${AIPERF_BIN:-$SITE_SCRATCH/envs/client/bin/aiperf}"

# What run/venv-prepare.sh staged, because a wheel ships enough to IMPORT a library
# and these engines COMPILE kernels on first use. All seven of its verification
# checks pass here, so the nine NYCU failures do not recur.
ENG_PY_INCLUDE="${ENG_PY_INCLUDE:-$SITE_SCRATCH/stage/pyinclude/python3.10:$SITE_SCRATCH/stage/pyinclude}"
ENG_PY_LIB="${ENG_PY_LIB:-$SITE_SCRATCH/stage/pylib}"

# PBS is not on PATH in a non-login shell, and myprojects lives elsewhere again.
SITE_PROLOGUE="${SITE_PROLOGUE:-export PATH=/opt/pbs/bin:/app/apps/local/bin:\$PATH; true}"

# Unused while PD_IMAGE is empty; kept so a future container run has it.
MOUNTS_BASE="${MOUNTS_BASE:-/scratch:/scratch}"

# ---------------------------------------------------------------- measured on a node
# Recon job 213456 on a2ap-dgx038, 2026-09-28. All of this is read, not assumed.
#
#   GPU            NVIDIA H100 80GB HBM3, 81559 MiB, driver 595.71.05, cc 9.0
#   CPU / RAM      14 cores per GPU chunk, 2015 GB visible
#   /dev/shm       1008 GB -- no --shm-size worry, unlike a default docker run
#   /raid          28 TB local NVMe, 27 TB free. Faster than /scratch for an image.
#   ulimit -n      16384 soft / 65536 hard
#   fabric         10 x 400 Gb/s NDR IB (mlx5_0,1,3..7,9,10,11) + 2 x 100 GbE (mlx5_2,8)
#   NVLink         26.562 GB/s per link
#   enroot         3.5.0, cache at /raid/local/containers
#   python3        3.10.12
#
# Ten 400 G ports, versus Firmus's eight. The per-request KV figure of 771 MiB
# (188 KiB/token x 4096) therefore moves in 1.6 ms across the full fabric here
# rather than 2.0 ms -- still an order of magnitude below the 15-18 ms per-step
# times measured on both machines, so the fabric is not the thing to tune.
SITE_GPU_MODEL="${SITE_GPU_MODEL:-H100 80GB HBM3}"

# PBS DOES pin the GPUs per chunk, and it does it BY UUID:
#   CUDA_VISIBLE_DEVICES=GPU-e6ef6154-f1c6-2bcd-0ffb-a20ea25fc611
# That is the opposite of the Slurm path, where this repo deliberately sets nothing
# and lets cgroups do the work. Two consequences for the PD port:
#   * nvidia-smi inside the job still numbers that GPU 0, so index arithmetic is
#     meaningless -- a worker cannot be given "GPUs 4-7".
#   * splitting a chunk between a prefill and a decode worker means slicing the
#     comma-separated UUID LIST, not computing indices. Each worker gets its own
#     CUDA_VISIBLE_DEVICES with $TP of those UUIDs.
# Getting that wrong does not error; it silently puts two workers on one GPU and
# reports the result as a performance number.
PBS_SETS_CUDA_VISIBLE=1
# PBS hands out GPU UUIDs. run/sched/pbs.sh translates them to the positional
# integers CUDA wants, because SGLang parses CUDA_VISIBLE_DEVICES with int() and
# silently disables custom all-reduce when that throws (job 214830, every rank).
# PBS_GPU_IDS=uuid restores the untranslated list for comparison.
PBS_GPU_IDS="${PBS_GPU_IDS:-index}"

# What one job will cost, in the unit the budget is denominated in. Printed before
# submission so the price is visible at the moment of the decision, not afterwards.
# NODES THAT ACCEPT A JOB AND RUN NOTHING. Two of them so far: a2ap-dgx038 and
# a2ap-dgx036. Both report state R, both answer ping in 0.3 ms, and pbsdsh, ssh and
# pbs_tmrsh all time out against them. PBS has no "exclude this host" for a plain
# select, so the defence is sched_preflight, which probes every node for 10 s before
# anything else runs: a bad draw then costs about 1 SU instead of the job's walltime.
# Three jobs were lost to dgx038 before that guard existed; it has since caught
# dgx038 once (215898) and dgx036 once (215903), both for 25-27 seconds.
#
# The right response is to resubmit immediately, not to wait.
NSCC_KNOWN_BAD_NODES="a2ap-dgx038 a2ap-dgx036"

nscc_su() {   # $1=gpus  $2=walltime hours
    awk -v g="$1" -v h="$2" 'BEGIN{printf "%.1f SU (%.2f GPU-hours)", g*h*128, g*h}'
}

nscc_ready() {
    local missing=()
    [[ -n "$SITE_PROJECT" ]] || missing+=(SITE_PROJECT)
    [[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]] || missing+=("MODEL_PATH($MODEL_PATH)")
    [[ -n "$SITE_GPU_MODEL" ]] || missing+=(SITE_GPU_MODEL)
    if (( ${#missing[@]} )); then
        echo "x run/sites/nscc.sh is not ready -- unset or unverified: ${missing[*]}" >&2
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------- the PBS port
# run/pd-lib.sh is written against Slurm and needs a second path for four things:
#
#   sbatch/squeue/scancel        -> qsub -P $SITE_PROJECT / qstat / qdel
#   srun --exact --gres=gpu:N    -> no equivalent. pbsdsh or ssh to the node, and
#                                   the per-worker GPU split has to be computed
#   scontrol show hostnames      -> cat $PBS_NODEFILE (one line per chunk)
#   squeue -o %L for time left   -> qstat -f | Walltime.Remaining
#
# The third of those is the one that needs care. On Slurm each worker is its own
# step with --gres=gpu:<tp> and cgroups hand it the right devices, which is why
# nothing in this repo sets CUDA_VISIBLE_DEVICES. If PBS does not isolate per
# chunk the same way, that guarantee disappears and the split becomes ours to get
# right -- and getting it wrong means two workers quietly sharing a GPU, which
# looks like a performance result rather than a bug.
#
# TWO OPERATIONAL FACTS THAT LOOK LIKE BUGS AND ARE NOT:
#
#   1. PBS COPIES stdout BACK ONLY WHEN THE JOB ENDS. There is no equivalent of
#      Slurm's live slurm-%j.out. A job that is clearly running has no output file
#      at all, which reads exactly like a hang. Job 213456 was declared wedged three
#      times on this basis and had in fact completed normally. Make every job
#      `| tee` into a file under $SITE_SCRATCH/logs so progress is readable while
#      it runs.
#
#   2. A NODE CAN BE PACKED, AND THEN STARTING IS SLOW. a2ap-dgx038 was holding
#      SEVEN other jobs -- 213281, 213440, 213406, 213454, 212925, 213145 and ours --
#      with all 112 CPU slots allocated. A job landing there took ~16 minutes to
#      record an stime. "State R with no stime" therefore means "queued behind a
#      busy node", not "wedged", and qdel is the wrong response. Check
#      `pbsnodes <node> | grep -c /` before concluding anything.
#
# DO NOT PIN A NODE. An early attempt at -l select=1:vnode=a2ap-dgx004 sat queued
# for an hour; the same request without vnode started immediately.
#
# DO NOT NAME A QUEUE. -q aiq1 / aiq2 / aidev are all denied ("Route queue only").
# Submit with no -q and the `normal` routing queue places the job.
