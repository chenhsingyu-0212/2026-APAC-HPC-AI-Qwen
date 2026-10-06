#!/bin/bash
# NCHC Nano5 / Forerunner1 (nano5.nchc.org.tw, login node cbi-lgn01). Native Slurm.
#
# THE REASON THIS SITE EXISTS: its credit can be topped up and NSCC's cannot. That
# changes which platform a speculative experiment belongs on -- an inconclusive
# sweep here spends money that can be replaced.
#
#   partition  nodes                 GPUs              MaxTime      MaxNodes
#   normal     hgpn[02-06,17-21]  10 x 8 H100 80GB   2-00:00:00   2   <- the only one
#   dev        hgpn[01-06,17-21]  11 x 8 H100           2:00:00   1      that can hold
#   4nodes     hgpn[02-06,17-21]  10 x 8 H100        1-00:00:00   ?      a 2-node PD run
#   normal2    hgpn[39-41,43-46]   7 x 8 H200        2-00:00:00   ?
#
# MaxNodes IS THE ONE TO READ. dev caps at ONE node, so it can build images and run
# single-node work but cannot hold a 2x8 PD deployment: sbatch answers "Node count
# specification invalid" and names neither the partition nor the limit. And
# `sbatch --test-only` does NOT enforce MaxNodes -- it accepted --nodes=2 on dev
# while the real submission was refused, so it cannot be used to pre-check this.
#
# normal caps at 2, which is exactly the 16 GPUs the task requires and also the
# ceiling: there is no way to ask for more on this machine.
#
# No GrpTRES and no MaxTRES on this association, so 16 GPUs is allowed outright --
# unlike NYCU, where defq-qos caps the group at gres/gpu=15 and the per-job
# association at 4, making a 2x8 H100 run impossible by policy.
#
# BILLING, measured rather than assumed. billing == cpu == 12 x GPUs, and SU is
# billing x hours, so a 16-GPU hour is 192 SU. Derived from u8924359's August jobs:
# summing billing x elapsed over them gives 583 billing-hours against sreport's
# 561, which is the same number. A 70-minute 16-GPU campaign costs about 224 SU.
# NSCC charges 2,389 for the same thing, so this machine is roughly 10x cheaper.
#
#   SU_BALANCE on MST114554 was 2,999.99 on 2026-10-05  ==  about 13 such jobs,
#   AND IT IS SHARED with the other users on that project. Re-read it before a
#   sweep: `sbatch --account=MST114554 ... --test-only` prints the wallet.
#
# THREE THINGS THE job_submit PLUGIN ENFORCES, each of which fails at submit time
# with a message that does not name the real rule:
#
#   1. --account is mandatory. Without it: "No project ID was assigned", plus the
#      wallet. Only MST114554 works for this user -- GOV113021 reports "not a
#      member" and MST113296 "is not allowed to use Forerunner1".
#   2. --gpus-per-node is mandatory and --gres=gpu:N is NOT accepted:
#      "The number of GPUs is not specified. Please try using --gpus-per-node=<num>."
#      Hence SITE_GPU_ARG_FMT below, which replaces the whole flag.
#   3. CPUs are capped at 12 PER GPU: "CPU per node specification of 16 exceeds the
#      limit (1 * 12)". NSCC allows 14, so SITE_CPUS_PER_GPU differs and a config
#      carried over from there is rejected at submit time.
#
# CONTAINERS: singularity 4.1.2, enroot and docker are all present, but the LOGIN
# node cannot build -- enroot import dies with "failed to set capabilities:
# Operation not permitted". The August TensorRT-LLM images were built inside a job
# on a compute node, and that is the path that works. run/nano5-build-image.sbatch
# does the same for SGLang.
#
# THE MODEL IS ALREADY HERE, 221 GB, put there in August:
#   /work/u8924359/qwen/models/Qwen3-235B-A22B-Instruct-2507-FP8
# config.json confirms it is the right checkpoint -- qwen3_moe, 94 layers,
# 4 kv heads, head_dim 128, 128 experts, quant_method fp8, and torch_dtype
# bfloat16 (which is why the KV cache defaults to BF16 and 192,512 B per token).
#
# Large-file egress is effectively blocked: huggingface.co answers in 21 ms but a
# safetensors shard downloads at 71 B/s. Do not plan on fetching anything big.

# billing == cpu == 12 x GPUs and SU is billing x hours. Derived from this account's
# August jobs: summing billing x elapsed gives 583 billing-hours against sreport's
# 561 for the same window. submit-site.sh calls this and prints it before submitting,
# because the wallet is shared and a 16-GPU hour is 6% of it.
nano5_su() {   # $1=total GPUs  $2=hours
    awk -v g="$1" -v h="$2" \
        'BEGIN{printf "%.0f SU (%.2f GPU-hours, billing %d)", g*h*12, g*h, g*12}'
}

# Refuse to submit into a half-built site rather than discover it 12 minutes into a
# model load. Each of these has cost a job at some point on some machine.
nano5_ready() {
    local missing=()
    [[ -n "$SITE_ACCOUNT" ]] || missing+=(SITE_ACCOUNT)
    [[ -n "$MODEL_PATH" && -r "$MODEL_PATH/config.json" ]] || missing+=("MODEL_PATH($MODEL_PATH)")
    # Check the name the campaign actually reads. Checking $AIPERF validated a
    # variable nothing downstream looks at, which is how 372918 got submitted.
    [[ -n "$AIPERF_BIN" && -x "$AIPERF_BIN" ]] || missing+=("AIPERF_BIN($AIPERF_BIN)")
    [[ -z "${PD_SINGULARITY_IMG:-}" || -s "$PD_SINGULARITY_IMG" ]] \
        || missing+=("PD_SINGULARITY_IMG($PD_SINGULARITY_IMG)")
    # THE KIT, which is what job 372908 died on. The campaign requires the
    # organisers' AIPerf scripts and checks them itself -- but it checks them AFTER
    # Slurm has granted the allocation, so on a partition where we wait six hours for
    # two nodes the cost of forgetting them is six hours, not a shell command.
    # 372908 was lucky: it ran on dev and failed in 0 s for 0 SU.
    # A non-empty SGLANG_IMG routes the image to pyxis, which cannot run here.
    [[ -z "${SGLANG_IMG:-}" ]] \
        || missing+=("SGLANG_IMG must be EMPTY here (it would go to pyxis): $SGLANG_IMG")
    local _kit="${KIT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/AIPerf}"
    local _s
    for _s in observe.sh gpu_sample.sh diagnose.sh official.sh; do
        [[ -x "$_kit/$_s" ]] || missing+=("$_kit/$_s")
    done
    if (( ${#missing[@]} )); then
        echo "x run/sites/nano5.sh is not ready -- missing or unreadable: ${missing[*]}" >&2
        return 1
    fi
    return 0
}

SITE_NAME="NCHC Nano5"
SITE_SCHEDULER=slurm
SITE_PARTITION="${SITE_PARTITION:-normal}"      # normal = H100; normal2 = H200
SITE_ACCOUNT="${SITE_ACCOUNT:-MST114554}"
SITE_GPUS_PER_NODE="${SITE_GPUS_PER_NODE:-8}"
SITE_GPU_ARG_FMT="${SITE_GPU_ARG_FMT:---gpus-per-node=%d}"
SITE_CPUS_PER_GPU="${SITE_CPUS_PER_GPU:-12}"    # a hard cap here, not a preference
SITE_MAX_MINUTES="${SITE_MAX_MINUTES:-2880}"    # normal = 2 days; dev = 2 hours
SITE_JOB_CPUS="${SITE_JOB_CPUS:-96}"            # 8 GPUs x 12
SITE_JOB_MEM="${SITE_JOB_MEM:-1600G}"           # 200G per GPU, what August used

SITE_SCRATCH="${SITE_SCRATCH:-/work/u8924359/hsingyu-hpcai}"
MODEL_PATH="${MODEL_PATH:-/work/u8924359/qwen/models/Qwen3-235B-A22B-Instruct-2507-FP8}"

# A step-level --gres is what breaks here: the job holds --gpus-per-node and a
# second gres request for the same cards is not satisfiable. SLURM_STEP_VIA=overlap
# skips it entirely and slices CUDA_VISIBLE_DEVICES itself -- the same mode the
# Firmus reference deployment uses to reach 3,190.
SLURM_STEP_VIA="${SLURM_STEP_VIA:-overlap}"

# Containers, not venvs: the image carries its own CUDA toolkit and Python headers,
# which is the whole class of failure that cost us jobs on NYCU and NSCC.
# THE .sif DOES NOT GO IN SGLANG_IMG. Job 372921 put it there and died in 12 s:
#
#   pyxis: failed to create container filesystem
#   pyxis: mkdir: cannot create directory '/run/user/37948': Permission denied
#
# because 10-campaign.sbatch:524 sets PD_IMAGE="$(eng_image)" and eng_image() returns
# $SGLANG_IMG for sglang -- so the .sif was handed to pyxis as --container-image.
# Setting PD_IMAGE empty in this file does nothing: line 524 overwrites it after the
# site is loaded.
#
# pyxis cannot work here at all, whatever it is given: enroot needs /run/user/$UID
# and the compute nodes refuse to create it. So SGLANG_IMG is EMPTY -- which
# engines/sglang.sh:11-13 documents as meaning bare metal -- and the image reaches
# the engine through PD_SINGULARITY_IMG instead, which sched/slurm.sh turns into a
# `singularity exec --nv ... <sif>` command prefix.
#
# v0.5.20-cu129 does not exist on Docker Hub (MANIFEST_UNKNOWN); v0.5.19-cu129 does.
NANO5_SIF="${NANO5_SIF:-$SITE_SCRATCH/images/sglang-v0.5.19-cu129.sif}"
SGLANG_IMG="${SGLANG_IMG-}"          # EMPTY on purpose: see above
PD_SINGULARITY_IMG="${PD_SINGULARITY_IMG:-$NANO5_SIF}"
PD_SINGULARITY_BIND="${PD_SINGULARITY_BIND:-/work,/tmp}"
VLLM_IMG="${VLLM_IMG-}"
# Also a .sif, so also NOT for pyxis. Left here as a record of what exists on the
# machine; a trtllm run would need the same PD_SINGULARITY_IMG treatment.
TRTLLM_IMG="${TRTLLM_IMG-}"
NANO5_TRTLLM_SIF="${NANO5_TRTLLM_SIF:-/work/u8924359/qwen/images/trtllm-1.3.0rc9.sif}"
LMDEPLOY_IMG="${LMDEPLOY_IMG-}"
AIPERF_IMG="${AIPERF_IMG-}"

SGLANG_VENV="${SGLANG_VENV:-$SITE_SCRATCH/envs/sglang}"
VLLM_VENV="${VLLM_VENV:-$SITE_SCRATCH/envs/vllm}"
TRTLLM_VENV="${TRTLLM_VENV:-$SITE_SCRATCH/envs/trtllm}"
# AIPerf 0.12.0 is already here from the August TensorRT-LLM work, in the venv that
# ran those jobs. Verified to carry every flag the official command needs:
# --goodput --isl --osl --num-dataset-entries --random-seed --warmup-duration
# --benchmark-duration --benchmark-grace-period --custom-endpoint --streaming
# --use-legacy-max-tokens --workers-max. No reason to build a second one.
# AIPERF_BIN, not AIPERF. The campaign reads AIPERF_BIN (10-campaign.sbatch:153,
# client_init at :207) and so do nscc.sh, nycu.sh and firmus.sh; I wrote AIPERF here
# and job 372918 died on it:
#
#   x cannot read 'aiperf profile --help' -- is the client installed?
#
# AIPERF_BIN fell back to the bare name `aiperf`, which `command -v` DOES find on the
# compute node -- so the precondition at :195 passed -- but which cannot run. The
# precondition and the thing it guards were looking at different binaries.
AIPERF_BIN="${AIPERF_BIN:-/work/u8924359/qwen/venv/bin/aiperf}"
AIPERF="${AIPERF:-$AIPERF_BIN}"      # kept: run/21-compile-deepgemm.sbatch reads this
