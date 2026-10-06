#!/bin/bash
# NYCU HPC (140.113.9.253, Bright Cluster Manager). The comparison machine: it
# has BOTH shapes the write-up needs, which Firmus does not.
#
#   defq   DGX-CN[01-02]       16x H100 80GB HBM3   defq-qos  GrpTRES gres/gpu=15
#   h200q  DGX-H200-CN[03-05]  24x H200             h200q-qos GrpTRES gres/gpu=21
#
# Both caps are group-wide across every user of that QOS, not per user. The H100
# one bites: 15 < 16, so a 2x8 H100 run is impossible here by policy, no matter
# how idle the machines are. H200 at 16 GPUs is legal (16 <= 21).
#
# Measured 2026-09-26, all of it by submitting jobs and reading what came back,
# because PrivateData=jobs,nodes,usage plus AdminLevel=None hides sinfo,
# scontrol show node, and every other user's jobs.
#
# NO CONTAINER RUNTIME EXISTS HERE -- singularity, apptainer, enroot, podman and
# docker are all absent. All three engines therefore run from pip venvs, which is
# the path TensorRT-LLM already took on Firmus. python3 is 3.10.6 and all three
# engines ship cp310 wheels (tensorrt_llm-1.2.1-cp310 on pypi.nvidia.com,
# sglang cp310, vllm abi3), so one interpreter covers everything.

SITE_NAME="NYCU HPC"
SITE_PARTITION="${SITE_PARTITION:-h200q}"   # defq is H100 and capped below 16
SITE_GPUS_PER_NODE="${SITE_GPUS_PER_NODE:-8}"

# A job_submit plugin rejects --gres=gpu:8 and demands the type be named:
#   "For full GPU: --gres=gpu:gpu:4"   /   "For MIG instances: --gres=gpu:1g.18gb:2"
SITE_GRES_FMT="${SITE_GRES_FMT:-gpu:gpu:%d}"

# Every partition here is MaxTime=UNLIMITED, so there is no cap to plan around --
# which also means other people's jobs can hold nodes indefinitely, and did.
SITE_MAX_MINUTES="${SITE_MAX_MINUTES:-0}"

MODEL_PATH="${MODEL_PATH:-/home/hsingyu/models/Qwen3-235B-A22B-Instruct-2507-FP8}"

# Empty images: bare metal everywhere. pd_container_args() already treats an
# empty PD_IMAGE as "no container", which is the path TensorRT-LLM validated.
SGLANG_IMG="${SGLANG_IMG-}"
VLLM_IMG="${VLLM_IMG-}"
TRTLLM_IMG="${TRTLLM_IMG-}"

SGLANG_VENV="${SGLANG_VENV:-/home/hsingyu/envs/sglang}"
VLLM_VENV="${VLLM_VENV:-/home/hsingyu/envs/vllm}"
TRTLLM_VENV="${TRTLLM_VENV:-/home/hsingyu/envs/trtllm}"
LMDEPLOY_IMG="${LMDEPLOY_IMG-}"
LMDEPLOY_VENV="${LMDEPLOY_VENV:-/home/hsingyu/envs/lmdeploy}"

AIPERF_IMG="${AIPERF_IMG-}"
AIPERF_BIN="${AIPERF_BIN:-/home/hsingyu/envs/client/bin/aiperf}"

# Bright Cluster Manager does not put Slurm on PATH for non-login shells, and
# /etc/profile.d is not sourced either, so `sbatch` and `squeue` are simply
# missing until the module is loaded. Note the module is `slurm`, not
# `slurm-client` -- and never `apt install slurm-client`, whose version will not
# match this slurmctld.
SITE_PROLOGUE="${SITE_PROLOGUE:-source /etc/profile.d/modules.sh 2>/dev/null; module load slurm 2>/dev/null; true}"

# Nothing to bind-mount without containers, but eng_mounts() still builds a
# string and the smoke tests still print it.
# PER-JOB CEILING, from the `test` account's association (measured 2026-09-26):
#
#   test  h200q  MaxTRES  cpu=96,gres/gpu=4,mem=1000G
#   test  defq   MaxTRES  cpu=96,gres/gpu=4,mem=1000G
#
# So FOUR GPUs per job, not eight, until someone raises it -- and --exclusive,
# which asks for the node's full 2 TB, is refused with AssocMaxMemPerJob before
# the GPU limit is even reached. At TP4 the 221 GB checkpoint is 55 GB/GPU, which
# fits an 80 GB H100 with room for KV, so a 4-GPU single instance is a real
# measurement; PD is not, because two instances need TP4 each.
# Python headers, staged out of the login node's /usr/include because the compute
# nodes do not have python3-dev. vLLM's triton needs them the first time it
# compiles a kernel; see eng_venv_env() in run/engine-lib.sh.
# Both entries are needed: the first holds Python.h, the second is what the
# pyconfig.h stub's <x86_64-linux-gnu/python3.10/pyconfig.h> resolves against.
ENG_PY_INCLUDE="${ENG_PY_INCLUDE:-/home/hsingyu/pyinclude/python3.10:/home/hsingyu/pyinclude}"
ENG_PY_LIB="${ENG_PY_LIB:-/home/hsingyu/pylib}"

# TRIALQ IS A FREE H100, AND IT DOES NOT QUEUE. MEASURED 2026-10-03: job 98519 sat
# on defq's QOSGrpGRES indefinitely (defq-qos caps the whole group at gres/gpu=15 and
# other people hold them), while the same script on trialq started in under 10 seconds
# on DGX-CN02 and reported
#
#   [DGX-CN02] gpu: NVIDIA H100 80GB HBM3
#
# which is the EXACT string our MoE configs are keyed on, so a config tuned there
# loads on NSCC unchanged. That makes trialq the reliable path for anything that fits
# inside its QOS, and defq the one to use only if a slot happens to be free.
#
#   trialq-qos   cpu=24, gres/gpu=1, mem=250G, MaxWall 00:30:00
#
# The site defaults below are written for the association's per-job MaxTRES and are
# too large for it -- job 98794 was refused with QOSGrpCpuLimit on cpu=96. Submitting
# to trialq means overriding both:
#
#   PD_SITE=nycu SITE_PARTITION=trialq NODES=1 GPUS=1 TIME_LIMIT=00:30:00 \
#   JOB_CPUS=24 JOB_MEM=250G MOE_BATCH_SIZES="48" run/submit-site.sh run/22-tune-moe.sbatch
#
# 30 minutes holds ONE batch size of the MoE search (measured at ~20 min each), so a
# multi-size sweep is a chain of trialq jobs rather than one job.
SITE_JOB_MEM="${SITE_JOB_MEM:-1000G}"
SITE_JOB_CPUS="${SITE_JOB_CPUS:-96}"
SITE_MAX_GPUS_PER_JOB="${SITE_MAX_GPUS_PER_JOB:-4}"

MOUNTS_BASE="${MOUNTS_BASE:-/home:/home}"
