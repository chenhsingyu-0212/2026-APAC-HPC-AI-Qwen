#!/bin/bash
# Firmus (apacsc33) -- the competition cluster. Everything here was measured on
# it; see docs/trtllm-plan.md section 1 for the recon that produced the numbers.

SITE_NAME="Firmus apacsc33"
SITE_PARTITION="${SITE_PARTITION:-hpcai}"
SITE_GRES_FMT="${SITE_GRES_FMT:-gpu:%d}"
SITE_GPUS_PER_NODE="${SITE_GPUS_PER_NODE:-8}"

# sinfo TIMELIMIT is 17:00 and Slurm kills at the cap with SIGKILL, which leaves
# no artefacts behind -- job 18466 is why pd_need_time() exists. Raising this
# does not get you a longer job, it gets the submission rejected.
SITE_MAX_MINUTES="${SITE_MAX_MINUTES:-17}"

MODEL_PATH="${MODEL_PATH:-/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8}"

# enroot squashfs images, one per engine. TRTLLM_IMG is empty on purpose:
# nvcr.io is unreachable from the compute nodes and /scratch/enroot-images is
# root-owned, so TensorRT-LLM runs from a pip venv on shared storage instead.
SGLANG_IMG="${SGLANG_IMG-/scratch/enroot-images/sglang-harbor.sqsh}"
VLLM_IMG="${VLLM_IMG-/scratch/enroot-images/vllm.sqsh}"
TRTLLM_IMG="${TRTLLM_IMG-}"
TRTLLM_VENV="${TRTLLM_VENV:-/scratch/home/$USER/hsingyu/trtllm-venv}"

AIPERF_IMG="${AIPERF_IMG-/scratch/enroot-images/aiperf.sqsh}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"      # inside that container

MOUNTS_BASE="${MOUNTS_BASE:-/scratch:/scratch}"
