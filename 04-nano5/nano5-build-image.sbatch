#!/bin/bash
#SBATCH --job-name=sgl-image
#SBATCH --partition=dev
#SBATCH --account=MST114554
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/work/u8924359/hsingyu-hpcai/logs/sgl-image-%j.log
#
# Build the SGLang image on a COMPUTE node. The login node cannot: enroot import
# dies with "failed to set capabilities: Operation not permitted", and the August
# TensorRT-LLM images on this machine were built this way for the same reason.
#
# No --gres and only one GPU because a build needs no GPU at all -- but the plugin
# requires --gpus-per-node, so one it is. billing is 12 x GPUs, so this costs about
# 12 SU per hour.
set -uo pipefail
R=/work/u8924359/hsingyu-hpcai

# TMPDIR ON THE NODE'S LOCAL DISK, NOT /work. Job 372144 put it on /work and after
# 48 minutes was writing at 1 MB/s with the process at 3% CPU. /work is not slow --
# dd from the same node measured 1.7 GB/s streaming, and /tmp 2.4 GB/s -- but
# extracting a container rootfs is hundreds of thousands of small files with a
# setxattr attempt on each, and every one of those is an NFS round trip. The log
# even says so once per file class: "ignoring (usually) harmless EPERM on setxattr".
# /tmp here is a local 28 TB md array where that metadata is free.
#
# CACHEDIR STAYS ON /work on purpose: the 18 GB of layer blobs are already fetched
# there, so a re-run skips the download. Putting the cache on node-local /tmp would
# throw it away every job.
export SINGULARITY_TMPDIR="/tmp/sing-$SLURM_JOB_ID"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="$R/sing-cache"
export APPTAINER_CACHEDIR="$R/sing-cache"
mkdir -p "$SINGULARITY_TMPDIR" "$R/images" "$R/logs"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "host    $(hostname)"
echo "sing    $(singularity --version)"
echo "tmpdir  $SINGULARITY_TMPDIR   ($(df -h /tmp | tail -1 | awk '{print $4}') free, node-local)"
echo "cache   $SINGULARITY_CACHEDIR   ($(du -sh "$R/sing-cache" 2>/dev/null | cut -f1) already fetched)"

# v0.5.20-cu129 does not exist on Docker Hub: MANIFEST_UNKNOWN. v0.5.19-cu129 does.
for ref in docker://lmsysorg/sglang:v0.5.19-cu129 docker://lmsysorg/sglang:latest ; do
    tag="${ref##*:}"; out="$R/images/sglang-$tag.sif"
    [ -s "$out" ] && { echo "== already have $out"; break; }
    echo "=============================================================="
    echo "== $ref  ->  $out"
    t0=$SECONDS
    if timeout 6000 singularity build --force "$out" "$ref"; then
        echo "== OK in $(( SECONDS - t0 ))s, $(du -h "$out" | cut -f1)"; break
    else
        echo "== FAILED after $(( SECONDS - t0 ))s"; rm -f "$out"
    fi
done

sif="$(ls -S "$R"/images/*.sif 2>/dev/null | head -1)"
if [ -n "$sif" ]; then
    echo; echo "===== what is inside $sif ====="
    singularity exec "$sif" python3 -c '
import importlib.metadata as md
for p in ("sglang","sglang-router","torch","triton","transformers",
          "flashinfer-python","deep-gemm","nvidia-nixl"):
    try: print(f"  {p:22} {md.version(p)}")
    except Exception: print(f"  {p:22} -")
import torch; print("  torch cuda          ", torch.version.cuda)' 2>&1 | tail -14
    # The MoE Triton configs are selected by a per-Triton-version directory, so the
    # version inside the image decides which tuned file would ever be read.
    echo "===== moe triton config dirs ====="
    singularity exec "$sif" python3 -c '
import sglang, os, glob
d = os.path.join(os.path.dirname(sglang.__file__), "srt", "layers", "moe",
                 "fused_moe_triton", "configs")
print("  ", d, os.path.isdir(d))
for x in sorted(glob.glob(d + "/*"))[:5]: print("    ", os.path.basename(x))' 2>&1 | tail -8
fi
echo DONE-BUILD
