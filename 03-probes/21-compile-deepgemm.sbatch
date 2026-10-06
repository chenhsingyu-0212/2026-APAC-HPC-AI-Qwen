#!/bin/bash
#SBATCH --job-name=compile-deepgemm
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --gres=gpu:gpu:4
#SBATCH --time=00:50:00
#PBS -N compile-deepgemm
#PBS -l walltime=00:50:00
#PBS -j oe
#
# Populate the DeepGEMM JIT cache once, so a served run does not have to.
#
# Job 86188 enabled --moe-runner-backend deep_gemm and the engine did start using it --
# it printed "DeepGEMM masked layout budget: 2.96 GiB from 11.83 GiB free" -- and then
# said:
#
#   Entering DeepGEMM JIT Pre-Compile session. It may take a long time (typically
#   10-20 mins) if you have not run `sglang.compile_deep_gemm`.
#
# The health check killed it at 155 s. So DeepGEMM was never broken here; it was
# compiling. SGLang ships the pre-compiler for exactly this, and the reference
# configuration behind 3190 does the same thing as a separate serial step before its
# engines start, noting that the organisers' rebuild starts from an empty cache.
#
# The cache lives under $HOME/.cache/sglang/deep_gemm -- NOT $HOME/.cache/deep_gemm,
# which is what this script reported until job 215218 and made a successful compile
# look like a failed one. $HOME is shared storage on both clusters, so one run serves
# every later job there. It has to be redone per cluster.
#
# WHAT THIS DOES NOT COVER. The tool forces SGLANG_ENABLE_JIT_DEEPGEMM, which is the
# DENSE FP8 GEMM path, and its warm-up request is a handful of tokens. Job 215218
# produced 28 sm90_fp8_gemm_1d2d kernels whose locks are named ..._g1 -- group of one,
# i.e. dense. The MoE grouped GEMM that --moe-runner-backend deep_gemm calls is keyed
# on the expert and token shapes of a real batch and is NOT compiled here, so a served
# run with that backend still JITs on first contact. Running this is still worth it:
# the dense FP8 layers use DeepGEMM even when the MoE runner is Triton.
#
#   run/submit-site.sh run/21-compile-deepgemm.sbatch
set -uo pipefail

_pbsid="${PBS_JOBID:-}"; _pbsid="${_pbsid%%.*}"
: "${SLURM_SUBMIT_DIR:=${PBS_O_WORKDIR:-}}"
: "${SLURM_JOB_ID:=$_pbsid}"
export SLURM_SUBMIT_DIR SLURM_JOB_ID

HERE=""
for _d in "${SLURM_SUBMIT_DIR:-}/run" "${SLURM_SUBMIT_DIR:-}" \
          "$(dirname "${BASH_SOURCE[0]}")" "$PWD/run" "$PWD"; do
    [[ -n "$_d" && -r "$_d/pd-lib.sh" ]] && { HERE="$(cd "$_d" && pwd)"; break; }
done
[[ -n "$HERE" ]] || { echo "x cannot find pd-lib.sh" >&2; exit 1; }
source "$HERE/site-lib.sh" || exit 1
site_load || exit 1
source "$HERE/pd-lib.sh"     || exit 1
source "$HERE/engine-lib.sh" || exit 1
eng_load sglang || exit 1

say() { echo "[$(date +%H:%M:%S)] $*"; }

mapfile -t NODES < <(sched_nodes)
sched_preflight "${NODES[@]}" || exit 1
sched_reset_pools "${NODES[@]}"
HEAD="${NODES[0]}"
TP="${TP:-4}"
MODEL_PATH="${MODEL_PATH:?the site should have set this}"

say "site ${SITE_NAME:-$PD_SITE}  $HEAD  compiling DeepGEMM for TP=$TP"
say "cache before: $(du -sh "$HOME/.cache/sglang/deep_gemm" 2>/dev/null | cut -f1 || echo empty)"

# COMPILE WITH THE SAME ARGS THE SERVER WILL USE. The cache is keyed on the GEMM
# shapes, which follow from the model and the tensor-parallel degree -- compiling for
# TP=4 does not populate what a TP=2 worker needs. A PD run that mixes TP2 prefill with
# TP4 decode therefore needs both, which is why TP is a parameter here.
PY="$(eng_python)"
mapfile -t env_lines < <(eng_pd_env 2>/dev/null)
CMD=( "$PY" -m sglang.compile_deep_gemm
      --model-path "$MODEL_PATH" --tp-size "$TP" --trust-remote-code
      --mem-fraction-static "${MEM_FRACTION:-0.90}" )
say "  ${CMD[*]}"

SCHED_CONTAINER_ARGS=()
mapfile -t SCHED_CONTAINER_ARGS < <(pd_container_args compile)
OUT="${SLURM_SUBMIT_DIR:-$PWD}/results/deepgemm-compile-${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
# The launcher exists to carry eng_pd_env into the step -- CUDA_HOME, the staged Python
# headers, LD_LIBRARY_PATH. Without it nvcc is not on PATH and the JIT fails the same
# way the nine bare-metal failures did.
{
    echo '#!/bin/bash'
    echo 'set -uo pipefail'
    printf '%s\n' "${env_lines[@]/#/export }"
    echo 'exec "$@"'
} > "$OUT/compile.sh"

sched_step "$HEAD" "$TP" "${JOB_CPUS:-32}" "$OUT/compile.log" \
    bash "$OUT/compile.sh" "${CMD[@]}" || { echo "x cannot launch" >&2; exit 1; }
say "compiling -- this takes 10-20 minutes on a cold cache"
sched_wait_all; rc=$?

say "cache after:  $(du -sh "$HOME/.cache/sglang/deep_gemm" 2>/dev/null | cut -f1 || echo empty)"
say "files: $(find "$HOME/.cache/sglang/deep_gemm" -type f 2>/dev/null | wc -l)"
tail -20 "$OUT/compile.log"
say "exit $rc -- log at $OUT/compile.log"
exit $rc
