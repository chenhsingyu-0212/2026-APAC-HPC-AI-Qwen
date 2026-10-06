#!/bin/bash
#SBATCH --job-name=probe-pfmem
#SBATCH --nodes=1
#SBATCH --gres=gpu:gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#PBS -l select=1:ngpus=4
#PBS -l walltime=02:00:00
#
# HOW MUCH MEMORY DOES A 12,288-TOKEN PREFILL BATCH ACTUALLY NEED?
#
#   PD_SITE=nycu SITE_PARTITION=defq NODES=1 GPUS=4 TIME_LIMIT=02:00:00 \
#   JOB_CPUS=96 JOB_MEM=1000G run/submit-site.sh run/23-probe-prefill-mem.sbatch
#
# This exists because two of our own logs disagree, and the number decides what to
# set on a 16-GPU scored run:
#
#   job 215921  chunked/max 12288, mem-fraction 0.85
#     "Available memory 9.78GB is less than required memory 25.31GB ...
#      reducing max_m to 6144"
#   job 215960  chunked 8192 / max 16384, mem-fraction 0.85
#     "Available memory 9.70GB is less than required memory 16.94GB ...
#      reducing max_m to 8192"
#
# A 12,288 batch cannot need more than a 16,384 one, so one of those readings is
# measuring something other than what its flags say. Guessing the mem-fraction from
# either of them risks a scored run that silently reduces max_m and measures a
# configuration we did not ask for.
#
# The cheap way out: `required memory` is a property of max_m and the model, NOT of
# mem-fraction -- mem-fraction only moves `Available memory`. So ONE pass reads the
# requirement, and the fraction that satisfies it follows by arithmetic. Pass 2 then
# verifies it, and the verdict is whether `reducing max_m` appears at all.
#
# A lone prefill worker with no decode peer never serves a request, which is fine:
# every number here is printed during startup. One node, TP4, and on NYCU it is free.
#
#   MEM_FRACTION_BASE   the fraction pass 1 runs at (default 0.85, what we ship)
#   MEM_FRACTION_FLOOR  refuse to go below this (default 0.72; the FP8 weights are
#                       55.2 GB at TP4, so a lower budget cannot hold them plus any
#                       KV at all, and the run would fail for a different reason)
#   PROBE_MARGIN_GB     headroom over the measured requirement (default 1.5)
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

ENGINE=sglang
TP="${TP:-4}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-12288}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-12288}"
MEM_FRACTION_BASE="${MEM_FRACTION_BASE:-0.85}"
MEM_FRACTION_FLOOR="${MEM_FRACTION_FLOOR:-0.72}"
PROBE_MARGIN_GB="${PROBE_MARGIN_GB:-1.5}"
READY_SECS="${READY_SECS:-1800}"
eng_load "$ENGINE" || exit 1
say() { echo "[$(date +%H:%M:%S)] $*"; }

OUT="${SLURM_SUBMIT_DIR:-$PWD}/results/probe-pfmem-${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

mapfile -t NODES < <(sched_nodes)
sched_preflight "${NODES[@]}" || exit 1
sched_reset_pools "${NODES[@]}"
HEAD="${NODES[0]}"

say "probing chunked=$CHUNKED_PREFILL max_prefill=$MAX_PREFILL_TOKENS tp=$TP on $HEAD"
say "out $OUT"

# Same shape as 22-tune-moe.sbatch: the launcher carries eng_pd_env into the step,
# and anything referring to $port is dropped because that line is written for a PD
# worker's launcher and is fatal under set -u here.
mapfile -t env_lines < <(eng_pd_env 2>/dev/null | grep -vw 'port')
{
    echo '#!/bin/bash'
    echo 'set -uo pipefail'
    printf '%s\n' "${env_lines[@]/#/export }"
    cat <<'GUARD'
# REFUSE THE WRONG GPU. The whole point is a number that transfers to NSCC's
# H100 80GB, and an H200's 141 GB would give a different and useless answer.
want="${EXPECT_GPU:-NVIDIA H100 80GB HBM3}"
got="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if [ "$got" != "$want" ]; then
    echo "x probing on '$got' but the answer is only valid for '$want' -- refusing" >&2
    exit 1
fi
GUARD
    cat "$HERE/probe-prefill-mem-inner.sh"
} > "$OUT/probe.sh"
chmod +x "$OUT/probe.sh"

SCHED_CONTAINER_ARGS=()
mapfile -t SCHED_CONTAINER_ARGS < <(pd_container_args probe)
sched_step "$HEAD" "$TP" "${JOB_CPUS:-32}" "$OUT/probe.log" \
    bash "$OUT/probe.sh" "$(eng_python)" "$MODEL_PATH" "$OUT" \
         "$CHUNKED_PREFILL" "$MAX_PREFILL_TOKENS" \
         "$MEM_FRACTION_BASE" "$MEM_FRACTION_FLOOR" "$PROBE_MARGIN_GB" \
         "$TP" "$READY_SECS" \
    || { echo "x cannot launch the probe" >&2; exit 1; }
sched_wait_all; rc=$?

say "exit $rc"
echo
echo "================ REPORT ================"
cat "$OUT/report.txt" 2>/dev/null || echo "(no report -- see $OUT/probe.log)"
echo "========================================"
exit $rc
