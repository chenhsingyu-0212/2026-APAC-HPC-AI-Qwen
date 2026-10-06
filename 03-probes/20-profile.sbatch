#!/bin/bash
#SBATCH --job-name=qwen3-profile
#SBATCH --partition=hpcai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#PBS -N qwen3-profile
#PBS -l walltime=00:50:00
#PBS -j oe
#
# Where does a prefill forward pass actually spend its time?
#
# The measured figure that prompted this: a TP4 prefill worker on NSCC ran at 290
# TFLOPS/GPU, 14.7% of H100 FP8 peak, with a 19-deep queue -- so prefill is the
# binding constraint on a 16-GPU PD deployment and 85% of its time is not peak GEMM.
# The arithmetic puts the TP all-reduce at 7-30% of a 310 ms 8192-token batch
# depending on the bandwidth actually achieved, which is the difference between
# ignoring collectives and attacking them. A profile settles it.
#
# This uses SGLang's own torch-profiler endpoints rather than nsys: they need no extra
# privileges, they annotate by stage, and `profile_by_stage` splits prefill from decode
# -- which is the whole question.
#
#   ENGINES=sglang SPECS=solo4 PROFILE_N=8 run/submit-site.sh run/20-profile.sbatch
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
source "$HERE/client-lib.sh" || exit 1

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "x $*" >&2; exit 1; }

ENGINE_NAME="${ENGINES:-sglang}"
SPEC="${SPECS:-solo4}"
PROFILE_N="${PROFILE_N:-8}"
PROFILE_REQS="${PROFILE_REQS:-16}"
ISL="${ISL:-4000}"; OSL="${OSL:-200}"
SERVED_NAME="${SERVED_NAME:-qwen}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
MODEL_PATH="${MODEL_PATH:?the site should have set this}"
KIT_DIR="${KIT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/AIPerf}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"
OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/results/profile-${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUT_ROOT" || die "cannot create $OUT_ROOT"

mapfile -t NODES < <(sched_nodes)
(( ${#NODES[@]} >= 1 )) || die "no nodes"
sched_preflight "${NODES[@]}" || die "an allocated node cannot run work"
sched_reset_pools "${NODES[@]}"
HEAD="${NODES[0]}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(sched_run "$HEAD" 1 nvidia-smi -L 2>/dev/null | grep -c '^GPU ')}"
(( GPUS_PER_NODE > 0 )) || die "cannot see any GPU on $HEAD"
say "site ${SITE_NAME:-$PD_SITE}  $HEAD  ${GPUS_PER_NODE} GPUs  engine $ENGINE_NAME  spec $SPEC"

# The profiler writes one trace per TP rank into this directory, which must exist and
# be on shared storage -- the ranks are separate processes and may be on another node.
PROF_DIR="$OUT_ROOT/traces"; mkdir -p "$PROF_DIR"
export SGLANG_TORCH_PROFILER_DIR="$PROF_DIR"
export SGLANG_PROFILE_RECORD_SHAPES="${SGLANG_PROFILE_RECORD_SHAPES:-1}"
export MODEL_PATH SERVED_NAME CONTEXT_LEN MEM_FRACTION SGLANG_TORCH_PROFILER_DIR

eng_load "$ENGINE_NAME" || die "cannot load engine $ENGINE_NAME"
PD_IMAGE="$(eng_image)"; export PD_IMAGE
export PD_WORK="$OUT_ROOT/deploy"; mkdir -p "$PD_WORK"

trap 'pd_teardown' EXIT INT TERM

if [[ "$SPEC" == solo* ]]; then
    tp="${SPEC#solo}"
    PD_ROLE=(solo); PD_NODE=("$HEAD"); PD_TP=("$tp")
    PD_PORT=(30000); PD_BOOT=(0)
    pd_resolve || die "cannot resolve $HEAD"
    GW="${PD_IP[0]}:30000"
    declare -F eng_pd_prepare >/dev/null && eng_pd_prepare "$PD_WORK"
    mapfile -t scmd < <(eng_serve_cmd "$MODEL_PATH" "$SERVED_NAME" 30000 "$tp")
    pd_write_launcher "$PD_WORK/worker.sh"
    mapfile -t SCHED_CONTAINER_ARGS < <(pd_container_args solo)
    PD_PIDS=()
    sched_step "$HEAD" "$tp" "$(( ${PD_CPUS_PER_WORKER:-16} + 8 ))" "$PD_WORK/solo.log" \
        bash "$PD_WORK/worker.sh" solo 30000 "${scmd[@]}" || die "cannot launch"
    PD_PIDS+=("${SCHED_PIDS[-1]}")
    pd_wait_healthy "http://$GW" "solo($ENGINE_NAME)" "${HEALTH_TIMEOUT:-1200}" \
                    "${PD_PIDS[0]}" "$PD_WORK/solo.log" || die "never became healthy"
else
    pd_parse "$SPEC" || die "bad spec"
    pd_place "$GPUS_PER_NODE" "${NODES[@]}" || die "cannot place $SPEC"
    pd_print_placement
    declare -F eng_pd_prepare >/dev/null && eng_pd_prepare "$PD_WORK"
    pd_write_launcher "$PD_WORK/worker.sh"
    pd_launch "$PD_WORK/worker.sh" || die "cannot launch workers"
    pd_wait_all "${HEALTH_TIMEOUT:-1200}" || die "workers never became healthy"
    pd_launch_gateway "${ROUTER_PORT:-8000}" || die "cannot launch gateway"
    GW="$(pd_gateway "${ROUTER_PORT:-8000}")"
    pd_wait_models "http://$GW" "${HEALTH_TIMEOUT:-1200}" >/dev/null || die "gateway never served"
fi
say "endpoint $GW"

# Warm up FIRST and profile SECOND. A cold profile is dominated by JIT compilation and
# graph capture, which is real time but not the time a scored run spends -- job 17914
# and two others were lost to conclusions drawn from a cold system.
say "warming up with $PROFILE_REQS requests"
mapfile -t wargs < <(client_prewarm_args "$OUT_ROOT/prewarm" "$GW" "$SERVED_NAME" "$PROFILE_REQS")
sched_run "$HEAD" "${CLIENT_CPUS:-8}" "$AIPERF_BIN" "${wargs[@]}" \
    > "$OUT_ROOT/prewarm.log" 2>&1 || say "  ! prewarm returned non-zero, continuing"

say "starting profiler -> $PROF_DIR"
curl -sS -X POST "http://$GW/start_profile" \
     -H 'Content-Type: application/json' \
     -d "{\"output_dir\": \"$PROF_DIR\", \"profile_by_stage\": true, \"record_shapes\": true}" \
     | sed 's/^/  /' || say "  ! start_profile failed -- is this an SGLang gateway?"

say "driving $PROFILE_REQS requests at concurrency $PROFILE_N"
mapfile -t pargs < <(client_prewarm_args "$OUT_ROOT/profiled" "$GW" "$SERVED_NAME" "$PROFILE_REQS")
sched_run "$HEAD" "${CLIENT_CPUS:-8}" "$AIPERF_BIN" "${pargs[@]}" \
    > "$OUT_ROOT/profiled.log" 2>&1 || say "  ! profiled run returned non-zero"

say "stopping profiler"
curl -sS -X POST "http://$GW/stop_profile" | sed 's/^/  /' || true
# The ranks flush asynchronously; without this the directory is empty or truncated.
for i in $(seq 1 24); do
    n=$(find "$PROF_DIR" -name '*.json*' -size +4k 2>/dev/null | wc -l)
    (( n > 0 )) && { sleep 5; break; }
    sleep 5
done
say "traces: $(find "$PROF_DIR" -name '*.json*' 2>/dev/null | wc -l) file(s), $(du -sh "$PROF_DIR" 2>/dev/null | cut -f1)"

pd_teardown
trap - EXIT INT TERM

echo
echo "===================== 時間去哪了 ====================="
"$HERE/profile-summary.py" "$PROF_DIR" 2>&1 | tee "$OUT_ROOT/breakdown.txt"
say "results under $OUT_ROOT"
