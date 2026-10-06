#!/bin/bash
#SBATCH --job-name=tune-moe
#SBATCH --nodes=1
#SBATCH --gres=gpu:gpu:1
#SBATCH --time=01:30:00
#PBS -N tune-moe
#PBS -l walltime=01:30:00
#PBS -j oe
#
# Tune the Triton fused-MoE kernel for THIS shape on THIS GPU.
#
# WHY. Our shape is E=128, N=384, fp8_w8a8, block_shape=[128,128]. The tuned
# configs both engines ship do not cover it on an H100:
#
#   vLLM   has E=128,N=384 for NVIDIA_H200, for B200 and for MI300X -- and an
#          H100 file only WITHOUT the fp8_w8a8/block_shape suffix, i.e. for a
#          different dtype. So on H100 it falls back to default heuristics.
#   SGLang has our exact file, but only under configs/triton_3_3_1 while the
#          installed Triton is 3.7.1. Its own loader logs "Performance might be
#          sub-optimal!" and the source says to re-tune in your environment. The
#          DOWN-projection file does not exist in any of its eight version
#          directories, so that GEMM reuses the up-projection config and loses
#          TMA -- the engine says so at startup.
#
# MoE GEMM is 46-49% of decode kernel time on this model (nsys, teammate's 20410),
# and vLLM is the engine bound by decode ITL, so this is aimed at vLLM first.
#
# WHAT IT COSTS. One GPU. The tuner reads config.json through
# vllm.transformers_utils.config and benchmarks synthetic tensors, so there is no
# 221 GB weight load. Tuning writes a JSON keyed on
# (E, N, device_name, dtype, block_shape) plus the Triton version.
#
# THE RESULT IS GPU-SPECIFIC. The filename carries device_name, so a config tuned
# on anything but NVIDIA_H100_80GB_HBM3 is useless for NSCC. This script refuses
# to run on the wrong GPU rather than produce a file that silently never loads.
#
# HOW LONG IT ACTUALLY TAKES. MEASURED on an H100 (NYCU job 98469): the search is
# ~1,920 configurations per batch size at ~1.6/s, so **about 20 minutes per batch
# size**. The tuner's default list is eighteen sizes, which is ~6 hours; the four
# sizes this deployment actually runs (decode residency 24-48, prefill ~4,096) are
# ~80 minutes. And benchmark_moe.py writes its JSON once, at the very end, so a
# walltime that runs out produces NOTHING. Ask for twice what you think you need.
#
# 98469 asked for 28 minutes on NYCU's trialq, got 1.5% of the way through one batch
# size, and timed out with an empty configs directory.
#
#   run/submit-site.sh run/22-tune-moe.sbatch          # defaults: vLLM, TP4
#   ENGINE=sglang run/submit-site.sh run/22-tune-moe.sbatch
#   MOE_BATCH_SIZES="24 32 48 4096" TIME_LIMIT=03:00:00 ...   # what we operate at
#
# TUNE_MODE=sep IS PARKED, AND THE REASON IS NOT THE TOOL. Five attempts on free NYCU
# nodes, in order: ModuleNotFoundError ray; ModuleNotFoundError common_utils; my own
# precondition running without CUDA_HOME; a 30-minute wall-clock timeout; and finally job
# 99894, which searched all 1,920 configurations in 45m32s on a 3-hour allocation,
# printed "Completed tuning for batch_size=32", exited 0 and wrote NOTHING -- no
# tuning_result_*.txt, no E=*.json, and none of main()'s completion prints. Root cause
# unknown. A batch-size-1 repro (job 100072) also ran past 25 minutes, so the search space
# is 1,920 configurations regardless of batch size: budget ~45 minutes per size.
#
# What parks it is job 216202, not the failures. The _down config exists to shorten the
# ITL tail, and 216202 took ITL failures from 69 to ZERO at N=88 by changing the WARM-UP
# -- no tuning, no new file, already implemented. The tail this tool was aimed at has
# already been addressed, so the tool is now speculative rather than targeted.
#
# ONE BATCH SIZE PER TRIALQ JOB. benchmark_moe.py writes its JSON only after every
# requested size has been searched, so a job that runs out of wall clock leaves
# NOTHING on disk -- job 98820 asked for three sizes, hit trialq's 30-minute cap, and
# produced zero files. The repo's "20 minutes per batch size" figure is right; it was
# the three-sizes-in-30-minutes guess that was wrong.
#
# WHAT THE DECODE WORKER ACTUALLY ASKS FOR, read out of job 215941's decode log
# rather than reasoned about. Two separate gaps, both of which this job can close:
#
#   Config file not found at .../configs/triton_3_7_1/E=128,N=384,
#     device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json.
#     Fallback to triton version 3.3.1 ... Performance might be sub-optimal!
#   Down MoE config file not found at .../triton_3_7_1/...[128, 128]_down.json;
#     reusing the tuned up-projection config without TMA. Performance might be sub-optimal.
#
# So: the venv ships Triton 3.7.1 and SGLang has no 3.7.1 config for this GPU, so it
# silently loads the 3.3.1 one; and there is no down-projection config at all, so the
# down GEMM runs the up-projection's tiling WITHOUT TMA. We are not choosing between
# "tuned" and "default" -- we are already running a config tuned for a different
# Triton and a down GEMM with no config of its own.
#
# WHICH BATCH SIZES, measured rather than assumed. 215941's two decode workers
# reported #running-req over 934 Decode-batch steps:
#
#            p10  p50  p90  max
#   decode-0  24   29   31   32
#   decode-1  25   29   31   33
#
# Tightly clustered at 28-32. SGLang picks the config whose key is NEAREST to M, so
# M=29 resolves to 32 and a config tuned at 48 is never selected at all -- which is
# what job 98797 was tuning before this measurement, and why it was cancelled. Tune
# 32 first, then 24 and 16 for the low tail. Note this is per DECODE WORKER, not the
# client's concurrency: N=80 over two decode workers leaves ~29 decoding on each and
# the rest in prefill or in KV transfer (#transfer-req ran 33-35).
#
# N=384 is 1536/TP at TP4 (Qwen3-235B moe_intermediate_size 1536), which is why TP
# below must stay 4 for the filename to match what the decode worker looks up. And
# decode's `moe_runner_backend: 'auto'` resolves to Triton -- the prefill role is on
# deep_gemm, so this config touches DECODE ONLY.
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

ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"
MOE_DTYPE="${MOE_DTYPE:-fp8_w8a8}"
# WHICH BATCH SIZES TO TUNE. Empty means the tuner's own list of eighteen, from 1 to
# 4096, which is what a config meant for general use needs. Narrow it only when a
# queue's walltime cap would otherwise truncate the run -- benchmark_moe.py writes
# the JSON once, at the end, so a timeout saves nothing at all.
#
# The sizes that matter for THIS deployment: decode runs a batch equal to its
# residency, which is 24-40 requests at the concurrencies we score, and prefill runs
# ~4,008 tokens. A config tuned only at those points is right where we operate and a
# nearest-key match everywhere else, so it is a probe, not the file to adopt.
MOE_BATCH_SIZES="${MOE_BATCH_SIZES:-}"
eng_load "$ENGINE" || exit 1
say() { echo "[$(date +%H:%M:%S)] $*"; }

# UP-ONLY OR BOTH GEMMS. SGLang runs TWO grouped GEMMs per MoE layer and looks up a
# config for each: the up projection, and the down projection under the same name plus
# "_down". When the _down file is absent the engine says so and borrows the up config
# WITHOUT TMA:
#
#   Down MoE config file not found at ..._down.json;
#   reusing the tuned up-projection config without TMA. Performance might be sub-optimal.
#
# That is not a small loss. The reference H200 campaign that measures 4,395-4,475 good
# tok/s tunes both, and its up and down configs disagree about the tiling at the batch
# size we run: up wants BLOCK_SIZE_N 64, down wants 128 with USE_TMA. It keeps an
# "up_only" tree purely as the A/B control arm, labelled "no _down, no TMA".
#
# Our job 216159 installed an up-only config and measured average ITL 19.03 -> 20.32 ms,
# over the gate, with 1,317 of 1,790 requests failing. We had measured the control arm.
#
# TUNE_MODE=sep uses SGLang's tuning_fused_moe_triton_sep.py, which writes both files and
# picks TMA-vs-no-TMA for the down GEMM. It is not in the pip wheel -- it ships only in
# the source tree -- so it is vendored next to the other tuner.
TUNE_MODE="${TUNE_MODE:-up}"
case "$ENGINE" in
  vllm)   TUNER="$HERE/benchmark_moe-v0.30.0.py"
          CFGDIR="$(dirname "$(eng_python)")/../lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs" ;;
  sglang) TUNER="$HERE/tuning_fused_moe_triton-v0.5.20.py"
          [[ "$TUNE_MODE" == sep ]] && TUNER="$HERE/tuning_fused_moe_triton_sep-v0.5.20.py"
          CFGDIR="$(dirname "$(eng_python)")/../lib/python3.10/site-packages/sglang/srt/layers/moe/moe_runner/triton_utils/configs" ;;
  *) echo "x no tuner for engine $ENGINE" >&2; exit 1 ;;
esac
[[ -r "$TUNER" ]] || { echo "x missing tuner $TUNER" >&2; exit 1; }
if [[ "$TUNE_MODE" == sep ]]; then
    [[ "$ENGINE" == sglang ]] || {
        echo "x TUNE_MODE=sep is SGLang's tuner; set ENGINE=sglang" >&2; exit 1; }
    [[ -r "$HERE/gen-topk-ids.py" ]] || {
        echo "x TUNE_MODE=sep needs run/gen-topk-ids.py for --topk-ids-dir" >&2; exit 1; }
    [[ -n "$MOE_BATCH_SIZES" ]] || {
        echo "x TUNE_MODE=sep takes one --batch-size per invocation, so MOE_BATCH_SIZES" >&2
        echo "  must be set; the tuner has no all-sizes mode here." >&2; exit 1; }
    # RAY, CHECKED HERE RATHER THAN IN THE ALLOCATION. The separate-GEMM tuner runs one
    # remote worker per GPU (`ray.init()`, `ray.available_resources()["GPU"]`), and the
    # SGLang venv does not ship it -- only the vLLM one does. NYCU job 99896 reached a
    # compute node, passed the GPU guard, generated all 100 topk_ids files and then died
    # on `ModuleNotFoundError: No module named 'ray'` with an empty config directory.
    [[ -r "$HERE/moe-tuner-deps/common_utils.py" ]] || {
        echo "x TUNE_MODE=sep needs run/moe-tuner-deps/common_utils.py" >&2
        echo "  the tuner does 'from common_utils import ...' and the wheel has neither" >&2
        exit 1; }
    # WALK THE WHOLE IMPORT CHAIN HERE, not one module at a time. NYCU job 99896 reached
    # a compute node and died on `import ray`; 99897 got past that and died on
    # `common_utils`. Each cost a node and a round trip to learn one name. Executing the
    # tuner as a module runs every top-level import and nothing else -- its main() is
    # behind `if __name__ == "__main__"` -- so one check on the submit host covers ray,
    # common_utils, and whatever the next one turns out to be.
    # IN THE ENVIRONMENT THE RUN ACTUALLY HAS. deep_gemm's import calls find_cuda_home()
    # and asserts on it, and sglang pulls deep_gemm in transitively, so the chain only
    # resolves where CUDA_HOME is set. The launcher exports eng_pd_env for exactly that
    # reason; a check that skips it tests a different environment and NYCU job 99899 died
    # on a bare `AssertionError` from deep_gemm/cuda_helpers.py line 20 -- my check's bug,
    # not the tuner's.
    mapfile -t _chk_env < <(eng_pd_env 2>/dev/null | grep -vw 'port')
    _sep_err="$(env "${_chk_env[@]}" \
                PYTHONPATH="$HERE/moe-tuner-deps${PYTHONPATH:+:$PYTHONPATH}" \
                "$(eng_python)" - "$TUNER" <<'SEPCHK' 2>&1
import importlib.util, sys, traceback
spec = importlib.util.spec_from_file_location("septuner", sys.argv[1])
try:
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
except Exception:
    # The last line alone was "AssertionError" with no message, which named nothing.
    for line in traceback.format_exc().strip().splitlines()[-6:]:
        print(line, file=sys.stderr)
    sys.exit(1)
SEPCHK
)" || {
        echo "x TUNE_MODE=sep: the tuner cannot be imported with the $ENGINE venv:" >&2
        printf '    %s\n' "$_sep_err" >&2
        echo "  ray is the usual one and lives only in the vLLM venv:" >&2
        echo "    $(eng_python) -m pip install 'ray==2.53.0'" >&2
        echo "  (2.53.0 is what the vLLM venv carries; adding it changed no other version)" >&2
        exit 1; }
fi

mapfile -t NODES < <(sched_nodes)
sched_preflight "${NODES[@]}" || exit 1
sched_reset_pools "${NODES[@]}"
HEAD="${NODES[0]}"
OUT="${SLURM_SUBMIT_DIR:-$PWD}/results/tune-moe-${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT/configs"

say "engine $ENGINE  tp $TP  dtype $MOE_DTYPE  batch ${MOE_BATCH_SIZES:-<tuner default, 18 sizes>}  on $HEAD"
say "tuner  $TUNER"
say "config dir $CFGDIR"

# The launcher carries eng_pd_env (CUDA_HOME, headers, LD_LIBRARY_PATH) into the
# step; the tuner JITs Triton kernels and needs all of it.
# eng_pd_env is written for a PD WORKER's launcher, where $port is in scope, so
# vLLM's side-channel line arrives as an expression: 5600 + port % 1000. The tuner
# has no port and no side channel, and under set -u that line is fatal -- NYCU job
# 97730 died one second in on "port: unbound variable". Drop anything that refers to
# $port rather than inventing a value for it, because a side channel the tuner never
# opens should not be configured at all.
mapfile -t env_lines < <(eng_pd_env 2>/dev/null | grep -vw 'port')   # -w: 'export' contains 'port'
{
    echo '#!/bin/bash'
    echo 'set -uo pipefail'
    printf '%s\n' "${env_lines[@]/#/export }"
    # REFUSE THE WRONG GPU. A config tuned elsewhere is named for that GPU and
    # would never be loaded on an H100 -- a silent no-op is worse than a failure.
    cat <<'GUARD'
want="${EXPECT_GPU:-NVIDIA H100 80GB HBM3}"
got="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if [ "$got" != "$want" ]; then
    echo "x tuning on '$got' but the configs are keyed on '$want' -- refusing" >&2
    echo "  set EXPECT_GPU to override if that is deliberate" >&2
    exit 1
fi
echo "[$(hostname)] gpu: $got"
GUARD
    if [[ "$TUNE_MODE" == sep ]]; then
        cat "$HERE/tune-moe-sep-inner.sh"
    else
        echo 'exec "$@"'
    fi
} > "$OUT/tune.sh"

SCHED_CONTAINER_ARGS=()
mapfile -t SCHED_CONTAINER_ARGS < <(pd_container_args tune)
if [[ "$TUNE_MODE" == sep ]]; then
    sched_step "$HEAD" 1 "${JOB_CPUS:-32}" "$OUT/tune.log" \
        bash "$OUT/tune.sh" "$(eng_python)" "$HERE/gen-topk-ids.py" "$TUNER" \
            "$MODEL_PATH" "$OUT" "$TP" "$MOE_DTYPE" "$MOE_BATCH_SIZES" \
        || { echo "x cannot launch the tuner" >&2; exit 1; }
else
    sched_step "$HEAD" 1 "${JOB_CPUS:-32}" "$OUT/tune.log" \
        bash "$OUT/tune.sh" "$(eng_python)" "$TUNER" \
            --model "$MODEL_PATH" --tp-size "$TP" --dtype "$MOE_DTYPE" \
            --trust-remote-code --tune --save-dir "$OUT/configs" \
            ${MOE_BATCH_SIZES:+--batch-size $MOE_BATCH_SIZES} \
        || { echo "x cannot launch the tuner" >&2; exit 1; }
fi
say "tuning -- the search is over BLOCK_SIZE_M x num_warps x num_stages per batch size"
sched_wait_all; rc=$?

say "exit $rc"
echo "--- produced:"; ls -la "$OUT/configs" 2>/dev/null
# Do NOT install it automatically. A config that changes every kernel launch is a
# change to the scored configuration, and it has to be A/B'd before it is adopted.
[[ -s "$OUT/sep-report.txt" ]] && { echo "--- sep report:"; cat "$OUT/sep-report.txt"; }
say "NOT installed. To use it:  cp $OUT/configs/*.json $CFGDIR/   (then A/B it)"
tail -25 "$OUT/tune.log"
exit $rc
