#!/bin/bash
#SBATCH --job-name=smoke-serve-aiperf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --partition=hpcai
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#
# The hpcai partition caps a job at 17 minutes (sinfo TIMELIMIT 17:00), and
# Slurm's kill at the cap is a SIGKILL that leaves no artefacts. Every
# reference script under /scratch/reference-scripts uses -t 00:17:00 or less.
# Do not raise this: the job will simply be rejected at submission.
#
# Smoke test 3/3: a server from the engine image and a client from the AIPerf
# image complete a full streaming request/response cycle.
#
#   sbatch run/03-smoke-serve-aiperf.sbatch                  # ENGINE=sglang
#   ENGINE=trtllm sbatch run/03-smoke-serve-aiperf.sbatch
#   SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/03-smoke-serve-aiperf.sbatch
#
# This is the last thing that can be wrong in a way that looks like a modelling
# problem: two container images, two processes, and a network path between them.
# Once a clean exit code comes out of here, a failure in the PD job is about the
# PD topology and not about plumbing.
#
# The AIPerf command below is deliberately NOT the official one. It is tiny on
# purpose -- a few short requests, just enough to prove the round trip. The rules
# leave your own exploratory runs unconstrained; only the scored re-run is fixed,
# and that one lives in official.sh.
#
# The client half is engine-neutral, and that is the point: it is the same
# instrument for every engine, which is what makes a later TRT-LLM vs vLLM
# comparison a finding rather than an anecdote.

set -uo pipefail

# Slurm copies the batch script to /var/spool/slurm/job<id>/, so ${BASH_SOURCE[0]}
# points at that copy and not into the repo -- job 17590 died in one second on
# exactly this. SLURM_SUBMIT_DIR is where sbatch was invoked from; fall back to
# BASH_SOURCE so the script still works when run directly.
HERE=""
for _d in "${SLURM_SUBMIT_DIR:-}/run" "${SLURM_SUBMIT_DIR:-}" \
          "$(dirname "${BASH_SOURCE[0]}")" "$PWD/run" "$PWD"; do
    [[ -n "$_d" && -r "$_d/engine-lib.sh" ]] && { HERE="$(cd "$_d" && pwd)"; break; }
done
[[ -n "$HERE" ]] || { echo "x cannot find engine-lib.sh -- run this from the repo, or set SLURM_SUBMIT_DIR" >&2; exit 1; }
# shellcheck source=run/engine-lib.sh
source "$HERE/engine-lib.sh" || { echo "x cannot source engine-lib.sh" >&2; exit 1; }

die() { echo "x $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

eng_load || exit 1
IMG="$(eng_image)"
AIPERF_IMG="${AIPERF_IMG:-/scratch/enroot-images/aiperf.sqsh}"
# NOT $HOME -- see eng_mounts() in engine-lib.sh
MOUNTS="${MOUNTS:-$(eng_mounts smoke3-${SLURM_JOB_ID:-manual})}"
SMOKE_MODEL="${SMOKE_MODEL:-/scratch/models/Qwen3-0.6B}"
SERVED_NAME="${SERVED_NAME:-qwen}"
PORT="${PORT:-31000}"
TP="${TP:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
OUT="${OUT:-${SLURM_SUBMIT_DIR:-$PWD}/results/smoke-$ENGINE-${SLURM_JOB_ID:-manual}}"

for f in "$IMG" "$AIPERF_IMG"; do [[ -e "$f" ]] || die "container image not found: $f"; done
if [[ "$SMOKE_MODEL" == /* && ! -d "$SMOKE_MODEL" ]]; then
    echo "x model path not found: $SMOKE_MODEL" >&2
    ls -1 "$(dirname "$SMOKE_MODEL")" 2>/dev/null | head -20 >&2
    exit 1
fi
mkdir -p "$OUT" || die "cannot create $OUT"

HOSTNAME_="$(hostname)"
eng_banner

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && { say "stopping the server"; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- serve
say "serving $SMOKE_MODEL on $HOSTNAME_:$PORT (tp=$TP)"
mapfile -t SERVE_CMD < <(eng_serve_cmd "$SMOKE_MODEL" "$SERVED_NAME" "$PORT" "$TP")
(( ${#SERVE_CMD[@]} )) || die "$ENGINE's eng_serve_cmd produced nothing"
say "  ${SERVE_CMD[*]}"

srun --overlap --container-image="$IMG" --container-mounts="$MOUNTS" \
     "${SERVE_CMD[@]}" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

eng_wait_healthy "http://$HOSTNAME_:$PORT" "$SERVER_PID" "$HEALTH_TIMEOUT" "$OUT/server.log" || exit 1

MODEL_ID="$(eng_model_id "http://$HOSTNAME_:$PORT")"
[[ -n "$MODEL_ID" ]] || die "healthy, but $(eng_models_path) returned nothing"
say "  ok serving '$MODEL_ID'"

# ---------------------------------------------------------------- round trip
# --tokenizer points at the same local path as the model so the client does not
# depend on outbound network, and its token counts match what the server did.
say "running a short aiperf round trip from the client image"
# --artifact-dir is explicit here on purpose. Relying on the default plus
# --container-workdir means a site whose pyxis honours workdir differently
# reports FAIL on a round trip that actually worked. The *official* command has
# no --artifact-dir and adding one would modify it, which is why submit.sbatch
# has to cd instead -- but this is a smoke test, so it can just say where.
srun --overlap --container-image="$AIPERF_IMG" --container-mounts="$MOUNTS" \
     aiperf profile \
        --artifact-dir "$OUT/artifacts" \
        --url "http://$HOSTNAME_:$PORT" \
        --endpoint-type chat --endpoint /v1/chat/completions \
        --model "$MODEL_ID" --tokenizer "$SMOKE_MODEL" \
        --streaming -H 'Accept: text/event-stream' \
        --concurrency 2 --request-count 10 \
        --synthetic-input-tokens-mean 128 --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean 16 --output-tokens-stddev 0 \
        --extra-inputs min_tokens:16 --extra-inputs max_tokens:16 \
        --extra-inputs ignore_eos:true \
        --ui simple
RC=$?

# A zero exit with no measurements is still a failure: check something was
# actually measured rather than trusting the return code alone.
CSV="$(find "$OUT" -name profile_export_aiperf.csv 2>/dev/null | head -1)"
if (( RC == 0 )) && [[ -n "$CSV" ]]; then
    say "results: $CSV"
    grep -E "^(Request Count|Time to First Token|Inter Token Latency)" "$CSV" | cut -d, -f1,2
    REQ="$(awk -F, '$1 == "Request Count" { print $2 }' "$CSV")"
    awk -v n="${REQ:-0}" 'BEGIN { exit !(n + 0 > 0) }' \
        || { say "aiperf exited 0 but measured no requests"; RC=1; }
else
    (( RC == 0 )) && { say "aiperf exited 0 but wrote no results"; RC=1; }
fi

# Worth keeping: which engine produced these numbers is the first thing you will
# want to know when two result directories sit side by side.
{
    echo "engine=$ENGINE"
    echo "image=$IMG"
    echo "model=$SMOKE_MODEL  model_id=$MODEL_ID  tp=$TP"
    echo "when=$(date -Is)  node=$HOSTNAME_"
} > "$OUT/engine.txt"

echo "=========================================="
(( RC == 0 )) && echo "PASS -- client, network path and $ENGINE are wired correctly" \
              || echo "FAIL -- exit $RC; see $OUT/server.log and the aiperf output above"
echo " next: run/submit.sbatch (the full PD deployment and the scored command)"
echo "=========================================="
exit $RC
