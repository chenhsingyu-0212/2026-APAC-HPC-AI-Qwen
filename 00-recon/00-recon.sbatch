#!/bin/bash
#SBATCH --job-name=recon
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
# Compute-node reconnaissance: what is inside the container, and what does the
# engine's CLI actually look like. Answers Q2/Q3/Q4/Q6 of docs/trtllm-plan.md.
#
#   sbatch run/00-recon.sbatch
#   ENGINE_IMG=/scratch/enroot-images/trtllm.sqsh sbatch run/00-recon.sbatch
#   SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/00-recon.sbatch   # also probes
#                                             /health, /v1/models and /metrics
#
# One GPU, half an hour, and it starts nothing you did not ask for. Run this
# before writing a line of Phase A: the whole point is that the engine backend
# gets written against the CLI that exists, not the one in a blog post.
#
# Submit it. Do not reach for `srun --pty` -- an interactive session holds the
# allocation for as long as it stays open, and most schedulers lower your
# priority the more time you have recently used.

set -uo pipefail

ENGINE_IMG="${ENGINE_IMG:-/scratch/enroot-images/sglang-harbor.sqsh}"
# NOT $HOME. Mounting the real home drags in .bashrc, whose conda init puts
# ~/miniforge3/bin first on PATH -- so python3 inside the container becomes the
# user's, not the image's, and nothing the image ships imports. Job 17593 spent
# eleven minutes discovering that. An empty directory is what the reference
# scripts bind, for the same reason.
RECON_HOME="${RECON_HOME:-${SLURM_SUBMIT_DIR:-$PWD}/.homes/recon-${SLURM_JOB_ID:-manual}}"
mkdir -p "$RECON_HOME"
MOUNTS="${MOUNTS:-/scratch:/scratch,$RECON_HOME:$HOME}"
SMOKE_MODEL="${SMOKE_MODEL:-}"
PROBE_PORT="${PROBE_PORT:-31999}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-600}"

die() { echo "x $*" >&2; exit 1; }
[[ -e "$ENGINE_IMG" ]] || die "container image not found: $ENGINE_IMG"

echo "=============================================================="
echo " recon  $(date -Is)"
echo " node:   $(hostname)"
echo " image:  $ENGINE_IMG"
echo " model:  ${SMOKE_MODEL:-<none -- inspection only, no server probe>}"
echo "=============================================================="

# ------------------------------------------------ the fabric, from a real node
# Host-side, outside the container, so /sys is the node's own.
echo ""
echo "--- InfiniBand on this compute node   [Q6] ---"
if [[ -d /sys/class/infiniband ]]; then
    for p in /sys/class/infiniband/*/ports/*; do
        [[ -r "$p/state" ]] || continue
        d="${p%/ports/*}"; d="${d##*/}"
        printf '  %-12s port %-3s state=%-14s rate=%-12s link=%s\n' "$d" "${p##*/}" \
            "$(cat "$p/state" 2>/dev/null)" "$(cat "$p/rate" 2>/dev/null)" \
            "$(cat "$p/phys_state" 2>/dev/null)"
    done
    IB=""
    for p in /sys/class/infiniband/*/ports/*; do
        [[ -r "$p/state" ]] || continue
        read -r st < "$p/state"; [[ "$st" == *ACTIVE* ]] || continue
        d="${p%/ports/*}"; d="${d##*/}"
        IB+="${IB:+,}${d}:${p##*/}"
    done
    echo ""
    echo "  detect_ib() would export: UCX_NET_DEVICES=${IB:-<empty>}"
    echo "  The rules' platform is 8x InfiniBand NDR. If you do not see eight ACTIVE"
    echo "  ports here, run/pd-lib.sh will put the KV cache on the wrong path and it"
    echo "  will read as 'PD is slow'. This never errors -- see checklist item 9."
    n=$(awk -F, '{print NF}' <<< "${IB:-}")
    [[ -n "$IB" ]] && echo "  -> ${n} ACTIVE port(s) found"
else
    echo "  x /sys/class/infiniband absent on a compute node -- that is a finding"
fi
echo ""
echo "--- GPU topology (which HCA is near which GPU) ---"
nvidia-smi topo -m 2>/dev/null || echo "  (nvidia-smi topo unavailable)"

# ------------------------------------------------ inside the container
echo ""
srun --container-image="$ENGINE_IMG" --container-mounts="$MOUNTS" \
     bash -c '
set -uo pipefail
echo "=============================================================="
echo " inside the container"
echo "=============================================================="

echo "--- gpu ---"
nvidia-smi -L || echo "  x nvidia-smi found no device"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null

echo ""
echo "--- python ---"
python3 -V
echo ""
echo "--- which engines are in here ---"
for m in tensorrt_llm sglang vllm dynamo; do
    python3 -c "import $m, sys; print(\"  ok $m\", getattr($m, \"__version__\", \"imported\"))" 2>/dev/null \
        || echo "  -  $m not importable"
done
echo ""
echo "--- KV transfer libraries   [Q6] ---"
for m in nixl ucp; do
    python3 -c "import $m; print(\"  ok $m importable\")" 2>/dev/null || echo "  -  $m not importable"
done
command -v ucx_info >/dev/null 2>&1 && { echo "  --- ucx_info -d (transports) ---"; \
    ucx_info -d 2>/dev/null | grep -E "Transport|Device:" | head -20; } \
    || echo "  -  no ucx_info binary"
echo "  UCX_NET_DEVICES currently: ${UCX_NET_DEVICES:-<unset>}"

echo ""
echo "=============================================================="
echo " the CLI, as it actually exists   [Q2]"
echo "=============================================================="
if command -v trtllm-serve >/dev/null 2>&1; then
    echo "--- trtllm-serve --help ---"
    trtllm-serve --help 2>&1 | head -80
    echo ""
    echo "--- subcommands that mention disagg ---"
    trtllm-serve --help 2>&1 | grep -i disagg
    for sub in disaggregated disaggregated_mpi_worker serve; do
        echo ""
        echo "--- trtllm-serve $sub --help ---"
        trtllm-serve "$sub" --help 2>&1 | head -60
    done
    echo ""
    echo "--- does it document the disagg config schema anywhere in the image? ---"
    find / -maxdepth 8 \( -name "*disagg*.y*ml" -o -name "*disagg*.py" \) 2>/dev/null \
        | grep -v "/proc/" | head -20
else
    echo "  -  no trtllm-serve in this image"
fi
if command -v vllm >/dev/null 2>&1; then
    echo ""
    echo "--- vllm serve: the KV-transfer flag   [Phase E] ---"
    vllm serve --help 2>&1 | grep -iA4 "kv-transfer\|kv_transfer" | head -30
fi
'
RC=$?
echo ""
echo "container inspection exited $RC"

# ------------------------------------------------ Q3/Q4: the served endpoints
if [[ -z "$SMOKE_MODEL" ]]; then
    echo ""
    echo "=============================================================="
    echo " Q3/Q4 skipped -- no SMOKE_MODEL."
    echo " /health, /v1/models and /metrics are what pd_wait_healthy(),"
    echo " submit.sbatch:135 and checklist 7/8 depend on. Re-run with a small"
    echo " model to find out whether this engine serves them:"
    echo "     SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/00-recon.sbatch"
    echo "=============================================================="
    exit $RC
fi
[[ -d "$SMOKE_MODEL" ]] || die "SMOKE_MODEL not found: $SMOKE_MODEL"

echo ""
echo "=============================================================="
echo " probing a single server on $SMOKE_MODEL   [Q3, Q4]"
echo "=============================================================="
echo "! This launches SGLang. For TRT-LLM or vLLM, replace the launch line below"
echo "! with what the --help output above actually showed -- that is the point of"
echo "! running this job before writing run/engines/*.sh."

NODE="$(hostname)"
srun --overlap -N1 -n1 --container-image="$ENGINE_IMG" --container-mounts="$MOUNTS" \
     python3 -m sglang.launch_server \
        --model-path "$SMOKE_MODEL" --served-model-name probe --trust-remote-code \
        --enable-metrics --host 0.0.0.0 --port "$PROBE_PORT" > recon-server-$SLURM_JOB_ID.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT INT TERM

waited=0
while (( waited < PROBE_TIMEOUT )); do
    curl -sf --max-time 5 "http://$NODE:$PROBE_PORT/health" >/dev/null 2>&1 && break
    sleep 5; waited=$(( waited + 5 ))
done

if (( waited >= PROBE_TIMEOUT )); then
    echo "x never became healthy in ${PROBE_TIMEOUT}s -- see recon-server-$SLURM_JOB_ID.log"
    tail -40 "recon-server-$SLURM_JOB_ID.log"
    exit 1
fi
echo "ok healthy after ${waited}s"

echo ""
echo "--- /v1/models   [Q3: submit.sbatch:135 parses .data[0].id from this] ---"
curl -s --max-time 10 "http://$NODE:$PROBE_PORT/v1/models" | head -c 2000; echo

echo ""
echo "--- /metrics: is checklist 7/8 satisfiable on this engine?   [Q4] ---"
RAW="$(curl -s --max-time 10 "http://$NODE:$PROBE_PORT/metrics" 2>/dev/null)"
if [[ -z "$RAW" ]]; then
    echo "  x /metrics served nothing."
    echo "    engine_metrics.csv will come out empty, and SCHEDULER CAP, KV CACHE"
    echo "    and the roofline's batch size degrade silently. See plan section 1, Q4."
else
    echo "  ok $(wc -l <<< "$RAW") lines"
    echo "  --- what observe.sh's filter would actually keep ---"
    echo "  (its regex: running|waiting|queue|pending|preempt|cache_usage|token_usage)"
    grep -E '^[a-zA-Z]' <<< "$RAW" | grep -Ei 'running|waiting|queue|pending|preempt|cache_usage|token_usage' \
        | grep -v '^#' | awk '{print "    " $0}' | head -30
    echo ""
    echo "  --- everything else it exposes, names only ---"
    grep '^# HELP' <<< "$RAW" | awk '{print "    " $3}' | head -60
fi

echo ""
echo "--- a real request, streaming (the shape the scored command uses) ---"
curl -s --max-time 60 "http://$NODE:$PROBE_PORT/v1/chat/completions" \
     -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
     -d '{"model":"probe","messages":[{"role":"user","content":"hi"}],
          "max_tokens":8,"stream":true}' | head -5

echo ""
echo "=============================================================="
echo " done. Write the answers into docs/trtllm-plan.md section 1."
echo "=============================================================="
