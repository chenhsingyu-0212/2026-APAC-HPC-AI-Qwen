#!/bin/bash
# Firmus / NSCC reconnaissance -- what is on this cluster, before you spend an
# allocation finding out.
#
#   ./run/00-recon.sh                       # report to stdout and recon-<host>.txt
#   ./run/00-recon.sh --out /path/file.txt
#
# Runs on a LOGIN NODE. Read-only: it lists, stats and greps, and starts nothing.
# No GPU allocation, no container, no downloads. Safe to run the minute you land.
#
# It answers Q1, Q5 and Q7 of docs/trtllm-plan.md, and hunts for the
# reference-scripts/ directory the application notes cite six times and which is
# not in the upstream repo -- if it exists anywhere, it is the authority for
# UCX_NET_DEVICES detection, which is currently the least-verified code here.
#
# Q2/Q3/Q4/Q6 need a GPU and a container: run/00-recon.sbatch does those.

set -uo pipefail

OUT=""
SCRATCH="${SCRATCH_DIR:-/scratch}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     OUT="${2:-}"; shift 2 ;;
        --scratch) SCRATCH="${2:-}"; shift 2 ;;
        -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
        *)         echo "x unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$OUT" ]] || OUT="recon-$(hostname -s)-$(date +%Y%m%d).txt"

exec > >(tee "$OUT") 2>&1

hdr()  { echo ""; echo "=============================================================="; \
         echo " $*"; echo "=============================================================="; }
ok()   { echo "  ok $*"; }
no()   { echo "  x  $*"; }
note() { echo "  !  $*"; }

echo "recon $(date -Is)  host $(hostname)  user ${USER:-?}"
echo "report -> $OUT"

# ---------------------------------------------------------------- the machine
hdr "0. Where am I"
echo "--- os / kernel ---"
cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)=' || echo "  (no os-release)"
uname -r
echo "--- scheduler ---"
if command -v sinfo >/dev/null 2>&1; then
    ok "slurm: $(sinfo --version 2>/dev/null)"
    echo "--- partitions ---"
    sinfo -o '%20P %5a %10l %6D %8t %N' 2>/dev/null | head -20
    echo "--- what a GPU node looks like ---"
    sinfo -N -o '%N %c %m %G %f' 2>/dev/null | grep -i gpu | head -10 \
        || note "no node advertises a GPU gres -- check 'scontrol show node'"
    echo "--- my limits ---"
    sacctmgr -nP show assoc user="${USER}" format=Account,Partition,GrpTRESMins,MaxJobs 2>/dev/null | head -10 \
        || note "sacctmgr gave nothing (accounting may be off)"
    squeue -u "${USER}" 2>/dev/null | head
elif command -v qstat >/dev/null 2>&1; then
    ok "pbs: $(qstat --version 2>&1 | head -1)"
    qstat -Q 2>/dev/null | head -20
else
    no "neither sinfo nor qstat found -- is this a login node?"
fi

# ---------------------------------------------------------------- Q1: images
hdr "1. Container images staged on the cluster   [Q1]"
for d in "$SCRATCH/enroot-images" "$SCRATCH/images" /opt/images; do
    [[ -d "$d" ]] || continue
    ok "$d"
    ls -lh "$d" 2>/dev/null | head -30
    echo "  --- anything that is not sglang ---"
    ls "$d" 2>/dev/null | grep -Ei 'trt|tensorrt|vllm|dynamo|nemo|triton' \
        && ok "a non-SGLang engine image is already staged -- Phase C may be a no-op" \
        || note "only SGLang is staged. You will have to build or import TRT-LLM yourself."
done
[[ -d "$SCRATCH/enroot-images" ]] || no "$SCRATCH/enroot-images does not exist -- the notes say it should"

echo "--- can I import an image myself? ---"
for c in enroot podman docker apptainer singularity; do
    command -v "$c" >/dev/null 2>&1 && ok "$c: $(command -v $c)"
done
command -v enroot >/dev/null 2>&1 || note "no enroot on the login node (it may still exist on compute nodes)"

# ---------------------------------------------------------------- reference-scripts
hdr "2. The reference-scripts/ the application notes keep citing"
echo "Not in the upstream git repo. If it is on shared storage, it is the authority"
echo "for UCX_NET_DEVICES detection -- the one thing run/pd-lib.sh guessed at."
FOUND=0
for d in "$SCRATCH" /global /home /opt /shared "$HOME"; do
    [[ -d "$d" ]] || continue
    while IFS= read -r hit; do
        ok "$hit"; FOUND=1
        ls -la "$hit" 2>/dev/null | head -20
        [[ -f "$hit/README.md" ]] && { echo "  --- its README ---"; sed -n '1,40p' "$hit/README.md"; }
    done < <(timeout 120 find "$d" -maxdepth 5 -type d -name 'reference-scripts' 2>/dev/null)
done
(( FOUND )) || note "not found. detect_ib() stays unverified; checklist 9 is the only check on it."

echo "--- any other worked examples lying around ---"
timeout 120 find "$SCRATCH" -maxdepth 4 \( -name '*.sbatch' -o -name 'disagg*.y*ml' -o -name '*hpcai*' \) \
     2>/dev/null | head -20 || true

# ---------------------------------------------------------------- Q5/Q7: weights
hdr "3. Model weights and disk headroom   [Q5, Q7]"
for d in "$SCRATCH/models" "$SCRATCH/weights" /models; do
    [[ -d "$d" ]] || continue
    ok "$d"
    ls -la "$d" 2>/dev/null | head -20
    for m in "$d"/*Qwen3-235B*; do
        [[ -d "$m" ]] || continue
        echo "  --- $m ---"
        echo "    size:  $(du -sh "$m" 2>/dev/null | cut -f1)"
        echo "    files: $(find "$m" -maxdepth 1 -type f 2>/dev/null | wc -l)"
        if [[ -f "$m/config.json" ]]; then
            echo "    --- quantisation (rules REQUIRE fp8 weights) ---"
            grep -iE 'quant|fp8|dtype|torch_dtype|max_position|num_experts|num_experts_per_tok' \
                 "$m/config.json" 2>/dev/null | head -20
        else
            no "no config.json in $m -- incomplete download?"
        fi
        ls "$m"/tokenizer* "$m"/*.jinja 2>/dev/null | head
    done
done

echo "--- headroom (a second checkpoint is ~230 GB) ---"
df -h "$SCRATCH" "$HOME" 2>/dev/null
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null | head
command -v lfs   >/dev/null 2>&1 && lfs quota -h -u "$USER" "$SCRATCH" 2>/dev/null | head

# ---------------------------------------------------------------- fabric
hdr "4. InfiniBand, as seen from here"
note "A login node often has no HCA. What matters is the COMPUTE node reading;"
note "run/00-recon.sbatch repeats this there."
if [[ -d /sys/class/infiniband ]]; then
    for p in /sys/class/infiniband/*/ports/*; do
        [[ -r "$p/state" ]] || continue
        d="${p%/ports/*}"; d="${d##*/}"
        printf '  %-12s port %-3s state=%-14s rate=%s\n' \
            "$d" "${p##*/}" "$(cat "$p/state" 2>/dev/null)" "$(cat "$p/rate" 2>/dev/null)"
    done
    echo "  --- what detect_ib() would export here ---"
    IB=""
    for p in /sys/class/infiniband/*/ports/*; do
        [[ -r "$p/state" ]] || continue
        read -r st < "$p/state"; [[ "$st" == *ACTIVE* ]] || continue
        d="${p%/ports/*}"; d="${d##*/}"
        IB+="${IB:+,}${d}:${p##*/}"
    done
    [[ -n "$IB" ]] && ok "UCX_NET_DEVICES=$IB" || note "no ACTIVE port here"
else
    no "/sys/class/infiniband absent -- no HCA on this node"
fi
command -v ibstat >/dev/null 2>&1 && { echo "--- ibstat ---"; ibstat 2>/dev/null | head -30; }

# ---------------------------------------------------------------- toolchain
hdr "5. Toolchain"
echo "--- python (AIPerf needs >=3.11,<3.14) ---"
for p in python3.13 python3.12 python3.11 python3; do
    command -v "$p" >/dev/null 2>&1 && echo "  $p -> $($p -V 2>&1)"
done
echo "--- modules ---"
command -v module >/dev/null 2>&1 && module avail 2>&1 | head -40 || note "no module command"
echo "--- outbound network (the AIPerf image prefetches a tokenizer at BUILD time,"
echo "    so a compute node with no egress is fine -- but the login node needs it) ---"
timeout 10 curl -sfI https://huggingface.co >/dev/null 2>&1 \
    && ok "huggingface.co reachable" || no "no egress to huggingface.co from here"
command -v hf >/dev/null 2>&1 && ok "hf cli: $(hf --version 2>&1 | head -1)"

hdr "Next"
cat <<'NEXT'
  1. Paste the answers into docs/trtllm-plan.md section 1.
  2. Q2/Q3/Q4/Q6 need a GPU and a container:
         sbatch run/00-recon.sbatch                    # whatever image is staged
         ENGINE_IMG=<trtllm.sqsh> sbatch run/00-recon.sbatch
  3. Only then write any Phase A code.
NEXT
echo ""
echo "report written to $OUT"
