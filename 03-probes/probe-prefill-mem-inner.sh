# Appended to a generated launcher by run/23-probe-prefill-mem.sbatch, after the
# environment exports and the GPU guard. Kept as its own file so it can be read and
# tested on its own; it is not executable by itself.
PY="$1"; MODEL="$2"; OUT="$3"; CP="$4"; MP="$5"
BASE="$6"; FLOOR="$7"; MARGIN="$8"; TP="$9"; READY="${10}"
REPORT="$OUT/report.txt"
: > "$REPORT"
rep() { echo "$*" | tee -a "$REPORT"; }

GPU_TOTAL_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
                | head -1 | awk '{printf "%.2f", $1/1024}')"
[ -n "$GPU_TOTAL_GB" ] || { echo "x cannot read GPU memory total" >&2; exit 1; }
rep "gpu           $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${GPU_TOTAL_GB} GB"
rep "prefill batch chunked=$CP max_prefill=$MP tp=$TP"
rep ""

# One pass: start a prefill-mode worker, wait only as long as it takes for the
# startup to declare itself, then stop it. Nothing is served.
run_pass() {    # $1=mem_fraction  $2=tag  -> echoes the verdict word
    local f="$1" tag="$2" log="$OUT/pass-$tag.log" pid t=0 verdict=timeout
    "$PY" -m sglang.launch_server \
        --model-path "$MODEL" --served-model-name qwen --trust-remote-code \
        --context-length 8192 --tp-size "$TP" --mem-fraction-static "$f" \
        --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
        --disaggregation-bootstrap-port 9000 \
        --moe-runner-backend deep_gemm --disable-prefill-cuda-graph \
        --disable-radix-cache --enable-metrics \
        --chunked-prefill-size "$CP" --max-prefill-tokens "$MP" \
        --host 0.0.0.0 --port 31100 > "$log" 2>&1 &
    pid=$!
    while [ "$t" -lt "$READY" ]; do
        if grep -qa 'reducing max_m to' "$log" 2>/dev/null; then verdict=reduced; break; fi
        if grep -qa 'fired up and ready to roll' "$log" 2>/dev/null; then verdict=fit; break; fi
        kill -0 "$pid" 2>/dev/null || { verdict=died; break; }
        sleep 5; t=$((t+5))
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    # The worker holds four GPUs; make sure they are free before the next pass.
    sleep 10
    echo "$verdict"
}

# >>> parsers  (run/test-probe-mem.sh extracts exactly this block; keep the markers)
# The largest requirement across the GEMM shapes is the binding one, and the smallest
# `Available memory` is what we actually had.
need_gb() { grep -ahoE 'required memory [0-9.]+GB' "$1" 2>/dev/null \
            | awk '{gsub("GB","",$3); if($3+0>m) m=$3+0} END{if(m)printf "%.2f", m}'; }
have_gb() { grep -ahoE 'Available memory [0-9.]+GB' "$1" 2>/dev/null \
            | awk '{gsub("GB","",$3); if(n==0||$3+0<m){m=$3+0;n=1}} END{if(n)printf "%.2f", m}'; }
maxm()    { grep -ahoE 'reducing max_m to [0-9]+' "$1" 2>/dev/null \
            | awk '{print $NF}' | sort -n | head -1; }
kvtok()   { grep -ahoE 'max_total_num_tokens=[0-9]+|#token: [0-9]+' "$1" 2>/dev/null \
            | grep -oE '[0-9]+' | sort -n | tail -1; }
# <<< parsers

rep "--- pass 1: mem-fraction $BASE (what we ship) ---"
v1="$(run_pass "$BASE" "1-$BASE")"
L1="$OUT/pass-1-$BASE.log"
NEED="$(need_gb "$L1")"; HAVE="$(have_gb "$L1")"; MM="$(maxm "$L1")"
rep "verdict       $v1"
rep "required      ${NEED:-<not reported>} GB"
rep "available     ${HAVE:-<not reported>} GB"
rep "max_m reduced ${MM:-<not reduced>}"
rep "kv tokens     $(kvtok "$L1" || echo '-')"
rep ""

if [ "$v1" = fit ]; then
    rep "RESULT  a $CP-token prefill batch already fits at mem-fraction $BASE."
    rep "        Nothing to free. The 25.31 GB in job 215921 was not this configuration."
    exit 0
fi
if [ "$v1" = died ]; then
    rep "RESULT  pass 1 died before declaring itself -- read $L1."
    exit 1
fi
if [ -z "$NEED" ] || [ -z "$HAVE" ]; then
    rep "RESULT  max_m was reduced but the memory lines were not both reported."
    rep "        Cannot compute a fraction from this; read $L1."
    exit 1
fi

# available scales 1:1 with the fraction we give back, on a card of GPU_TOTAL_GB.
TARGET="$(awk -v b="$BASE" -v need="$NEED" -v have="$HAVE" -v m="$MARGIN" -v g="$GPU_TOTAL_GB" \
          'BEGIN{ f = b - (need - have + m)/g; printf "%.2f", int(f*100)/100 }')"
rep "--- arithmetic ---"
rep "need ${NEED} - have ${HAVE} + margin ${MARGIN} = $(awk -v n="$NEED" -v h="$HAVE" -v m="$MARGIN" 'BEGIN{printf "%.2f", n-h+m}') GB to free"
rep "on a ${GPU_TOTAL_GB} GB card that is mem-fraction $BASE -> $TARGET"
rep ""

if awk -v t="$TARGET" -v fl="$FLOOR" 'BEGIN{exit !(t < fl)}'; then
    rep "RESULT  $TARGET is below the floor $FLOOR, so a ${CP}-token prefill batch"
    rep "        does not fit an H100 at TP4 even with the KV pool emptied."
    rep "        This is the same answer job 215921 gave, now with the number behind it."
    exit 0
fi

rep "--- pass 2: mem-fraction $TARGET (computed) ---"
v2="$(run_pass "$TARGET" "2-$TARGET")"
L2="$OUT/pass-2-$TARGET.log"
rep "verdict       $v2"
rep "required      $(need_gb "$L2" || echo '-') GB"
rep "available     $(have_gb "$L2" || echo '-') GB"
rep "max_m reduced $(maxm "$L2" || echo '<not reduced>')"
rep "kv tokens     $(kvtok "$L2" || echo '-')"
rep ""
if [ "$v2" = fit ]; then
    rep "RESULT  MEM_FRACTION_PREFILL=$TARGET lets a ${CP}-token prefill batch start"
    rep "        without reducing max_m. That is the setting to carry to NSCC."
else
    rep "RESULT  $TARGET was still not enough (verdict $v2). Re-run with"
    rep "        PROBE_MARGIN_GB raised, or accept that this batch does not fit."
fi
