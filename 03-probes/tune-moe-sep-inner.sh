# Appended to a generated launcher by run/22-tune-moe.sbatch when TUNE_MODE=sep, after
# the environment exports and the GPU guard. Not executable on its own.
#
# tuning_fused_moe_triton_sep.py differs from the up-only tuner in three ways that all
# have to be handled here rather than in the argv:
#   * --batch-size takes ONE int, so the sweep is a loop
#   * --topk-ids-dir is REQUIRED and wants 100 tensors of shape [batch_size, topk], so
#     they are generated per batch size
#   * there is no --save-dir: it writes next to the current directory, so each
#     invocation runs inside the output directory
PY="$1"; GEN="$2"; TUNER="$3"; MODEL="$4"; OUT="$5"; TP="$6"; DTYPE="$7"; SIZES="$8"
REPORT="$OUT/sep-report.txt"
: > "$REPORT"
rep() { echo "$*" | tee -a "$REPORT"; }

# `from common_utils import (...)` resolves against the TUNER's directory, but the tuner
# is vendored under run/ while common_utils.py lives in its own directory so a name that
# generic cannot collide. Put it on the path explicitly.
DEPS="$(dirname "$TUNER")/moe-tuner-deps"
[ -r "$DEPS/common_utils.py" ] || { echo "x no $DEPS/common_utils.py" >&2; exit 1; }
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT/configs" "$OUT/topk" || exit 1
rep "gpu      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
rep "tuner    $(basename "$TUNER")"
rep "tp $TP  dtype $DTYPE  batch sizes: $SIZES"
rep ""

rc_all=0
for m in $SIZES; do
    rep "--- batch size $m ---"
    td="$OUT/topk/bs$m"
    if ! "$PY" "$GEN" --model "$MODEL" --batch-size "$m" --out "$td" >>"$REPORT" 2>&1; then
        rep "x could not generate topk_ids for batch size $m"; rc_all=1; continue
    fi
    # No --save-dir: run where the files should land.
    ( cd "$OUT/configs" && "$PY" "$TUNER" \
        --model "$MODEL" --tp-size "$TP" --dtype "$DTYPE" \
        --batch-size "$m" --tune --topk-ids-dir "$td" ) >>"$OUT/sep-bs$m.log" 2>&1
    rc=$?
    rep "   exit $rc"
    # The tuner is the only thing that can name these files correctly, so report what
    # it actually produced rather than what we hoped for.
    while IFS= read -r f; do rep "   produced $(basename "$f")"; done \
        < <(find "$OUT/configs" -name 'E=*.json' -newer "$td" 2>/dev/null | sort)
    (( rc == 0 )) || rc_all=1
    # 100 tensors per batch size add up; the configs are what we keep.
    rm -rf "$td"
done

rep ""
up=$(find "$OUT/configs" -name 'E=*.json' ! -name '*_down.json' | wc -l | tr -d ' ')
dn=$(find "$OUT/configs" -name 'E=*_down.json' | wc -l | tr -d ' ')
rep "RESULT   $up up config(s), $dn down config(s)"
if (( dn == 0 )); then
    rep "         NO _down file. That is the up-only case, which measured worse than no"
    rep "         tuned config at all (job 216159). Do not install this."
    rc_all=1
else
    rep "         Both GEMMs covered. Install with run/install-moe-config.sh and A/B it."
fi
exit $rc_all
