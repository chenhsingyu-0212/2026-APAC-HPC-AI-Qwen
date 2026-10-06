#!/bin/bash
# The probe's whole value is in parsing two log lines correctly and doing one piece of
# arithmetic. Both are fed the REAL text from jobs 215921 and 215960.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

# Extract the whole parser block by its markers. Slicing it by line offsets cut a
# two-line function in half and eval'd a syntax error that the passing checks hid.
block="$(sed -n '/^# >>> parsers/,/^# <<< parsers/p' run/probe-prefill-mem-inner.sh)"
[[ -n "$block" ]] || { echo "FAIL cannot find the parser block"; exit 1; }
eval "$block" || { echo "FAIL the parser block does not even parse"; exit 1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT

# Verbatim from job 215960's prefill-30100.log: two GEMM shapes, so two pairs of
# numbers, and the binding requirement is the larger one.
cat > "$T/960.log" <<'LOG'
[TP0] Entering DeepGEMM JIT Pre-Compile session.
Available memory 9.702392578125GB is less than required memory 16.937500476837158GB for warmup, reducing max_m to 8192 to avoid out of memory
Available memory 9.780517578125GB is less than required memory 11.375000476837158GB for warmup, reducing max_m to 8192 to avoid out of memory
[TP0] max_total_num_tokens=1523487
LOG
ck "takes the LARGEST requirement across GEMM shapes" "$(need_gb "$T/960.log")" "16.94"
ck "takes the SMALLEST available"                     "$(have_gb "$T/960.log")" "9.70"
ck "reads the reduced max_m"                          "$(maxm    "$T/960.log")" "8192"
ck "reads the kv pool size"                           "$(kvtok   "$T/960.log")" "1523487"

# Verbatim from job 215921.
cat > "$T/921.log" <<'LOG'
Available memory 9.78GB is less than required memory 25.31GB for warmup, reducing max_m to 6144 to avoid out of memory
LOG
ck "215921's numbers"      "$(need_gb "$T/921.log")/$(have_gb "$T/921.log")/$(maxm "$T/921.log")" "25.31/9.78/6144"

# A log where DeepGEMM fit: no such lines at all, and the parsers must come back
# empty rather than printing 0 -- a 0 would compute a fraction of 0.85 and look fine.
cat > "$T/fit.log" <<'LOG'
[TP0] Entering DeepGEMM JIT Pre-Compile session.
The server is fired up and ready to roll!
LOG
ck "an untroubled log yields nothing, not 0" \
   "[$(need_gb "$T/fit.log")][$(have_gb "$T/fit.log")][$(maxm "$T/fit.log")]" "[][][]"

# The arithmetic, as the inner script computes it. 80 GB card.
f() { awk -v b="$1" -v need="$2" -v have="$3" -v m="$4" -v g="$5" \
      'BEGIN{ f = b - (need - have + m)/g; printf "%.2f", int(f*100)/100 }'; }
# 215960's reading: free 16.94-9.70+1.5 = 8.74 GB -> 0.85 - 0.109 = 0.74
ck "from 215960's numbers -> 0.74" "$(f 0.85 16.94 9.70 1.5 79.65)" "0.74"
# 215921's reading: free 25.31-9.78+1.5 = 17.03 GB -> 0.85 - 0.214 = 0.63, under the floor
ck "from 215921's numbers -> 0.63 (below the 0.72 floor)" "$(f 0.85 25.31 9.78 1.5 79.65)" "0.63"
ck "and 0.63 trips the floor check" \
   "$(awk -v t=0.63 -v fl=0.72 'BEGIN{exit !(t < fl)}' && echo tripped)" "tripped"
ck "0.74 does not trip it" \
   "$(awk -v t=0.74 -v fl=0.72 'BEGIN{exit !(t < fl)}' && echo tripped)" ""

# The floor itself has to be above what the weights alone need: 55.2 GB of 79.65.
ck "the floor leaves room for the 55.2 GB of weights" \
   "$(awk 'BEGIN{printf "%s", (0.72*79.65 > 55.2) ? "yes" : "no"}')" "yes"

# --- the guard that keeps a scored run honest ---
eval "$(sed -n '/^pd_assert_prefill_batch()/,/^}/p' run/pd-lib.sh)"
D="$T/cell"; mkdir -p "$D"
cat > "$D/prefill-30100.log" <<'LOG'
Available memory 9.78GB is less than required memory 25.31GB for warmup, reducing max_m to 6144 to avoid out of memory
LOG
cp "$D/prefill-30100.log" "$D/prefill-30101.log"
out="$(pd_assert_prefill_batch "$D" 12288 2>&1)"; rc=$?
ck "a reduced max_m fails the cell"            "$rc" "1"
ck "and it names both workers"                 "$(grep -c 'reduced max_m to 6144' <<<"$out")" "2"
# Once per failing worker, not once per cell: 215960's two workers reported
# different available figures (9.70 and 9.78), so each one's line is worth printing.
ck "and quotes each worker's memory line"      "$(grep -c 'required memory 25.31GB' <<<"$out")" "2"

# A reduction that still covers the request is not a failure.
cat > "$D/prefill-30100.log" <<'LOG'
Available memory 9.70GB is less than required memory 16.94GB for warmup, reducing max_m to 8192 to avoid out of memory
LOG
rm -f "$D/prefill-30101.log"
pd_assert_prefill_batch "$D" 8192 >/dev/null 2>&1
ck "a reduction down TO the request passes" "$?" "0"
pd_assert_prefill_batch "$D" "" >/dev/null 2>&1
ck "no request means no assertion"          "$?" "0"
: > "$D/prefill-30100.log"
pd_assert_prefill_batch "$D" 12288 >/dev/null 2>&1
ck "an untroubled log passes"               "$?" "0"
rm -rf "$D"; mkdir -p "$D"
pd_assert_prefill_batch "$D" 12288 >/dev/null 2>&1
ck "no prefill logs at all is not a failure" "$?" "0"

# And it has to be wired in, before the first paid measurement.
ck "the campaign calls it" \
   "$(grep -c 'pd_assert_prefill_batch "$PD_WORK" "${CHUNKED_PREFILL:-}"' run/10-campaign.sbatch)" "1"
ck "and calls it before prewarm" \
   "$(awk '/pd_assert_prefill_batch "\$PD_WORK"/{a=NR} /prewarm "\$PD_WORK"/{if(a&&NR>a){print "yes";exit}}' run/10-campaign.sbatch)" "yes"

exit $fail
