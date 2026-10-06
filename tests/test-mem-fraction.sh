#!/bin/bash
# A per-role mem-fraction is the lever that unblocks a bigger prefill batch, so the
# role split has to be exact: a value on one role must not reach the other, and the
# global must still apply where no per-role value is set.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

mf() { ( set -uo pipefail
         for kv in "$@"; do export "${kv?}"; done
         eval "$(sed -n '/^eng_mem_fraction()/,/^}/p' run/engines/sglang.sh)"
         echo "prefill=$(eng_mem_fraction prefill) decode=$(eng_mem_fraction decode)" ); }

ck "with only the global, both roles get it" \
   "$(mf MEM_FRACTION=0.85)" "prefill=0.85 decode=0.85"
ck "a prefill value does not reach decode" \
   "$(mf MEM_FRACTION=0.85 MEM_FRACTION_PREFILL=0.76)" "prefill=0.76 decode=0.85"
ck "a decode value does not reach prefill" \
   "$(mf MEM_FRACTION=0.85 MEM_FRACTION_DECODE=0.90)" "prefill=0.85 decode=0.90"
ck "both can be set at once" \
   "$(mf MEM_FRACTION=0.85 MEM_FRACTION_PREFILL=0.76 MEM_FRACTION_DECODE=0.90)" \
   "prefill=0.76 decode=0.90"
ck "an EMPTY per-role value falls back, it does not emit nothing" \
   "$(mf MEM_FRACTION=0.85 MEM_FRACTION_PREFILL=)" "prefill=0.85 decode=0.85"
# An empty --mem-fraction-static would be a stray argv entry; sglang would read the
# next flag as its value. The helper must never print an empty line.
ck "never empty" "$(mf MEM_FRACTION=0.85 MEM_FRACTION_PREFILL= MEM_FRACTION_DECODE= | grep -c '=[^ ]')" "1"

# And the knob has to survive the trip through the campaign's export list, or it is
# set in the batch script and absent in the cell -- the way MAX_PREFILL_TOKENS was.
for f in run/10-campaign.sbatch run/sweep-pd.sbatch; do
  ck "$f forwards MEM_FRACTION_PREFILL" \
     "$(grep -c 'MEM_FRACTION_PREFILL' "$f")" "1"
  ck "$f forwards MEM_FRACTION_DECODE" \
     "$(grep -c 'MEM_FRACTION_DECODE' "$f")" "1"
done

# The worker command must actually call the helper rather than the raw global.
ck "the PD worker cmd uses the helper" \
   "$(grep -c 'mem-fraction-static "$(eng_mem_fraction "$role")"' run/engines/sglang.sh)" "1"

exit $fail
