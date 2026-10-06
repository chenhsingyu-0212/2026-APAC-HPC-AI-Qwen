#!/bin/bash
# The warm-up is the only thing that touches the ten seconds where every one of our
# failures lives, so its shape is checked here: a burst wide enough to open every
# prefill-decode pair, then a sequential tail, and a verdict that still reads both.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

# --- client_prewarm_args: request count and concurrency are separate now ---
args() { ( set -uo pipefail
           CLIENT_DIALECT=new; ISL=4000
           client_has() { return 1; }
           eval "$(sed -n '/^client_prewarm_args()/,/^}/p' run/client-lib.sh)"
           client_prewarm_args /tmp/d host:8000 qwen "$@" ) ; }

ck "one argument keeps the old behaviour" \
   "$(args 4 | grep -A1 -- '--concurrency' | tail -1)" "4"
ck "a second argument sets concurrency only" \
   "$(args 8 1 | grep -A1 -- '--concurrency' | tail -1)" "1"
ck "and the request count is still the first" \
   "$(args 8 1 | grep -A1 -- '--request-count' | tail -1)" "8"
ck "a 32-wide burst asks for 32 of each" \
   "$(args 32 32 | grep -A1 -E -- '--(concurrency|request-count)' | grep -c '^32$')" "2"
ck "the artifact dir is still passed through" \
   "$(args 4 | grep -c '/tmp/d')" "1"
ck "no empty argument is emitted" "$(args 8 1 | grep -c '^$')" "0"

# --- the campaign's two phases ---
S=run/10-campaign.sbatch
ck "the campaign parses"            "$(bash -n $S 2>&1; echo rc=$?)" "rc=0"
ck "the burst defaults to 32"       "$(grep -c 'PREWARM_CONCURRENT:-32' $S)" "1"
ck "the sequential tail defaults to 8" "$(grep -c 'PREWARM_SEQUENTIAL:-8' $S)" "1"
ck "the burst phase runs at its own width" \
   "$(grep -c 'prewarm_phase "\$dir" "\$gw" "\$mid" "\$burst" "\$burst" burst' $S)" "1"
ck "the tail runs at concurrency 1" \
   "$(grep -c 'prewarm_phase "\$dir" "\$gw" "\$mid" "\$PREWARM_SEQUENTIAL" 1 seq' $S)" "1"
ck "the tail is skipped when the burst failed" \
   "$(grep -c 'rc == 0 && PREWARM_SEQUENTIAL > 0' $S)" "1"
ck "the tail can be turned off"  "$(grep -c 'PREWARM_SEQUENTIAL > 0' $S)" "1"
# The old caller passes nworkers*2; the burst must never be narrower than that.
ck "the burst is never narrower than the caller asked" \
   "$(grep -c '(( n > burst )) && burst="\$n"' $S)" "1"
# The verdict grep that stopped job 215492 from sweeping a dead cell has to keep working.
ck "both phase logs feed the verdict" \
   "$(grep -c 'cat "\$dir"/prewarm-\*.log > "\$dir/prewarm.log"' $S)" "1"
ck "the no-successful-response guard is still there" \
   "$(grep -c 'No successful responses were collected' $S)" "1"
ck "and it still fails the cell" \
   "$(awk '/No successful responses were collected/,/return 1/' $S | grep -c 'return 1')" "1"

# A variant must be able to set the warm-up per cell, or the A/B runs the same warm-up
# twice. The assignments are exported inside the cell subshell AFTER the top-level
# defaults are assigned, so the override only works because prewarm() reads them at call
# time -- which is exactly the shape of bug that lost MAX_PREFILL_TOKENS once.
ck "the defaults are assigned with :-, so a variant can override them" \
   "$(grep -cE 'PREWARM_(CONCURRENT|SEQUENTIAL)="\$\{PREWARM_(CONCURRENT|SEQUENTIAL):-[0-9]+\}"' $S)" "2"
ck "prewarm reads them at call time, not at parse time" \
   "$(awk '/^prewarm\(\)/,/^}/' $S | grep -c 'PREWARM_CONCURRENT\|PREWARM_SEQUENTIAL')" "3"
# Reproduce the subshell: top-level default, then the variant's export, then read.
over() { ( set -uo pipefail
           PREWARM_CONCURRENT="${PREWARM_CONCURRENT:-32}"
           PREWARM_SEQUENTIAL="${PREWARM_SEQUENTIAL:-8}"
           ( export PREWARM_CONCURRENT=4 PREWARM_SEQUENTIAL=0
             n=4; burst="$PREWARM_CONCURRENT"; (( n > burst )) && burst="$n"
             echo "burst=$burst seq=$PREWARM_SEQUENTIAL" ) ) ; }
ck "a variant setting 4/0 reproduces the OLD warm-up exactly" "$(over)" "burst=4 seq=0"
dflt() { ( set -uo pipefail
           PREWARM_CONCURRENT="${PREWARM_CONCURRENT:-32}"; PREWARM_SEQUENTIAL="${PREWARM_SEQUENTIAL:-8}"
           n=4; burst="$PREWARM_CONCURRENT"; (( n > burst )) && burst="$n"
           echo "burst=$burst seq=$PREWARM_SEQUENTIAL" ) ; }
ck "and the default is the new one"  "$(dflt)" "burst=32 seq=8"

exit $fail
