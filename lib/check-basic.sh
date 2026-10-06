#!/bin/bash
# Is this PD deployment "Basic" as the 2026-09-25 addendum defines it?
#
#   run/check-basic.sh <results dir>
#
# The rule (Qwen3 Task Rules, Q1):
#
#   In Basic, every request is prefilled on a prefill worker and decoded on a decode
#   worker. Sending a request's prefill to a decode worker (for example when prefill
#   workers are full) is colocated serving for that request. It is not Basic.
#
# So the question is not "did we configure PD" -- it is whether any request was ever
# prefilled somewhere it should not have been. That is answered by the engine's own
# logs rather than by reading the router: a worker that runs a prefill logs a
# "Prefill batch" line, and a decode worker should have none.
#
# Worth running on every PD result, because the condition the rule names DOES occur
# here: job 214750 reached a 19-deep prefill queue and job 214760 a TTFT p90 of
# 4795 ms. Both are exactly "prefill workers are full", and in both the decode
# workers stayed at zero prefills -- SGLang queues rather than reassigning. That is
# evidence for the result package, not just reassurance.
set -uo pipefail
ROOT="${1:-}"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "usage: ${0##*/} <results dir>" >&2; exit 2; }

fail=0 cells=0
printf "  %-40s %-18s %8s %8s\n" cell worker Prefill Decode
while IFS= read -r d; do
    cell="$(basename "$(dirname "$d")")"
    seen=0
    for f in "$d"/prefill-*.log "$d"/decode-*.log "$d"/solo*.log; do
        [[ -f "$f" ]] || continue
        pf=$(grep -c 'Prefill batch' "$f" 2>/dev/null); pf=${pf:-0}
        dc=$(grep -c 'Decode batch'  "$f" 2>/dev/null); dc=${dc:-0}
        (( pf == 0 && dc == 0 )) && continue     # this cell never came up
        seen=1
        w="$(basename "$f" .log)"
        printf "  %-40s %-18s %8d %8d\n" "$cell" "$w" "$pf" "$dc"
        if [[ "$w" == decode-* ]] && (( pf > 0 )); then
            echo "    x $w ran $pf prefill batch(es) -- that request was served colocated," >&2
            echo "      which the addendum says is not Basic." >&2
            fail=1
        fi
    done
    (( seen )) && cells=$(( cells + 1 ))
done < <(find "$ROOT" -type d -name deploy 2>/dev/null | sort)

echo
if (( cells == 0 )); then
    echo "  ! no cell in $ROOT produced engine logs -- nothing to check"
    exit 0
fi
if (( fail )); then
    echo "  x NOT Basic: at least one prefill ran on a decode worker."
    exit 1
fi
echo "  ok Basic: across $cells cell(s), every prefill ran on a prefill worker."
echo "     Prefill and decode are strictly disjoint in the engine's own accounting."
