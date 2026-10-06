#!/usr/bin/env bash
# Does this run satisfy the Basic tier? Reads the engines' own accounting.
#
# The rules require prefill and decode to be deployed disaggregated and every
# prefill to run on a prefill worker. SGLang logs one line per batch naming which
# kind it was, so the claim is checkable from the logs rather than from the flags:
# a decode worker that logs even one "Prefill batch" was doing prefill work.
#
#   ./verify-basic.sh runs/215675
set -uo pipefail
D="${1:-}"
[[ -d "$D" ]] || { echo "usage: $0 <rundir>" >&2; exit 2; }

shopt -s nullglob
pre=( "$D"/prefill-*.log ); dec=( "$D"/decode-*.log )
(( ${#pre[@]} && ${#dec[@]} )) || { echo "x no prefill-*.log / decode-*.log in $D" >&2; exit 2; }

printf '  %-18s %10s %10s\n' worker prefill decode
bad=0; tot_p=0; tot_d=0
for f in "${pre[@]}" "${dec[@]}"; do
    # grep -c prints 0 AND exits 1 when it finds nothing, so `|| echo 0` would
    # append a second zero and every later arithmetic test would blow up on "0\n0".
    p=$(grep -c 'Prefill batch' "$f" 2>/dev/null) || true
    d=$(grep -c 'Decode batch'  "$f" 2>/dev/null) || true
    printf '  %-18s %10d %10d\n' "$(basename "${f%.log}")" "$p" "$d"
    case "$(basename "$f")" in
        decode-*)  (( p > 0 )) && bad=1; tot_d=$(( tot_d + d )) ;;
        prefill-*) (( d > 0 )) && bad=1; tot_p=$(( tot_p + p )) ;;
    esac
done
echo
if (( bad )); then
    echo "  x NOT Basic: a worker did work belonging to the other role."
    exit 1
fi
(( tot_p > 0 && tot_d > 0 )) || { echo "  x inconclusive: one role logged no batches at all."; exit 1; }
echo "  ok Basic: every prefill ran on a prefill worker ($tot_p prefill, $tot_d decode batches)."

# The flags that must NOT be there, and the ones that must.
echo
for f in "${pre[@]}" "${dec[@]}"; do
    line=$(grep -m1 -oE "exec: .*" "$f" 2>/dev/null) || continue
    n=$(basename "${f%.log}")
    case "$line" in
        *--ep-size*)   echo "  ! $n has --ep-size: expert parallelism is on"; bad=1 ;;
    esac
    case "$line" in
        *--disaggregation-mode*) ;;
        *) echo "  ! $n is not disaggregated"; bad=1 ;;
    esac
done
for f in "${pre[@]}" "${dec[@]}"; do
    s=$(grep -m1 -oE "'speculative_algorithm': [^,]*" "$f" 2>/dev/null)
    [[ -z "$s" || "$s" == *None* ]] || { echo "  ! $(basename "${f%.log}") $s -- that is Bonus, not Basic"; bad=1; }
done
(( bad )) || echo "  ok no expert parallelism, no speculative decoding, both roles disaggregated."
exit $bad
