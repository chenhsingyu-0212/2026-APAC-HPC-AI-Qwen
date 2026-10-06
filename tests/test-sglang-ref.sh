#!/usr/bin/env bash
# Can our harness reproduce the organisers' reference deployment?
#
# reference-scripts/07 passes NO routing policy, so sglang_router falls back to
# cache_aware. Our default is round_robin because job 17682 measured cache_aware
# giving one prefill worker 126 requests and the other three one apiece. Measuring the
# score floor means running 07's choice, not ours -- so the flags have to be omittable,
# and an empty value has to mean "omit" rather than "pass an empty string", which
# sglang_router would reject.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

eng_python(){ echo /usr/bin/python3; }
pd_worker_url(){ echo "http://10.0.0.$1:3${1}100"; }
PD_ROLE=(prefill prefill decode decode) PD_BOOT=(9000 9001 - -) PD_PORT=(30100 30101 30200 30201)
src(){ ( for kv in "$@"; do export "${kv?}"; done
        eval "$(sed -n '/^PD_PREFILL_POLICY=/,/^PD_DECODE_POLICY=/p;/^eng_pd_gateway_cmd()/,/^}/p' run/engines/sglang.sh)"
        eng_pd_gateway_cmd 8000 ); }

fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

mapfile -t D < <(src)
ck "our default is round_robin on both" \
   "$(printf '%s\n' "${D[@]}" | grep -c -E 'round_robin')" "2"

mapfile -t R < <(src PD_PREFILL_POLICY= PD_DECODE_POLICY=)
ck "empty omits --prefill-policy" "$(printf '%s\n' "${R[@]}" | grep -c -- '--prefill-policy')" "0"
ck "empty omits --decode-policy"  "$(printf '%s\n' "${R[@]}" | grep -c -- '--decode-policy')"  "0"
ck "and leaves no empty argument"  "$(printf '%s\n' "${R[@]}" | grep -c '^$')" "0"
# -x, because `grep -c -- '--prefill'` also counts --prefill-policy and would then
# report a different number for the two cases for the wrong reason.
ck "the workers are still wired"   "$(printf '%s\n' "${R[@]}" | grep -cx -- '--prefill')" "2"

mapfile -t H < <(src PD_PREFILL_POLICY=round_robin PD_DECODE_POLICY=)
ck "one side can be omitted alone" \
   "$(printf '%s\n' "${H[@]}" | grep -c -- '--prefill-policy')$(printf '%s\n' "${H[@]}" | grep -c -- '--decode-policy')" "10"

# The Prometheus port must move with the cell, or the second cell's router cannot
# bind -- sglang_router hardcodes 29000 and the campaign's port stride misses it.
mapfile -t G0 < <(src)                      # router port 8000 (cell 0)
ck "cell 0 keeps the router's own default" \
   "$(printf '%s\n' "${G0[@]}" | grep -A1 -- '--prometheus-port' | tail -1)" "29000"
_g(){ ( for kv in "$@"; do export "${kv?}"; done
        eval "$(sed -n '/^PD_PREFILL_POLICY=/,/^PD_DECODE_POLICY=/p;/^eng_pd_gateway_cmd()/,/^}/p' run/engines/sglang.sh)"
        eng_pd_gateway_cmd 8001 ); }
mapfile -t G1 < <(_g)
ck "cell 1 shifts with it" \
   "$(printf '%s\n' "${G1[@]}" | grep -A1 -- '--prometheus-port' | tail -1)" "29001"

exit $fail
