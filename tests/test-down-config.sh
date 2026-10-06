#!/bin/bash
# A _down.json whose values match the up config changes exactly one thing: TMA becomes
# available to the engine. That is only true if nothing ELSE drifts, so this checks the
# generated file against its base field by field.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }
python3 -c 'import json' 2>/dev/null || { echo "skip"; exit 0; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
B="$T/E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json"
cat > "$B" <<'JSON'
{"16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":128,"GROUP_SIZE_M":32,"num_warps":4,"num_stages":3},
 "32":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":128,"GROUP_SIZE_M":32,"num_warps":4,"num_stages":3}}
JSON
D="$T/out/${B##*/}"; D="${D%.json}_down.json"

out="$(run/make-down-config.py --base "$B" --out "$T/out" 2>&1)"
ck "it writes the _down name sglang looks up" "$([[ -e "$D" ]] && echo yes)" "yes"
ck "and keeps every batch size" \
   "$(python3 -c 'import json,sys;print(sorted(json.load(open(sys.argv[1])),key=int))' "$D")" "['16', '32']"
ck "TMA is on" \
   "$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(all(c["USE_TMA"] is True for c in d.values()))' "$D")" "True"
# The whole claim is "one variable". Verify every other field is byte-identical.
ck "nothing else changed" "$(python3 - "$B" "$D" <<'PY'
import json,sys
b=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2]))
diff=set()
for k in b:
    for f in set(b[k])|set(d[k]):
        if f=="USE_TMA": continue
        if b[k].get(f)!=d[k].get(f): diff.add(f)
print(sorted(diff) or "none")
PY
)" "none"
ck "it says what it changed" "$(grep -c 'USE_TMA=true' <<<"$out")" "1"

# The optional second arm, where more than one thing moves -- and it must say so.
out2="$(run/make-down-config.py --base "$B" --out "$T/o2" --group-size-m 1 --num-stages 2 2>&1)"
D2="$T/o2/${B##*/}"; D2="${D2%.json}_down.json"
ck "the reference shape can be requested" \
   "$(python3 -c 'import json,sys;c=json.load(open(sys.argv[1]))["32"];print(c["GROUP_SIZE_M"],c["num_stages"],c["USE_TMA"])' "$D2")" \
   "1 2 True"
ck "and all three changes are named" "$(grep -oc 'USE_TMA=true, GROUP_SIZE_M=1, num_stages=2' <<<"$out2")" "1"

# A null arm: the same file without TMA, to separate "a _down file exists" from "TMA".
run/make-down-config.py --base "$B" --out "$T/o3" --no-tma >/dev/null 2>&1
D3="$T/o3/${B##*/}"; D3="${D3%.json}_down.json"
ck "the no-tma arm omits the key entirely" \
   "$(python3 -c 'import json,sys;print(any("USE_TMA" in c for c in json.load(open(sys.argv[1])).values()))' "$D3")" "False"

# Refusals, each of which otherwise fails inside the engine.
out="$(run/make-down-config.py --base "$D" --out "$T/o4" 2>&1)"
ck "a down config as base is refused" "$(grep -c 'already a down config' <<<"$out")" "1"
echo '{"triton_version":"3.7.1","bogus":{}}' > "$T/bad.json"
out="$(run/make-down-config.py --base "$T/bad.json" --out "$T/o5" 2>&1)"
ck "non-integer keys are refused" "$(grep -c 'non-integer keys' <<<"$out")" "1"
out="$(run/make-down-config.py --base "$T/nope.json" --out "$T/o6" 2>&1)"
ck "a missing base is refused" "$(grep -c 'base config not found' <<<"$out")" "1"

exit $fail
