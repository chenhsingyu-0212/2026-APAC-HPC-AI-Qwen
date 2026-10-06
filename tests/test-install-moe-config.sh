#!/bin/bash
# Three separate things silently turn a tuned MoE config into a no-op, and this covers
# all three plus the merge that keeps an A/B honest:
#   * vLLM's "triton_version" key makes sglang's loader raise on int(key)
#   * SGLANG_MOE_CONFIG_DIR replaces the packaged tree instead of layering on it
#   * the filename is a lookup key, and block_shape needs the space str(list) writes
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT

# A fake packaged tree: two version dirs, one of which is the serving version.
mkdir -p "$T/pkg/configs/triton_3_3_1" "$T/pkg/configs/triton_3_7_1"
echo '{"1":{"BLOCK_SIZE_M":16}}' > "$T/pkg/configs/triton_3_3_1/E=128,N=384,other.json"
echo '{"1":{"BLOCK_SIZE_M":32}}' > "$T/pkg/configs/triton_3_7_1/E=1,N=1,keepme.json"
echo 'x' > "$T/pkg/configs/loose_file.json"
mkdir -p "$T/venv/lib/python3.10/site-packages/sglang/srt/layers/moe/moe_runner"
ln -s "$T/pkg" "$T/venv/lib/python3.10/site-packages/sglang/srt/layers/moe/moe_runner/triton_utils"

# What vllm's benchmark_moe.py actually writes: triton_version FIRST, keys as strings,
# and block_shape WITHOUT the space sglang spells it with.
CANON='E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json'
SPACELESS='E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json'
cat > "$T/$SPACELESS" <<'JSON'
{
    "triton_version": "3.5.1",
    "32": {"BLOCK_SIZE_M": 64, "num_warps": 4},
    "16": {"BLOCK_SIZE_M": 32, "num_warps": 4}
}
JSON

out="$(TRITON_VERSION=3.7.1 SGLANG_VENV="$T/missing" run/install-moe-config.sh "$T/$SPACELESS" "$T/o0" 2>&1)"
ck "a missing venv is named, not guessed" "$(grep -c 'set SGLANG_VENV' <<<"$out")" "1"

out="$(TRITON_VERSION=3.7.1 SGLANG_VENV="$T/venv" run/install-moe-config.sh "$T/$SPACELESS" "$T/out" 2>&1)"
ck "it installs both batch sizes" "$(grep -c 'installed 2 batch size' <<<"$out")" "1"

D="$T/out/configs"
ck "the serving version dir is REAL, not a symlink" \
   "$([[ -d "$D/triton_3_7_1" && ! -L "$D/triton_3_7_1" ]] && echo yes)" "yes"
ck "the OTHER version dir is mirrored, so fallback still works" \
   "$([[ -e "$D/triton_3_3_1/E=128,N=384,other.json" ]] && echo yes)" "yes"
ck "loose packaged files are mirrored too" "$([[ -e "$D/loose_file.json" ]] && echo yes)" "yes"
ck "what shipped in our version dir survives" \
   "$([[ -e "$D/triton_3_7_1/E=1,N=1,keepme.json" ]] && echo yes)" "yes"

ck "block_shape gains the space sglang writes" "$([[ -e "$D/triton_3_7_1/$CANON" ]] && echo yes)" "yes"
ck "and the spaceless name is NOT what landed" "$([[ -e "$D/triton_3_7_1/$SPACELESS" ]] && echo yes)" ""

F="$D/triton_3_7_1/$CANON"
ck "triton_version is stripped" "$(grep -c triton_version "$F")" "0"
ck "sglang's loader can read it" \
   "$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sorted({int(k):v for k,v in d.items()}))' "$F" | grep -c '^\[16, 32\]$')" "1"
ck "provenance is kept" "$(grep -c 'tuned-by-triton: 3.5.1' "$F.provenance")" "1"

# --- the filename is a lookup key, and MERGE_BASE supplies it ---
# Our tuner output came back from the tuning cluster renamed, which is the real case.
printf '%s\n' '{"1":{"BLOCK_SIZE_M":1},"16":{"BLOCK_SIZE_M":2},"32":{"BLOCK_SIZE_M":3},"64":{"BLOCK_SIZE_M":4}}' > "$T/$CANON"
printf '%s\n' '{"triton_version":"3.7.1","32":{"BLOCK_SIZE_M":99,"BLOCK_SIZE_N":64}}' > "$T/h100-tuned-bs32.json"
out="$(MERGE_BASE="$T/$CANON" TRITON_VERSION=3.7.1 SGLANG_VENV="$T/venv" \
       run/install-moe-config.sh "$T/h100-tuned-bs32.json" "$T/outm" 2>&1)"
M="$T/outm/configs/triton_3_7_1/$CANON"
ck "a renamed source lands under the LOOKUP name" "$([[ -e "$M" ]] && echo yes)" "yes"
ck "and not under its own name" \
   "$([[ -e "$T/outm/configs/triton_3_7_1/h100-tuned-bs32.json" ]] && echo yes)" ""
ck "merging keeps every base key" \
   "$(python3 -c 'import json,sys;print(sorted(json.load(open(sys.argv[1])),key=int))' "$M")" \
   "['1', '16', '32', '64']"
ck "and replaces only ours" \
   "$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["32"]["BLOCK_SIZE_M"],d["64"]["BLOCK_SIZE_M"])' "$M")" "99 4"
ck "and names the key it replaced" "$(grep -c "replaced: \['32'\]" <<<"$out")" "1"
ck "provenance records the merge base" "$(grep -c 'merge-base:' "$M.provenance")" "1"
ck "provenance has real newlines, not a literal backslash-n" \
   "$(wc -l < "$M.provenance" | tr -d ' ')" "3"

# Without a base, a source whose name is not a lookup key is a guaranteed silent
# no-op, so it is refused rather than installed.
out="$(TRITON_VERSION=3.7.1 SGLANG_VENV="$T/venv" \
       run/install-moe-config.sh "$T/h100-tuned-bs32.json" "$T/outbad" 2>&1)"
ck "a non-key source name is refused" "$(grep -c 'is not a name sglang looks up' <<<"$out")" "1"

# A bad base is reported as a bad base, not as a filename problem: the base is
# validated first on purpose.
BB="$T/E=1,N=1,device_name=B,block_shape=[1,1].json"
printf '%s\n' '{"triton_version":"x","bogus":{}}' > "$BB"
out="$(MERGE_BASE="$BB" TRITON_VERSION=3.7.1 SGLANG_VENV="$T/venv" \
       run/install-moe-config.sh "$T/h100-tuned-bs32.json" "$T/outbb" 2>&1)"
ck "a base with non-integer keys is refused as such" \
   "$(grep -c 'MERGE_BASE has non-integer keys' <<<"$out")" "1"
out="$(MERGE_BASE="$T/nope" TRITON_VERSION=3.7.1 SGLANG_VENV="$T/venv" \
       run/install-moe-config.sh "$T/h100-tuned-bs32.json" "$T/outnb" 2>&1)"
ck "a missing base is refused as such" "$(grep -c 'MERGE_BASE is not a file' <<<"$out")" "1"

# And the tuned config itself: a stray non-integer key, or none at all, fails inside
# the engine twelve minutes into a bring-up rather than here.
printf '%s\n' '{"triton_version":"3.5.1","bogus":{"x":1}}' > "$T/$CANON.bad"
mv "$T/$CANON.bad" "$T/E=2,N=2,device_name=C,block_shape=[1,1].json"
out="$(TRITON_VERSION=3.7.1 SGLANG_VENV="$T/venv" \
       run/install-moe-config.sh "$T/E=2,N=2,device_name=C,block_shape=[1,1].json" "$T/o2" 2>&1)"
ck "a non-integer key in the tuned config is refused" "$(grep -c 'non-integer keys' <<<"$out")" "1"
printf '%s\n' '{"triton_version":"3.5.1"}' > "$T/E=3,N=3,device_name=C,block_shape=[1,1].json"
out="$(TRITON_VERSION=3.7.1 SGLANG_VENV="$T/venv" \
       run/install-moe-config.sh "$T/E=3,N=3,device_name=C,block_shape=[1,1].json" "$T/o3" 2>&1)"
ck "an empty tuned config is refused" "$(grep -c 'no batch-size entries' <<<"$out")" "1"

exit $fail
