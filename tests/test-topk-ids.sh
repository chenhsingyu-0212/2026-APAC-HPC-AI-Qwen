#!/bin/bash
# The filenames are the contract: the tuner builds them with ITS hardcoded layer counts
# and then loads them, so a name built from the model's real layer count is a file the
# tuner will never find -- and --topk-ids-dir is required, so the run dies at argparse
# time rather than silently.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

python3 -c 'import torch' 2>/dev/null || { echo "skip  torch is not importable here"; exit 0; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/model"
cat > "$T/model/config.json" <<'JSON'
{"num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 94,
 "moe_intermediate_size": 1536, "hidden_size": 4096}
JSON

out="$(run/gen-topk-ids.py --model "$T/model" --batch-size 32 --out "$T/topk" 2>&1)"
ck "it writes 100 files"        "$(ls "$T/topk" | wc -l | tr -d ' ')" "100"
ck "shape comes from the model" "$(grep -c 'shape \[32, 8\]' <<<"$out")" "1"
ck "over 128 experts"           "$(grep -c 'over 128 experts' <<<"$out")" "1"

# The exact names the tuner will ask for, derived from its own formula.
ck "first name"  "$([[ -e "$T/topk/topk_ids_layer3_idx0.pt" ]] && echo yes)" "yes"
ck "last name"   "$([[ -e "$T/topk/topk_ids_layer44_idx1.pt" ]] && echo yes)" "yes"
ck "the wrap point is layer 60 idx 0" \
   "$([[ -e "$T/topk/topk_ids_layer60_idx0.pt" ]] && echo yes)" "yes"
ck "nothing is named for the model's 94 layers" \
   "$(ls "$T/topk" | grep -c 'layer9[0-9]')" "0"

# Reproduce the tuner's loader exactly and check all 100 resolve.
ck "all 100 names the tuner builds exist" "$(python3 - "$T/topk" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1]); n_layers, dense = 61, 3; moe = n_layers - dense
missing = [i for i in range(100)
           if not (d / f"topk_ids_layer{i % moe + dense}_idx{i // moe}.pt").is_file()]
print(len(missing))
PY
)" "0"

ck "the tensors load with the right shape" "$(python3 - "$T/topk" <<'PY'
import pathlib, sys, torch
t = torch.load(pathlib.Path(sys.argv[1]) / "topk_ids_layer3_idx0.pt")
print(f"{tuple(t.shape)} {int(t.min())>=0} {int(t.max())<128}")
PY
)" "(32, 8) True True"

# A batch size of 1 and a large one both have to work, since the sweep covers both.
run/gen-topk-ids.py --model "$T/model" --batch-size 1 --out "$T/t1" >/dev/null 2>&1
ck "batch size 1 works" "$(ls "$T/t1" 2>/dev/null | wc -l | tr -d ' ')" "100"

# A config without the expert fields must be refused, not filled with a guess.
echo '{"num_hidden_layers": 94}' > "$T/model/bad.json"
mkdir -p "$T/bad" && cp "$T/model/bad.json" "$T/bad/config.json"
out="$(run/gen-topk-ids.py --model "$T/bad" --batch-size 8 --out "$T/t2" 2>&1)"; rc=$?
ck "a config without expert counts is refused" "$rc" "1"
ck "and says which fields are missing" "$(grep -c 'num_experts' <<<"$out")" "1"
out="$(run/gen-topk-ids.py --model "$T/nothing" --batch-size 8 --out "$T/t3" 2>&1)"; rc=$?
ck "a missing config.json is refused" "$rc" "1"

# Determinism: the seed is what makes a re-tune comparable.
run/gen-topk-ids.py --model "$T/model" --batch-size 8 --out "$T/a" >/dev/null 2>&1
run/gen-topk-ids.py --model "$T/model" --batch-size 8 --out "$T/b" >/dev/null 2>&1
ck "the same seed gives the same files" \
   "$(cmp -s "$T/a/topk_ids_layer3_idx0.pt" "$T/b/topk_ids_layer3_idx0.pt" && echo same)" "same"

exit $fail
