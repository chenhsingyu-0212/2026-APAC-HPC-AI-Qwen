#!/bin/bash
# The NIXL backend params are the first knob whose VALUE contains a comma, and the
# campaign splits a variant's assignments on commas -- so writing the JSON directly in a
# VARIANTS token tears it into fragments that are not KEY=VALUE and abort the cell.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

env_of() { ( set -uo pipefail
             for kv in "$@"; do export "${kv?}"; done
             eng_venv_env() { :; }
             eval "$(sed -n '/^eng_pd_env()/,/^}/p' run/engines/sglang.sh)"
             eng_pd_env ) ; }

ck "nothing set means nothing emitted" "$(env_of | grep -c NIXL)" "0"
ck "both fields assemble into one JSON value" \
   "$(env_of NIXL_NUM_THREADS=16 NIXL_SPLIT_BATCH_SIZE=256 | grep '^SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS=')" \
   'SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS={"num_threads":"16","split_batch_size":"256"}'
ck "one field alone is still valid JSON" \
   "$(env_of NIXL_NUM_THREADS=8 | grep '^SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS=')" \
   'SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS={"num_threads":"8"}'
ck "the assembled value really is parseable JSON" \
   "$(env_of NIXL_NUM_THREADS=16 NIXL_SPLIT_BATCH_SIZE=256 \
      | sed 's/^[^=]*=//' | python3 -c 'import json,sys;print(sorted(json.load(sys.stdin)))')" \
   "['num_threads', 'split_batch_size']"
ck "the staging buffer is forwarded" \
   "$(env_of SGLANG_DISAGG_STAGING_BUFFER=1 | grep -c '^SGLANG_DISAGG_STAGING_BUFFER=1$')" "1"
ck "and the knobs that were already forwarded still are" \
   "$(env_of SGLANG_MOE_CONFIG_DIR=/x SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=131072 \
      | grep -cE '^SGLANG_(MOE_CONFIG_DIR|CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB)=')" "2"
# And the knobs must be writable in a VARIANTS token, i.e. contain no comma.
ck "the knob names carry no comma" \
   "$(printf 'NIXL_NUM_THREADS=16\nNIXL_SPLIT_BATCH_SIZE=256\n' | grep -c ,)" "0"

exit $fail
