#!/usr/bin/env bash
# The knobs taken from the 3,591 H200 track: a bigger prefill batch and the three
# environment variables SGLang reads instead of argv.
#
# Each of these aims at prefill capacity or decode ITL, which are the two numbers that
# decide our score, and each must be absent unless asked for -- a scored run has to be
# reproducible from its flags, so a knob that leaks in by default is a defect.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
eng_python(){ echo python3; }; eng_venv_env(){ echo "CUDA_HOME=/x"; }
pd_worker_name(){ echo "w$1"; }; pd_worker_url(){ echo "http://10.0.0.$1:30100"; }
export MODEL_PATH=/M SERVED_NAME=qwen CONTEXT_LEN=8192 MEM_FRACTION=0.88 MAX_RUNNING=256
PD_ROLE=(prefill decode) PD_TP=(4 4) PD_PORT=(30100 30200) PD_BOOT=(9000 -)
source run/engines/sglang.sh
fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }
argv(){ ( for kv in "${@:2}"; do export "${kv?}"; done; eng_pd_worker_cmd "$1" ); }
envv(){ ( for kv in "$@"; do export "${kv?}"; done; eng_pd_env ); }

ck "max-prefill-tokens absent by default" \
   "$(argv 0 | grep -c -- '--max-prefill-tokens')" "0"
ck "set, it reaches prefill" \
   "$(argv 0 MAX_PREFILL_TOKENS=12288 | grep -A1 -- '--max-prefill-tokens' | tail -1)" "12288"
ck "and NOT decode -- a decoder has no prefill batch" \
   "$(argv 1 MAX_PREFILL_TOKENS=12288 | grep -c -- '--max-prefill-tokens')" "0"
ck "chunked-prefill-size still reaches both" \
   "$(argv 0 CHUNKED_PREFILL=12288 | grep -c -- '--chunked-prefill-size')$(argv 1 CHUNKED_PREFILL=12288 | grep -c -- '--chunked-prefill-size')" "11"

ck "no SGLANG_ knob leaks when unset" \
   "$(envv | grep -c '^SGLANG_')" "0"
ck "the MoE config dir is forwarded" \
   "$(envv SGLANG_MOE_CONFIG_DIR=/m | grep -c '^SGLANG_MOE_CONFIG_DIR=/m$')" "1"
ck "the all-reduce ceiling is forwarded" \
   "$(envv SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=131072 | grep -c '^SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=131072$')" "1"
ck "the disaggregation timeout is forwarded" \
   "$(envv SGLANG_DISAGGREGATION_WAITING_TIMEOUT=20 | grep -c '^SGLANG_DISAGGREGATION_WAITING_TIMEOUT=20$')" "1"
ck "all three together, and the venv line survives" \
   "$(envv SGLANG_MOE_CONFIG_DIR=/m SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=1 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=2 | grep -cE '^(SGLANG_|CUDA_HOME)')" "4"
ck "no empty line is emitted" "$(envv | grep -c '^$')" "0"
exit $fail
