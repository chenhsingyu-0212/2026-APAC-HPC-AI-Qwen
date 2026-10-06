#!/usr/bin/env bash
# vLLM argv tests: the three flags the Firmus 3,176 recipe turns.
#
# The speculative config is JSON and contains spaces, so the failure mode that matters
# is it arriving as two argv elements -- which vLLM would reject in a way that looks
# like a bad draft rather than a bad quoting. Each case below inspects the built argv
# directly instead of a rendered string.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

eng_python(){ echo /usr/bin/python3; }
eng_exe(){ echo "/usr/bin/$1"; }
eng_venv_env(){ :; }
pd_worker_name(){ echo "w$1"; }
export MODEL_PATH=/m SERVED_NAME=qwen CONTEXT_LEN=4608 MEM_FRACTION=0.90
export VLLM_KV_CONNECTOR=NixlConnector
PD_ROLE=(prefill decode) PD_TP=(4 4) PD_PORT=(31100 31200)
# Source it, the way the harness does: vllm.sh defines eng_pd_env twice and only the
# later definition is the live one, which a per-function sed extraction gets wrong.
eng_exe(){ echo "$1"; }
source run/engines/vllm.sh

fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }
argv(){ ( for kv in "${@:2}"; do export "${kv?}"; done; eng_pd_worker_cmd "$1" ); }

JSON='{"model": "/d/eagle3", "num_speculative_tokens": 3}'

mapfile -t A < <(argv 0 SPEC_DRAFT_PATH=/d/eagle3 SPEC_NUM_TOKENS=3)
n=0; for x in "${A[@]}"; do [[ "$x" == "$JSON" ]] && n=$((n+1)); done
ck "speculative JSON stays one argument" "$n" "1"
ck "the flag is present once" "$(printf '%s\n' "${A[@]}" | grep -c -- '--speculative-config')" "1"

mapfile -t P < <(argv 0 MOE_RUNNER_BACKEND_PREFILL=deep_gemm MOE_RUNNER_BACKEND_DECODE=triton)
mapfile -t D < <(argv 1 MOE_RUNNER_BACKEND_PREFILL=deep_gemm MOE_RUNNER_BACKEND_DECODE=triton)
ck "prefill takes deep_gemm" "$(printf '%s\n' "${P[@]}" | grep -A1 -- '--moe-backend' | tail -1)" "deep_gemm"
ck "decode takes triton"     "$(printf '%s\n' "${D[@]}" | grep -A1 -- '--moe-backend' | tail -1)" "triton"

# The NIXL compatibility hash covers the speculative config, so one-sided is unusable.
mapfile -t P2 < <(argv 0 SPEC_DRAFT_PATH=/d/eagle3)
mapfile -t D2 < <(argv 1 SPEC_DRAFT_PATH=/d/eagle3)
ck "spec config on both roles" \
   "$(printf '%s\n' "${P2[@]}" | grep -c -- '--speculative-config')$(printf '%s\n' "${D2[@]}" | grep -c -- '--speculative-config')" "11"
ck "default K is 3" "$(printf '%s\n' "${P2[@]}" | grep -o '"num_speculative_tokens": [0-9]*')" '"num_speculative_tokens": 3'

mapfile -t C < <(argv 1 DISABLE_CHUNKED_PREFILL=1)
ck "chunked prefill can be turned off" "$(printf '%s\n' "${C[@]}" | grep -c -- '--no-enable-chunked-prefill')" "1"

mapfile -t B < <(argv 0)
ck "no spec flag when unset"    "$(printf '%s\n' "${B[@]}" | grep -c -- '--speculative-config')" "0"
ck "no moe flag when unset"     "$(printf '%s\n' "${B[@]}" | grep -c -- '--moe-backend')" "0"
ck "no chunk flag when unset"   "$(printf '%s\n' "${B[@]}" | grep -c -- '--no-enable-chunked-prefill')" "0"
ck "no empty argument"          "$(printf '%s\n' "${B[@]}" | grep -c '^$')" "0"

# The NIXL side channel is advertised to the peer, not just bound locally, so
# 0.0.0.0 is a wrong answer that looks right -- job 215686's decode worker was told
# to fetch KV from 0.0.0.0 and waited until the lease expired.
mapfile -t E < <(eng_pd_env)
ck "the side-channel host is resolved, not 0.0.0.0" \
   "$(printf '%s\n' "${E[@]}" | grep -c 'SIDE_CHANNEL_HOST=0.0.0.0')" "0"
ck "it is an expression the launcher evaluates" \
   "$(printf '%s\n' "${E[@]}" | grep -c 'SIDE_CHANNEL_HOST=\$(getent ahostsv4')" "1"
ck "and the port is still per-worker" \
   "$(printf '%s\n' "${E[@]}" | grep -c 'SIDE_CHANNEL_PORT=\$(( 5600 + port % 1000 ))')" "1"
# What the launcher will actually run has to be valid shell.
host_expr="$(printf '%s\n' "${E[@]}" | sed -n 's/^VLLM_NIXL_SIDE_CHANNEL_HOST=//p')"
ck "the expression parses" "$(bash -n <<< "export H=$host_expr" 2>&1; echo rc=$?)" "rc=0"
ck "and resolves to one IPv4 here" \
   "$(eval "echo $host_expr" 2>/dev/null | grep -cE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$')" "1"

# The tuned-config folder has to reach the worker, because that is the whole A/B.
mapfile -t E2 < <( VLLM_TUNED_CONFIG_FOLDER=/tmp/moecfg eng_pd_env )
ck "the tuned folder is forwarded when set" \
   "$(printf '%s\n' "${E2[@]}" | grep -c '^VLLM_TUNED_CONFIG_FOLDER=/tmp/moecfg$')" "1"
mapfile -t E3 < <(eng_pd_env)
ck "and absent when unset" \
   "$(printf '%s\n' "${E3[@]}" | grep -c 'VLLM_TUNED_CONFIG_FOLDER')" "0"

exit $fail
