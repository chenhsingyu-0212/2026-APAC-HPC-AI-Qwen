#!/usr/bin/env bash
# Expert parallelism must be opt-in on every engine, not defaulted to the TP degree.
#
# SGLang defaults --ep-size to the TP degree for a PD decode worker, and every PD run
# this project made before job 214830 carried it. Removing it was worth 2.27x --
# 986 -> 2,240 good tok/s. vllm.sh and trtllm.sh had the same default copied into
# them; this test exists so a fourth engine cannot quietly reintroduce it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

eng_python(){ echo python3; }; eng_exe(){ echo "$1"; }; eng_venv_env(){ :; }
pd_worker_name(){ echo "w$1"; }; pd_worker_url(){ echo "http://10.0.0.$1:30100"; }
export MODEL_PATH=/M SERVED_NAME=qwen CONTEXT_LEN=4608 MEM_FRACTION=0.90 MAX_RUNNING=256
export PD_WORK=/tmp TRTLLM_VENV=/v TRTLLM_BACKEND=pytorch TRTLLM_BACKEND_FLAG=--backend
export TRTLLM_TP_FLAG=--tp_size TRTLLM_EXTRA=""
export VLLM_KV_CONNECTOR=NixlConnector
PD_ROLE=(prefill decode) PD_TP=(4 4) PD_PORT=(30100 30200) PD_BOOT=(9000 -) PD_IP=(10.0.0.1 10.0.0.2)

for eng in sglang vllm trtllm; do
    ( source "run/engines/$eng.sh"
      # decode worker, DECODE_EP unset
      out="$(eng_pd_worker_cmd 1 | paste -sd' ')"
      n=$(grep -oE -- '--ep[-_]size|--enable-expert-parallel' <<< "$out" | wc -l)
      echo "$eng unset=$n"
      # and set
      out2="$(DECODE_EP=4 eng_pd_worker_cmd 1 | paste -sd' ')"
      n2=$(grep -oE -- '--ep[-_]size|--enable-expert-parallel' <<< "$out2" | wc -l)
      echo "$eng set=$n2" )
done > /tmp/ep-optin.$$ 2>/dev/null

for eng in sglang vllm trtllm; do
    ck "$eng: EP absent when DECODE_EP is unset" "$(grep -c "^$eng unset=0$" /tmp/ep-optin.$$)" "1"
    ck "$eng: EP present when DECODE_EP is set"  "$(grep -c "^$eng set=1$"   /tmp/ep-optin.$$)" "1"
done
rm -f /tmp/ep-optin.$$
exit $fail
