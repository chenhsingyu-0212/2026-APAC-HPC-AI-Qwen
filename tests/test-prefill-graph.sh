#!/usr/bin/env bash
# DISABLE_PREFILL_GRAPH reaches the prefill role and nothing else.
#
# Jobs 215267 and 215459 both lost their DeepGEMM prefill cell to
# "markCaptureEnd called with no captures in progress" -- first use of a DeepGEMM
# cubin does a module load, and a module load inside a CUDA-graph capture is illegal
# whether the kernel was just compiled or came from a warm cache. The knob has to
# land on prefill workers only: a decode worker in PD never prefills, and switching
# its graphs off would cost ITL for nothing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
eng_python(){ echo python3; }; eng_venv_env(){ :; }; pd_worker_name(){ echo w$1; }
pd_worker_url(){ echo http://10.0.0.$1:30100; }
MODEL_PATH=/M SERVED_NAME=qwen CONTEXT_LEN=8192 MEM_FRACTION=0.85
PD_ROLE=(prefill decode) PD_TP=(4 4) PD_PORT=(30100 30200) PD_BOOT=(9000 -)
source run/engines/sglang.sh
f=0; ck(){ [[ "$2" == "$3" ]] && echo "ok   $1" || { echo "FAIL $1: [$2] want [$3]"; f=1; }; }
g(){ ( for kv in "${@:2}"; do export "$kv"; done; eng_pd_worker_cmd "$1" ); }
ck "預設不帶"              "$(g 0 | grep -c -- --disable-prefill-cuda-graph)" "0"
ck "開了之後 prefill 帶"    "$(g 0 DISABLE_PREFILL_GRAPH=1 | grep -c -- --disable-prefill-cuda-graph)" "1"
ck "decode 不帶"           "$(g 1 DISABLE_PREFILL_GRAPH=1 | grep -c -- --disable-prefill-cuda-graph)" "0"
ck "沒有空引數"            "$(g 0 DISABLE_PREFILL_GRAPH=1 | grep -c '^$')" "0"
ck "bootstrap port 還在"    "$(g 0 DISABLE_PREFILL_GRAPH=1 | grep -c -- --disaggregation-bootstrap-port)" "1"
exit $f
