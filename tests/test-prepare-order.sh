#!/usr/bin/env bash
# pd_resolve must run before eng_pd_prepare, in every script that drives a PD cell.
#
# TensorRT-LLM's prepare step writes the disagg gateway's config, which lists the
# workers by address, so it reads PD_IP -- and PD_IP is filled by pd_resolve. SGLang's
# and vLLM's prepare are no-ops, so the order was wrong in both drivers for weeks and
# nothing noticed until job 215874 died eighteen seconds in on
# "PD_IP[$i]: unbound variable".
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

for f in run/10-campaign.sbatch run/sweep-pd.sbatch; do
    # The PD branch only: the solo path resolves separately and has no placement.
    r=$(grep -n '^\s*pd_resolve ||' "$f" | tail -1 | cut -d: -f1)
    p=$(grep -n 'eng_pd_prepare "\$PD_WORK"' "$f" | tail -1 | cut -d: -f1)
    ck "$(basename "$f"): both calls present" "$( [[ -n "$r" && -n "$p" ]] && echo yes || echo no )" "yes"
    ck "$(basename "$f"): resolve precedes prepare" "$( (( r < p )) && echo yes || echo no )" "yes"
done

# And the engine says which precondition failed, rather than letting set -u name a
# variable the reader has to go looking for.
eng_python(){ echo python3; }; eng_venv_env(){ :; }
export MODEL_PATH=/M CONTEXT_LEN=4608 MEM_FRACTION=0.90 PD_WORK=/tmp/po.$$
export TRTLLM_VENV=/v TRTLLM_BACKEND=pytorch TRTLLM_BACKEND_FLAG=--backend TRTLLM_TP_FLAG=--tp_size
PD_ROLE=(prefill decode) PD_TP=(4 4) PD_PORT=(30100 30200) PD_NODE=(a b) PD_IP=()
out="$( source run/engines/trtllm.sh; eng_pd_prepare "$PD_WORK" 2>&1 )"; rc=$?
rm -rf "$PD_WORK"
ck "trtllm names the precondition" "$(grep -c 'call pd_resolve before this' <<< "$out")" "1"
ck "and fails rather than continuing" "$rc" "1"
exit $fail
