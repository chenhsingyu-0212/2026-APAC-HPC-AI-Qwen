#!/bin/bash
# The escape hatch has to land on the right role and nowhere else: a decode flag on a
# prefill worker is a configuration nobody chose, which is the failure mode this project
# has already paid for more than once.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

# Drive eng_pd_worker_cmd with the minimum one worker of each role needs.
w() { ( set -uo pipefail
        idx="$1"; shift
        for kv in "$@"; do export "${kv?}"; done
        MODEL_PATH=/m SERVED_NAME=qwen CONTEXT_LEN=8192 MEM_FRACTION=0.85
        PD_ROLE=(prefill decode); PD_TP=(4 4); PD_PORT=(30100 30200); PD_BOOT=(9000 0)
        eng_python() { echo python3; }
        eval "$(sed -n '/^eng_mem_fraction()/,/^}/p;/^eng_pd_worker_cmd()/,/^}/p' run/engines/sglang.sh)"
        eng_pd_worker_cmd "$idx" ) ; }
pre() { w 0 "$@"; }
dec() { w 1 "$@"; }

ck "it builds a prefill worker at all"  "$(pre | grep -c -- '--disaggregation-mode')" "1"
ck "no extras means no extra flags"     "$(pre | grep -c -- '^--aa$')" "0"

ck "PD_EXTRA reaches prefill" "$(pre PD_EXTRA=--aa | grep -c -- '^--aa$')" "1"
ck "PD_EXTRA reaches decode"  "$(dec PD_EXTRA=--aa | grep -c -- '^--aa$')" "1"

ck "DECODE_EXTRA reaches decode"  "$(dec DECODE_EXTRA=--bb | grep -c -- '^--bb$')" "1"
ck "and never prefill"            "$(pre DECODE_EXTRA=--bb | grep -c -- '^--bb$')" "0"
ck "PREFILL_EXTRA reaches prefill" "$(pre PREFILL_EXTRA=--cc | grep -c -- '^--cc$')" "1"
ck "and never decode"              "$(dec PREFILL_EXTRA=--cc | grep -c -- '^--cc$')" "0"

# Unique dummy flags on purpose: --max-running-requests is already emitted by the decode
# branch, so reusing it would count twice and hide whether the split happened.
ck "a multi-flag string becomes several argv entries" \
   "$(dec 'DECODE_EXTRA=--aa --bb cc' | grep -cE -- '^(--aa|--bb|cc)$')" "3"

# An empty value must add nothing: a stray "" in argv makes sglang read the next flag as
# this one's value, which is a silent misconfiguration rather than an error.
ck "an empty value emits no argument" \
   "$(( $(pre PD_EXTRA= PREFILL_EXTRA= | wc -l) - $(pre | wc -l) ))" "0"
ck "and no blank line is produced" "$(pre PD_EXTRA= | grep -c '^$')" "0"

# PD_EXTRA first, then the role-specific one, so a cell can set a common flag and still
# override it on one side.
ck "PD_EXTRA is emitted before the role-specific one" \
   "$(dec PD_EXTRA=--aa DECODE_EXTRA=--bb | grep -nE -- '^--(aa|bb)$' | cut -d: -f2 | tr '\n' ' ')" \
   "--aa --bb "

ck "the solo path is untouched" \
   "$(awk '/^eng_solo_cmd\(\)/,/^}/' run/engines/sglang.sh | grep -c 'PD_EXTRA\|DECODE_EXTRA\|PREFILL_EXTRA')" "0"
for f in run/10-campaign.sbatch run/sweep-pd.sbatch; do
  ck "$f forwards all three" "$(grep -c 'PD_EXTRA PREFILL_EXTRA DECODE_EXTRA' "$f")" "1"
done

# The variant parser cannot carry whitespace inside an assignment, and job 216204 paid
# 2,576 SU to find that out. The campaign now refuses it before allocating anything.
guard() { ( set -uo pipefail
            eval "$(sed -n '/^_vcheck()/,/^}$/p' run/10-campaign.sbatch)"
            _vcheck "$1" && echo OK ) 2>&1 ; }
ck "a space inside an assignment is refused" \
   "$(guard 'a:PD_EXTRA=--flag value' | grep -c 'whitespace inside an assignment')" "1"
ck "and it names the --flag=value form" \
   "$(guard 'a:PD_EXTRA=--flag value' | grep -c -- '--flag=value')" "1"
ck "the equals form passes"       "$(guard 'a:PD_EXTRA=--flag=value' | tail -1)" "OK"
ck "a bare variant passes"        "$(guard 'base' | tail -1)" "OK"
ck "several assignments pass"     "$(guard 'a:X=1,Y=2+b:Z=3' | tail -1)" "OK"
ck "the real 216204 string is caught" \
   "$(guard 'symm:A=1,PD_EXTRA=--enable-torch-symm-mem+fifus:A=1,PD_EXTRA=--flashinfer-allreduce-fusion-backend trtllm+bigpf:A=1'       | grep -c 'has whitespace inside an assignment')" "1"

exit $fail
