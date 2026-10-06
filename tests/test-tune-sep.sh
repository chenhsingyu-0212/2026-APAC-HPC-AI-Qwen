#!/bin/bash
# TUNE_MODE=sep is what makes the difference between tuning both GEMMs and reproducing
# the control arm of somebody else's A/B, so the wiring is checked here: the right
# tuner, the preconditions, and that the launcher carries the sep loop rather than
# `exec "$@"`.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }
S=run/22-tune-moe.sbatch

ck "the job script parses"      "$(bash -n $S 2>&1; echo rc=$?)" "rc=0"
ck "the sep inner script parses" "$(bash -n run/tune-moe-sep-inner.sh 2>&1; echo rc=$?)" "rc=0"
ck "the sep tuner is vendored"  "$([[ -r run/tuning_fused_moe_triton_sep-v0.5.20.py ]] && echo yes)" "yes"
ck "and it is the one with _down and TMA" \
   "$(grep -c 'USE_TMA' run/tuning_fused_moe_triton_sep-v0.5.20.py | awk '{print ($1>0)?"yes":"no"}')" "yes"
ck "the generator is there"     "$([[ -x run/gen-topk-ids.py ]] && echo yes)" "yes"

# Selector: sep picks the _sep tuner, up keeps the old one.
sel() { ( set -uo pipefail; ENGINE=sglang; TUNE_MODE="$1"; HERE=run
          eval "$(sed -n '/^TUNE_MODE=/,/^esac$/p' '"'"$S"'"' 2>/dev/null)" 2>/dev/null
          echo "${TUNER:-unset}" ) ; }
ck "sep selects the separate-GEMM tuner" \
   "$(grep -c 'tuning_fused_moe_triton_sep-v0.5.20.py' $S)" "1"
ck "and up is still the default" "$(grep -c 'TUNE_MODE:-up' $S)" "1"

# Preconditions, each of which otherwise fails inside the allocation.
ck "sep refuses the vllm engine"       "$(grep -c "TUNE_MODE=sep is SGLang's tuner" $S)" "1"
ck "sep requires the generator"        "$(grep -c 'needs run/gen-topk-ids.py' $S)" "1"
ck "sep requires explicit batch sizes" "$(grep -c 'takes one --batch-size per invocation' $S)" "1"

# The launcher must carry the loop in sep mode, not exec the argv.
ck "sep appends the inner loop" "$(grep -c 'cat "\$HERE/tune-moe-sep-inner.sh"' $S)" "1"
ck "and up still execs the argv" "$(grep -c "echo 'exec" $S)" "1"
ck "sep passes the generator and tuner as arguments" \
   "$(grep -c 'bash "\$OUT/tune.sh" "\$(eng_python)" "\$HERE/gen-topk-ids.py" "\$TUNER"' $S)" "1"
ck "sep does NOT pass --save-dir, which the tuner has no option for" \
   "$(awk '/TUNE_MODE" == sep/,/^else$/' $S | grep -c -- '--save-dir')" "0"
ck "sep does NOT pass --trust-remote-code either" \
   "$(awk '/TUNE_MODE" == sep/,/^else$/' $S | grep -c -- '--trust-remote-code')" "0"

# The inner script's own contract.
I=run/tune-moe-sep-inner.sh
ck "it runs the tuner inside the output dir" "$(grep -c 'cd "\$OUT/configs"' $I)" "1"
ck "it generates topk ids per batch size"    "$(grep -c 'td="\$OUT/topk/bs\$m"' $I)" "1"
ck "it cleans the tensors up afterwards"     "$(grep -c 'rm -rf "\$td"' $I)" "1"
# The whole point is the _down file; a run that produces none must fail, because
# installing an up-only config is what measured 1,317 ITL failures.
ck "no _down file fails the job" "$(grep -c 'NO _down file' $I)" "1"
ck "and it counts both kinds"    "$(grep -c "name 'E=\*_down.json'" $I)" "1"

ck "the closing report includes the sep report" "$(grep -c 'sep-report.txt' $S)" "1"

# The import preconditions, each of which cost a compute node to discover: job 99896
# died on `import ray`, 99897 on `common_utils`. The check is now the whole chain.
ck "sep requires the vendored common_utils" \
   "$(grep -c '\[\[ -r "\$HERE/moe-tuner-deps/common_utils.py" \]\]' $S)" "1"
ck "sep walks the tuner's whole import chain" "$(grep -c 'spec.loader.exec_module' $S)" "1"
ck "with the deps dir on PYTHONPATH"        "$(grep -c 'PYTHONPATH="\$HERE/moe-tuner-deps' $S)" "1"
ck "and it uses the engine's python"        "$(grep -c '"\$(eng_python)" - "\$TUNER"' $S)" "1"
ck "ray is still named as the likely cause" "$(grep -c "pip install 'ray==2.53.0'" $S)" "1"
ck "the inner script also sets PYTHONPATH"  "$(grep -c 'export PYTHONPATH="\$DEPS' run/tune-moe-sep-inner.sh)" "1"
ck "and refuses a missing common_utils"     "$(grep -c 'no \$DEPS/common_utils.py' run/tune-moe-sep-inner.sh)" "1"

# The check has to run in the environment the launcher builds, or it tests something
# else: deep_gemm asserts on CUDA_HOME at import and sglang pulls it in transitively.
ck "the import check carries eng_pd_env" "$(grep -c 'mapfile -t _chk_env < <(eng_pd_env' $S)" "1"
ck "and drops the \$port line like the launcher does" \
   "$(grep -c "_chk_env.*grep -vw 'port'" $S)" "1"
ck "it runs the import under that env"   "$(grep -c 'env "\${_chk_env\[@\]}"' $S)" "1"
ck "and reports more than the last line" "$(grep -c 'splitlines()\[-6:\]' $S)" "1"

exit $fail
