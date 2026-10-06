#!/usr/bin/env bash
# What can be checked about qwen3-pd-h100-16gpu.pbs without an allocation.
#
# It is 300 lines that run once, on 16 GPUs, after a 10-minute model load. Most of
# what can go wrong is shell-level and costs nothing to catch here: the embedded
# worker script's syntax, the GPU-id translation, and whether the AIPerf block
# still matches the command the measured run actually issued.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
S=qwen3-pd-h100-16gpu.pbs
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

ck "the job script parses" "$(bash -n $S 2>&1; echo rc=$?)" "rc=0"

# 1. the worker heredoc is a script in its own right and is never syntax-checked
#    by the outer parse, because to bash it is just text.
awk "/^cat > \"\\\$RUNDIR\/worker.sh\" <<'WORKER'$/{f=1;next} /^WORKER$/{f=0} f" $S > "$TMP/worker.sh"
ck "the worker heredoc was found" "$( [[ -s $TMP/worker.sh ]] && echo yes || echo no )" "yes"
ck "the worker script parses" "$(bash -n "$TMP/worker.sh" 2>&1; echo rc=$?)" "rc=0"
ck "the worker builds a prefill and a decode branch" \
   "$(grep -c -- '--moe-runner-backend deep_gemm\|--max-running-requests' "$TMP/worker.sh")" "2"
ck "prefill graphs are off on prefill only" \
   "$(awk '/if \[ "\$role" = prefill \]/,/^else$/' "$TMP/worker.sh" | grep -c -- '--disable-prefill-cuda-graph')" "1"
# Comments mention --ep-size on purpose (that is the finding); what must not exist
# is a line that appends it to the command.
ck "no branch appends --ep-size" \
   "$(grep -c 'cmd+=.*--ep-size' "$TMP/worker.sh")" "0"

# 2. the GPU-id translation, against a stub driver. The whole point is that a UUID
#    becomes a POSITION, so a node whose cgroup renumbers must still start at 0.
awk "/^gpu_indices\(\)/,/^}$/" $S | sed -n "/rsh \"\\\$1\" '/,/^done'$/p" \
  | sed "1s/.*rsh \"\$1\" '//" | sed "\$s/'\$//" > "$TMP/snip.sh"
ck "the translation snippet was found" "$( [[ -s $TMP/snip.sh ]] && echo yes || echo no )" "yes"
mkdir -p "$TMP/bin"
smi(){ { echo '#!/bin/sh'; echo 'cat <<X'; printf '%s\n' "$@"; echo X; } > "$TMP/bin/nvidia-smi"; chmod +x "$TMP/bin/nvidia-smi"; }
run(){ ( export PATH="$TMP/bin:$PATH" CUDA_VISIBLE_DEVICES="$1"; bash "$TMP/snip.sh" ) | paste -sd,; }
smi GPU-a GPU-b GPU-c GPU-d GPU-e GPU-f GPU-g GPU-h
ck "a UUID list becomes positions"      "$(run 'GPU-a,GPU-b,GPU-c,GPU-d')" "0,1,2,3"
ck "the second half too"                "$(run 'GPU-e,GPU-f,GPU-g,GPU-h')" "4,5,6,7"
ck "order is preserved, not sorted"     "$(run 'GPU-d,GPU-a')"             "3,0"
ck "integers pass through"              "$(run '2,3')"                     "2,3"
ck "an unknown id passes through"       "$(run 'GPU-zz')"                  "GPU-zz"
smi GPU-p GPU-q GPU-r GPU-s
ck "a cgroup-restricted node starts at 0" "$(run 'GPU-r,GPU-s')"           "2,3"
ck "an empty variable lists every GPU"    "$(run '')"                      "0,1,2,3"

# 3. the official command. Every flag of the measured run (job 215675) must still
#    be here, because this is the part the rules lock.
for f in --endpoint-type --custom-endpoint --isl --osl --tokenizer --streaming \
         --warmup-duration --benchmark-duration --benchmark-grace-period \
         --use-legacy-max-tokens --num-dataset-entries --random-seed --goodput \
         --concurrency --workers-max; do
    grep -q -- "$f" $S || { echo "FAIL the AIPerf block lost $f"; fail=1; }
done
ck "warm-up is 60"            "$(grep -c -- '--warmup-duration 60' $S)"        "1"
ck "benchmark is 120"         "$(grep -c -- '--benchmark-duration 120' $S)"    "1"
ck "grace is 30"              "$(grep -c -- '--benchmark-grace-period 30' $S)" "1"
ck "the dataset is 12800"     "$(grep -c -- '--num-dataset-entries 12800' $S)" "1"
ck "the seed is 2026"         "$(grep -c -- '--random-seed 2026' $S)"          "1"
ck "the gate is 3000/20"      "$(grep -c 'time_to_first_token:3000 inter_token_latency:20' $S)" "1"
ck "min_tokens is 200"        "$(grep -c -- '--extra-inputs min_tokens:200' $S)" "1"
ck "greedy"                   "$(grep -c -- '--extra-inputs temperature:0.0' $S)" "1"
ck "ignore_eos"               "$(grep -c -- '--extra-inputs ignore_eos:true' $S)" "1"

# 4. paths are checked before any GPU work. 215695 found a wrong aiperf path after a
#    fifteen-minute model load and a full bring-up, so the guard matters more than any
#    flag in this file.
ck "the model is checked"      "$(grep -c 'MODEL/config.json' $S)" "1"
ck "the engine venv is checked" "$(grep -c "cannot import sglang" $S)" "1"
ck "the client binary is checked and run" "$(grep -c -- '--version' $S)" "1"
ck "the guard precedes the launch" \
   "$( [[ $(grep -n 'paths ok:' $S | cut -d: -f1) -lt $(grep -n 'launching 4 workers' $S | cut -d: -f1) ]] && echo yes || echo no )" "yes"
ck "the client venv is not the engine venv" "$(grep -c 'AIPERF=${AIPERF:-\$VENV/bin/aiperf}' $S)" "0"

# 5. the PBS request has to be the scored shape.
ck "two nodes of eight GPUs" "$(grep -c '^#PBS -l select=2:ncpus=112:ngpus=8' $S)" "1"
ck "scattered, so select=2 means two hosts" "$(grep -c '^#PBS -l place=scatter' $S)" "1"

exit $fail
