#!/usr/bin/env bash
# Teardown tests: the reap-as-you-wait loop in pd-lib.sh.
#
# Jobs 214830 and 214966 each lost a cell because sched_cleanup matches `--port <n>`
# and SGLang's sglang::scheduler_TP* children carry no such argument, so ~77 GB
# survived the teardown and the next engine died on "The memory capacity is
# unbalanced". pd_wait_gpus_free now asks the driver who holds memory.
#
# This file covers the loop's decisions: when it reaps, when it must not, and what it
# returns when it gives up. Whether the reaper survives how PBS ships a command is a
# different question, and the layer this stubs out is exactly where that bug lived --
# run/test-reap-quoting.sh covers it with the real sched_run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
CALLS="$TMP/calls"; MIB="$TMP/mib"

pd_say() { echo "$*"; }
pd_die() { echo "die: $*"; }
PD_NODE=(nodeA nodeA nodeB)
PD_LIB_DIR="$PWD/run"
sched_nodes() { printf '%s\n' nodeA nodeB; }
sched_run() {                                   # $1=node $2=ncpu then argv
    local node="$1"; shift 2
    if [[ "$*" == *query-gpu=memory.used* ]]; then
        cat "$MIB"
    elif [[ "$*" == *reap-gpus.sh* ]]; then
        # Record the node and the signal, which is the last argument.
        echo "reap $node ${*: -1}" >> "$CALLS"
        echo 3
    fi
}
eval "$(sed -n '/^pd_reap_gpu_procs()/,/^}/p;/^pd_wait_gpus_free()/,/^}/p' run/pd-lib.sh)"

fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

echo 300 > "$MIB"; : > "$CALLS"
PD_GPU_FREE_TIMEOUT=6 pd_wait_gpus_free >/dev/null
rc=$?
ck "idle GPUs are never reaped" "$(wc -l < "$CALLS")" "0"
ck "and the wait succeeds"      "$rc"                 "0"

echo 77562 > "$MIB"; : > "$CALLS"
PD_GPU_REAP_AFTER=0 PD_GPU_FREE_TIMEOUT=40 pd_wait_gpus_free >/dev/null
rc=$?
ck "TERM once per node"              "$(grep -c 'TERM' "$CALLS")" "2"
ck "KILL once per node"              "$(grep -c 'KILL' "$CALLS")" "2"
ck "both nodes covered"              "$(awk '{print $2}' "$CALLS" | sort -u | tr '\n' ' ')" "nodeA nodeB "
ck "giving up is reported as failure" "$rc"                        "1"

echo 77562 > "$MIB"; : > "$CALLS"
PD_GPU_REAP_AFTER=9999 PD_GPU_FREE_TIMEOUT=20 pd_wait_gpus_free >/dev/null
ck "the delay is honoured" "$(wc -l < "$CALLS")" "0"

# A reaper that cannot report a count must say so and not be mistaken for an idle
# node -- the discarded stderr is why job 214966 looked like it had nothing to reap.
sched_run() { local node="$1"; shift 2
    if [[ "$*" == *query-gpu=memory.used* ]]; then cat "$MIB"
    else echo "ssh: Permission denied" ; fi; }
echo 77562 > "$MIB"
out="$(PD_GPU_REAP_AFTER=0 PD_GPU_FREE_TIMEOUT=12 pd_wait_gpus_free 2>&1)"
ck "a broken reaper is named, not silent" \
   "$(grep -c 'did not report a count' <<< "$out")" "2"

exit $fail
