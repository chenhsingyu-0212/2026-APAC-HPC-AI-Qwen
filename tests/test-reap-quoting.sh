#!/usr/bin/env bash
# The GPU reaper has to survive how PBS sends a command.
#
# sched_run runs printf '%q ' over the argv and _pbs_remote embeds the result in
#   ssh <node> "bash -c '<string>'"
# so only an argv free of shell-special characters arrives intact. Job 214966's
# reaper was an inline multi-line snippet; %q returned it as $'...', whose single
# quotes closed that wrapper early, and it never ran -- while its stderr was
# discarded, so a reaper that could not execute looked like a node with nothing to
# reap. These tests use the REAL sched_run with an _pbs_remote that reproduces the
# ssh quoting exactly, instead of stubbing the layer where the bug lived.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO="$PWD"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin"
export PATH="$TMP/bin:$PATH"
stub_smi() {   # $@ = pids the driver reports as holding memory
    { echo '#!/bin/sh'; echo 'cat <<X'; printf '%s\n' "$@"; echo 'X'; } > "$TMP/bin/nvidia-smi"
    chmod +x "$TMP/bin/nvidia-smi"
}

PBS_STEP_VIA=ssh
eval "$(sed -n '/^sched_run()/,/^}/p' run/sched/pbs.sh)"
# Exactly the shape of the ssh branch in _pbs_remote, locally.
_pbs_remote() { local node="$1" snippet="$2"; bash -c "bash -c '$snippet'" 2>&1; }

fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

# 1. The real thing: a victim we own, reported by the stub driver.
sleep 300 & victim=$!
stub_smi "$victim"
out="$(sched_run nodeX 1 bash "$REPO/run/reap-gpus.sh" TERM)"
ck "the reaper runs and reports a count" "$out" "1"
sleep 0.3
ck "and the process is gone" "$(kill -0 "$victim" 2>/dev/null && echo alive || echo gone)" "gone"
wait "$victim" 2>/dev/null

# 2. A process we do not own is left alone. PID 1 is never ours.
stub_smi 1
ck "someone else's process is not touched" "$(sched_run nodeX 1 bash "$REPO/run/reap-gpus.sh" TERM)" "0"

# 3. Nothing on the GPUs.
stub_smi
ck "an idle node reports zero" "$(sched_run nodeX 1 bash "$REPO/run/reap-gpus.sh" TERM)" "0"

# 4. The negative control. The inline form that job 214966 shipped must NOT survive
#    this path -- if it does, this test cannot detect the bug it exists for.
inline='
me=$(id -u); n=0
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(stat -c %u /proc/$p 2>/dev/null)" = "$me" ] || continue
    kill -TERM "$p" 2>/dev/null && n=$((n+1))
done
echo "$n"'
sleep 300 & victim2=$!
stub_smi "$victim2"
got="$(sched_run nodeX 1 bash -c "$inline" 2>&1 | tail -1)"
if [[ "$got" == "1" ]]; then
    echo "FAIL the inline form was expected to be mangled by this path, but it worked"
    fail=1
else
    echo "ok   the inline form is still mangled (control)"
fi
kill -9 "$victim2" 2>/dev/null; wait "$victim2" 2>/dev/null

# 5. A bad signal name is refused rather than passed to kill.
stub_smi 1
sched_run nodeX 1 bash "$REPO/run/reap-gpus.sh" 9 >/dev/null 2>&1
ck "a signal that is not in the allow-list is refused" "$?" "2"

exit $fail
