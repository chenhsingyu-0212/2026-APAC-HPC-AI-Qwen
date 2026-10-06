#!/usr/bin/env bash
# Kill our own processes that still hold GPU memory on this node, identified by the
# driver rather than by argv.
#
# Why this is a file and not an inline snippet: run/sched/pbs.sh sends a command by
# running printf '%q ' over the argv and embedding the result in
# ssh ... "bash -c '<string>'". A multi-line script comes back from %q as $'...',
# whose single quotes terminate that wrapper, so the whole thing is silently
# mangled -- job 214966's reaper never ran and left 77,594 MiB held. Invoked by
# path, the argv has no shell-special characters and survives.
#
# Why the driver and not argv: sched_cleanup matches `--port <n>`, which finds the
# server SGLang launches but not the sglang::scheduler_TP* children it forks, and
# those hold ~19 GB per rank.
#
# Only our own processes are touched -- the PID must be ours by /proc ownership.
# Prints the number signalled, so a caller can report it.
set -uo pipefail
sig="${1:-TERM}"
case "$sig" in TERM|KILL|INT) ;; *) echo "0"; exit 2 ;; esac

me="$(id -u)"
n=0
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    [ -n "$p" ] || continue
    [ -d "/proc/$p" ] || continue
    [ "$(stat -c %u "/proc/$p" 2>/dev/null)" = "$me" ] || continue
    kill -"$sig" "$p" 2>/dev/null && n=$((n + 1))
done
echo "$n"
