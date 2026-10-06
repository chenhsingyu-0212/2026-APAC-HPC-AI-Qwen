#!/usr/bin/env bash
# PBS hands GPUs out as UUIDs; SGLang parses CUDA_VISIBLE_DEVICES with int() and
# silently drops to NCCL when that fails. These tests check the translation to
# positional indices, with a stub nvidia-smi standing in for the node.
#
# The position matters, not nvidia-smi's reported index: when a cgroup exposes a
# subset the driver renumbers from zero, so test 4 feeds a list whose reported
# indices would be 4..7 and expects 0..3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin"
make_smi() {   # $@ = uuids in enumeration order
    { echo '#!/bin/sh'; printf 'cat <<X\n'; printf '%s\n' "$@"; printf 'X\n'; } > "$TMP/bin/nvidia-smi"
    chmod +x "$TMP/bin/nvidia-smi"
}
snippet="$(sed -n "/^_PBS_GPU_XLAT_SNIPPET='$/,/^'$/p" run/sched/pbs.sh | sed '1d;$d')"
[[ -n "$snippet" ]] || { echo "FAIL could not extract the snippet"; exit 1; }
run() { ( export PATH="$TMP/bin:$PATH" CUDA_VISIBLE_DEVICES="$1"; bash -c "$snippet" ) | paste -sd,; }

fail=0
ck(){ if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

A=GPU-aaaa; B=GPU-bbbb; C=GPU-cccc; D=GPU-dddd
make_smi $A $B $C $D

ck "a UUID becomes its position"        "$(run "$A")"             "0"
ck "the fourth becomes 3"               "$(run "$D")"             "3"
ck "a slice keeps order"                "$(run "$C,$D")"          "2,3"
ck "out-of-order input is not sorted"   "$(run "$D,$B")"          "3,1"
ck "plain integers pass through"        "$(run "2,3")"            "2,3"
ck "an unknown id passes through"       "$(run "GPU-zzzz")"       "GPU-zzzz"
ck "empty CUDA_VISIBLE_DEVICES lists all" "$(run "")"             "0,1,2,3"

# A cgroup-restricted node: only four GPUs visible, and CUDA numbers them 0..3
# however nvidia-smi labels them.
make_smi GPU-e GPU-f GPU-g GPU-h
ck "a restricted node still starts at 0" "$(run "GPU-g,GPU-h")"   "2,3"

exit $fail
