#!/bin/bash
# Offline tests for the scheduler seam. No cluster, no allocation, no SU.
#
#   run/test-sched.sh
#
# The one that matters is "no GPU is handed out twice". Under Slurm that is the
# allocator's job and cannot go wrong here; under PBS it is our arithmetic, and
# when it goes wrong NOTHING FAILS -- two workers open contexts on one device and
# the run reports a throughput number that looks like a result. The first version
# of sched_take_gpus printed its answer instead of setting a global, so every
# caller invoked it in a subshell, the pool never shrank, and all six workers of a
# 2x4:4x2 deployment were handed the same two GPUs. This file is why that was
# caught on a laptop rather than in a 1024-SU allocation.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok   $*"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $*"; }

# ---------------------------------------------------------------- 1. resolution
for site in nycu firmus nscc; do
    want=slurm; [[ $site == nscc ]] && want=pbs
    got="$(PD_SITE=$site bash -c "
        source '$HERE/site-lib.sh' && site_load >/dev/null 2>&1
        source '$HERE/pd-lib.sh'   >/dev/null 2>&1
        echo \$SCHED_NAME" 2>/dev/null)"
    [[ "$got" == "$want" ]] && ok "$site -> $want" || bad "$site -> '$got', wanted '$want'"
done

# sched_gpus_on must exist for PBS and must NOT exist for Slurm: its presence is
# what switches sched_reset_pools on, and a stray definition on a Slurm site would
# start second-guessing an allocator that is already correct.
for site in nycu nscc; do
    got="$(PD_SITE=$site bash -c "
        source '$HERE/site-lib.sh' && site_load >/dev/null 2>&1
        source '$HERE/pd-lib.sh'   >/dev/null 2>&1
        declare -F sched_gpus_on >/dev/null 2>&1 && echo yes || echo no" 2>/dev/null)"
    want=no; [[ $site == nscc ]] && want=yes
    [[ "$got" == "$want" ]] && ok "$site sched_gpus_on=$want" || bad "$site sched_gpus_on='$got'"
done

# ---------------------------------------------------------------- 2. the pool
_pool_case() {   # $1=spec  $2=gpus-per-node  $3...=nodes
    local spec="$1" per="$2"; shift 2
    local out
    out="$(PD_SITE=nscc bash -c '
        set -uo pipefail
        source '"$HERE"'/site-lib.sh && site_load >/dev/null 2>&1
        source '"$HERE"'/pd-lib.sh   >/dev/null 2>&1
        sched_gpus_on() { local n=$1 i; for ((i=0;i<'"$per"';i++)); do printf "GPU-%s-%02d\n" "$n" "$i"; done; }
        sched_reset_pools '"$*"'
        pd_parse '"$spec"' >/dev/null && pd_place '"$per"' '"$*"' >/dev/null || exit 9
        for i in "${!PD_ROLE[@]}"; do
            sched_take_gpus "${PD_NODE[$i]}" "${PD_TP[$i]}" || exit 8
            echo "$SCHED_TAKEN"
        done' 2>&1)"
    echo "$out"
}

run_pool_test() {   # $1=label $2=spec $3=per-node $4=expected-total $5...=nodes
    local label="$1" spec="$2" per="$3" total="$4"; shift 4
    local out; out="$(_pool_case "$spec" "$per" "$@")"
    local n_uniq n_all
    n_all="$(tr ',' '\n' <<< "$out" | sed '/^$/d' | wc -l)"
    n_uniq="$(tr ',' '\n' <<< "$out" | sed '/^$/d' | sort -u | wc -l)"
    if [[ "$n_all" != "$total" ]]; then
        bad "$label: handed out $n_all GPUs, expected $total"; echo "$out" | sed 's/^/       /'
    elif [[ "$n_uniq" != "$n_all" ]]; then
        bad "$label: $n_all handed out but only $n_uniq distinct -- A GPU IS SHARED"
        tr ',' '\n' <<< "$out" | sed '/^$/d' | sort | uniq -d | sed 's/^/       dup: /'
    else
        ok "$label: $n_all GPUs, all distinct"
    fi
}

run_pool_test "2x4:4x2 on 2x8"  2x4:4x2 8  16 nodeA nodeB
run_pool_test "4x2:8x1 on 2x8"  4x2:8x1 8  16 nodeA nodeB
run_pool_test "1x4:4x1 on 1x8"  1x4:4x1 8   8 nodeA
run_pool_test "2x2:4x1 on 1x8"  2x2:4x1 8   8 nodeA

# ---------------------------------------------------------------- 2b. cell reuse
# A campaign runs many cells in one allocation, each deploying and tearing down.
# sched_take_gpus only ever removes, so without a reset per cell the second cell
# is refused for want of devices that are sitting idle. Caught by reading, then
# pinned here.
out="$(PD_SITE=nscc bash -c '
    set -uo pipefail
    source '"$HERE"'/site-lib.sh && site_load >/dev/null 2>&1
    source '"$HERE"'/pd-lib.sh   >/dev/null 2>&1
    sched_gpus_on() { local n=$1 i; for ((i=0;i<8;i++)); do printf "GPU-%s-%02d\n" "$n" "$i"; done; }
    for cell in 1 2 3; do
        sched_reset_pools nodeA nodeB
        pd_parse 2x4:4x2 >/dev/null && pd_place 8 nodeA nodeB >/dev/null || exit 9
        for i in "${!PD_ROLE[@]}"; do
            sched_take_gpus "${PD_NODE[$i]}" "${PD_TP[$i]}" || { echo "cell $cell refused"; exit 8; }
        done
        echo "cell $cell ok"
    done' 2>&1)"
[[ "$out" == "cell 1 ok
cell 2 ok
cell 3 ok" ]] && ok "three cells in one allocation each get a full pool" \
    || { bad "cells do not get a fresh pool"; echo "$out" | sed 's/^/       /'; }

# ---------------------------------------------------------------- 2c. all shapes
for site in nycu nscc; do
    miss="$(PD_SITE=$site bash -c "
        source '$HERE/site-lib.sh' && site_load >/dev/null 2>&1
        source '$HERE/pd-lib.sh'   >/dev/null 2>&1
        for fn in sched_nodes sched_time_left sched_step sched_step_nogpu sched_run sched_run_all; do
            declare -F \$fn >/dev/null 2>&1 || echo \$fn
        done" 2>/dev/null)"
    [[ -z "$miss" ]] && ok "$site defines every shape the seam needs" \
        || bad "$site is missing: $(tr '\n' ' ' <<< "$miss")"
done

# ---------------------------------------------------------------- 2d. no empty argv
# "${A[@]:-}" on an unset array expands to ONE EMPTY STRING. srun takes that empty
# argument as the command to run, the real command becomes its argument, output is
# empty and the exit status is 0 -- so the failure surfaces somewhere else entirely.
# Job 85542 died on it: the GPU census read zero GPUs on a node with four.
#
# THE FIRST VERSION OF THIS TEST COULD NOT FAIL. sched_step redirects into the logfile
# it is given, the test passed /dev/null, so the capture was always empty and "no empty
# argument found" was always true. It now writes to real files and reads them back.
for site in nycu nscc; do
    tmp="$(mktemp -d)"
    PD_SITE=$site bash -c '
        source '"$HERE"'/site-lib.sh && site_load >/dev/null 2>&1
        source '"$HERE"'/pd-lib.sh   >/dev/null 2>&1
        srun()   { printf "%s\n" "$@"; }
        ssh()    { printf "%s\n" "$@"; }
        pbsdsh() { printf "%s\n" "$@"; }
        unset SCHED_CONTAINER_ARGS
        sched_gpus_on() { local i; for i in 0 1 2 3; do echo "GPU-fake-0$i"; done; }
        sched_reset_pools nodeA
        SCHED_PIDS=()
        sched_step       nodeA 2 8 '"$tmp"'/step.log  echo hello
        sched_step_nogpu nodeA 2   '"$tmp"'/nogpu.log echo hello
        sched_run        nodeA 2 echo hello           > '"$tmp"'/run.log 2>&1
        wait 2>/dev/null' >/dev/null 2>&1
    cap="$(cat "$tmp"/*.log 2>/dev/null)"
    if [[ -z "$cap" ]]; then
        bad "$site: the argv capture is empty -- this test cannot fail as written"
    elif grep -qx '' <<< "$cap"; then
        bad "$site: a launcher built an EMPTY argument"
        cat -A "$tmp"/*.log 2>/dev/null | grep -n '^\$' | head -3 | sed 's/^/       line /'
    else
        ok "$site: no launcher builds an empty argument ($(wc -l <<< "$cap") 個參數)"
    fi
    rm -rf "$tmp"
done

# ---------------------------------------------------------------- 2e. slurm overlap
# SLURM_STEP_VIA=overlap is a second way to place a worker, added because --exact would
# not create a SECOND step on NYCU (job 86436: prefill up and healthy, decode stuck on
# "step creation temporarily disabled"). An untested code path is how the last three
# bugs got in, so: the mode must define sched_gpus_on, build --overlap rather than
# --exact, carry an explicit CUDA_VISIBLE_DEVICES, and still hand out disjoint devices.
tmp="$(mktemp -d)"
have="$(PD_SITE=nycu SLURM_STEP_VIA=overlap bash -c '
    set -uo pipefail
    source '"$HERE"'/site-lib.sh && site_load >/dev/null 2>&1
    source '"$HERE"'/pd-lib.sh   >/dev/null 2>&1
    declare -F sched_gpus_on >/dev/null 2>&1 && echo HAVE || echo NO
    srun() { printf "%s\n" "$@"; }
    sched_gpus_on() { local i; for i in 0 1 2 3 4 5 6 7; do echo "$i"; done; }
    sched_reset_pools nodeA
    SCHED_PIDS=(); unset SCHED_CONTAINER_ARGS
    pd_parse 2x2:4x1 >/dev/null && pd_place 8 nodeA >/dev/null || exit 9
    for i in "${!PD_ROLE[@]}"; do
        sched_step nodeA "${PD_TP[$i]}" 8 '"$tmp"'/w$i.log true || exit 8
    done
    wait 2>/dev/null' 2>/dev/null)"
cap="$(cat "$tmp"/w*.log 2>/dev/null)"
[[ "$have" == HAVE ]] && ok "overlap 模式定義了 sched_gpus_on" || bad "overlap 模式沒有 sched_gpus_on"
if [[ -z "$cap" ]]; then
    bad "overlap: 捕捉不到 argv -- 測試本身無效"
else
    if grep -qx -- '--overlap' <<< "$cap" && ! grep -qx -- '--exact' <<< "$cap"; then
        ok "overlap 模式用 --overlap 而非 --exact"
    else bad "overlap 的 srun 旗標不對"; head -6 <<< "$cap" | sed 's/^/       /'; fi
    cvd="$(grep -c '^CUDA_VISIBLE_DEVICES=' <<< "$cap")"
    [[ "$cvd" == 3 ]] && ok "三個 worker 各帶一個 CUDA_VISIBLE_DEVICES" \
        || bad "CUDA_VISIBLE_DEVICES 出現 $cvd 次，應該是 3"
    dev="$(grep '^CUDA_VISIBLE_DEVICES=' <<< "$cap" | sed 's/^[^=]*=//' | tr ',' '\n' | sed '/^$/d')"
    na="$(grep -c . <<< "$dev")"; nu="$(sort -u <<< "$dev" | grep -c .)"
    [[ "$na" == 8 && "$nu" == 8 ]] && ok "overlap 發出 8 個互不重複的 GPU" \
        || { bad "overlap 的 GPU 分配不對：$na 個、$nu 個不重複"; tr '\n' ' ' <<< "$dev" | sed 's/^/       /'; echo; }
fi
rm -rf "$tmp"

# The default must be unchanged: every validated number in this project came from it.
tmp="$(mktemp -d)"
PD_SITE=nycu bash -c '
    source '"$HERE"'/site-lib.sh && site_load >/dev/null 2>&1
    source '"$HERE"'/pd-lib.sh   >/dev/null 2>&1
    srun() { printf "%s\n" "$@"; }
    SCHED_PIDS=(); unset SCHED_CONTAINER_ARGS
    sched_step nodeA 4 8 '"$tmp"'/d.log true
    wait 2>/dev/null' >/dev/null 2>&1
cap2="$(cat "$tmp"/d.log 2>/dev/null)"
if [[ -z "$cap2" ]]; then bad "預設模式: 捕捉不到 argv"
elif grep -qx -- '--exact' <<< "$cap2" && ! grep -q '^CUDA_VISIBLE_DEVICES=' <<< "$cap2"; then
    ok "預設仍是 --exact，且不自己設 CUDA_VISIBLE_DEVICES"
else bad "預設模式被改動了"; head -5 <<< "$cap2" | sed 's/^/       /'; fi
rm -rf "$tmp"

# ---------------------------------------------------------------- 3. refusal
out="$(PD_SITE=nscc bash -c '
    source '"$HERE"'/site-lib.sh && site_load >/dev/null 2>&1
    source '"$HERE"'/pd-lib.sh   >/dev/null 2>&1
    sched_gpus_on() { local i; for i in 0 1 2 3; do echo "GPU-x-$i"; done; }
    sched_reset_pools n1
    sched_take_gpus n1 4 || echo UNEXPECTED
    sched_take_gpus n1 1 && echo "UNEXPECTED: got a 5th GPU from a 4-GPU node"' 2>&1)"
grep -q 'has 0 GPU(s) left but a worker wants 1' <<< "$out" \
    && ok "an over-subscribed node is refused, not silently shared" \
    || { bad "over-subscription was not refused"; echo "$out" | sed 's/^/       /'; }

# ---------------------------------------------------------------- 4. HH:MM:SS
out="$(PD_SITE=nscc bash -c '
    source '"$HERE"'/site-lib.sh && site_load >/dev/null 2>&1
    source '"$HERE"'/pd-lib.sh   >/dev/null 2>&1
    for v in 00:10:00 01:00:00 24:00:00 05:30 42; do printf "%s=%s " "$v" "$(_pbs_hms "$v")"; done' 2>&1)"
want="00:10:00=600 01:00:00=3600 24:00:00=86400 05:30=330 42=42 "
[[ "$out" == "$want" ]] && ok "PBS walltime parsing" \
    || { bad "PBS walltime parsing"; echo "       got:  $out"; echo "       want: $want"; }

echo
echo "  $PASS passed, $FAIL failed"
(( FAIL == 0 ))
