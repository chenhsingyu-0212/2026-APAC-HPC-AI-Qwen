#!/bin/bash
# Every site file must define the names the campaign READS, not near-misses.
#
# Job 372918 was submitted with AIPERF set and AIPERF_BIN unset, because nano5.sh
# used a name the other three site files do not. AIPERF_BIN fell back to the bare
# `aiperf`, which `command -v` finds on a compute node and which cannot run -- so the
# precondition at 10-campaign.sbatch:195 passed and client_init at :207 failed. The
# guard and the thing it guarded were reading different variables.
#
# No indirect expansion (${!v}) here. The first version of this file used it and I
# could not tell whether a reported failure was the site file or the harness; a plain
# `printf '%s'` per variable in a subshell cannot be ambiguous.
set -uo pipefail
cd "$(dirname "$0")/.."
pass=0 fail=0
bad(){ echo "x $*" >&2; fail=$((fail+1)); }
ok(){  pass=$((pass+1)); }

val() {   # $1=site file  $2=variable -- prints its value after sourcing, or nothing
    bash -c 'set +u; . "$1" >/dev/null 2>&1; printf "%s" "${'"$2"':-}"' _ "$1"
}

for site in run/sites/*.sh; do
    name="$(basename "$site" .sh)"

    # MODEL_PATH and the two shape variables must be non-empty everywhere.
    for v in MODEL_PATH SITE_PARTITION SITE_GPUS_PER_NODE; do
        [[ -n "$(val "$site" "$v")" ]] && ok || bad "$name leaves $v empty"
    done

    # AIPERF_BIN must be non-empty everywhere, because client_init runs it.
    binv="$(val "$site" AIPERF_BIN)"
    [[ -n "$binv" ]] && ok || bad "$name leaves AIPERF_BIN empty -- client_init runs it"

    # A BARE name is legal only when the client runs inside a container: Firmus has
    # AIPERF_IMG=/scratch/enroot-images/aiperf.sqsh and AIPERF_BIN=aiperf, resolved
    # inside it. Without a container the bare name is exactly the 372918 failure.
    imgv="$(val "$site" AIPERF_IMG)"
    if [[ "$binv" != /* ]]; then
        [[ -n "$imgv" ]] && ok \
            || bad "$name has a bare AIPERF_BIN=$binv and no AIPERF_IMG -- that is 372918"
    else ok; fi

    # And a site must not define AIPERF while leaving AIPERF_BIN at its default.
    if grep -qE '^AIPERF=' "$site" && ! grep -qE '^AIPERF_BIN=' "$site"; then
        bad "$name defines AIPERF but not AIPERF_BIN -- the campaign reads AIPERF_BIN"
    else ok; fi
done

# A site cannot use both container mechanisms. SGLANG_IMG feeds eng_image() ->
# PD_IMAGE -> pyxis's --container-image, and PD_SINGULARITY_IMG feeds a
# `singularity exec` command prefix. Job 372921 set both and pyxis got the .sif:
#   pyxis: mkdir: cannot create directory '/run/user/37948': Permission denied
for site in run/sites/*.sh; do
    name="$(basename "$site" .sh)"
    sing="$(val "$site" PD_SINGULARITY_IMG)"
    img="$(val "$site" SGLANG_IMG)"
    if [[ -n "$sing" && -n "$img" ]]; then
        bad "$name sets both PD_SINGULARITY_IMG and SGLANG_IMG -- pyxis would get the .sif (372921)"
    else ok; fi
    # and a .sif must never be what SGLANG_IMG points at, on any site
    case "$img" in
        *.sif) bad "$name has SGLANG_IMG=$img -- a .sif cannot be a pyxis image" ;;
        *) ok ;;
    esac
done

# nano5_ready must guard the three things that have actually failed a submission.
guard="$(sed -n '/nano5_ready()/,/^}/p' run/sites/nano5.sh)"
grep -q 'AIPERF_BIN' <<<"$guard" && ok || bad "nano5_ready does not check AIPERF_BIN (372918)"
grep -q 'observe.sh'  <<<"$guard" && ok || bad "nano5_ready does not check the AIPerf kit (372908)"
grep -q 'SGLANG_IMG'  <<<"$guard" && ok || bad "nano5_ready does not check SGLANG_IMG is empty (372921)"

echo "test-site-contract: $pass passed, $fail failed"
(( fail == 0 ))
