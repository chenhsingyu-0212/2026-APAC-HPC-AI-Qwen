#!/bin/bash
# The singularity command prefix. Nano5 runs engines from a .sif, which is a command
# PREFIX and cannot ride in SCHED_CONTAINER_ARGS the way pyxis's --container-image
# does -- and pyxis cannot run there at all (enroot needs /run/user/$UID).
#
# EVERY CHECK GOES THROUGH sched_load, which is how production loads the backend.
# The first version of this file sourced run/sched/slurm.sh at top level and passed
# while the real thing was broken: the prefix was a `declare -a` at file scope, and
# because sched_load is a FUNCTION that sources the file, declare made the array
# local to sched_load and it vanished on return. Job 372975 ran the engine on bare
# metal and died with "No module named 'sglang'" from /usr/bin/python3.
set -uo pipefail
cd "$(dirname "$0")/.."
pass=0 fail=0
ck(){ if [[ "$2" == "$3" ]]; then (( pass++ )); else echo "x $1: want '$2' got '$3'" >&2; (( fail++ )); fi; }

# prod <env assignments...> -- loads the backend the way the campaign does and runs
# whatever is left in "$@" afterwards.
prod() {
    local script="$1"; shift
    env "$@" SITE_SCHEDULER=slurm bash -c '
        cd "'"$PWD"'"
        . run/sched-lib.sh >/dev/null 2>&1 || exit 9
        sched_load        >/dev/null 2>&1 || exit 9
        '"$script"
}

# 1-3. through sched_load: set, unset, empty
ck "via sched_load, set"   "8" "$(prod 'sched_sing_args | wc -l | tr -d " "' PD_SINGULARITY_IMG=/w/x.sif)"
ck "via sched_load, unset" "0" "$(prod 'sched_sing_args | wc -l | tr -d " "')"
ck "via sched_load, empty" "0" "$(prod 'sched_sing_args | wc -l | tr -d " "' PD_SINGULARITY_IMG=)"

# 4-5. the exact invocation the August build job printed on that machine
ck "the words" "singularity exec --nv --env PYTHONNOUSERSITE=1 --bind /work,/tmp /w/x.sif" \
   "$(prod 'sched_sing_args | tr "\n" " " | sed "s/ $//"' PD_SINGULARITY_IMG=/w/x.sif)"
# The words are: singularity exec --nv --env PYTHONNOUSERSITE=1 --bind <binds> <sif>
# so the bind VALUE is the 7th, not the 6th. Line 6 is --bind itself.
ck "bind override" "/a,/b" \
   "$(prod 'sched_sing_args | sed -n "7p"' PD_SINGULARITY_IMG=/w/x.sif PD_SINGULARITY_BIND=/a,/b)"

# 6. a path with a space must stay ONE element, or the engine is launched against a
# nonexistent image and srun reports the wrong half
ck "space in path" "/a b/x.sif" "$(prod 'sched_sing_args | tail -1' 'PD_SINGULARITY_IMG=/a b/x.sif')"

# 7. and top-level sourcing must agree with sched_load -- if these ever differ again,
# the prefix has gone back to being computed at source time
ck "both load paths agree" \
   "$(prod 'sched_sing_args | wc -l | tr -d " "' PD_SINGULARITY_IMG=/w/x.sif)" \
   "$(env PD_SINGULARITY_IMG=/w/x.sif SITE_SCHEDULER=slurm bash -c 'cd "'"$PWD"'"; . run/sched/slurm.sh >/dev/null 2>&1; sched_sing_args | wc -l | tr -d " "')"

# 8-11. the prefix is spliced at every srun site, built LOCALLY, and always before
# the command. If it landed after "$@" the engine would run on the host while
# singularity got the engine's flags -- which fails minutes later about the wrong
# program.
ck "spliced at 4 sites" "4" "$(grep -c '_sing\[@\]' run/sched/slurm.sh)"
ck "built in 3 functions" "3" "$(grep -c 'mapfile -t _sing' run/sched/slurm.sh)"
ck "no file-scope array"  "0" "$(grep -c '^declare -a SCHED_SING' run/sched/slurm.sh)"
bad=0
while IFS=: read -r ln _; do
    line="$(sed -n "${ln}p" run/sched/slurm.sh)"
    if [[ "$line" == *'"$@"'* ]]; then
        [[ "${line%%\"\$@\"*}" == *_sing* ]] || { echo "x line $ln: prefix after command" >&2; bad=1; }
    else
        nxt="$(sed -n "$((ln+1))p" run/sched/slurm.sh)"
        [[ "$nxt" == *'env "CUDA_VISIBLE_DEVICES'* || "$nxt" == *'"$@"'* ]] \
            || { echo "x line $ln: next line is not the command: '$nxt'" >&2; bad=1; }
    fi
done < <(grep -n '_sing\[@\]+' run/sched/slurm.sh)
(( bad == 0 )) && (( pass++ )) || (( fail++ ))

# 12-14. nano5 must use singularity and NOT pyxis: a non-empty PD_IMAGE would make
# pd_container_args emit --container-image for the same cards, and that is 372921.
ck "nano5 SGLANG_IMG empty" "EMPTY" "$(bash -c 'set +u; . run/sites/nano5.sh >/dev/null 2>&1; printf "%s" "${SGLANG_IMG:-EMPTY}"')"
ck "nano5 sif" "/work/u8924359/hsingyu-hpcai/images/sglang-v0.5.19-cu129.sif" \
   "$(bash -c 'set +u; . run/sites/nano5.sh >/dev/null 2>&1; printf "%s" "$PD_SINGULARITY_IMG"')"
for s in nscc nycu firmus; do
    got="$(bash -c 'set +u; . run/sites/'"$s"'.sh >/dev/null 2>&1; printf "%s" "${PD_SINGULARITY_IMG:-EMPTY}"')"
    [[ "$got" == EMPTY ]] || { echo "x $s should not set PD_SINGULARITY_IMG (got $got)" >&2; (( fail++ )); }
done
(( pass++ ))

echo "test-singularity-prefix: $pass passed, $fail failed"
(( fail == 0 ))
