#!/bin/bash
# Slurm backend. This is exactly what run/pd-lib.sh did inline before the seam
# existed, moved here unchanged so that the NYCU and Firmus numbers already
# measured stay reproducible and the PBS port can be reviewed as a diff.
#
# NOTE THE ARRAY EXPANSION, which is not a style preference. "${A[@]:-}" on an
# unset array expands to ONE EMPTY STRING rather than to nothing, and srun takes
# that empty argument as the command to run -- the real command becomes its
# argument, output is empty, and the exit status is 0. Job 85542 died that way: the
# GPU census read zero GPUs on a node with four, and the campaign correctly refused
# to guess the allocation's size. ${A[@]+"${A[@]}"} expands to nothing.

sched_nodes() {
    scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}"
}

# TWO WAYS TO PUT A WORKER ON GPUS, and Slurm needs both.
#
#   exact    (default) `srun --exact --gres=gpu:N`. Slurm's own per-step allocator
#            puts the step in a cgroup holding N devices and nothing here has to
#            compute an index. This produced every validated number in this project,
#            including Firmus's 1908, so it stays the default.
#
#   overlap  `srun --overlap` plus an explicit CUDA_VISIBLE_DEVICES slice, which is
#            what the PBS backend already does with UUIDs.
#
# The second exists because the first does not always allow a SECOND step. Job 86436
# on NYCU brought its prefill worker up perfectly -- TP2, gpus=0,1, NixlConnector
# configured, /health returning 200 -- and then:
#
#   srun: Job 86436 step creation temporarily disabled, retrying (Requested nodes are busy)
#
# for the decode worker, with 4 GPUs and 96 CPUs allocated and the two steps asking
# for gpu:2/cpu:24 each. NYCU's job_submit plugin requires the type-named form
# --gres=gpu:gpu:N, and a step-level gres in that form appears not to be satisfiable
# a second time. The reference deployment that reaches 3190 on Firmus uses --overlap
# with explicit indices for exactly this reason: its own spec is written as
# "node:0,1 node:2,3 ...".
#
# SLURM_STEP_VIA=overlap switches to it. Solo runs need only one step and are
# unaffected either way.
SLURM_STEP_VIA="${SLURM_STEP_VIA:-exact}"

# Only defined in overlap mode, because in exact mode the allocator owns this and
# second-guessing it would be worse than useless. Inside a job Slurm renumbers the
# allocated devices from 0, so the indices a step needs are 0..N-1 of what it sees.
if [[ "$SLURM_STEP_VIA" == overlap ]]; then
    sched_gpus_on() {   # $1=node -- one index per line
        local n="$1" cnt
        cnt="$(srun --overlap -N1 -n1 -w "$n" --cpus-per-task=1 \
                  nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
        (( ${cnt:-0} > 0 )) || return 0
        seq 0 $(( cnt - 1 ))
    }
fi

sched_time_left() {   # seconds remaining, or 999999 if unknown
    local raw d h m sec a b c
    raw="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o '%L' 2>/dev/null | tr -d ' ')"
    [[ -n "$raw" && "$raw" != "INVALID" && "$raw" != "UNLIMITED" ]] || { echo 999999; return; }
    d=0
    [[ "$raw" == *-* ]] && { d="${raw%%-*}"; raw="${raw#*-}"; }
    IFS=: read -r a b c <<< "$raw"
    if   [[ -n "$c" ]]; then h=$a; m=$b; sec=$c
    elif [[ -n "$b" ]]; then h=0;  m=$a; sec=$b
    else                     h=0;  m=0;  sec=$a; fi
    echo $(( 10#$d * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${sec:-0} ))
}

# ---------------------------------------------------------------- singularity
# A COMMAND PREFIX, not launcher flags, which is why it cannot ride inside
# SCHED_CONTAINER_ARGS the way pyxis's --container-image does.
#
# Nano5 has pyxis as well (--container-image accepts a local PATH), but what we
# built there is a .sif from `singularity build`, not the .sqsh that enroot wants,
# and re-fetching 18 GB only to change container format is not worth it. The
# August TensorRT-LLM runs on that machine used singularity too, and the build
# job printed the exact invocation to copy:
#
#   singularity exec --nv --env PYTHONNOUSERSITE=1 --bind /work,/tmp <sif> <cmd>
#
# PYTHONNOUSERSITE matters: without it a ~/.local/lib built against a different
# torch shadows the image's own packages and the engine imports a mismatched
# extension. --nv is what exposes the driver, and the bind list has to cover both
# the model path and wherever CUDA_VISIBLE_DEVICES-sliced workers write.
# COMPUTED WHEN CALLED, NOT WHEN SOURCED, and that is the whole point. The first
# version was `declare -a SCHED_SING=()` at file scope -- but this file is sourced BY
# sched_load(), which is a function, so `declare` made the array LOCAL TO
# sched_load and it vanished when sched_load returned. Every srun then expanded
# ${SCHED_SING[@]+...} to nothing and the engine ran on bare metal:
#
#   /usr/bin/python3: No module named 'sglang'          (job 372975)
#
# The engine's own python is /opt/sglang/bin/python3 inside the image, and the
# container's PATH puts it first, so that error is proof the prefix never applied
# rather than a PATH problem. I had verified the array by sourcing THIS FILE at top
# level, which is not how production loads it -- 8 words there, unset through
# sched_load.
#
# One line per word so a path containing a space stays one element.
sched_sing_args() {
    [[ -n "${PD_SINGULARITY_IMG:-}" ]] || return 0
    printf '%s\n' singularity exec --nv --env PYTHONNOUSERSITE=1 \
        --bind "${PD_SINGULARITY_BIND:-/work,/tmp}" "$PD_SINGULARITY_IMG"
}

sched_step() {   # $1=node $2=ngpu $3=ncpu $4=logfile $5...=cmd
    local node="$1" ngpu="$2" ncpu="$3" log="$4"; shift 4
    local -a _sing=(); mapfile -t _sing < <(sched_sing_args)
    if [[ "$SLURM_STEP_VIA" == overlap ]]; then
        # NOT ids=$(sched_take_gpus ...): that is a subshell and the pool would never
        # shrink. See the note on sched_take_gpus in sched-lib.sh.
        sched_take_gpus "$node" "$ngpu" || return 1
        local ids="$SCHED_TAKEN"
        srun --overlap -N1 -n1 -w "$node" --cpus-per-task="$ncpu" \
             ${SCHED_CONTAINER_ARGS[@]+"${SCHED_CONTAINER_ARGS[@]}"} \
             ${_sing[@]+"${_sing[@]}"} \
             env "CUDA_VISIBLE_DEVICES=$ids" "$@" > "$log" 2>&1 &
    else
        srun --exact -N1 -n1 -w "$node" \
             --gres="$(pd_gres "$ngpu")" --cpus-per-task="$ncpu" \
             ${SCHED_CONTAINER_ARGS[@]+"${SCHED_CONTAINER_ARGS[@]}"} \
             ${_sing[@]+"${_sing[@]}"} "$@" > "$log" 2>&1 &
    fi
    SCHED_PIDS+=($!)
    return 0
}

# --overlap, not --exact: the gateway holds no GPU and must be allowed to sit on
# a node whose devices are already spoken for by the workers.
sched_step_nogpu() {   # $1=node $2=ncpu $3=logfile $4...=cmd
    local node="$1" ncpu="$2" log="$3"; shift 3
    local -a _sing=(); mapfile -t _sing < <(sched_sing_args)
    srun --overlap -N1 -n1 -w "$node" --cpus-per-task="$ncpu" \
         ${SCHED_CONTAINER_ARGS[@]+"${SCHED_CONTAINER_ARGS[@]}"} \
         ${_sing[@]+"${_sing[@]}"} "$@" > "$log" 2>&1 &
    SCHED_PIDS+=($!)
    return 0
}

# Synchronous, no GPU. --overlap so it can share a node with the workers.
sched_run() {   # $1=node $2=ncpu $3...=cmd
    local node="$1" ncpu="$2"; shift 2
    local -a _sing=(); mapfile -t _sing < <(sched_sing_args)
    srun --overlap -N1 -n1 -w "$node" --cpus-per-task="$ncpu" \
         ${SCHED_CONTAINER_ARGS[@]+"${SCHED_CONTAINER_ARGS[@]}"} \
         ${_sing[@]+"${_sing[@]}"} "$@"
}

sched_run_all() {   # $@=cmd -- one task per node
    local -a nodes=(); mapfile -t nodes < <(sched_nodes)
    srun --overlap -N"${#nodes[@]}" --ntasks-per-node=1 "$@"
}

# Nothing to do. Slurm DRAINs a node that stops answering, so an allocation never
# contains one -- and this cluster has proved it does: two H200 nodes sat DRAINed
# for three days after a stuck job ("Kill task failed : Not responding") rather
# than being handed to the next person. PBS has no such reflex, which is why the
# PBS backend's version of this function is forty lines.
sched_preflight() { return 0; }
