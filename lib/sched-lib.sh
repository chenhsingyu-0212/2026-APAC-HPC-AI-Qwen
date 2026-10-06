#!/bin/bash
# The scheduler seam.
#
# Everything in this repo was written against Slurm, and NSCC ASPIRE 2A+ runs PBS
# Pro. That is not a setting -- it is four different mechanisms:
#
#   sbatch / squeue / scancel       ->  qsub -P <project> / qstat / qdel
#   srun --exact --gres=gpu:N       ->  NO EQUIVALENT. There is no per-step GPU
#                                       allocator; the split becomes ours.
#   scontrol show hostnames         ->  cat $PBS_NODEFILE  (one line per chunk)
#   squeue -o %L                    ->  qstat -f | Resource_List.walltime minus
#                                       resources_used.walltime
#
# The third is the dangerous one, and it is why this file exists rather than a
# handful of `if [[ $SITE_SCHEDULER == pbs ]]` branches spread through pd-lib.sh.
# On Slurm each PD worker is its own step with --gres=gpu:<tp>, and cgroups hand
# it exactly those devices -- which is why nothing in this repo sets
# CUDA_VISIBLE_DEVICES. Take that allocator away and a mistake does not raise an
# error: two workers quietly share a GPU and the run reports a performance number.
# So the isolation lives in ONE function per scheduler, where it can be read.
#
# WHAT A BACKEND MUST PROVIDE
#   sched_nodes            prints one unique node name per line, in allocation order
#   sched_gpus_here        prints this node's GPU identifiers, one per line
#   sched_time_left        seconds remaining in the allocation, or 999999 if unknown
#   sched_step  <node> <ngpu> <ncpu> <logfile> <cmd...>
#                          runs cmd on <node> confined to <ngpu> GPUs, backgrounded,
#                          appending its pid to SCHED_PIDS
#   sched_step_nogpu <node> <ncpu> <logfile> <cmd...>
#                          the same for something that needs no GPU (the gateway)
#   sched_run    <node> <ncpu> <cmd...>
#                          runs cmd on <node> SYNCHRONOUSLY, no GPU, stdout and
#                          stderr left to the caller's redirection. The benchmark
#                          client must block -- backgrounding it would measure
#                          nothing and return success.
#   sched_preflight <nodes...>
#                          called ONCE before anything is deployed. Nothing to do
#                          under Slurm, which drains a node that cannot run work;
#                          PBS hands such a node out anyway, so it probes.
#   sched_run_all <cmd...> one task on every node, synchronously. Used for the
#                          hardware census, which has to be taken from inside the
#                          allocation: two of the five wrong hypotheses about
#                          TensorRT-LLM came from reading ulimit on a login node.
#
# sched_step is stateful on purpose: it draws from a per-node pool of GPU
# identifiers that sched_reset_pools fills, so consecutive calls on one node get
# disjoint devices. That mirrors what Slurm's allocator does for free and makes
# double-assignment a thing the code cannot express.

set -uo pipefail

SCHED_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SCHED_PIDS=()

sched_load() {
    local which="${SITE_SCHEDULER:-slurm}"
    local f="$SCHED_LIB_DIR/sched/$which.sh"
    [[ -r "$f" ]] || { echo "x no scheduler backend for '$which' ($f)" >&2; return 1; }
    # shellcheck disable=SC1090
    source "$f" || return 1
    SCHED_NAME="$which"
    local fn
    for fn in sched_nodes sched_time_left sched_step sched_step_nogpu \
              sched_run sched_run_all sched_preflight; do
        declare -F "$fn" >/dev/null 2>&1 \
            || { echo "x $which backend does not define $fn" >&2; return 1; }
    done
    return 0
}

# ---------------------------------------------------------------- GPU pools
# Filled once per allocation. Keys are node names; each value is a space-separated
# list of the identifiers still unclaimed on that node. A backend that has a real
# per-step allocator (Slurm) leaves these empty and ignores them.
declare -gA SCHED_POOL=()

sched_reset_pools() {   # $@ = node names
    SCHED_POOL=()
    declare -F sched_gpus_on >/dev/null 2>&1 || return 0
    local n ids
    for n in "$@"; do
        ids="$(sched_gpus_on "$n" | tr '\n' ' ')"
        SCHED_POOL["$n"]="${ids% }"
    done
    return 0
}

# Take the next $2 identifiers from node $1's pool. The answer comes back in the
# global SCHED_TAKEN, NOT on stdout, and that is not a style choice: a function
# that both mutates state and prints its result has to be called as $(...), which
# is a subshell, so the mutation is discarded. The first version of this did print,
# and the unit test showed all six workers of a 2x4:4x2 deployment being handed
# GPUs 0-1 of the same node -- sixteen GPUs' worth of placement collapsing onto
# six devices, with no error anywhere. That is precisely the failure this whole
# file exists to make impossible, so the caller takes the extra line.
#
# Refuses rather than over-subscribing: a short pool is a placement bug, and the
# symptom of ignoring it is two workers sharing a GPU and a plausible-looking
# throughput number.
SCHED_TAKEN=""
sched_take_gpus() {   # $1=node  $2=count  -> SCHED_TAKEN
    local node="$1" want="$2"
    SCHED_TAKEN=""
    local -a pool=(${SCHED_POOL["$node"]:-})
    if (( ${#pool[@]} < want )); then
        echo "x $node has ${#pool[@]} GPU(s) left but a worker wants $want" >&2
        return 1
    fi
    local taken="${pool[*]:0:want}"
    SCHED_POOL["$node"]="${pool[*]:want}"
    SCHED_TAKEN="${taken// /,}"
    return 0
}

sched_pools_print() {
    local n
    for n in "${!SCHED_POOL[@]}"; do
        printf '  pool %-26s %s\n' "$n" "${SCHED_POOL[$n]:-<empty>}"
    done
}

sched_wait_all() {   # returns the first non-zero exit, or 0
    local p rc=0 r
    for p in "${SCHED_PIDS[@]:-}"; do
        [[ -n "$p" ]] || continue
        wait "$p"; r=$?
        (( r != 0 && rc == 0 )) && rc=$r
    done
    return $rc
}

sched_kill_all() {
    local p
    for p in "${SCHED_PIDS[@]:-}"; do
        [[ -n "$p" ]] && kill "$p" 2>/dev/null
    done
    SCHED_PIDS=()
    declare -F sched_cleanup >/dev/null 2>&1 && sched_cleanup
    return 0
}
