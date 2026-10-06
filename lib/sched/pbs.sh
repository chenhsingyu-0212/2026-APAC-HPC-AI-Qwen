#!/bin/bash
# PBS Pro backend (NSCC ASPIRE 2A+).
#
# The hard part is not the command names. It is that PBS HAS NO PER-STEP GPU
# ALLOCATOR. Slurm's `srun --exact --gres=gpu:4` creates a cgroup holding four
# devices and the step sees them as 0..3; PBS gives the whole chunk to the job and
# expects the job to divide it. So the isolation that Slurm did for us is written
# out here, and it is written once.
#
# What PBS does give us, measured on a node (recon job 213456):
#
#     CUDA_VISIBLE_DEVICES=GPU-e6ef6154-f1c6-2bcd-0ffb-a20ea25fc611
#
# GPU UUIDs, not indices. That is the whole design constraint: nvidia-smi inside
# the job renumbers the visible device to 0, so "give this worker GPUs 4-7" cannot
# be expressed. A worker is given a SLICE OF THE UUID LIST, which is what
# sched_gpus_on / sched_take_gpus in sched-lib.sh exist to hand out.
#
# Why UUIDs are in fact safer than indices here: a UUID names one physical device
# for the life of the machine. An index is relative to whatever the caller could
# already see, so an off-by-one in an index scheme silently overlaps two workers,
# while the same mistake with UUIDs hands out the same string twice -- which
# sched_take_gpus refuses, because it removes what it hands out.

PBS_BIN="${PBS_BIN:-/opt/pbs/bin}"

sched_nodes() {
    # One line per CHUNK, so a 2-node x 8-GPU job repeats each host. Unique, in
    # allocation order -- not sorted, because chunk order is the placement order
    # pd_place assumes.
    if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
        awk '!seen[$1]++ {print $1}' "$PBS_NODEFILE"
    else
        hostname
    fi
}

# How a remote command gets onto a node. Two mechanisms exist and they differ in
# whether the PBS cgroup follows:
#
#   pbsdsh  -- PBS's own task launcher. The task is a real PBS task, so it lands
#              inside the job's cgroup and is cleaned up with the job.
#   ssh     -- ordinary ssh. On many PBS sites the pbs_cgroups hook attaches
#              incoming ssh sessions to the job; where it does not, the process
#              escapes the job's accounting and survives qdel.
#
#   mpirun  -- what a PBS site normally EXPECTS for multi-node work. It reads
#              $PBS_NODEFILE itself, and NSCC provides hpcx/2.20-mt-gcc11-cuda12
#              and openmpi/5.0.5-gcc11-cuda12.6 as modules. `--map-by node` puts
#              one rank per node, which is the shape a PD worker wants.
#
# The default is chosen by measurement, and the measurement changed it. Job 213474
# scattered three chunks over a2ap-dgx035, dgx037 and dgx038 and probed each from
# the mother superior:
#
#            pbsdsh          ssh             ping
#   dgx035   exit 0, NO OUT  exit 0, "dgx035"  0.045 ms
#   dgx037   exit 0, NO OUT  exit 0, "dgx037"  0.137 ms
#   dgx038   TIMEOUT         TIMEOUT           (fine)
#
# Two things fall out of that. THE FAILURE FOLLOWS THE HOST, NOT THE MECHANISM:
# every launcher works against two nodes and every launcher hangs against dgx038 --
# which is the node that had been holding seven jobs and took sixteen minutes to
# start one, and which hung recon2 and recon4 as well. Remote execution is open
# here; 16-GPU PD is possible. And PBSDSH IS NOT USABLE ANYWAY: it exits 0 and
# produces nothing, on a healthy node, for `pbsdsh -n 0 -- /bin/hostname`. An exit
# code of 0 from a launcher that never ran the command is the worst possible
# failure mode for this code path, because pd_wait_healthy would then be waiting on
# a worker that was never started.
#
# So: ssh. It also turns out to land inside the job's cgroup -- an ssh session on a
# 1-GPU chunk reports cgroup pbs_jobs.service/jobs/213471.1/3 and sees one GPU --
# so there is no escape hazard and no orphan surviving qdel, which was the only
# reason to prefer pbsdsh in the first place.
PBS_STEP_VIA="${PBS_STEP_VIA:-ssh}"

# Where mpirun comes from, when PBS_STEP_VIA=mpirun. Empty means "whatever is on
# PATH after SITE_PROLOGUE has loaded its modules".
PBS_MPIRUN="${PBS_MPIRUN:-mpirun}"

# Which pbsdsh task index corresponds to a node. pbsdsh numbers by CHUNK, and a
# node holding two chunks therefore has two indices; we want the first.
_pbs_index_of() {   # $1=node
    [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]] || { echo 0; return; }
    awk -v want="$1" '$1 == want { print NR - 1; exit }' "$PBS_NODEFILE"
}

# PBS hands us GPUs as UUIDs, and SGLang parses CUDA_VISIBLE_DEVICES with int().
#
#   Setup Custom allreduce failed with invalid literal for int() with base 10:
#   GPU-3f70e3fb-6c96-fa36-b623-d77b30e3b686
#
# Every TP rank of every worker in job 214830 printed that, on both roles: custom
# all-reduce was never once enabled on this cluster, and every collective fell back
# to NCCL. The campaign banner says "gpus-held 0,1,2,3", which is the intended
# layout, not what the process received -- that is how this hid.
#
# The fix is to hand CUDA integers. The integer must be the GPU's POSITION in the
# node's enumeration, not the index nvidia-smi reports: if the cgroup exposes a
# subset, the driver renumbers from zero and the reported index would be wrong,
# while the position is right either way.
#
# PBS_GPU_IDS=uuid restores the old behaviour, so the two can be compared.
_PBS_GPU_XLAT_SNIPPET='
order=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d " " | sed "/^$/d")
ids=${CUDA_VISIBLE_DEVICES:-}
mapfile -t ord <<< "$order"
if [ -z "$ids" ]; then
    for k in "${!ord[@]}"; do echo "$k"; done
    exit 0
fi
IFS=, read -r -a toks <<< "$ids"
for t in "${toks[@]}"; do
    [ -n "$t" ] || continue
    found=
    for k in "${!ord[@]}"; do
        if [ "${ord[$k]}" = "$t" ]; then found=$k; break; fi
    done
    if [ -n "$found" ]; then echo "$found"; else echo "$t"; fi
done
'

sched_gpus_on() {   # $1=node -- prints one GPU identifier per line
    local node="$1" out
    local snippet="$_PBS_GPU_XLAT_SNIPPET"
    if [[ "${PBS_GPU_IDS:-index}" == uuid ]]; then
        # The old behaviour, kept so the two can be measured against each other.
        snippet='if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then tr "," "\n" <<< "$CUDA_VISIBLE_DEVICES"; else nvidia-smi --query-gpu=uuid --format=csv,noheader; fi'
    fi
    # On the node running this script PBS has already exported the list, so the
    # translation runs here rather than over ssh.
    if [[ "$node" == "$(hostname -s)" || "$node" == "$(hostname)" ]]; then
        out="$(bash -c "$snippet" 2>/dev/null)"
    else
        out="$(_pbs_remote "$node" "$snippet" 2>/dev/null)"
    fi
    sed '/^$/d' <<< "$out"
}

_pbs_remote() {   # $1=node  $2=shell snippet -- synchronous, stdout captured
    local node="$1" snippet="$2"
    case "$PBS_STEP_VIA" in
        pbsdsh) "$PBS_BIN/pbsdsh" -n "$(_pbs_index_of "$node")" -- \
                    /bin/bash -c "$snippet" ;;
        ssh)    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node" \
                    "bash -c '$snippet'" ;;
        mpirun) "$PBS_MPIRUN" -np 1 --host "$node" --map-by node \
                    /bin/bash -c "$snippet" ;;
    esac
}

sched_time_left() {   # seconds remaining, or 999999 if unknown
    local lim used
    lim="$("$PBS_BIN/qstat" -f "${PBS_JOBID:-0}" 2>/dev/null \
            | awk -F= '/Resource_List.walltime/ {gsub(/ /,"",$2); print $2; exit}')"
    [[ -n "$lim" ]] || { echo 999999; return; }
    used="$("$PBS_BIN/qstat" -f "${PBS_JOBID:-0}" 2>/dev/null \
            | awk -F= '/resources_used.walltime/ {gsub(/ /,"",$2); print $2; exit}')"
    echo $(( $(_pbs_hms "$lim") - $(_pbs_hms "${used:-00:00:00}") ))
}

_pbs_hms() {   # HH:MM:SS (or MM:SS, or SS) -> seconds
    local a b c; IFS=: read -r a b c <<< "$1"
    if   [[ -n "$c" ]]; then echo $(( 10#$a*3600 + 10#$b*60 + 10#$c ))
    elif [[ -n "$b" ]]; then echo $(( 10#$a*60 + 10#$b ))
    else                     echo $(( 10#${a:-0} )); fi
}

# The GPU split. Note what is NOT here: no index arithmetic, and no --gres.
# sched_take_gpus removes the identifiers it hands out, so the same device cannot
# be given to two workers even if the caller's placement is wrong.
sched_step() {   # $1=node $2=ngpu $3=ncpu $4=logfile $5...=cmd
    local node="$1" ngpu="$2" ncpu="$3" log="$4"; shift 4
    # NOT ids=$(sched_take_gpus ...): that is a subshell and the pool would never
    # shrink. See the note on sched_take_gpus in sched-lib.sh.
    sched_take_gpus "$node" "$ngpu" || return 1
    local ids="$SCHED_TAKEN"
    _pbs_bg "$node" "$log" \
        "export CUDA_VISIBLE_DEVICES=$ids OMP_NUM_THREADS=$ncpu" "$@"
    return 0
}

# The gateway is a pure proxy in front of workers that are already listening, so
# it gets no GPU at all -- and is told so explicitly. An empty CUDA_VISIBLE_DEVICES
# means "no devices" to every CUDA runtime; leaving it unset would let the proxy
# initialise a context on a worker's device and cost that worker memory.
sched_step_nogpu() {   # $1=node $2=ncpu $3=logfile $4...=cmd
    local node="$1" ncpu="$2" log="$3"; shift 3
    _pbs_bg "$node" "$log" \
        "export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$ncpu" "$@"
    return 0
}

# One place that knows how to put a backgrounded command on a node with a log.
# The command is passed as an argv array and re-quoted with printf %q, because a
# model path with a space in it must not become two arguments after crossing into
# a remote shell.
_pbs_bg() {   # $1=node $2=logfile $3=env-prelude $4...=cmd
    local node="$1" log="$2" prelude="$3"; shift 3
    local q; q="$(printf '%q ' "$@")"
    _pbs_remote_bg "$node" "$prelude; exec $q" "$log"
    SCHED_PIDS+=($!)
}

_pbs_remote_bg() {   # $1=node $2=snippet $3=logfile
    local node="$1" snippet="$2" log="$3"
    case "$PBS_STEP_VIA" in
        pbsdsh) "$PBS_BIN/pbsdsh" -n "$(_pbs_index_of "$node")" -- \
                    /bin/bash -c "$snippet" > "$log" 2>&1 & ;;
        ssh)    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node" \
                    "bash -c $(printf '%q' "$snippet")" > "$log" 2>&1 & ;;
        mpirun) "$PBS_MPIRUN" -np 1 --host "$node" --map-by node \
                    /bin/bash -c "$snippet" > "$log" 2>&1 & ;;
    esac
}

# Synchronous, no GPU, and told so: an empty CUDA_VISIBLE_DEVICES means "no
# devices" to every CUDA runtime, where leaving it unset would let the benchmark
# client initialise a context on a worker's GPU and take memory from the thing it
# is measuring.
sched_run() {   # $1=node $2=ncpu $3...=cmd
    local node="$1" ncpu="$2"; shift 2
    local q; q="$(printf '%q ' "$@")"
    _pbs_remote "$node" "export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$ncpu; exec $q"
}

# No -N equivalent: PBS has no "one task per node" broadcast, so this is a loop.
# Serial rather than parallel on purpose -- it is used for the hardware census,
# where interleaved output from several nodes is worse than waiting a second.
sched_run_all() {   # $@=cmd
    local node q rc=0
    q="$(printf '%q ' "$@")"
    for node in $(sched_nodes); do
        _pbs_remote "$node" "exec $q" || rc=$?
    done
    return $rc
}

# KILLING THE SSH CLIENT DOES NOT KILL THE WORKER, and that cost three cells of job
# 214760. Slurm's srun forwards a signal to its step; ssh does not -- the remote
# process group outlives the local client, so pd_teardown's kill/wait returned while
# 77,562 MiB of HBM was still held 180 seconds later and the next cell's engine
# refused to start.
#
# An earlier version of this function pkill'd by engine name. That is worse than
# nothing on shared nodes: `pkill -f sglang` would kill workers belonging to another
# of my own jobs on the same node, and the symptom would be a healthy run collapsing
# for no visible reason.
#
# The port is the precise handle. Every worker in an allocation serves on its own
# port, the campaign gives each cell its own range, and no other job can have ours.
# So: kill exactly the processes whose command line contains our ports, on our nodes.
sched_cleanup() {
    local -a ports=()
    (( ${#PD_PORT[@]} )) && ports+=("${PD_PORT[@]}")
    [[ -n "${ROUTER_PORT:-}" ]] && ports+=("$ROUTER_PORT")
    # And the router's Prometheus listener, which is a SECOND port on the same
    # process. Killing by --port only matches argv, and the exporter's port never
    # appears there unless we pass it -- which sglang.sh now does, so this entry
    # finds it. Job 215941's second cell was lost to exactly this.
    [[ -n "${ROUTER_PORT:-}" ]] && ports+=("$(( ROUTER_PORT + 21000 ))")
    (( ${#ports[@]} )) || return 0
    local -a nodes=()
    if (( ${#PD_NODE[@]} )); then
        mapfile -t nodes < <(printf '%s\n' "${PD_NODE[@]}" | awk '!seen[$0]++')
    else
        mapfile -t nodes < <(sched_nodes)
    fi
    local n pt
    for n in "${nodes[@]}"; do
        for pt in "${ports[@]}"; do
            # --port <pt> as a whole word, so port 3010 never matches 30100. TERM
            # first; SIGKILL after a grace period, because a worker that is mid-CUDA
            # teardown can hold HBM through a TERM it never processes.
            _pbs_remote "$n" "pkill -u \$(id -u) -f -- '--port +${pt}( |\$)' 2>/dev/null; \
                              sleep 3; \
                              pkill -9 -u \$(id -u) -f -- '--port +${pt}( |\$)' 2>/dev/null; \
                              true" >/dev/null 2>&1
        done
    done
    return 0
}

# Fail in ten seconds instead of hanging for seventeen minutes.
#
# a2ap-dgx038 accepts a job, reports state R, and then answers no launcher at all:
# pbsdsh, ssh and pbs_tmrsh every one time out against it while ping is 0.3 ms.
# Three jobs were lost to it. Under Slurm this class of node is drained and you
# never see it; here it is handed out, and on a metered machine a worker that never
# starts still bills for the walltime it holds.
#
# So every node is probed before anything is deployed, and an unreachable one is a
# hard error naming the node -- which is also the message to put in a ticket.
sched_preflight() {   # $@ = nodes
    local node bad=() t="${PBS_PREFLIGHT_TIMEOUT:-10}"
    for node in "$@"; do
        if timeout -k 2 "$t" ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
                -o ConnectTimeout=6 "$node" true >/dev/null 2>&1; then
            echo "  ok  $node answers"
        else
            echo "  x   $node does not answer a launcher in ${t}s" >&2
            bad+=("$node")
        fi
    done
    if (( ${#bad[@]} )); then
        echo "x unreachable node(s): ${bad[*]}" >&2
        echo "  PBS gave us a node that accepts jobs and runs nothing. Resubmit --" >&2
        echo "  do not wait, the walltime is billed either way." >&2
        return 1
    fi
    return 0
}
