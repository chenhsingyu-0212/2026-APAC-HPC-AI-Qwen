#!/bin/bash
# Shared PD deployment: parse a topology spec, place workers on GPUs, launch,
# wait for health, start the router, tear down.
#
# Sourced by run/submit.sbatch (the scored run) and run/sweep-pd.sbatch (the
# P:D ratio sweep) so both place workers the same way. A ratio that wins the
# sweep is then one PD_SPEC away from a scored run, instead of an edit.
#
# The spec is <prefill-tp>x<count>:<decode-tp>x<count>:
#
#   2x4:4x2   4 prefill at TP2 + 2 decode at TP4   -> 8 + 8
#   4x2:4x2   2 prefill at TP4 + 2 decode at TP4   -> 8 + 8
#   4x1:4x3   1 prefill at TP4 + 3 decode at TP4   -> 4 + 12
#
# MINIMUM TP IS SET BY THE WEIGHTS, NOT BY TASTE. 221 GB of FP8 at TP2 is 110.5 GB
# per GPU: it fits an H200's 141 GB with 30.5 GB to spare and does NOT fit an H100's
# 80 GB at all. So 2x4:4x2 -- which an earlier version of this comment called "the
# verified split" -- was verified on Firmus H200 and cannot start on NSCC or on
# NYCU's defq. On H100 every worker is TP4 (55.2 GB/GPU, 24.8 left) or TP8.
# See docs/campaign-16gpu.md for the shape space that leaves.
#
# Workers are placed by filling one node before spilling to the next, and no
# worker ever straddles a node -- tensor parallelism across an InfiniBand hop
# would cost far more than any ratio gains back.

# The scheduler is a plug-in, not an assumption. NYCU and Firmus are Slurm, NSCC
# is PBS Pro, and the difference is not command names -- PBS has no per-step GPU
# allocator, so the isolation Slurm gave us for free is code there. See
# run/sched-lib.sh for the seam and run/sched/pbs.sh for why UUIDs.
_PD_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=run/sched-lib.sh
source "$_PD_LIB_DIR/sched-lib.sh" || return 1
sched_load || return 1

# ---------------------------------------------------------------- state
PD_P_TP=0; PD_P_N=0; PD_D_TP=0; PD_D_N=0
PD_ROLE=(); PD_NODE=(); PD_GPUS=(); PD_PORT=(); PD_BOOT=(); PD_TP=(); PD_IP=()
PD_PIDS=()

# Set by the caller before pd_launch:
#   PD_WORK              a directory on shared storage. The launcher script, the
#                        per-worker $HOME directories and the worker logs all
#                        live here, and it is bind-mounted into every container.
#   PD_IMAGE             the serving container image
#   PD_MOUNTS            base mounts, default /scratch:/scratch
#   PD_CPUS_PER_WORKER   default 16, as in the reference scripts
#   PD_ROUTER_CPUS       default 2, likewise

pd_die() { echo "x $*" >&2; return 1; }
pd_say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- parse
pd_parse() {   # $1=spec
    local spec="$1"
    [[ "$spec" =~ ^([0-9]+)x([0-9]+):([0-9]+)x([0-9]+)$ ]] \
        || { pd_die "bad PD spec '$spec' -- want <ptp>x<pn>:<dtp>x<dn>, e.g. 2x4:4x2"; return 1; }
    PD_P_TP="${BASH_REMATCH[1]}"; PD_P_N="${BASH_REMATCH[2]}"
    PD_D_TP="${BASH_REMATCH[3]}"; PD_D_N="${BASH_REMATCH[4]}"
    (( PD_P_N > 0 && PD_D_N > 0 )) \
        || { pd_die "PD disaggregation needs at least one worker of each role: '$spec'"; return 1; }
    # a tp of 0 places a worker on no GPUs at all, and every later check passes
    (( PD_P_TP > 0 && PD_D_TP > 0 )) \
        || { pd_die "tensor-parallel degree must be at least 1: '$spec'"; return 1; }
    return 0
}

pd_gpus_used() { echo $(( PD_P_TP * PD_P_N + PD_D_TP * PD_D_N )); }
pd_label()     { echo "${PD_P_TP}x${PD_P_N}:${PD_D_TP}x${PD_D_N}"; }
pd_split()     { echo "$(( PD_P_TP * PD_P_N ))+$(( PD_D_TP * PD_D_N ))"; }

# ---------------------------------------------------------------- placement
# Fill each node before spilling to the next, prefill first. A worker whose TP
# does not fit in what is left of a node moves on whole rather than splitting.
pd_place() {   # $1=gpus-per-node  $2...=node names
    local per_node="$1"; shift
    local nodes=("$@")
    PD_ROLE=(); PD_NODE=(); PD_GPUS=(); PD_PORT=(); PD_BOOT=(); PD_TP=(); PD_IP=()

    local ni=0 used=0 i
    # every one of these is local to the inner function: bash's dynamic scoping
    # means a loop variable named `i` in here would otherwise clobber the outer
    # loop's `i` and silently place one worker per role
    _place_one() {   # $1=role $2=tp $3=port $4=boot
        local k gpus first
        while (( ni < ${#nodes[@]} )); do
            if (( used + $2 <= per_node )); then
                gpus=""; first=$used
                for (( k = 0; k < $2; k++ )); do gpus+="${gpus:+,}$(( first + k ))"; done
                PD_ROLE+=("$1"); PD_NODE+=("${nodes[$ni]}"); PD_GPUS+=("$gpus")
                PD_PORT+=("$3");  PD_BOOT+=("$4");  PD_TP+=("$2")
                used=$(( used + $2 ))
                return 0
            fi
            ni=$(( ni + 1 )); used=0
        done
        pd_die "cannot place a tp=$2 worker: only ${#nodes[@]} node(s) of $per_node GPUs"
        return 1
    }

    for (( i = 0; i < PD_P_N; i++ )); do
        _place_one prefill "$PD_P_TP" $(( ${PD_PORT_BASE:-30000} + 100 + i )) \
                                     $(( ${PD_BOOT_BASE:-9000} + i )) || return 1
    done
    for (( i = 0; i < PD_D_N; i++ )); do
        _place_one decode "$PD_D_TP" $(( ${PD_PORT_BASE:-30000} + 200 + i )) "" || return 1
    done
    return 0
}

# The gpus column is what the packing reserved, not an assignment. Under Slurm each
# worker is its own step with --gres=gpu:<tp> and the allocator picks the devices;
# under PBS there is no such allocator and sched_take_gpus hands out a slice of the
# chunk's GPU UUID list instead. Either way these indices are bookkeeping, not
# something a worker is ever told.
pd_print_placement() {
    local i
    # spec-gpus is the LOGICAL layout the shape asked for. The devices a worker is
    # actually handed are printed by pd_launch, per worker, as "ids:".
    printf '  %-8s %-14s %-5s %-10s %-7s %s\n' role node tp spec-gpus port bootstrap
    for i in "${!PD_ROLE[@]}"; do
        printf '  %-8s %-14s %-5s %-10s %-7s %s\n' \
               "${PD_ROLE[$i]}" "${PD_NODE[$i]}" "${PD_TP[$i]}" "${PD_GPUS[$i]}" \
               "${PD_PORT[$i]}" "${PD_BOOT[$i]:--}"
    done
}

# ---------------------------------------------------------------- addressing
# Resolve to IPv4 and address everything by it, as the reference scripts do.
# A hostname that resolves on the login node need not resolve inside the
# container, and a router pointed at a name it cannot resolve fails in a way
# that reads as a routing bug rather than as DNS.
pd_ip_of() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }

pd_resolve() {
    local i ip
    PD_IP=()
    for i in "${!PD_NODE[@]}"; do
        ip="$(pd_ip_of "${PD_NODE[$i]}")"
        [[ -n "$ip" ]] || { pd_die "cannot resolve ${PD_NODE[$i]} to IPv4"; return 1; }
        PD_IP+=("$ip")
    done
    return 0
}

pd_worker_name() { echo "${PD_ROLE[$1]}-${PD_PORT[$1]}"; }
pd_worker_url()  { echo "http://${PD_IP[$1]}:${PD_PORT[$1]}"; }

# HOW THIS SITE SPELLS A GPU REQUEST. Firmus takes --gres=gpu:4; NYCU has a
# job_submit plugin that rejects that form outright and demands the type be
# named, --gres=gpu:gpu:4. A hardcoded spelling here means every worker step on
# one of the two clusters is refused, so the format comes from the site file
# (run/sites/*.sh sets SITE_GRES_FMT; PD_GRES_FMT overrides it for one run).
pd_gres() { printf "${PD_GRES_FMT:-${SITE_GRES_FMT:-gpu:%d}}" "${1:-1}"; }

# Everything that builds a URL goes through this first. Without it an unresolved
# node yields "http://:30100" -- a string that looks like an address, is accepted
# by every caller, and fails as a connection error much later.
pd_resolved() {
    (( ${#PD_IP[@]} == ${#PD_NODE[@]} && ${#PD_IP[@]} > 0 )) \
        || { pd_die "addresses not resolved -- call pd_resolve first"; return 1; }
    return 0
}

# Each container gets its own writable $HOME. Four workers sharing one would
# race on everything that writes there on import, and PD_WORK is mounted at its
# own path so the launcher script is readable from inside.
pd_mounts() {   # $1=worker name
    local home="${PD_WORK:?PD_WORK is not set}/home/$1"
    mkdir -p "$home"
    echo "${PD_MOUNTS:-/scratch:/scratch},${home}:${HOME},${PD_WORK}:${PD_WORK}"
}

# ---------------------------------------------------------------- launcher
# ONE WORKER PER CONTAINER, one Slurm step each -- the shape the organisers'
# /scratch/reference-scripts use and the only shape verified on this hardware.
#
# The alternative, one container per node with the workers backgrounded inside
# it and split by CUDA_VISIBLE_DEVICES, is fewer moving parts and was what this
# file did until 2026-09-16. It was changed because with a dozen jobs queued for
# two nodes, a job that fails costs an hour in the queue rather than its own 17
# minutes, and "fewer moving parts" is not worth as much as "known to work".
#
# What changes with it:
#   * no CUDA_VISIBLE_DEVICES. Slurm gives each step --gres=gpu:<tp> and the
#     step sees exactly those, so the placement's gpu list is a record of the
#     packing, not an assignment.
#   * --exact, because concurrent steps in one allocation otherwise each try to
#     take the node's whole share of CPUs and the second one blocks forever.
#   * a writable $HOME per worker, and the log redirected by the caller rather
#     than by the worker.
pd_write_launcher() {   # $1=path
    {
        cat <<'HEAD_EOF'
#!/bin/bash
# worker.sh <role> <port> <command...>
# One worker. Runs inside its own container, as its own Slurm step.
#
# This wrapper is engine-neutral: it sets up the node environment every engine
# needs, then execs whatever command the engine backend supplied. The engine
# decides what to run; the wrapper decides what the process runs inside.
set -uo pipefail
role="$1"; port="$2"; shift 2

# NIXL/UCX needs the right HCA for cross-node KV transfer, and the two nodes do
# not have to enumerate their devices the same way, so detect per node.
#
# VERIFIED 2026-09-16 against /scratch/reference-scripts/{04,05,06,07,08,09}.
# Three things it does that a written-from-prose version did not, each of which
# fails silently:
#   * link_layer == InfiniBand. Without it an Ethernet/RoCE HCA on the same host
#     lands in UCX_NET_DEVICES and the KV cache goes out over the wrong fabric.
#   * port 1 only, state matched EXACTLY as "4: ACTIVE" rather than a substring.
#   * FATAL when nothing matches, rather than letting UCX pick.
#
# MEASURED 2026-09-17 (jobs 17593, 17603): g1 and g4 have TEN active ports, not
# the eight the rules describe. Eight are 400 Gb/s NDR mapping one-to-one onto
# the GPUs; mlx5_1 and mlx5_2 are 200 Gb/s HDR beside GPU0, almost certainly
# storage. PD_IB_MIN_RATE=400 keeps only the NDR fabric.
detect_ib() {
    local d rate name
    for d in /sys/class/infiniband/*; do
        [ "$(cat "$d/ports/1/link_layer" 2>/dev/null)" = InfiniBand ] || continue
        [ "$(cat "$d/ports/1/state" 2>/dev/null)" = "4: ACTIVE" ] || continue
        name="$(basename "$d")"
        rate="$(cat "$d/ports/1/rate" 2>/dev/null | awk '{print int($1)}')"
        if [ -n "${PD_IB_MIN_RATE:-}" ] && [ "${rate:-0}" -lt "$PD_IB_MIN_RATE" ]; then
            echo "[$(hostname)] ib: skipping $name at ${rate} Gb/s (< ${PD_IB_MIN_RATE})" >&2
            continue
        fi
        echo "$name:1"
    done | paste -sd,
}
# Report the rates of the devices we ACTUALLY selected, which means applying the
# same link_layer and PD_IB_MIN_RATE filters detect_ib() does. Listing every
# active port instead made NYCU look like a mixed-rate fabric -- mlx5_2 and
# mlx5_8 are 100 Gb/s ETHERNET there, next to ten 400 Gb/s NDR InfiniBand ports,
# and detect_ib() had already excluded them. A warning naming UCX_NET_DEVICES has
# to be computed from UCX_NET_DEVICES.
ib_rates() {   # $1=comma-separated "dev:port" list, as detect_ib prints it
    local spec="${1:-}" name
    for name in $(echo "$spec" | tr ',' ' '); do
        name="${name%%:*}"
        [ -n "$name" ] || continue
        echo "$name=$(cat "/sys/class/infiniband/$name/ports/1/rate" 2>/dev/null | awk "{print \$1}")"
    done | paste -sd' '
}
IB="$(detect_ib)"
[ -n "$IB" ] || { echo "[$(hostname)] FATAL: no active IB" >&2; exit 1; }
export UCX_NET_DEVICES="$IB"
# From the reference scripts, all six of them. Undocumented there; carried over
# because it is what was validated on this hardware.
export UCX_POSIX_USE_PROC_LINK=n
echo "[$(hostname)] ib rates: $(ib_rates "$IB")"
_n="$(ib_rates "$IB" | tr ' ' '\n' | sed 's/.*=//' | sort -u | grep -c .)"
[ "${_n:-1}" -gt 1 ] && \
    echo "[$(hostname)] ! mixed IB link rates in UCX_NET_DEVICES -- checklist 9, and PD_IB_MIN_RATE"

# File descriptors. Slurm hands the step whatever slurmd's default is -- and
# these scripts carry #SBATCH --propagate=NONE, so nothing is inherited from the
# submitting shell either. On this cluster that default is 1024.
#
# MEASURED 2026-09-25: that is what breaks TensorRT-LLM. Its Ray executor starts
# a local Ray instance, the raylet cannot accept the connections it needs, and
# the worker dies 63 seconds later with
#   Failed to register worker to Raylet: IOError: ... End of file
# Raising nofile to 65535 (the hard limit here is 1048576) makes ray.init()
# succeed and run a remote task. Ray's own guidance is 65535.
#
# Engine-neutral on purpose: no serving stack is happier with 1024 descriptors.
if [ "$(ulimit -n)" -lt 65535 ] 2>/dev/null; then
    ulimit -n 65535 2>/dev/null \
        && echo "[$(hostname)] nofile 1024 -> $(ulimit -n)" \
        || echo "[$(hostname)] ! could not raise nofile past $(ulimit -n) (hard limit $(ulimit -Hn))"
fi

# Per-worker compile caches on node-local disk. Without them two workers on one
# node race on the same Triton and Inductor directories.
export TRITON_CACHE_DIR="/tmp/pdw-$role-$port/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/pdw-$role-$port/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
HEAD_EOF

        # Engine-supplied environment, baked in at generation time so it travels
        # with the script rather than depending on srun's environment surviving.
        if declare -F eng_pd_env >/dev/null 2>&1; then
            echo ""
            echo "# --- from the engine backend ---"
            # QUOTE THE VALUE. `export K=V` is re-parsed by the shell that runs this
            # launcher, so an unquoted value is subject to brace expansion, globbing and
            # word-splitting. NSCC job 216211 passed
            #   SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS={"num_threads":"16","split_batch_size":"256"}
            # and bash turned it into {num_threads:16,split_batch_size:256} -- every quote
            # eaten by brace expansion -- and the engine died on
            #   json.decoder.JSONDecodeError: Expecting value: line 1 column 1
            # The value reached worker.sh intact; it was the shell reading worker.sh that
            # destroyed it, which is why grepping the file looked fine.
            eng_pd_env | while IFS= read -r _l; do
                printf 'export %s=%q\n' "${_l%%=*}" "${_l#*=}"
            done
        fi

        cat <<'TAIL_EOF'

echo "[$(hostname)] $role port=$port gpus=${CUDA_VISIBLE_DEVICES:-<slurm>}"
echo "[$(hostname)] nofile=$(ulimit -n) (hard $(ulimit -Hn))  shm=$(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2}')"
echo "[$(hostname)] exec: $*"

# Copy the engine's own logs somewhere that outlives this step.
#
# Job 19368 tried to collect them afterwards from a separate srun and got
# nothing: a later step does not reliably see the earlier step's /tmp. Anything
# worth reading has to be copied out from INSIDE the step that produced it.
#
# This is why the wrapper no longer execs: exec replaces the shell, and a
# replaced shell cannot run a trap.
save_engine_logs() {
    local d out
    out="${PD_WORK:-/tmp}/enginelog-$role-$port"
    # Job 19380: the first version only looked under RAY_TMPDIR and found
    # nothing. TensorRT-LLM calls ray.init() WITHOUT _temp_dir, so Ray ignores
    # RAY_TMPDIR for its session directory and uses its default /tmp/ray --
    # which is not where the earlier login-node test had put it, because that
    # test passed _temp_dir explicitly. Search everywhere Ray might have gone.
    mkdir -p "$out" 2>/dev/null || return 0
    local found=0
    # NEWEST session only. Job 19386 copied a raylet.err dated two days earlier:
    # /tmp/ray accumulates a session directory per attempt, and the glob took
    # whichever came first alphabetically. Stale evidence is worse than none --
    # it sends you after a bug that was already fixed.
    local newest
    newest="$(ls -1dt "${RAY_TMPDIR:-/nonexistent}"/session_*/logs \
                      /tmp/ray-$role-$port/session_*/logs \
                      /tmp/ray/session_*/logs \
                      "${TMPDIR:-/tmp}"/ray/session_*/logs 2>/dev/null | head -1)"
    for d in $newest; do
        [ -d "$d" ] || continue
        echo "$d" > "$out/SESSION-DIR.txt" 2>/dev/null
        cp -f "$d"/raylet.err "$d"/raylet.out "$d"/gcs_server.err \
              "$d"/python-core-worker-*.log "$d"/python-core-driver-*.log \
              "$out"/ 2>/dev/null
        found=1
    done
    # If none of the guesses hit, record what IS under /tmp so the next attempt
    # does not have to guess either.
    if [ "$found" = 0 ]; then
        { echo "no ray session dir found. RAY_TMPDIR=${RAY_TMPDIR:-unset}"
          echo "--- /tmp ---"; ls -la /tmp 2>/dev/null | head -40
          echo "--- find ray dirs ---"; find /tmp -maxdepth 3 -name "session_*" 2>/dev/null | head -10
        } > "$out/WHERE-ARE-THE-LOGS.txt" 2>/dev/null
    fi
    echo "[$(hostname)] engine logs -> $out (found=$found)"
}
trap save_engine_logs EXIT

"$@"
rc=$?
save_engine_logs
exit $rc
TAIL_EOF
    } > "$1"
    chmod +x "$1"
}

# ---------------------------------------------------------------- launch
# PD_IMAGE empty means BARE METAL, deliberately. TensorRT-LLM here is a pip venv
# on shared storage, not a container -- nvcr.io is unreachable from this cluster
# and /scratch/enroot-images is root-owned. The rules allow it: "container image
# plus build files, OR a self-built setup on shared storage".
# Pyxis flags for srun. PBS has no pyxis: NSCC enters a container with `enroot
# start`, which is a command PREFIX rather than launcher flags, so the PBS backend
# ignores what this prints and wraps the command itself. Both paths are behind the
# scheduler seam; neither is in this function.
pd_container_args() {   # prints srun container arguments, or nothing
    [[ -n "${PD_IMAGE:-}" ]] || return 0
    printf '%s\n' "--container-image=$PD_IMAGE" "--container-mounts=$(pd_mounts "$1")"
}

# ---------------------------------------------------------------- ports
# BIND IT BEFORE THE ENGINE DOES. Nano5 jobs 372421 and 372456 each lost their second
# cell to one line:
#
#   Server error: [Errno 98] error while attempting to bind on address
#                 ('0.0.0.0', 9100): address already in use
#
# PD_BOOT_BASE was 9000 + cell*100, so the second cell's first prefill worker asked
# for 9100 -- which on that cluster is a Prometheus IPMI exporter, running on every
# node. That worker came up with no bootstrap server, both decode workers answered
# every request with "Error fetching prefill server info from bootstrap", and the
# deployment served 36 requests on one pair and 215 on the other instead of 1,362
# each. R fell from 14.3 to 0.53 and two scored points came back at pass 0.66.
#
# Both jobs looked like the flag under test had failed. It had not: the second cell of
# BOTH produced R = 0.53 / 0.59 with n=36 and n=215, byte for byte, with different
# flags. The experiments were never run.
#
# This is the fourth port-enumeration collision in this project, after the router's
# Prometheus port landing on 29000. Picking a less popular base is not the fix,
# because the next cluster has its own occupants -- testing the bind is.
pd_assert_ports_free() {
    local i node port role rc=0 taken=""
    for i in "${!PD_ROLE[@]}"; do
        node="${PD_NODE[$i]}"; role="${PD_ROLE[$i]}"
        for port in "${PD_PORT[$i]}" "${PD_BOOT[$i]}"; do
            [[ -n "$port" && "$port" != 0 && "$port" != - ]] || continue
            # python, not nc: nc is absent on these images and ss needs the listener
            # to already exist. A failed bind is the exact condition the engine hits.
            if ! sched_run "$node" 1 "$(eng_python 2>/dev/null || echo python3)" -c "
import socket,sys
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
try: s.bind(('0.0.0.0',$port))
except OSError as e: sys.exit(1)
s.close()" >/dev/null 2>&1; then
                taken+=" $node:$port($role)"; rc=1
            fi
        done
    done
    if (( rc )); then
        pd_die "ports already in use --$taken"
        pd_say "  a bootstrap port that cannot bind leaves the worker with no bootstrap"
        pd_say "  server, and the decode side then fails every request with"
        pd_say "  'Error fetching prefill server info from bootstrap' -- see 372421/372456"
        return 1
    fi
    return 0
}

pd_launch() {   # $1=launcher path
    local launcher="$1" i name
    [[ -n "${PD_WORK:-}"  ]] || { pd_die "PD_WORK is not set";  return 1; }
    [[ -n "${PD_IMAGE:-}" ]] || pd_say "  no container image -- running bare metal"
    pd_resolve || return 1
    PD_PIDS=()

    # Refuse to launch into GPUs somebody is still holding. The previous cell's
    # teardown is supposed to have cleared them, but when it has not, an engine gets
    # 160 s into loading and then dies on "The memory capacity is unbalanced", which
    # names neither the cause nor the holder (jobs 214830 and 214966, one cell each).
    # pd_wait_gpus_free reaps as it waits, so this is also the second chance to
    # recover rather than only to report.
    if ! pd_wait_gpus_free "${PD_GPU_FREE_PRELAUNCH_TIMEOUT:-150}"; then
        pd_die "GPUs are still held -- not launching into them"
        return 1
    fi

    # Every port this cell needs, on the node that needs it, before a checkpoint load.
    if ! pd_assert_ports_free; then
        return 1
    fi

    declare -F eng_pd_worker_cmd >/dev/null 2>&1 \
        || { pd_die "the engine backend does not implement eng_pd_worker_cmd"; return 1; }

    for i in "${!PD_ROLE[@]}"; do
        name="$(pd_worker_name "$i")"
        local -a wcmd=()
        mapfile -t wcmd < <(eng_pd_worker_cmd "$i")
        (( ${#wcmd[@]} )) || { pd_die "eng_pd_worker_cmd produced nothing for $name"; return 1; }

        mapfile -t SCHED_CONTAINER_ARGS < <(pd_container_args "$name")
        pd_say "  $name -> ${PD_NODE[$i]} (${PD_IP[$i]}) gpu:${PD_TP[$i]} cpu:${PD_CPUS_PER_WORKER:-16}"
        sched_step "${PD_NODE[$i]}" "${PD_TP[$i]}" "${PD_CPUS_PER_WORKER:-16}" \
                   "$PD_WORK/$name.log" \
                   bash "$launcher" "${PD_ROLE[$i]}" "${PD_PORT[$i]}" "${wcmd[@]}" || return 1
        # What the worker actually got, not what the spec asked for. The banner above
        # prints the spec's logical devices, and for job 214830 it said 0,1,2,3 while
        # every worker received four UUIDs -- which is how a disabled custom
        # all-reduce went unnoticed for the whole campaign. Empty under Slurm's
        # --exact, where the allocator and not this script picks the devices.
        [[ -n "${SCHED_TAKEN:-}" ]] && pd_say "     ids: $SCHED_TAKEN"
        PD_PIDS+=("${SCHED_PIDS[-1]}")
    done
    pd_say "launched ${#PD_PIDS[@]} worker step(s)"
    return 0
}

# The gateway needs no GPU, so --overlap rather than --exact, as in the
# reference. Both SGLang's router and TensorRT-LLM's disaggregated server are
# pure proxies in front of workers that are already listening, so the same
# launch shape serves both -- which is what the engine seam was for.
pd_launch_gateway() {   # $1=port  -- appends its pid to PD_PIDS
    local port="$1"
    pd_resolved || return 1
    declare -F eng_pd_gateway_cmd >/dev/null 2>&1 \
        || { pd_die "the engine backend does not implement eng_pd_gateway_cmd"; return 1; }

    local -a gcmd=()
    mapfile -t gcmd < <(eng_pd_gateway_cmd "$port")
    (( ${#gcmd[@]} )) || { pd_die "eng_pd_gateway_cmd produced nothing"; return 1; }

    pd_say "  gateway -> ${PD_NODE[0]} (${PD_IP[0]}:$port)"
    pd_say "  ${gcmd[*]}"
    mapfile -t SCHED_CONTAINER_ARGS < <(pd_container_args gateway)
    sched_step_nogpu "${PD_NODE[0]}" "${PD_ROUTER_CPUS:-2}" "$PD_WORK/gateway.log" \
                     "${gcmd[@]}" || return 1
    PD_PIDS+=("${SCHED_PIDS[-1]}")
    return 0
}
# The old name, kept so nothing breaks mid-refactor.
pd_launch_router() { pd_launch_gateway "$@"; }

pd_gateway() { pd_resolved || return 1; echo "${PD_IP[0]}:${1:-8000}"; }

# ---------------------------------------------------------------- health
# Give up the moment the step dies. A worker that exited thirty seconds in
# otherwise costs the full timeout, and in a 17-minute job that is the job.
pd_wait_healthy() {   # $1=url $2=label $3=timeout [$4=pid] [$5=log]
    local url="$1" label="$2" timeout="$3" pid="${4:-}" log="${5:-}" waited=0
    while (( waited < timeout )); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "x $label exited after ${waited}s, before becoming healthy" >&2
            [[ -r "$log" ]] && tail -30 "$log" >&2
            return 1
        fi
        if curl -sf --max-time 5 "$url/health" >/dev/null 2>&1; then
            pd_say "  ok $label healthy after ${waited}s"; return 0
        fi
        sleep 5; waited=$(( waited + 5 ))
    done
    echo "x $label never became healthy in ${timeout}s" >&2
    [[ -r "$log" ]] && tail -30 "$log" >&2
    return 1
}

pd_wait_all() {   # $1=timeout
    local i name
    for i in "${!PD_ROLE[@]}"; do
        name="$(pd_worker_name "$i")"
        pd_wait_healthy "$(pd_worker_url "$i")" "$name" "$1" \
                        "${PD_PIDS[$i]:-}" "$PD_WORK/$name.log" || return 1
    done
    return 0
}

# Queue depth and KV usage are engine-internal: the gateway does not have them,
# and prefill and decode differ, so every worker is polled.
#
# The PATH is the engine's business. TensorRT-LLM serves iteration stats as JSON
# at /metrics and Prometheus at /prometheus/metrics -- pointed at the wrong one,
# observe.sh writes an empty CSV and never errors.
# /health means the process is listening. It does NOT mean the gateway has its
# workers: sglang_router answers /health 200 while it still reports
# "Workers initialized: 0 total, 0 healthy" and registers them in the background
# a second later. Job 17743 asked /v1/models once at that moment, got 503, and
# died with six healthy workers and a 235B checkpoint already loaded.
#
# So poll for the thing we actually need -- a servable model id -- rather than
# for a liveness signal that arrives earlier.
pd_wait_models() {   # $1=base url  $2=timeout  [$3=pid] [$4=log]  -- prints the id
    local url="$1" timeout="${2:-120}" pid="${3:-}" log="${4:-}" waited=0 id=""
    while (( waited < timeout )); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "x the gateway exited while waiting for $(pd_models_path)" >&2
            [[ -r "$log" ]] && tail -20 "$log" >&2
            return 1
        fi
        id="$(curl -sf --max-time 5 "${url}$(pd_models_path)" 2>/dev/null \
              | python3 -c 'import sys,json;d=json.load(sys.stdin)["data"];print(d[0]["id"] if d else "")' 2>/dev/null)"
        if [[ -n "$id" ]]; then
            pd_say "  ok gateway serving '$id' after ${waited}s" >&2
            echo "$id"; return 0
        fi
        sleep 3; waited=$(( waited + 3 ))
    done
    echo "x gateway healthy but no model id after ${timeout}s" >&2
    [[ -r "$log" ]] && tail -20 "$log" >&2
    return 1
}

pd_models_path() {
    declare -F eng_models_path >/dev/null 2>&1 && { eng_models_path; return; }
    echo "/v1/models"
}

pd_metrics_args() {
    pd_resolved || return 1
    local i path="/metrics"
    declare -F eng_metrics_path >/dev/null 2>&1 && path="$(eng_metrics_path)"
    for i in "${!PD_ROLE[@]}"; do
        echo "--metrics-url"
        echo "$(pd_worker_url "$i")${path}"
    done
}

# ---------------------------------------------------------------- prewarm
# Move the first-request cost OUT of the scored command.
#
# The rules fix the official command; they say nothing about what happens before
# it. Warming a server before benchmarking it is ordinary practice -- and the
# official command carries its own --warmup-duration 60 regardless.
#
# WHY, measured:
#   17682  --policy cache_aware (the default) sent all 40 concurrent requests to
#          ONE prefill worker. 129 requests got through in ~11 min; the nominal
#          60s warmup phase ran 503s.
#   17810  --prefill-policy round_robin spread them 11/1/1/4 -- and only 17
#          requests got through in ~9.5 min. SLOWER, not faster.
#
# The hypothesis that fits both: each worker pays a large ONE-TIME cost on its
# first request of a given shape (CUDA graph capture at ISL 4000). cache_aware
# made one worker pay it; round_robin made all four pay it, serially, inside the
# measurement window.
#
# If that is right, the fix is neither policy: it is paying that cost for every
# worker CONCURRENTLY, before the clock starts. This function measures how long
# that actually takes, which is the number the hypothesis lives or dies on.
#
# The prompt has to be the RIGHT SHAPE. A "hello" does not capture the graph for
# a 4000-token prefill, so this sends the real length.
pd_prewarm() {   # $1=gateway url  $2=model id  $3=requests  $4=isl  $5=osl  $6=timeout
    local url="$1" model="$2" n="${3:-8}" isl="${4:-4000}" osl="${5:-16}" timeout="${6:-600}"
    local started=$SECONDS prompt body i
    local -a pids=() rcs=()

    # ~4 characters per token is close enough for a warm-up; the point is the
    # shape of the work, not an exact token count.
    prompt="$(python3 -c "print('warm up the prefill graph. ' * $(( isl / 6 )))" 2>/dev/null)"
    [[ -n "$prompt" ]] || { pd_die "could not build a prewarm prompt"; return 1; }

    pd_say "prewarming: $n concurrent requests at ~${isl} input tokens"
    for (( i = 0; i < n; i++ )); do
        # STREAMING, exactly as the official command sends it. Jobs 17914 and
        # 18315 warmed with a plain non-streaming POST: all eight requests
        # reached the gateway, none was ever dispatched to a worker, and the
        # job died with six healthy workers and the 235B already resident.
        # 17682, which ran the official command with no warm-up at all, served
        # 129 requests through the same deployment and logged 241 decode
        # batches -- so the deployment was never the problem.
        #
        # Warm the path you are going to measure. A warm-up that exercises a
        # code path nobody benchmarks proves nothing and, here, hangs.
        body="$(python3 - "$model" "$prompt" "$osl" <<'PYJ'
import json,sys
print(json.dumps({"model":sys.argv[1],
                  "messages":[{"role":"user","content":sys.argv[2]}],
                  "max_tokens":int(sys.argv[3]),"temperature":0.0,"stream":True}))
PYJ
)"
        curl -sf --max-time "$timeout" "$url/v1/chat/completions" \
             -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
             -d "$body" -o /dev/null 2>/dev/null &
        pids+=($!)
    done

    local ok=0 fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}" 2>/dev/null; then ok=$(( ok + 1 )); else fail=$(( fail + 1 )); fi
    done
    local took=$(( SECONDS - started ))
    pd_say "  prewarm: $ok ok, $fail failed, ${took}s"
    PD_PREWARM_SECONDS="$took"

    if (( ok == 0 )); then
        pd_die "every prewarm request failed -- the deployment is not serving"
        return 1
    fi
    (( fail )) && pd_say "  ! $fail prewarm request(s) failed; continuing"
    return 0
}

# ---------------------------------------------------------------- time budget
# The hpcai partition caps a job at 17 minutes, and loading the 235B checkpoint
# spends 2-4 of them. Slurm's kill is a SIGKILL: a run that overruns leaves no
# artefacts at all, and with a dozen jobs queued for two nodes you do not get
# that slot back for an hour. So every script that is about to spend minutes
# asks first whether it has them.
pd_time_left() {   # seconds remaining in this allocation, or 999999 if unknown
    # Delegated: Slurm answers with squeue -o %L, PBS with Resource_List.walltime
    # minus resources_used.walltime. See run/sched/.
    if declare -F sched_time_left >/dev/null 2>&1; then
        sched_time_left
    else
        echo 999999
    fi
}

# Refuse to start something that cannot finish. Better a clear message now than
# a SIGKILL two minutes into a 210-second measurement.
pd_need_time() {   # $1=seconds needed  $2=what for (optional)
    local left what="${2:-the next step}"
    left="$(pd_time_left)"
    if (( left < $1 )); then
        echo "x ${left}s left in this allocation, $what needs $1s -- not starting it" >&2
        echo "  Deployment ate the budget. Submit again; the queue is the cost, not the run." >&2
        return 1
    fi
    pd_say "  ok ${left}s left, $what needs $1s"
    return 0
}

# Waiting for a port to be free is not the same as teardown returning: the worker
# process is gone but its listening socket can still be held, and the next cell's
# bind then fails outright rather than retrying.
pd_wait_port_free() {   # $1=port  $2=timeout seconds (default 90)
    local port="$1" timeout="${2:-90}" waited=0
    while (( waited < timeout )); do
        if command -v ss >/dev/null 2>&1; then
            ss -Hltn "sport = :$port" 2>/dev/null | grep -q . || { pd_say "  port $port free after ${waited}s"; return 0; }
        else
            (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null || { pd_say "  port $port free after ${waited}s"; return 0; }
            exec 3>&- 2>/dev/null
        fi
        sleep 3; waited=$(( waited + 3 ))
    done
    pd_say "  ! port $port still held after ${timeout}s"
    return 1
}

pd_teardown() {
    local p
    for p in "${PD_PIDS[@]:-}"; do
        [[ -n "$p" ]] || continue
        kill "$p" 2>/dev/null
        # wait on this job only. A bare `wait` waits for every background job the
        # caller owns -- including a telemetry sampler with no --duration, which
        # runs until it is stopped, so teardown would simply never return.
        wait "$p" 2>/dev/null
    done
    PD_PIDS=()
    # Kill what the local signal could not reach. On Slurm this is a no-op -- srun
    # forwards signals to its step -- but on PBS the workers are behind ssh and outlive
    # the client that started them.
    declare -F sched_cleanup >/dev/null 2>&1 && sched_cleanup
    # AND THEN WAIT FOR THE HBM. The step exiting is not the memory coming back:
    # four workers holding 221 GB across sixteen GPUs take tens of seconds to
    # release, and job 214750 lost THREE cells to starting before they had. SGLang
    # checks free memory against its own expectation at startup and refuses:
    #
    #   RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied
    #   by other processes. pre_model_load_memory=1.67, local_gpu_memory=2.37
    #
    # which reads like a hardware or placement fault and is really the previous
    # cell still exiting. Cheap to wait for, expensive to skip.
    #
    # Swallowed on purpose: teardown reports, it does not decide. pd_launch runs the
    # same wait and refuses to start, which is where a held GPU has to be fatal --
    # and this function is an EXIT trap in 20-profile.sbatch, where a non-zero last
    # command would change the job's status.
    pd_wait_gpus_free || true
}

# REFUSE A CELL THAT IS NOT MEASURING WHAT WE ASKED FOR. When DeepGEMM cannot get the
# workspace its warmup wants, it does not fail -- it says so and halves max_m:
#
#   Available memory 9.70GB is less than required memory 16.94GB for warmup,
#   reducing max_m to 8192 to avoid out of memory
#
# The server then comes up healthy and benchmarks happily with a prefill batch smaller
# than the flags requested. Job 215921 spent 1,471 SU producing a number for a
# configuration nobody chose, and the line was in the log the whole time. So when a
# run asks for a specific prefill batch, check that it got it.
#
# Returns 0 when nothing was reduced or the reduction still covers the request, 1
# otherwise, naming the file and the numbers.
pd_assert_prefill_batch() {   # $1=dir holding prefill-*.log  $2=requested batch tokens
    local dir="$1" want="${2:-}"
    [[ -n "$want" ]] || return 0           # nothing requested, nothing to assert
    local f got worst=""
    for f in "$dir"/prefill-*.log; do
        [[ -e "$f" ]] || continue
        got="$(grep -ahoE 'reducing max_m to [0-9]+' "$f" 2>/dev/null \
               | awk '{print $NF}' | sort -n | head -1)"
        [[ -n "$got" ]] || continue
        if (( got < want )); then
            echo "x $(basename "$f"): asked for a ${want}-token prefill batch, DeepGEMM reduced max_m to $got" >&2
            grep -ahoE 'Available memory [0-9.]+GB is less than required memory [0-9.]+GB' "$f" \
                | sort -u | head -2 | sed 's/^/    /' >&2
            worst="$got"
        fi
    done
    [[ -z "$worst" ]] || {
        echo "    this cell would measure a smaller batch than requested -- refusing" >&2
        return 1
    }
    return 0
}


# Kill whatever still holds HBM on a node, identified by the driver rather than by
# argv. sched_cleanup matches `--port <n>`, which finds the server SGLang launches but
# NOT the `sglang::scheduler_TP*` children it forks -- and those are the processes that
# actually hold the 19 GB per rank. Job 214830 lost its second cell to exactly this:
# 77,562 MiB survived a full port-based teardown and the next engine died with
# "The memory capacity is unbalanced ... local_gpu_memory=4.28".
#
# Only our own processes are touched: the PID must be ours by /proc ownership. PBS and
# Slurm both hand us whole nodes here (ngpus=8 of 8), so there is no other tenant, but
# the uid check means a shared node would still be safe.
pd_reap_gpu_procs() {   # $1=node  $2=signal (TERM|KILL)  -> prints the count
    local node="$1" sig="${2:-TERM}" out
    declare -F sched_run >/dev/null 2>&1 || return 0
    local script="${PD_LIB_DIR:-${BASH_SOURCE[0]%/*}}/reap-gpus.sh"
    [[ -r "$script" ]] || { pd_say "  ! cannot read $script -- not reaping" >&2; return 0; }
    # Stderr is kept: the previous version discarded it, which is why a reaper that
    # could not run at all looked exactly like a node with nothing to reap.
    out="$(sched_run "$node" 1 bash "$script" "$sig" 2>&1)"
    if [[ ! "$out" =~ ^[0-9]+$ ]]; then
        # stderr, not stdout: the caller reads this function's stdout as the count.
        pd_say "  ! reaper on ${node} did not report a count: ${out//$'\n'/ | }" >&2
        echo 0
        return 0
    fi
    echo "$out"
}

# Poll until the allocation's GPUs are actually idle again, or give up and say so.
# HEAD is where the campaign's own steps run; on a multi-node PD deployment every
# node is checked, because the cell that failed was the one whose workers were on
# the OTHER node.
pd_wait_gpus_free() {   # $1=timeout seconds (default 420)
    # 180 was not enough: job 214760 still held 77,562 MiB after it, because on PBS the
    # ssh client had been killed and the remote worker had not. sched_cleanup now kills
    # by port, so the wait should be short -- but it stays generous, because the cost of
    # waiting is seconds and the cost of not waiting was three cells.
    local timeout="${1:-${PD_GPU_FREE_TIMEOUT:-420}}"
    local -a nodes=()
    if (( ${#PD_NODE[@]} )); then
        mapfile -t nodes < <(printf '%s\n' "${PD_NODE[@]}" | awk '!seen[$0]++')
    elif declare -F sched_nodes >/dev/null 2>&1; then
        mapfile -t nodes < <(sched_nodes)
    fi
    (( ${#nodes[@]} )) || return 0
    declare -F sched_run >/dev/null 2>&1 || return 0

    local t0=$SECONDS used n worst killed
    local reaped_term=0 reaped_kill=0
    while (( SECONDS - t0 < timeout )); do
        worst=0
        for n in "${nodes[@]}"; do
            # MiB in use across the node's visible GPUs. A freshly idle H100 still
            # reports a few hundred MiB of driver overhead, so the threshold is not
            # zero -- 2 GB is far below a loaded worker's 55 GB per GPU and far above
            # anything idle.
            used="$(sched_run "$n" 1 nvidia-smi --query-gpu=memory.used \
                        --format=csv,noheader,nounits 2>/dev/null \
                    | awk '{if ($1+0 > m) m = $1+0} END{print m+0}')"
            (( ${used:-0} > worst )) && worst=${used:-0}
        done
        if (( worst <= ${PD_GPU_FREE_MIB:-2048} )); then
            (( SECONDS - t0 > 2 )) && pd_say "  GPUs free after $(( SECONDS - t0 ))s (max ${worst} MiB in use)"
            return 0
        fi
        # Politeness first, then the driver's own list. Each stage runs once.
        local elapsed=$(( SECONDS - t0 ))
        if (( reaped_term == 0 && elapsed >= ${PD_GPU_REAP_AFTER:-45} )); then
            reaped_term=1
            for n in "${nodes[@]}"; do
                killed="$(pd_reap_gpu_procs "$n" TERM)"
                (( ${killed:-0} > 0 )) && pd_say "  reaped ${killed} GPU process(es) on ${n} with TERM"
            done
        elif (( reaped_kill == 0 && elapsed >= ${PD_GPU_REAP_AFTER:-45} + 30 )); then
            reaped_kill=1
            for n in "${nodes[@]}"; do
                killed="$(pd_reap_gpu_procs "$n" KILL)"
                (( ${killed:-0} > 0 )) && pd_say "  reaped ${killed} GPU process(es) on ${n} with KILL"
            done
        fi
        sleep 5
    done
    pd_say "  ! GPUs still hold ${worst} MiB after ${timeout}s, reaper included"
    return 1
}
