#!/bin/bash
#SBATCH --job-name=qwen3-campaign
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#
# Every engine, every shape that fits, one allocation.
#
#   run/submit-site.sh run/10-campaign.sbatch                    # 8 GPUs, 1 node
#
# No --exclusive here on purpose. sbatch has no way to cancel an #SBATCH
# directive from the command line, and on NYCU --exclusive means "give me the
# node's entire 2 TB", which that site's association refuses outright
# (Reason=AssocMaxMemPerJob). run/submit-site.sh adds it where a site wants it
# and requests explicit memory where a site caps it.
#   ENGINES=sglang SPECS=solo8 run/submit-site.sh run/10-campaign.sbatch
#
# This exists because the two clusters have opposite constraints. Firmus caps a
# job at 17 minutes, so run/sweep-pd.sbatch deliberately does ONE topology per
# job and pays the checkpoint load once per job. NYCU has no cap at all, and
# getting an allocation there is the hard part -- two H200 nodes sat DRAINed for
# three days, and the H100 partition is capped at 15 of its 16 GPUs group-wide.
# So the economics invert: hold the allocation and sweep everything inside it.
#
# WHAT FITS IN 8 GPUs. The checkpoint is 221 GB, which sets a floor on tensor
# parallelism per instance before any KV cache exists:
#
#   TP2   110 GB/GPU   impossible on H100 80GB, ~30 GB left on H200 141GB
#   TP4    55 GB/GPU   ~25 GB left on H100, ~86 GB on H200
#   TP8    28 GB/GPU   comfortable on both
#
# So at 8 GPUs there are exactly two shapes, not a ratio grid: one TP8 instance,
# or PD with TP4 on each side. `2x4:4x2` and its neighbours need 16 GPUs to exist
# at all. That is why SPECS defaults to two entries and the interesting axes here
# are engine, shape, and concurrency rather than the P:D ratio.
#
# The solo shape is not filler. It is the baseline the published single-node
# numbers use, and without it "PD wins" is unfalsifiable -- you cannot tell a PD
# gain from an H200-versus-H100 gain.
#
# Exploratory, not scored: short windows, own seeds. What stays identical to the
# official command is everything that decides comparability -- the 4000/200
# shape, the goodput thresholds, --cache-bust first_turn_prefix, closed loop.

set -uo pipefail

# ============================================================ 0. where, and where am I
HERE=""
# PBS names its variables differently and starts the job in $HOME rather than the
# directory it was submitted from, so every "${SLURM_SUBMIT_DIR:-$PWD}" below would
# resolve to the wrong place. Map the two that matter onto the names this repo
# already reads, once and here, so nothing downstream has to know which scheduler
# it is running under.
_pbsid="${PBS_JOBID:-}"; _pbsid="${_pbsid%%.*}"
: "${SLURM_SUBMIT_DIR:=${PBS_O_WORKDIR:-}}"
: "${SLURM_JOB_ID:=$_pbsid}"
export SLURM_SUBMIT_DIR SLURM_JOB_ID

for _d in "${SLURM_SUBMIT_DIR:-}/run" "${SLURM_SUBMIT_DIR:-}" \
          "$(dirname "${BASH_SOURCE[0]}")" "$PWD/run" "$PWD"; do
    [[ -n "$_d" && -r "$_d/pd-lib.sh" ]] && { HERE="$(cd "$_d" && pwd)"; break; }
done
[[ -n "$HERE" ]] || { echo "x cannot find pd-lib.sh -- set SLURM_SUBMIT_DIR" >&2; exit 1; }
# shellcheck source=run/site-lib.sh
source "$HERE/site-lib.sh" || exit 1
site_load || exit 1

# ============================================================ 1. configuration
ENGINES="${ENGINES:-sglang vllm trtllm}"
# `sbatch --export=ALL,ENGINES=a b` cannot carry a space, so a caller who goes
# that route has to join the list some other way. Accept + and , as separators
# too: getting this wrong does not error, it silently runs one cell named
# "trtllm+sglang" and reports that the engine does not exist.
norm_list() { tr '+,' '  ' <<< "$1" | tr -s ' '; }
# VARIANTS uses ',' INSIDE a token to separate assignments, so it only gets the
# '+' translation. Job 79704 lost a cell to this: `cg32kv:CUDA_GRAPH_MAX_BS=32,
# KV_CACHE_DTYPE=fp8` was split into `cg32kv:CUDA_GRAPH_MAX_BS=32` and a phantom
# variant literally named `KV_CACHE_DTYPE=fp8`, which had no assignments at all and
# therefore silently re-ran `base` under a misleading name.
norm_plus() { tr '+' ' ' <<< "$1" | tr -s ' '; }
ENGINES="$(norm_list "$ENGINES")"
SPECS="${SPECS:-solo8 4x1:4x1}"
# MEASURED, not guessed. Jobs 79456/79457 swept TP4 on 4 GPUs and put the gate
# right between 8 and 16:
#
#            N=8                    N=16                   N=24
#   H100  itl 16.55  pass 1.000  itl 23.79  pass 0.562  itl 31.35  pass 0.000
#   H200  itl 15.44  pass 1.000  itl 21.00  pass 0.812  itl 28.92  pass 0.167
#
# H200 at N=16 misses by almost nothing -- 21.00 ms against a 20 ms gate, 0.812
# against 0.90 -- so the best scoring point is inside a gap the old list stepped
# straight over. Everything above 24 is there for the slope only; those points
# score zero and cost a minute each.
CONCURRENCIES="${CONCURRENCIES:-8 10 12 14 16 20 24}"

SPECS="$(norm_list "$SPECS")"
CONCURRENCIES="$(norm_list "$CONCURRENCIES")"

# THE VARIANT AXIS. One token per configuration to try, as `name` or
# `name:KEY=VALUE,KEY=VALUE`. The assignments are exported for that cell only, so
# the engine files pick them up when they build their argv and nothing leaks into
# the next cell.
#
#   VARIANTS="base kv8:KV_CACHE_DTYPE=fp8_e4m3 ep4:EP_SIZE=4"
#
# This is the axis that matters at four GPUs. The shape axis has collapsed --
# 221 GB of weights puts the floor at TP4, so there is exactly one shape -- and
# concurrency is already swept inside a cell for free. Everything else that the
# rules leave to us is a knob, and jobs 79456/79457 said which knobs: we are
# ITL-bound (21.00 ms against a 20 ms gate at N=16) and nowhere near KV-bound
# (1,523,487 tokens allocated, about 360 concurrent requests' worth), so the
# levers worth spending an allocation on are the ones that touch decode latency.
# A VALUE CONTAINING A SPACE CANNOT SURVIVE THIS, SO REFUSE IT BEFORE IT IS SPLIT.
# norm_plus turns '+' into a space and the cell loop then word-splits, so an assignment
# whose value contains whitespace is torn apart: job 216204 passed
#   PD_EXTRA=--flashinfer-allreduce-fusion-backend trtllm
# and got four variants instead of three -- one with the flag and no value, one named
# literally "trtllm" carrying no settings at all, and the control pushed to fourth where
# it ran out of allocation and measured nothing. 2,576 SU for one unpaired cell.
#
# The check has to run on the RAW string, split only on '+'. A first version of it ran
# after the word-split, where no token can contain whitespace any more and the condition
# is unreachable -- a guard that cannot fire.
#
# argparse accepts --flag=value as a single word, so that is how to write it:
#   PD_EXTRA=--flashinfer-allreduce-fusion-backend=trtllm
# For a value that must contain a comma too, add a named knob instead (see the NIXL
# backend params in engines/sglang.sh).
_vcheck() {   # $1 = the raw VARIANTS string, '+'-separated
    local raw="$1" tok assign
    local IFS='+'
    for tok in $raw; do
        assign="${tok#*:}"
        [[ "$assign" == "$tok" ]] && continue          # no ':' -- a bare variant
        if [[ "$assign" != "${assign//[[:space:]]/}" ]]; then
            echo "x VARIANTS: '$tok'" >&2
            echo "  has whitespace inside an assignment, and the parser splits on it." >&2
            echo "  Write a flag with a value as --flag=value instead." >&2
            return 1
        fi
    done
    return 0
}
_vcheck "${VARIANTS:-base}" || exit 1
VARIANTS="$(norm_plus "${VARIANTS:-base}")"

MODEL_PATH="${MODEL_PATH:?the site should have set this}"
KIT_DIR="${KIT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/AIPerf}"
AIPERF_IMG="${AIPERF_IMG-}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"
MOUNTS="${MOUNTS:-${MOUNTS_BASE:-/home:/home}}"

SERVED_NAME="${SERVED_NAME:-qwen}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"      # chat template + 4000 ISL + 200 OSL
MEM_FRACTION="${MEM_FRACTION:-0.90}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-nixl}"
MAX_RUNNING="${MAX_RUNNING:-256}"
# EMPTY BY DEFAULT, deliberately. This used to be 4, so every PD decode worker ran
# with expert parallelism -- and EP4 was the worst of the six knobs we swept at 4 GPUs
# (-26.3%), while the reference 3190 configuration measured a DP8+EP8 decode group as
# 4.3-5.2 ms per step SLOWER than plain TP4. Set it only to measure it.
DECODE_EP="${DECODE_EP:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
# Each cell gets its own port range. Two cells in one allocation otherwise reuse
# 30000/30100/30200, and a torn-down worker's listener can still be held when the
# next one binds -- jobs 79456 and 79457 lost their TensorRT-LLM cell to exactly
# that, 25 seconds in, with nothing wrong in TensorRT-LLM.
SOLO_PORT_BASE="${SOLO_PORT_BASE:-30000}"
CELL_PORT_STRIDE="${CELL_PORT_STRIDE:-1000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"    # 221 GB off shared storage, three times over
PD_CPUS_PER_WORKER="${PD_CPUS_PER_WORKER:-24}"
PD_ROUTER_CPUS="${PD_ROUTER_CPUS:-4}"

ISL="${ISL:-4000}"; OSL="${OSL:-200}"
SLA="${SLA:-time_to_first_token:3000 inter_token_latency:20}"
WARMUP_SECONDS="${WARMUP_SECONDS:-20}"
BENCH_SECONDS="${BENCH_SECONDS:-20}"
GRACE_SECONDS="${GRACE_SECONDS:-20}"
POINT_SECONDS="${POINT_SECONDS:-$(( WARMUP_SECONDS + BENCH_SECONDS + GRACE_SECONDS + 40 ))}"
# What one cell costs before its first measured point: bring-up plus prewarm.
# Nothing is started unless a whole cell fits, because half a cell is a
# checkpoint load bought for nothing.
CELL_OVERHEAD="${CELL_OVERHEAD:-$(( HEALTH_TIMEOUT / 3 + 180 ))}"

OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/results/${SLURM_JOB_ID:-manual}}"

die() { echo "x $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================ 2. preconditions
[[ -d "$MODEL_PATH" ]] || die "model path not found: $MODEL_PATH"
[[ -z "$AIPERF_IMG" || -e "$AIPERF_IMG" ]] || die "client image not found: $AIPERF_IMG"
[[ -n "$AIPERF_IMG" ]] || command -v "$AIPERF_BIN" >/dev/null 2>&1 \
    || die "no client container and no aiperf at: $AIPERF_BIN"
for s in observe.sh gpu_sample.sh diagnose.sh; do
    [[ -x "$KIT_DIR/$s" ]] || die "$KIT_DIR/$s missing or not executable"
done

# shellcheck source=run/pd-lib.sh
source "$HERE/pd-lib.sh" || die "cannot source pd-lib.sh"
# shellcheck source=run/client-lib.sh
source "$HERE/client-lib.sh" || die "cannot source client-lib.sh"
client_init "$AIPERF_BIN" || die "cannot work out what the client accepts"
client_banner
# No --cache-bust in this client, so prefix reuse gets turned off in the engine.
# Exported because the engine files read it when they build their argv.
if (( CLIENT_NEEDS_SERVER_PREFIX_OFF )); then
    DISABLE_PREFIX_CACHE="${DISABLE_PREFIX_CACHE:-1}"
else
    DISABLE_PREFIX_CACHE="${DISABLE_PREFIX_CACHE:-0}"
fi
export DISABLE_PREFIX_CACHE

# Slurm expands a range like "DGX-H200-CN[05-06]"; PBS lists one line per chunk
# in $PBS_NODEFILE. Same answer, two mechanisms -- see run/sched-lib.sh.
mapfile -t NODES < <(sched_nodes)
(( ${#NODES[@]} >= 1 )) || die "no nodes"
# Before anything is deployed: is every node actually able to run work? PBS hands
# out nodes that accept a job, report R and then answer no launcher at all, and a
# worker that never starts still bills for the walltime it holds. A no-op on Slurm.
sched_preflight "${NODES[@]}" || die "one of the allocated nodes cannot run work"
sched_reset_pools "${NODES[@]}"
HEAD="${NODES[0]}"
CAMPAIGN_GPUS="${CAMPAIGN_GPUS:-$(( ${#NODES[@]} * GPUS_PER_NODE ))}"
# What the per-GPU column divides by, and what labels the rows in the results
# table. Without the node name in the label, an H100 row and an H200 row are
# indistinguishable once they are side by side in the dashboard -- and telling
# them apart is the entire reason this cluster is in the plan.
GPU_LABEL="${GPU_LABEL:-${SITE_NAME:-$PD_SITE} ${HEAD}}"
export CAMPAIGN_GPUS GPU_LABEL BENCH_SECONDS
# WHAT THE ALLOCATION ACTUALLY HAS, not what we were told it has. GPUS_PER_NODE was
# a hardcoded 8 with no check against reality, and in job 81755 reality was 4: every
# per-GPU figure came out halved, and a PD spec would have placed workers on GPUs
# that did not exist. The measurement wins over the argument, and a disagreement is
# printed rather than resolved silently.
_measured=$(sched_run "$HEAD" 1 nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
if (( _measured == 0 )); then
    die "cannot see any GPU on $HEAD -- refusing to guess the allocation's size"
elif (( _measured != GPUS_PER_NODE )); then
    say "! GPUS_PER_NODE was $GPUS_PER_NODE, $HEAD actually has $_measured -- using $_measured"
    GPUS_PER_NODE=$_measured
    CAMPAIGN_GPUS=$(( ${#NODES[@]} * GPUS_PER_NODE ))
fi
export GPUS_PER_NODE CAMPAIGN_GPUS

say "site ${SITE_NAME:-$PD_SITE}  nodes ${NODES[*]}  ${GPUS_PER_NODE} GPUs each  ($CAMPAIGN_GPUS total)"
say "engines [$ENGINES] x shapes [$SPECS] x variants [$VARIANTS]"
say "concurrencies [$CONCURRENCIES]"

# Reject a shape that cannot be placed before spending an hour finding out. A
# soloN is placeable when N fits on one node; a PD spec goes through pd_place.
for spec in $SPECS; do
    if [[ "$spec" == solo* ]]; then
        (( ${spec#solo} <= GPUS_PER_NODE )) || die "$spec needs more than $GPUS_PER_NODE GPUs on one node"
    else
        ( pd_parse "$spec" && pd_place "$GPUS_PER_NODE" "${NODES[@]}" ) >/dev/null 2>&1 \
            || die "spec '$spec' cannot be placed on ${#NODES[@]}x$GPUS_PER_NODE GPUs"
    fi
done

SWEEP="$OUT_ROOT/sweep"; mkdir -p "$SWEEP" || die "cannot create $SWEEP"

# ============================================================ 2b. what we are on
# Recorded from inside the allocation, because the login node is not the machine:
# it has no GPU, no driver, and a different /dev/shm and fd limit. Two of the five
# wrong hypotheses about TensorRT-LLM on Firmus came from measuring ulimit and
# /dev/shm on a login node and assuming they applied to the compute nodes.
say "hardware, from inside the allocation:"
sched_run_all bash -c '
  printf "  %s: %s\n" "$(hostname)" "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  printf "     GPUs %s   shm %s   nofile %s   IB "     "$(nvidia-smi -L | wc -l)" "$(df -h /dev/shm | awk "NR==2{print \$2}")" "$(ulimit -n)"
  for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue
    [ "$(cat $d/ports/1/link_layer 2>/dev/null)" = InfiniBand ] || continue
    [ "$(cat $d/ports/1/state 2>/dev/null)" = "4: ACTIVE" ] || continue
    printf "%s(%s) " "$(basename $d)" "$(cat $d/ports/1/rate 2>/dev/null | awk "{print \$1}")"
  done; echo' 2>&1 | tee "$OUT_ROOT/hardware.txt"

# ============================================================ 3. telemetry
# One sampler per node for the whole campaign; each point records its own window
# so diagnose.sh slices the right span out of a shared series.
TEL_PIDS=()
for n in "${NODES[@]}"; do
    # No GPU: nvidia-smi reports on a node's devices without holding any, and a
    # sampler that took a GPU would be taking it from the thing it samples.
    SCHED_CONTAINER_ARGS=()
    sched_step_nogpu "$n" 2 "$SWEEP/tel-$n.log" \
         "$KIT_DIR/gpu_sample.sh" --out "$OUT_ROOT/tel-$n" --label "$n" --interval 2
    TEL_PIDS+=("${SCHED_PIDS[-1]}")
done
campaign_cleanup() {
    pd_teardown
    local p
    for p in "${TEL_PIDS[@]:-}"; do [[ -n "$p" ]] || continue; kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    TEL_PIDS=()
}
trap campaign_cleanup EXIT INT TERM

# aiperf builds its synthetic dataset before each measurement and gives itself 300s.
# That is enough most of the time and not always: in job 85746 the c4 point built
# fine, c5 and c6 both died with "timed out waiting for dataset configuration after
# 300.0s", and c7 and c8 built fine again -- so two points out of five were lost to a
# flake rather than to anything about the engine. The error message names the knob.
# BOTH of them, because aiperf validates one against the other and refuses to start:
# "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT: 600.0 must be greater than or equal to
# AIPERF_DATASET_CONFIGURATION_TIMEOUT: 900.0". Raising only the dataset timeout
# turned a 5-minute flake into an instant hard failure -- job 85853 lost both of its
# points in two seconds each.
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-900}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1200}"

export MODEL_PATH SERVED_NAME CONTEXT_LEN MEM_FRACTION \
       MEM_FRACTION_PREFILL MEM_FRACTION_DECODE TRANSFER_BACKEND \
       PD_EXTRA PREFILL_EXTRA DECODE_EXTRA \
       DECODE_EP MAX_RUNNING PD_CPUS_PER_WORKER PD_ROUTER_CPUS \
       MOE_RUNNER_BACKEND MOE_RUNNER_BACKEND_PREFILL MOE_RUNNER_BACKEND_DECODE

# ============================================================ 4. one measured point
# Split out because solo and PD differ only in how the endpoint came to exist.
measure_point() {   # $1=dir  $2=gateway host:port  $3=model id  $4=N  $5=label
    local dir="$1" gw="$2" mid="$3" N="$4" label="$5"
    local -a cargs=(); mapfile -t cargs < <(site_client_args)
    local -a margs=(); mapfile -t margs < <(pd_metrics_args)
    local -a pargs=(); mapfile -t pargs < <(client_profile_args "$dir/c$N/artifacts" "$gw" "$mid" "$N")
    say "    $label @ c$N (${BENCH_SECONDS}s)"
    SCHED_CONTAINER_ARGS=(${cargs[@]+"${cargs[@]}"})
    sched_run "$HEAD" "${CLIENT_CPUS:-8}" \
         "$KIT_DIR/observe.sh" --out "$dir" --concurrency "$N" \
            --isl "$ISL" --osl "$OSL" --sla "$SLA" --model "$mid" \
            ${margs[@]+"${margs[@]}"} -- \
         "$AIPERF_BIN" "${pargs[@]}" \
         > "$dir/c$N.log" 2>&1
}

# TWO PHASES, BECAUSE ONE DOES NOT OPEN EVERY ROUTE. The reference H200 campaign that
# measures 4,395-4,475 good tok/s warms the gateway with 32 requests at once, waits for
# all of them, and only then sends 8 one at a time. Both halves do work: the burst opens
# every prefill-decode pair simultaneously, where our old `n` requests at concurrency `n`
# walked round_robin and touched two of four pairs twice each; the sequential tail then
# gives each pair a warm second pass.
#
# This matters here more than it looks. Every failure we have, at every concurrency, is a
# TTFT failure inside the first ten seconds of the measurement window, and aiperf fires
# all N profiling requests within 0.1 s of the warm-up draining. A route that is cold at
# that instant pays for it in exactly the requests that are failing.
PREWARM_CONCURRENT="${PREWARM_CONCURRENT:-32}"
PREWARM_SEQUENTIAL="${PREWARM_SEQUENTIAL:-8}"
# A prewarm that errors on more than this fraction is a broken deployment, not a slow
# one. 10% is below 372421's kv8 cell (14/32 burst = 44%) and above the zero its
# control produced, so it separates the two cases that job contained.
PREWARM_MAX_ERROR_PCT="${PREWARM_MAX_ERROR_PCT:-10}"

prewarm_phase() {   # $1=dir  $2=gateway  $3=model id  $4=requests  $5=concurrency  $6=tag
    local dir="$1" gw="$2" mid="$3" n="$4" conc="$5" tag="$6"
    local -a cargs=(); mapfile -t cargs < <(site_client_args)
    local -a wargs=(); mapfile -t wargs < <(client_prewarm_args "$dir/prewarm-$tag" "$gw" "$mid" "$n" "$conc")
    say "    prewarm/$tag: $n requests at concurrency $conc, ISL $ISL"
    SCHED_CONTAINER_ARGS=(${cargs[@]+"${cargs[@]}"})
    local rc=0
    sched_run "$HEAD" "${CLIENT_CPUS:-8}" \
         "$AIPERF_BIN" "${wargs[@]}" \
         > "$dir/prewarm-$tag.log" 2>&1 || rc=$?
    return $rc
}

prewarm() {   # $1=dir  $2=gateway  $3=model id  $4=requests (floor for the burst)
    local dir="$1" gw="$2" mid="$3" n="$4"
    (( n > 0 )) || return 0
    # The burst has to be at least as wide as the old behaviour asked for.
    local burst="$PREWARM_CONCURRENT"
    (( n > burst )) && burst="$n"
    local t0=$SECONDS rc=0
    prewarm_phase "$dir" "$gw" "$mid" "$burst" "$burst" burst || rc=$?
    if (( rc == 0 && PREWARM_SEQUENTIAL > 0 )); then
        prewarm_phase "$dir" "$gw" "$mid" "$PREWARM_SEQUENTIAL" 1 seq || rc=$?
    fi
    # The verdict reads whichever phase ran; concatenate so one grep covers both.
    cat "$dir"/prewarm-*.log > "$dir/prewarm.log" 2>/dev/null
    say "    prewarm done in $(( SECONDS - t0 ))s"
    # THE PREWARM IS ALSO A SMOKE TEST, AND ITS VERDICT WAS BEING THROWN AWAY.
    # Job 215492's prewarm said "All 4 inference request(s) failed. No successful
    # responses were collected" and the cell went on to sweep anyway: 25 minutes and
    # 110,306 rejected requests later it produced nothing. If the client cannot get
    # one answer out of the gateway, no concurrency is going to work either.
    if (( rc != 0 )) || grep -q "No successful responses were collected" "$dir/prewarm.log" 2>/dev/null; then
        say "    x prewarm got no successful response (rc=$rc) -- not sweeping this cell"
        grep -m3 -E "Error:|Reason:" "$dir/prewarm.log" 2>/dev/null | sed 's/^/      /'
        return 1
    fi

    # AND A FRACTION OF ERRORS IS ALSO A BROKEN DEPLOYMENT. The test above only fires
    # when NOTHING succeeded, which is too lenient: job 372421's kv8 cell prewarmed
    # with 14 errors out of 32 burst requests and 3 out of 8 sequential, the campaign
    # swept it anyway, and the two scored points came back at pass 0.662 and 0.568 --
    # about 90 SU spent on a cell whose own prewarm had already said it was broken.
    #
    # The control cell on the same two nodes minutes earlier prewarmed 32/32 and 8/8
    # with zero errors, so the signal is per-cell and it is available before the first
    # scored request.
    local errs tot
    errs="$(grep -ohE 'error_records=[0-9]+' "$dir"/prewarm-*.log 2>/dev/null \
            | cut -d= -f2 | awk '{s+=$1} END{print s+0}')"
    tot="$(grep -ohE 'final_requests_completed=[0-9]+' "$dir"/prewarm-*.log 2>/dev/null \
           | cut -d= -f2 | awk '{s+=$1} END{print s+0}')"
    if (( tot > 0 )) && (( errs * 100 > tot * PREWARM_MAX_ERROR_PCT )); then
        say "    x prewarm: $errs of $tot requests errored (>${PREWARM_MAX_ERROR_PCT}%) -- not sweeping this cell"
        grep -m3 -E "Error:|Reason:" "$dir"/prewarm-*.log 2>/dev/null | sed 's/^/      /'
        return 1
    fi
    (( errs > 0 )) && say "    ! prewarm: $errs of $tot requests errored (under the ${PREWARM_MAX_ERROR_PCT}% limit)"
    return 0
}

# ============================================================ 5. one cell
# A cell is (engine, shape): bring it up once, measure every concurrency, tear it
# down. Returns non-zero if the cell produced nothing, so the caller can record
# that and move on rather than abandoning the allocation.
run_cell() {   # $1=engine  $2=spec  $3=dir
    local eng="$1" spec="$2" dir="$3"
    export PD_WORK="$dir/deploy"; mkdir -p "$PD_WORK"
    local gw mid nworkers

    # Hand every GPU back before this cell takes any. A cell has the whole
    # allocation to itself -- the previous one was torn down -- but the pool does
    # not know that: sched_take_gpus only ever removes, so without this reset the
    # second cell of a campaign would be refused for want of devices that are in
    # fact idle. A no-op under Slurm, which has a real allocator.
    sched_reset_pools "${NODES[@]}"

    if [[ "$spec" == solo* ]]; then
        # One instance, no gateway: the client talks to the server. This is the
        # shape every published single-node number uses, so it is the only way to
        # tell a PD gain apart from a hardware gain.
        local tp="${spec#solo}"
        PD_ROLE=(solo); PD_NODE=("$HEAD"); PD_TP=("$tp")
        PD_PORT=("$SOLO_PORT"); PD_BOOT=(0)
        pd_resolve || { say "  ! cannot resolve $HEAD"; return 1; }
        gw="${PD_IP[0]}:$SOLO_PORT"
        # Some engines take their knobs through a file rather than argv, and only
        # eng_pd_prepare knows how to write one -- so the solo path calls it too.
        # SGLang's is a no-op, which is why this was not needed before.
        declare -F eng_pd_prepare >/dev/null && { eng_pd_prepare "$PD_WORK" || return 1; }
        local -a scmd=(); mapfile -t scmd < <(eng_serve_cmd "$MODEL_PATH" "$SERVED_NAME" "$SOLO_PORT" "$tp")
        (( ${#scmd[@]} )) || { say "  ! eng_serve_cmd produced nothing"; return 1; }
        pd_write_launcher "$PD_WORK/worker.sh"
        mapfile -t SCHED_CONTAINER_ARGS < <(pd_container_args solo)
        say "  solo: $tp GPUs on $HEAD port $SOLO_PORT"
        PD_PIDS=()
        sched_step "$HEAD" "$tp" "$(( PD_CPUS_PER_WORKER + 8 ))" "$PD_WORK/solo.log" \
             bash "$PD_WORK/worker.sh" solo "$SOLO_PORT" "${scmd[@]}" || return 1
        PD_PIDS+=("${SCHED_PIDS[-1]}")
        pd_wait_healthy "http://$gw" "solo($eng)" "$HEALTH_TIMEOUT" \
                        "${PD_PIDS[0]}" "$PD_WORK/solo.log" || return 1
        nworkers=1
    else
        pd_parse "$spec" || return 1
        pd_place "$GPUS_PER_NODE" "${NODES[@]}" || return 1
        pd_print_placement
        # RESOLVE BEFORE PREPARE. TensorRT-LLM's eng_pd_prepare writes the gateway's
        # disagg.yaml, which lists the workers by address, so it needs PD_IP -- and
        # PD_IP is what pd_resolve fills. With prepare first it died on
        # "PD_IP[$i]: unbound variable" eighteen seconds into job 215874. SGLang's
        # and vLLM's prepare are no-ops, which is why the order was wrong for weeks
        # without anything noticing.
        pd_resolve || return 1
        declare -F eng_pd_prepare >/dev/null && { eng_pd_prepare "$PD_WORK" || return 1; }
        pd_write_launcher "$PD_WORK/worker.sh"
        gw="$(pd_gateway "$ROUTER_PORT")"
        pd_launch "$PD_WORK/worker.sh" || return 1
        pd_wait_all "$HEALTH_TIMEOUT" || return 1
        pd_launch_gateway "$ROUTER_PORT" || return 1
        pd_wait_healthy "http://$gw" "gateway($eng)" 300 \
                        "${PD_PIDS[-1]}" "$PD_WORK/gateway.log" || return 1
        nworkers="${PD_P_N:-1}"
    fi

    # /health goes true before the gateway has registered its workers, so ask for
    # a servable model id rather than trusting one probe (job 17743).
    mid="$(pd_wait_models "http://$gw" 300 "${PD_PIDS[-1]}" "$PD_WORK/gateway.log")" || return 1

    { echo "engine=$eng spec=$spec gateway=$gw model_id=$mid"
      [[ "$spec" == solo* ]] || pd_print_placement; } > "$dir/deployment.txt"

    # Before paying for any measured point, check that the prefill batch we asked for
    # is the prefill batch we got. DeepGEMM reduces max_m silently and benchmarks on.
    if [[ "$spec" != solo* ]]; then
        pd_assert_prefill_batch "$PD_WORK" "${CHUNKED_PREFILL:-}" || return 1
    fi

    prewarm "$PD_WORK" "$gw" "$mid" "$(( nworkers * 2 ))" || return 1

    local measured=0 N
    for N in $CONCURRENCIES; do
        pd_need_time "$POINT_SECONDS" "$eng/$spec c$N" || { say "    out of allocation"; break; }
        if measure_point "$dir" "$gw" "$mid" "$N" "$eng/$spec"; then
            measured=$(( measured + 1 ))
        else
            say "    ! $eng/$spec @ c$N failed (see $dir/c$N.log)"
        fi
    done
    pd_teardown
    (( measured > 0 ))
}

# ============================================================ 6. the campaign
FAILED=(); DONE=(); CELL_INDEX=0
for eng in $ENGINES; do
    for spec in $SPECS; do
     for variant in $VARIANTS; do
        vname="${variant%%:*}"
        vassign="${variant#*:}"; [[ "$vassign" == "$variant" ]] && vassign=""
        cell="$eng-${spec//:/_}+$vname"
        dir="$SWEEP/$cell"; mkdir -p "$dir"
        SOLO_PORT=$(( SOLO_PORT_BASE + CELL_INDEX * CELL_PORT_STRIDE ))
        PD_PORT_BASE=$SOLO_PORT
        # 9000 + cell*100 put the second cell on 9100, which is a Prometheus IPMI
        # exporter on Nano5 -- see pd_assert_ports_free. 18000 is clear of the
        # exporter range (9100 node, 9400 ipmi, 9835 dcgm and friends) and of our own
        # allocations: engines at 30100+, router Prometheus at ROUTER_PORT+21000.
        # The bind check is the guarantee; this only makes a collision unlikely.
        PD_BOOT_BASE=$(( ${PD_BOOT_BASE_FLOOR:-18000} + CELL_INDEX * 100 ))
        ROUTER_PORT=$(( 8000 + CELL_INDEX ))
        export SOLO_PORT PD_PORT_BASE PD_BOOT_BASE ROUTER_PORT
        CELL_INDEX=$(( CELL_INDEX + 1 ))
        echo ""
        echo "=============================================================="
        say "CELL $cell"
        echo "=============================================================="

        if ! pd_need_time $(( CELL_OVERHEAD + POINT_SECONDS )) "cell $cell"; then
            say "  not enough allocation left for a whole cell -- stopping here"
            FAILED+=("$cell:no-time"); break 2
        fi

        # Each engine in its own subshell. engine-lib defines the defaults
        # (eng_health_path and friends) and each engine file may override them;
        # loading a second engine into the same shell would leave the first one's
        # overrides behind, so the function table is rebuilt per cell rather than
        # patched.
        # The variant's assignments live only in this subshell.
        if ( if [[ -n "$vassign" ]]; then
                 IFS=',' read -r -a _kv <<< "$vassign"
                 for _a in "${_kv[@]}"; do
                     [[ "$_a" == *=* ]] || { echo "x variant '$vname': '$_a' is not KEY=VALUE" >&2; exit 1; }
                     export "${_a?}"
                 done
                 say "  variant $vname: $vassign"
             fi
             source "$HERE/engine-lib.sh" \
             && eng_load "$eng" \
             && { eng_banner; PD_IMAGE="$(eng_image)"; export PD_IMAGE; } \
             && run_cell "$eng" "$spec" "$dir" ) ; then
            DONE+=("$cell"); say "CELL $cell done"
        else
            FAILED+=("$cell"); say "CELL $cell produced nothing"
            # Whatever the subshell left running belongs to this shell's job
            # table, so clear it here too before the next cell asks for the GPUs.
            pd_teardown
        fi
     done
    done
done

# ============================================================ 7. summary
say "cells done: ${DONE[*]:-none}"
say "cells failed: ${FAILED[*]:-none}"
"$HERE/campaign-summary.py" "$SWEEP" "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/summary.txt"
say "results under $OUT_ROOT"

# EXIT NON-ZERO WHEN NOTHING WORKED, because a scheduler dependency is only worth
# anything if this script tells the truth about its outcome.
#
# Job 214742 was a 1-node PD shakedown whose whole purpose was to gate a 16-GPU run
# behind `-W depend=afterok:`. Its single cell died on a missing nixl, it printed
# "cells failed", and it exited 0 -- so PBS considered afterok satisfied and started
# the 16-GPU job, which was on course to fail identically on 16 GPUs instead of 8.
# The dependency was not wrong; this script was lying to it.
#
# A partial failure still exits 0: some cells failing while others produce numbers is
# a normal sweep, and there is no reason to withhold the ones that worked.
if (( ${#DONE[@]} == 0 )); then
    say "x every cell failed -- exiting non-zero so a dependent job does not start"
    exit 1
fi
exit 0
