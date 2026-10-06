#!/bin/bash
# Resource sampler -- run this ON THE SERVER NODE while run.sh drives load from
# elsewhere, so you can tell whether a goodput plateau is the GPU, the memory
# system, the interconnect, the CPU, or none of them.
#
#   ./gpu_sample.sh --out ./telemetry                 # sample until Ctrl-C, then summarize
#   ./gpu_sample.sh --out ./telemetry --duration 600  # sample for 10 minutes
#   ./gpu_sample.sh --summary ./telemetry             # re-print the summary later
#
# Why not just watch nvidia-smi: its GPU-Util% only means "at least one kernel
# was resident during the sample window", so LLM decode sits at 95-100% while
# the SMs are mostly idle. Power draw against the board limit, the achieved SM
# clock against the card's maximum, PCIe/NVLink traffic, and the busiest single
# CPU core are the signals that actually distinguish "saturated" from "starved".
#
# Three things this records that the obvious version misses:
#   * clocks.max.sm alongside clocks.sm -- throttle *flags* are permanently lit
#     on thermally-limited cards, so a clock ratio is the honest evidence
#   * PCIe rx/tx (nvidia-smi dmon), NVLink counters, and InfiniBand port
#     counters -- for TP > 1 or PD disaggregation the interconnect is often the
#     ceiling, and cross-node KV transfer rides on IB
#   * every compute process on each GPU, at start and end -- on a shared node
#     these numbers are machine-wide and someone else's job is in them
#
# Writes host.txt (static config), gpu.csv, cpu.csv, pcie.csv, nvlink.csv, ib.csv.
# Needs nothing but nvidia-smi, /proc/stat and sysfs -- no DCGM, no root, and no
# rdma tooling: the IB counters are read straight out of
# /sys/class/infiniband (point --ib-root elsewhere to override).

set -uo pipefail

die()   { echo "x $*" >&2; exit 1; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; }
need()  { [[ $# -ge 2 ]] || die "$1 requires a value"; }

OUT=""
SUMMARY_ONLY=""
INTERVAL=1
DURATION=0          # 0 = until interrupted
LABEL=""
IB_ROOT="${IB_ROOT:-/sys/class/infiniband}"
DCGM_URL="${DCGM_URL:-http://localhost:9400/metrics}"
NO_DCGM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)       need "$@"; OUT="$2";          shift 2 ;;
        --summary)   need "$@"; SUMMARY_ONLY="$2"; shift 2 ;;
        --interval)  need "$@"; INTERVAL="$2";     shift 2 ;;
        --duration)  need "$@"; DURATION="$2";     shift 2 ;;
        --label)     need "$@"; LABEL="$2";        shift 2 ;;
        --ib-root)   need "$@"; IB_ROOT="$2";      shift 2 ;;
        --dcgm-url)  need "$@"; DCGM_URL="$2";     shift 2 ;;
        --no-dcgm)   DCGM_URL="";  NO_DCGM=1;      shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "x unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------- compute apps
# On a shared node every number in gpu.csv and cpu.csv is machine-wide. Record
# who else was resident so a run can be judged clean or contaminated afterwards.
compute_apps() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r uuid pid mem; do
        pid="${pid// /}"
        [[ -n "$pid" ]] || continue
        user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        echo "${uuid// /},${pid},${user:-?},${mem// /}"
    done
}

# IB error counters are worth a before/after rather than a time series: a link
# that retrained or dropped packets mid-run invalidates the numbers around it.
ib_errors() {
    local i p n v
    for i in "${!IB_PATH[@]}"; do
        p="${IB_PATH[$i]}"
        for n in link_downed link_error_recovery symbol_error port_rcv_errors \
                 port_xmit_discards VL15_dropped; do
            v=0; [[ -r "$p/counters/$n" ]] && read -r v < "$p/counters/$n"
            echo "${IB_DEV[$i]}:${IB_PORT[$i]},$n,${v:-0}"
        done
    done
}

# ---------------------------------------------------------------- summary
summarize() {
    local dir="$1"
    [[ -d "$dir" ]] || die "no such directory: $dir"

    echo ""
    echo "=========================================="
    echo " telemetry summary -- $dir"
    echo "=========================================="

    if [[ -s "$dir/gpu.csv" ]]; then
        echo ""
        echo "GPU (per device, over the whole sampling window)"
        printf "  %-4s %8s %8s %10s %12s %12s %8s %8s %10s\n" \
               "gpu" "util%" "memio%" "power" "power/limit" "mem peak" "temp" "clk/max" "throttled"
        awk -F, 'NR > 1 {
            n[$2]++
            uavg[$2] += $3; mavg[$2] += $4
            gmem[$2]  = ($5 > gmem[$2] ? $5 : gmem[$2]); mtot[$2] = $6
            pavg[$2] += $7; plim[$2] = $8
            if ($10 > 0) { cn[$2]++; cr[$2] += $9/$10; if ($9/$10 < 0.85) clow[$2]++ }
            if ($11 > tmax[$2]) tmax[$2] = $11
            if ($12 ~ /hw|powerbrake/)          hard[$2]++
            else if ($12 != "none" && $12 != "") soft[$2]++
        }
        END {
            for (g in n) {
                printf "  %-4s %7.0f%% %7.0f%% %6.0f W av %11.0f%% %7.0f/%.0f MiB %6.0fC %7s %4.0f%%h %3.0f%%s\n", \
                       g, uavg[g]/n[g], mavg[g]/n[g], pavg[g]/n[g], \
                       (plim[g] > 0 ? 100*(pavg[g]/n[g])/plim[g] : 0), \
                       gmem[g], mtot[g], tmax[g], \
                       (cn[g] > 0 ? sprintf("%.0f%%", 100*cr[g]/cn[g]) : "-"), \
                       100*hard[g]/n[g], 100*soft[g]/n[g]
            }
        }' "$dir/gpu.csv" | sort -k1,1 -s
        echo "  clk/max = achieved SM clock as % of the card maximum; throttled: h=hardware, s=software flags"

        # Flags lie on thermally-limited cards; the clock is the evidence that matters.
        awk -F, 'NR > 1 && $10 > 0 { n++; if ($9/$10 < 0.85) low++ }
                 NR > 1 && ($12 ~ /hw|powerbrake/) { hard++ }
                 NR > 1 && $12 != "none" && $12 != "" { any++ }
        END {
            if (n == 0) { print "  ! no clocks.max.sm from this driver -- clock evidence unavailable"; exit }
            lp = 100*low/n; ap = 100*any/n; hp = 100*hard/n
            if (lp > 20)
                printf "  ! clocks below 85%% of maximum in %.0f%% of samples -- really throttled, fix before tuning\n", lp
            else if (ap > 5)
                printf "  . throttle flags lit in %.0f%% of samples but clocks held at >=85%% of maximum\n", ap
            if (hp > 5)
                printf "  ! hardware slowdown (thermal/power brake) in %.0f%% of samples\n", hp
        }' "$dir/gpu.csv"
    else
        echo "  (no gpu.csv -- nvidia-smi unavailable during sampling?)"
    fi

    # ------------------------------------------------------------ interconnect
    if [[ -s "$dir/pcie.csv" ]]; then
        local cap
        cap="$(awk -F= '/^pcie_cap_gbs=/ { print $2 }' "$dir/host.txt" 2>/dev/null)"
        echo ""
        echo "PCIe (host <-> device, per GPU)"
        awk -F, -v cap="${cap:-0}" 'NR > 1 {
            n[$2]++; rx[$2] += $3; tx[$2] += $4
            if ($3 > rxm[$2]) rxm[$2] = $3
            if ($4 > txm[$2]) txm[$2] = $4
        }
        END {
            for (g in n) {
                peak = (rxm[g] > txm[g] ? rxm[g] : txm[g]) / 1000.0
                printf "  gpu%-3s rx %6.2f GB/s av (peak %5.2f)   tx %6.2f GB/s av (peak %5.2f)%s\n", \
                       g, rx[g]/n[g]/1000, rxm[g]/1000, tx[g]/n[g]/1000, txm[g]/1000, \
                       (cap > 0 ? sprintf("   peak %.0f%% of link", 100*peak/cap) : "")
            }
        }' "$dir/pcie.csv" | sort -k1,1 -s
        [[ -n "${cap:-}" && "${cap:-0}" != "0" ]] && \
            echo "  link capacity ${cap} GB/s per direction (from PCIe gen x width)"
    fi

    if [[ -s "$dir/nvlink.csv" ]]; then
        echo ""
        echo "NVLink (cumulative counters, converted to rate)"
        awk -F, 'NR > 1 {
            if (!(($2) in first)) { first[$2] = $3; firstr[$2] = $4; t0[$2] = $1 }
            last[$2] = $3; lastr[$2] = $4; t1[$2] = $1; unit[$2] = $5
        }
        END {
            for (g in last) {
                dt = t1[g] - t0[g]
                if (dt <= 0) continue
                k = (unit[g] ~ /MiB/) ? 1048576 : ((unit[g] ~ /GiB/) ? 1073741824 : 1024)
                printf "  gpu%-3s tx %6.2f GB/s   rx %6.2f GB/s   (over %.0fs)\n", \
                       g, (last[g]-first[g])*k/dt/1e9, (lastr[g]-firstr[g])*k/dt/1e9, dt
            }
        }' "$dir/nvlink.csv" | sort -k1,1 -s
    fi

    if [[ -s "$dir/dcgm.csv" ]]; then
        echo ""
        echo "DCGM profiling counters (what nvidia-smi cannot tell you)"
        awk -F, 'NR > 1 {
            key = $2 "," $3; n[key]++; sum[key] += $4
            if ($4 > max[key]) max[key] = $4
            if (!($2 in seen)) { seen[$2] = 1; order[++g] = $2 }
        }
        END {
            printf "  %-4s %10s %10s %10s %10s\n", "gpu", "gr%", "tensor%", "dram%", "sm%"
            for (i = 1; i <= g; i++) {
                d = order[i]
                printf "  %-4s", d
                split("DCGM_FI_PROF_GR_ENGINE_ACTIVE DCGM_FI_PROF_PIPE_TENSOR_ACTIVE DCGM_FI_PROF_DRAM_ACTIVE DCGM_FI_PROF_SM_ACTIVE", f, " ")
                for (j = 1; j <= 4; j++) {
                    k = d "," f[j]
                    if (n[k] > 0) printf " %9.0f%%", 100 * sum[k] / n[k]
                    else          printf " %10s", "-"
                }
                printf "\n"
            }
            print "  ratios of cycles the unit was active, averaged over the window"
        }' "$dir/dcgm.csv"
        awk -F, 'NR > 1 && $3 == "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE" { tn++; tsum += $4 }
                 NR > 1 && $3 == "DCGM_FI_PROF_DRAM_ACTIVE"        { dn++; dsum += $4 }
        END {
            if (tn && dn) {
                t = tsum/tn; d = dsum/dn
                if (d > 0.5 && t < 0.2)
                    printf "  -> memory interface busy %.0f%% of cycles with the tensor pipe at %.0f%%: decode-bound, as expected\n", 100*d, 100*t
                else if (t > 0.5)
                    printf "  -> tensor pipe active %.0f%% of cycles: this really is compute\n", 100*t
            }
        }' "$dir/dcgm.csv"
    fi

    if [[ -s "$dir/ib.csv" ]]; then
        local ibcap ibtot
        ibcap="$(awk -F= '/^ib_cap_gbs=/ { print $2 }' "$dir/host.txt" 2>/dev/null)"
        ibtot="$(awk -F= '/^ib_total_cap_gbs=/ { print $2 }' "$dir/host.txt" 2>/dev/null)"
        echo ""
        echo "InfiniBand (per port, from the sysfs counters)"
        # first-seen order is already sorted (the sysfs glob is), so the rows keep
        # their order without piping the totals and warnings through sort with them
        awk -F, -v cap="${ibcap:-0}" 'NR > 1 {
            if (!(($2) in t0)) { t0[$2] = $1; r0[$2] = $3; x0[$2] = $4; w0[$2] = $5
                                 order[++n] = $2 }
            t1[$2] = $1; r1[$2] = $3; x1[$2] = $4; w1[$2] = $5
        }
        END {
            for (i = 1; i <= n; i++) {
                p = order[i]
                dt = t1[p] - t0[p]
                if (dt <= 0) continue
                dr = r1[p] - r0[p]; dx = x1[p] - x0[p]
                if (dr < 0 || dx < 0) { printf "  %-14s counter wrapped -- rate unavailable\n", p; continue }
                rx = dr / dt / 1e9; tx = dx / dt / 1e9
                peak = (rx > tx ? rx : tx)
                printf "  %-14s rx %6.2f GB/s   tx %6.2f GB/s%s%s\n", p, rx, tx, \
                       (cap > 0 ? sprintf("   %3.0f%% of link", 100 * peak / cap) : ""), \
                       (w1[p] - w0[p] > 0 ? sprintf("   xmit_wait +%d", w1[p] - w0[p]) : "")
                tr += rx; tt += tx; tw += w1[p] - w0[p]
            }
            printf "  %-14s rx %6.2f GB/s   tx %6.2f GB/s\n", "all ports", tr, tt
            if (tw > 0)
                printf "  ! xmit_wait grew by %d across the fabric -- ports had data to send and no\n     credit, which is congestion, not bandwidth\n", tw
        }' "$dir/ib.csv"
        [[ -n "${ibcap:-}" && "${ibcap:-0}" != "0" ]] && \
            echo "  link capacity ${ibcap} GB/s per port, ${ibtot:-?} GB/s across the node"

        # a link that retrained mid-run invalidates everything measured around it
        awk -F, '
            /IB error counters at start/ { where = 1; next }
            /IB error counters at end/   { where = 2; next }
            /^---/ { where = 0; next }
            where == 1 && NF == 3 { a[$1 "," $2] = $3 }
            where == 2 && NF == 3 { b[$1 "," $2] = $3 }
            END {
                for (k in b) if (b[k] + 0 > a[k] + 0) {
                    split(k, f, ",")
                    printf "  ! %s %s rose by %d during sampling\n", f[1], f[2], b[k] - a[k]
                    bad++
                }
                if (bad) print "    IB errors during the run -- the numbers around them are not trustworthy"
            }' "$dir/host.txt" 2>/dev/null
    fi

    if [[ -s "$dir/cpu.csv" ]]; then
        echo ""
        echo "CPU (host running this sampler)"
        awk -F, 'NR > 1 {
            n++; mean += $2; hot += $3; io += $5
            if ($2 > meanmax) meanmax = $2
            if ($3 > hotmax) { hotmax = $3; hotcore = $4 }
        }
        END {
            if (n == 0) exit
            printf "  all-core busy   avg %5.1f%%   peak %5.1f%%\n", mean/n, meanmax
            printf "  busiest core    avg %5.1f%%   peak %5.1f%% (cpu%s)\n", hot/n, hotmax, hotcore
            printf "  iowait          avg %5.1f%%\n", io/n
            if (hotmax > 95 && meanmax < 60)
                printf "  ! a single core is pegged while the machine is idle -- single-threaded bottleneck\n"
        }' "$dir/cpu.csv"
    fi

    # ------------------------------------------------------------ tenancy
    if [[ -f "$dir/host.txt" ]]; then
        local users
        users="$(awk -F= '/^compute_app_users=/ { print $2 }' "$dir/host.txt" | tr ' ' '\n' | grep -v '^$' | sort -u)"
        if [[ -n "$users" ]] && (( $(wc -l <<<"$users") > 1 )); then
            echo ""
            echo "  ! shared node: compute processes from $(wc -l <<<"$users") users ($(tr '\n' ' ' <<<"$users"))"
            echo "    every GPU and CPU number above is machine-wide, not your job's -- treat"
            echo "    GPU COMPUTE and SERIAL CPU verdicts as unproven until you have the node"
        fi
    fi

    echo ""
    echo "How to read this:"
    echo "  power near the limit + clocks at max  -> GPU is genuinely saturated, tune the model/engine"
    echo "  power well under the limit            -> GPU is starved: batch too small, client-bound, or queueing"
    echo "  clocks < 85% of max                   -> really throttled; flags alone are not evidence"
    echo "  mem peak far below total              -> KV cache is under-provisioned, raise gpu_memory_utilization"
    echo "  PCIe peak near link capacity          -> interconnect bound (TP, PD-disagg KV transfer)"
    echo "  tensor% high                          -> genuinely compute bound (prefill looks like this)"
    echo "  dram% high with tensor% low           -> memory bound (decode looks like this)"
    echo "  IB near link capacity                 -> cross-node KV transfer is the ceiling"
    echo "  IB far from capacity but xmit_wait up -> congestion or credit starvation, not bandwidth"
    echo "  busiest core pegged, mean low         -> a serial stage (HTTP frontend / detokenize) is the limit"
    echo ""
}

if [[ -n "$SUMMARY_ONLY" ]]; then
    summarize "$SUMMARY_ONLY"
    exit 0
fi

# ---------------------------------------------------------------- setup
[[ -n "$OUT" ]] || { usage >&2; die "--out is required"; }
[[ "$INTERVAL" =~ ^[0-9]*\.?[0-9]+$ ]] || die "--interval must be a number, got '$INTERVAL'"
[[ "$DURATION" =~ ^[0-9]+$ ]]          || die "--duration must be a non-negative integer, got '$DURATION'"
mkdir -p "$OUT" || die "cannot create $OUT"

command -v nvidia-smi >/dev/null 2>&1 || echo "! nvidia-smi not found -- CPU only" >&2

# Driver 5xx renamed clocks_throttle_reasons.* to clocks_event_reasons.*; probe both
THROTTLE_FIELD=""
for f in clocks_event_reasons.active clocks_throttle_reasons.active; do
    if nvidia-smi --query-gpu="$f" --format=csv,noheader -i 0 >/dev/null 2>&1; then
        THROTTLE_FIELD="$f"; break
    fi
done

QUERY="index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.max.sm,temperature.gpu"
[[ -n "$THROTTLE_FIELD" ]] && QUERY="$QUERY,$THROTTLE_FIELD"

# NVLink counters exist only on NVLink-equipped boards; probe once
# Probes capture into a variable rather than piping into `grep -q`: with
# pipefail on, grep exiting early SIGPIPEs nvidia-smi and the whole probe reads
# as a failure, silently disabling the sampler.
NVLINK=""
if command -v nvidia-smi >/dev/null 2>&1; then
    NVLINK_PROBE="$(nvidia-smi nvlink -gt d 2>/dev/null || true)"
    [[ "$NVLINK_PROBE" == *"Data Tx"* ]] && NVLINK=1
fi

# PCIe throughput comes from dmon (query-gpu has no such field on most drivers)
DMON=""
if command -v nvidia-smi >/dev/null 2>&1; then
    DMON_PROBE="$(timeout 10 nvidia-smi dmon -s t -c 1 -d 1 2>/dev/null || true)"
    [[ "$DMON_PROBE" == *rxpci* ]] && DMON=1
fi
DMON_D="$(awk -v v="$INTERVAL" 'BEGIN { d = int(v + 0.5); if (d < 1) d = 1; print d }')"

# DCGM exporter, if the site runs one. This is the only source here that can
# answer what nvidia-smi cannot: PIPE_TENSOR_ACTIVE says whether the tensor
# cores are actually busy, and DRAM_ACTIVE measures the memory interface rather
# than inferring it from power. Everything still works without it.
DCGM=""
if (( NO_DCGM == 0 )) && [[ -n "$DCGM_URL" ]]; then
    DCGM_PROBE="$(curl -sf --max-time 5 "$DCGM_URL" 2>/dev/null | head -c 65536 || true)"
    [[ "$DCGM_PROBE" == *DCGM_FI_* ]] && DCGM=1
fi

# InfiniBand / RoCE ports, read straight from sysfs. Names are resolved once so
# the sample loop is nothing but builtin reads.
IB_PATH=(); IB_DEV=(); IB_PORT=()
IB_CAP=0            # per-port GB/s, from the slowest active port
IB_TOTAL_CAP=0
if [[ -d "$IB_ROOT" ]]; then
    for _p in "$IB_ROOT"/*/ports/*; do
        [[ -r "$_p/counters/port_xmit_data" ]] || continue
        _st=""; [[ -r "$_p/state" ]] && read -r _st < "$_p/state"
        # only ports carrying traffic; a DOWN port would dilute the capacity
        [[ "$_st" == *ACTIVE* || -z "$_st" ]] || continue
        _d="${_p%/ports/*}"
        IB_PATH+=("$_p"); IB_DEV+=("${_d##*/}"); IB_PORT+=("${_p##*/}")
        _rate=""; [[ -r "$_p/rate" ]] && read -r _rate < "$_p/rate"
        # "400 Gb/sec (4X NDR)" -> 50 GB/s
        _cap="$(awk -v s="${_rate:-0}" 'BEGIN { n = s + 0; printf "%.1f", (n > 0 ? n / 8 : 0) }')"
        IB_TOTAL_CAP="$(awk -v a="$IB_TOTAL_CAP" -v b="$_cap" 'BEGIN { printf "%.1f", a + b }')"
        awk -v c="$IB_CAP" -v n="$_cap" 'BEGIN { exit !(c == 0 || (n > 0 && n < c)) }' \
            && IB_CAP="$_cap"
    done
fi
STDBUF=""
command -v stdbuf >/dev/null 2>&1 && STDBUF="stdbuf -oL"

# ---------------------------------------------------------------- static fingerprint
# One `ps` per resident process is enough: capture once, format three ways.
APPS="$(compute_apps)"

# The key=value lines are what diagnose.sh parses; the blocks below them are for humans.
PCIE_GEN="$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')"
PCIE_W="$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')"
# per-lane GB/s after encoding: gen3 0.985, gen4 1.969, gen5 3.938, gen6 7.563
PCIE_CAP="$(awk -v g="${PCIE_GEN:-0}" -v w="${PCIE_W:-0}" 'BEGIN {
    split("1:0.25 2:0.5 3:0.985 4:1.969 5:3.938 6:7.563", a, " ")
    for (i in a) { split(a[i], kv, ":"); if (kv[1] + 0 == g + 0) r = kv[2] }
    printf "%.1f", (r > 0 && w > 0) ? r * w : 0
}')"

{
    echo "date=$(date -Is)"
    echo "epoch=$(date +%s)"
    echo "host=$(hostname)"
    echo "label=${LABEL:-}"
    echo "interval=$INTERVAL duration=${DURATION:-until-interrupted}"
    echo "cores=$(nproc)"
    echo "cpu_model=$(awk -F: '/^model name/ {gsub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo)"
    echo "throttle_field=${THROTTLE_FIELD:-unavailable}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
        echo "gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
        echo "sm_clock_max_mhz=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')"
        echo "pcie_gen=${PCIE_GEN:-?} pcie_width=${PCIE_W:-?}"
        echo "pcie_cap_gbs=${PCIE_CAP:-0}"
        echo "nvlink=${NVLINK:-none} dmon=${DMON:-none} dcgm=${DCGM:-none}"
        echo "dcgm_url=$( ((DCGM)) && echo "$DCGM_URL" || echo none)"
        echo "ib_ports=${#IB_PATH[@]}"
        echo "ib_cap_gbs=${IB_CAP:-0}"
        echo "ib_total_cap_gbs=${IB_TOTAL_CAP:-0}"
        echo "compute_app_users=$(cut -d, -f3 <<<"$APPS" | sort -u | tr '\n' ' ')"
        echo "compute_app_count=$([[ -n "$APPS" ]] && wc -l <<<"$APPS" || echo 0)"
    fi
    echo ""
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "--- nvidia-smi ---"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit,clocks.max.sm,persistence_mode,compute_mode \
                   --format=csv 2>&1
        echo ""
        echo "--- MIG ---"
        nvidia-smi --query-gpu=index,mig.mode.current --format=csv 2>&1
        echo ""
        echo "--- compute apps at start (gpu_uuid,pid,user,used_memory) ---"
        printf '%s\n' "$APPS"
    fi
    if (( ${#IB_PATH[@]} > 0 )); then
        echo ""
        echo "--- InfiniBand ports (device:port, rate, state, link_layer) ---"
        for _i in "${!IB_PATH[@]}"; do
            _p="${IB_PATH[$_i]}"
            _r=""; [[ -r "$_p/rate" ]]       && read -r _r  < "$_p/rate"
            _s=""; [[ -r "$_p/state" ]]      && read -r _s  < "$_p/state"
            _l=""; [[ -r "$_p/link_layer" ]] && read -r _l  < "$_p/link_layer"
            echo "${IB_DEV[$_i]}:${IB_PORT[$_i]}, ${_r:-?}, ${_s:-?}, ${_l:-?}"
        done
        echo ""
        echo "--- IB error counters at start (port,counter,value) ---"
        ib_errors
    fi
} > "$OUT/host.txt"
cat "$OUT/host.txt"

echo "ts,gpu,util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,power_w,power_limit_w,sm_clock_mhz,sm_clock_max_mhz,temp_c,throttle" > "$OUT/gpu.csv"
echo "ts,busy_mean_pct,busy_max_core_pct,max_core,iowait_pct" > "$OUT/cpu.csv"
[[ -n "$NVLINK" ]] && echo "ts,gpu,tx_cum,rx_cum,unit" > "$OUT/nvlink.csv"
# port_rcv_data / port_xmit_data are reported in units of 4 octets, so these are
# already multiplied out to bytes; xmit_wait counts ticks with data to send and
# no credit, which is the congestion signal that matters for PD KV transfer.
(( ${#IB_PATH[@]} > 0 )) && echo "ts,port,rcv_bytes,xmit_bytes,xmit_wait" > "$OUT/ib.csv"
# long format: which DCGM fields an exporter publishes is a site decision, so
# do not bake a column order into the file
[[ -n "$DCGM" ]] && echo "ts,gpu,metric,value" > "$OUT/dcgm.csv"

PREV_STAT="$(</proc/stat)"      # kept in memory: no temp file to clean up or collide on

# ---------------------------------------------------------------- pcie sampler
# dmon runs its own loop; stamping each line as it arrives keeps pcie.csv on the
# same epoch clock as gpu.csv, so diagnose.sh can slice both by work point.
PCIE_PID=""
if [[ -n "$DMON" ]]; then
    echo "ts,gpu,rx_mbps,tx_mbps" > "$OUT/pcie.csv"
    (
        # shellcheck disable=SC2086
        $STDBUF nvidia-smi dmon -s t -d "$DMON_D" 2>/dev/null \
        | while IFS= read -r line; do
            case "$line" in \#*) continue ;; esac
            # shellcheck disable=SC2086
            set -- $line
            [[ $# -ge 3 ]] && printf '%s,%s,%s,%s\n' "$(date +%s)" "$1" "$2" "$3"
        done
    ) >> "$OUT/pcie.csv" &
    PCIE_PID=$!
fi

cleanup() {
    if [[ -n "$PCIE_PID" ]]; then
        pkill -P "$PCIE_PID" 2>/dev/null
        kill "$PCIE_PID" 2>/dev/null
        wait "$PCIE_PID" 2>/dev/null
        PCIE_PID=""
    fi
}

STOP=0
trap 'STOP=1' INT TERM
trap cleanup EXIT

echo "sampling every ${INTERVAL}s -> $OUT  (Ctrl-C to stop and summarize)"
[[ -n "$DMON"   ]] && echo "  + PCIe throughput via nvidia-smi dmon (every ${DMON_D}s)"
[[ -n "$NVLINK" ]] && echo "  + NVLink counters"

# ---------------------------------------------------------------- sample loop
START=$(date +%s)
while (( STOP == 0 )); do
    TS=$(date +%s)
    (( DURATION > 0 && TS - START >= DURATION )) && break

    if [[ -n "$QUERY" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu="$QUERY" --format=csv,noheader,nounits 2>/dev/null \
        | while IFS=, read -r idx ugpu umem mused mtot pdraw plim clk clkmax temp thr; do
            flags=""
            if [[ -n "${thr:-}" ]]; then
                # bitmask, split by severity: software caps are advisory and stay
                # lit on consumer cards, hardware slowdown means real lost clocks
                t=$(( ${thr// /} ))
                (( t & 0x4  )) && flags="${flags}swpower|"
                (( t & 0x20 )) && flags="${flags}swthermal|"
                (( t & 0x8  )) && flags="${flags}hw|"
                (( t & 0x40 )) && flags="${flags}hwthermal|"
                (( t & 0x80 )) && flags="${flags}powerbrake|"
            fi
            clkmax="${clkmax// /}"
            [[ "$clkmax" =~ ^[0-9]+$ ]] || clkmax=0      # [N/A] on some drivers
            echo "$TS,${idx// /},${ugpu// /},${umem// /},${mused// /},${mtot// /},${pdraw// /},${plim// /},${clk// /},${clkmax},${temp// /},${flags:-none}"
        done >> "$OUT/gpu.csv"
    fi

    if [[ -n "$NVLINK" ]]; then
        nvidia-smi nvlink -gt d 2>/dev/null | awk -v ts="$TS" '
            /^GPU [0-9]+:/ { g = $2; gsub(/:/, "", g); next }
            /Data Tx/ { tx[g] += $(NF-1); u[g] = $NF }
            /Data Rx/ { rx[g] += $(NF-1); u[g] = $NF }
            END { for (g in u) printf "%s,%s,%.0f,%.0f,%s\n", ts, g, tx[g], rx[g], u[g] }
        ' >> "$OUT/nvlink.csv"
    fi

    if (( ${#IB_PATH[@]} > 0 )); then
        for i in "${!IB_PATH[@]}"; do
            p="${IB_PATH[$i]}"; rcv=0; xmt=0; wt=0
            [[ -r "$p/counters/port_rcv_data"  ]] && read -r rcv < "$p/counters/port_rcv_data"
            [[ -r "$p/counters/port_xmit_data" ]] && read -r xmt < "$p/counters/port_xmit_data"
            [[ -r "$p/counters/port_xmit_wait" ]] && read -r wt  < "$p/counters/port_xmit_wait"
            # 10# so a leading zero is never read as octal
            printf '%s,%s,%s,%s,%s\n' "$TS" "${IB_DEV[$i]}:${IB_PORT[$i]}" \
                   "$(( 10#${rcv:-0} * 4 ))" "$(( 10#${xmt:-0} * 4 ))" "$(( 10#${wt:-0} ))"
        done >> "$OUT/ib.csv"
    fi

    if [[ -n "$DCGM" ]]; then
        curl -sf --max-time 5 "$DCGM_URL" 2>/dev/null | awk -v ts="$TS" '
            /^DCGM_FI_/ {
                name = $1; sub(/\{.*/, "", name)
                gpu = "?"
                if (match($0, /gpu="[0-9]+"/))
                    gpu = substr($0, RSTART + 5, RLENGTH - 6)
                v = $NF
                if (v + 0 == v) printf "%s,%s,%s,%s\n", ts, gpu, name, v
            }' >> "$OUT/dcgm.csv"
    fi

    # per-core busy% from /proc/stat deltas; the busiest core matters more than the mean
    awk -F' +' -v ts="$TS" '
        NR == FNR { if ($1 ~ /^cpu[0-9]+$/) prev[$1] = $0; next }
        $1 ~ /^cpu[0-9]+$/ {
            split(prev[$1], o, " ")
            tot = 0
            for (i = 2; i <= NF; i++) tot += $i - o[i]
            if (tot <= 0) next
            idle = ($5 - o[5]) + ($6 - o[6])
            busy = 100 * (tot - idle) / tot
            iow  = 100 * ($6 - o[6]) / tot
            n++; sum += busy; iosum += iow
            if (busy > hot) { hot = busy; hotcore = substr($1, 4) }
        }
        END { if (n > 0) printf "%s,%.2f,%.2f,%s,%.2f\n", ts, sum/n, hot, hotcore, iosum/n }
    ' <(printf '%s\n' "$PREV_STAT") /proc/stat >> "$OUT/cpu.csv"
    PREV_STAT="$(</proc/stat)"

    sleep "$INTERVAL"
done

cleanup

# Tenancy can change mid-run: a job that started after us contaminates the tail.
if (( ${#IB_PATH[@]} > 0 )); then
    {
        echo ""
        echo "--- IB error counters at end (port,counter,value) ---"
        ib_errors
    } >> "$OUT/host.txt"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    APPS_END="$(compute_apps)"
    {
        echo ""
        echo "--- compute apps at end (gpu_uuid,pid,user,used_memory) ---"
        printf '%s\n' "$APPS_END"
        echo "compute_app_users_end=$(cut -d, -f3 <<<"$APPS_END" | sort -u | tr '\n' ' ')"
        echo "end_epoch=$(date +%s)"
    } >> "$OUT/host.txt"
fi

summarize "$OUT"
