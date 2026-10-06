#!/bin/bash
# Bottleneck finder -- reads a run directory and says WHERE the ceiling is,
# instead of leaving you to eyeball six CSVs.
#
#   ./diagnose.sh ./aiperf-20260817-143022
#   ./diagnose.sh ./aiperf-20260817-143022 --telemetry ./telemetry   # much stronger
#   ./diagnose.sh ./run --telemetry ./tel --tp 2 --mem-bw 3350       # override the roofline inputs
#
# Combines what each source knows:
#   metrics.csv        -- goodput, pass rate, which SLA is being missed
#   engine_metrics.csv -- queue depth, KV cache usage, preemptions
#   points.csv         -- per-point start/end, used to slice the telemetry
#   gpu/cpu/pcie/nvlink/ib/dcgm.csv -- from gpu_sample.sh on the server node
#   the model's own config.json + the GPU's HBM bandwidth -- the roofline
#
# Each source alone is ambiguous: a low GPU power draw could be a starved
# client, a scheduler cap, or a serial CPU stage, and only the combination
# tells them apart.
#
# The roofline is what makes "memory bandwidth bound" a measurement rather than
# a guess. Every decode step re-reads the weights plus the KV cache, so
#     MBU = bytes_moved_per_step x steps_per_second / peak_HBM_bandwidth
# and an MBU near 1 means the memory system is the ceiling -- no profiler, no
# DCGM, just the model shape and the clock.
#
# Roofline inputs are auto-detected (weights from the model directory, HBM
# bandwidth from the GPU name) and can all be overridden:
#   --weights GiB   weights size of the whole model
#   --tp N          tensor-parallel degree (weights and KV are sharded by this)
#   --mem-bw GBPS   peak HBM bandwidth per GPU
#   --model-dir DIR where to look for config.json / *.safetensors
#
# Runs fine with just metrics.csv, and says more the more you give it.

set -uo pipefail

die()   { echo "x $*" >&2; exit 1; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; }
need()  { [[ $# -ge 2 ]] || die "$1 requires a value"; }

RUN=""
TELEMETRY=""
WEIGHTS_GIB=""
TP=""
MEM_BW=""
MODEL_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --telemetry) need "$@"; TELEMETRY="$2";   shift 2 ;;
        --weights)   need "$@"; WEIGHTS_GIB="$2"; shift 2 ;;
        --tp)        need "$@"; TP="$2";          shift 2 ;;
        --mem-bw)    need "$@"; MEM_BW="$2";      shift 2 ;;
        --model-dir) need "$@"; MODEL_DIR="$2";   shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        -*)          echo "x unknown argument: $1" >&2; usage >&2; exit 1 ;;
        *)           RUN="$1"; shift ;;
    esac
done

[[ -n "$RUN" ]]                || { usage >&2; die "give me a run directory"; }
[[ -d "$RUN" ]]                || die "no such directory: $RUN"
[[ -f "$RUN/metrics.csv" ]]    || die "$RUN/metrics.csv not found -- was this a completed run?"
[[ -z "$TELEMETRY" || -d "$TELEMETRY" ]] || die "no such telemetry directory: $TELEMETRY"
[[ -z "$WEIGHTS_GIB" || "$WEIGHTS_GIB" =~ ^[0-9]*\.?[0-9]+$ ]] \
    || die "--weights must be a number in GiB, got '$WEIGHTS_GIB'"
[[ -z "$MEM_BW" || "$MEM_BW" =~ ^[0-9]*\.?[0-9]+$ ]] \
    || die "--mem-bw must be a number in GB/s, got '$MEM_BW'"
[[ -z "$TP" || ( "$TP" =~ ^[0-9]+$ && "$TP" -gt 0 ) ]] || die "--tp must be a positive integer, got '$TP'"

PY=""
for c in "${AIPERF_VENV:-$(pwd)/aiperf-venv}/bin/python" python3 python; do
    [[ -x "$c" ]] && { PY="$c"; break; }
    command -v "$c" >/dev/null 2>&1 && { PY="$(command -v "$c")"; break; }
done
[[ -n "$PY" ]] || die "no python found"

"$PY" - "$RUN" "$TELEMETRY" "$WEIGHTS_GIB" "$TP" "$MEM_BW" "$MODEL_DIR" <<'PYEOF'
import csv, glob, json, os, re, sys

run  = sys.argv[1]
tel  = sys.argv[2] or None
argw = sys.argv[3] or None
argtp = sys.argv[4] or None
argbw = sys.argv[5] or None
argmd = sys.argv[6] or None

GIB = 1024 ** 3

def num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

_cache = {}
def read_cached(path):
    """gpu.csv is re-sliced once per work point; read it once."""
    if path not in _cache:
        _cache[path] = read_csv(path)
    return _cache[path]

def kv_lines(path):
    """key=value lines out of host.txt. Some lines pack two pairs
    (pcie_gen=4 pcie_width=16) and some hold a value with spaces
    (gpu_name=NVIDIA A100-SXM4-80GB), so the pair count decides how to split."""
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("-") or "=" not in line:
            continue
        pairs = re.findall(r'(\w+)=(\S*)', line)
        if len(pairs) > 1:
            for k, v in pairs:
                out[k] = v.strip("'")
        else:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'")
    return out

# ---------------------------------------------------------------- inputs
points = read_csv(os.path.join(run, "metrics.csv"))
if not points:
    print("x metrics.csv is empty -- every work point failed"); sys.exit(1)

env, cfg = {}, {}
envp = os.path.join(run, "env.txt")
if os.path.isfile(envp):
    text = open(envp).read()
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip("'")
    for k, v in re.findall(r'\b(isl|osl|anchor_c|shape_c|req_per_c|total_requests)=(\d+)', text):
        cfg[k] = int(v)

host = kv_lines(os.path.join(tel, "host.txt")) if tel else {}

sla = {}
for tok in env.get("sla", "").split():
    if ":" in tok:
        k, v = tok.rsplit(":", 1)
        if num(v) is not None:
            sla[k] = num(v)
ttft_thr = sla.get("time_to_first_token")
e2e_thr  = sla.get("request_latency")
itl_thr  = sla.get("inter_token_latency")     # the Qwen3 PD gate uses this, not request_latency

# ---------------------------------------------------------------- engine metrics
# A PD deployment has one endpoint per worker, and how they combine depends on
# the metric: a cache-usage *fraction* summed over six workers reads 600%, while
# a queue depth only means anything as a fleet total. So collect per endpoint
# first, then combine per kind.
def kind_of(series):
    s = series.lower()
    if "cache_usage" in s or "token_usage" in s:                  return "kv"
    if "waiting" in s or "queue" in s or "pending" in s:          return "queue"
    if "preempt" in s:                                            return "preempt"
    if "running" in s:                                            return "running"
    return None

raw = {}
for r in read_csv(os.path.join(run, "engine_metrics.csv")):
    ep, sep, series = r["metric"].partition("|")
    if not sep:                      # single-endpoint runs have no label
        ep, series = "", r["metric"]
    k = kind_of(series)
    if not k:
        continue
    d = raw.setdefault(r["point"], {}).setdefault(k, {})
    if k == "preempt":               # a counter: accumulate within the endpoint
        d[ep] = d.get(ep, 0.0) + num(r["delta"], 0.0)
    else:                            # a gauge: the endpoint's own peak
        d[ep] = max(d.get(ep, 0.0), num(r["peak"], 0.0))

engine = {}
for point, kinds in raw.items():
    e = engine.setdefault(point, {})
    if "kv" in kinds:      e["kv"] = max(kinds["kv"].values())          # worst worker
    if "queue" in kinds:   e["queue"] = sum(kinds["queue"].values())    # fleet backlog
    if "preempt" in kinds: e["preempt"] = sum(kinds["preempt"].values())
    if "running" in kinds:
        e["running"] = sum(kinds["running"].values())        # in flight fleet-wide
        e["running_max"] = max(kinds["running"].values())    # per worker
        e["endpoints"] = len(kinds["running"])

# ---------------------------------------------------------------- clock alignment
# points.csv is stamped by the client's clock, the telemetry by the server's.
# Unsynchronized clocks silently slice the wrong window, so correct by the skew
# run.sh measured from the server's HTTP Date header, and verify the result.
skew = num(env.get("server_clock_skew_s"), 0.0) or 0.0

window, requested, duration = {}, {}, {}
for r in read_csv(os.path.join(run, "points.csv")):
    s, e = num(r.get("start_ts")), num(r.get("end_ts"))
    if s is not None and e is not None:
        window[r["point"]] = (s + skew, e + skew)
    # the shape sweeps deliberately differ from the run-wide --isl/--osl, so the
    # target for each point is per-point, not global
    requested[r["point"]] = {"isl": num(r.get("isl")), "osl": num(r.get("osl")),
                             "conc": num(r.get("concurrency"))}
    if s is not None and e is not None:
        duration[r["point"]] = e - s

align_note = []
if tel and window:
    gts = [num(r["ts"]) for r in read_cached(os.path.join(tel, "gpu.csv")) if num(r["ts"])]
    if gts:
        tlo, thi = min(gts), max(gts)
        # warmup is excluded from the results, so its coverage is not interesting
        measured = {k: v for k, v in window.items() if k != "warmup"}
        covered = sum(1 for lo, hi in measured.values() if thi >= lo and tlo <= hi)
        if covered == 0:
            run_mid = sum(sum(w) / 2 for w in measured.values()) / len(measured)
            tel_mid = (tlo + thi) / 2
            align_note.append(
                "telemetry does not overlap the run at all (off by %.0f s)" % (tel_mid - run_mid))
            align_note.append(
                "either the sampler ran at a different time, or the two nodes' clocks disagree")
            align_note.append(
                "if the clocks are the problem: ./diagnose.sh ... after adding "
                "server_clock_skew_s=%.0f to %s/env.txt" % (tel_mid - run_mid, run))
        elif covered < len(measured):
            align_note.append("telemetry covers only %d of %d work points -- the rest have no "
                              "GPU evidence" % (covered, len(measured)))
        if abs(skew) > 2:
            align_note.append("applied a %+.0f s clock skew measured against the server" % skew)

def slice_telemetry(point):
    """-> dict of GPU/CPU/interconnect aggregates for this point's time window."""
    out = {}
    if not tel or point not in window:
        return out
    lo, hi = window[point]
    inwin = lambda rows: [r for r in rows if lo <= (num(r.get("ts")) or 0) <= hi]

    rows = inwin(read_cached(os.path.join(tel, "gpu.csv")))
    if rows:
        per_gpu, hard, soft, clk_low, clk_n = {}, 0, 0, 0, 0
        for r in rows:
            g = r["gpu"]
            p, lim = num(r["power_w"], 0.0), num(r["power_limit_w"], 0.0)
            d = per_gpu.setdefault(g, {"ratio": [], "util": [], "membw": [], "vram": []})
            if lim > 0:
                d["ratio"].append(p / lim)
            d["util"].append(num(r["util_gpu_pct"], 0.0))
            # two unrelated memory signals: controller busy time (bandwidth
            # proxy) and footprint (capacity). Do not conflate them.
            d["membw"].append(num(r["util_mem_pct"], 0.0))
            used, total = num(r["mem_used_mib"], 0.0), num(r["mem_total_mib"], 0.0)
            if total > 0:
                d["vram"].append(used / total)
            # Throttle FLAGS are permanently lit on thermally-limited cards, so
            # the clock ratio is the evidence and the flags are only a hint.
            cmax = num(r.get("sm_clock_max_mhz"), 0.0) or 0.0
            if cmax > 0:
                clk_n += 1
                if (num(r.get("sm_clock_mhz"), 0.0) or 0.0) / cmax < 0.85:
                    clk_low += 1
            f = r.get("throttle", "none") or "none"
            if re.search(r'\bhw|powerbrake', f):
                hard += 1
            elif f != "none":
                soft += 1
        def avg_of(key):
            per = [sum(d[key]) / len(d[key]) for d in per_gpu.values() if d[key]]
            return sum(per) / len(per) if per else None
        for src, dst in (("ratio", "power_ratio"), ("util", "util"), ("membw", "membw")):
            v = avg_of(src)
            if v is not None:
                out[dst] = v
        peaks = [max(d["vram"]) for d in per_gpu.values() if d["vram"]]
        if peaks:
            out["vram"] = max(peaks)
        out["hard_pct"] = 100.0 * hard / len(rows)
        out["soft_pct"] = 100.0 * soft / len(rows)
        if clk_n:
            out["clk_low_pct"] = 100.0 * clk_low / clk_n

    crows = inwin(read_cached(os.path.join(tel, "cpu.csv")))
    if crows:
        out["cpu_mean_peak"] = max(num(r["busy_mean_pct"], 0.0) for r in crows)
        out["cpu_hot"]       = max(num(r["busy_max_core_pct"], 0.0) for r in crows)

    # interconnect: peak of (rx, tx) across GPUs, as a fraction of link capacity
    prows = inwin(read_cached(os.path.join(tel, "pcie.csv")))
    cap = num(host.get("pcie_cap_gbs"), 0.0) or 0.0
    if prows:
        peak = max(max(num(r["rx_mbps"], 0.0), num(r["tx_mbps"], 0.0)) for r in prows) / 1000.0
        out["pcie_gbs"] = peak
        if cap > 0:
            out["pcie_ratio"] = peak / cap

    # DCGM profiling counters, where a site runs the exporter. These are the only
    # measurements here that answer occupancy directly instead of inferring it.
    drows = inwin(read_cached(os.path.join(tel, "dcgm.csv")))
    if drows:
        acc = {}
        for r in drows:
            acc.setdefault(r["metric"], []).append(num(r["value"], 0.0))
        for field, key in (("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "tensor"),
                           ("DCGM_FI_PROF_DRAM_ACTIVE", "dram"),
                           ("DCGM_FI_PROF_GR_ENGINE_ACTIVE", "gr"),
                           ("DCGM_FI_PROF_SM_ACTIVE", "sm")):
            v = acc.get(field)
            if v:
                out["dcgm_" + key] = sum(v) / len(v)

    # InfiniBand: the cross-node path, which is where PD disaggregation moves KV
    irows = inwin(read_cached(os.path.join(tel, "ib.csv")))
    if len(irows) >= 2:
        by_port = {}
        for r in irows:
            by_port.setdefault(r["port"], []).append(r)
        ib_cap = num(host.get("ib_cap_gbs"), 0.0) or 0.0
        peak = rx_tot = tx_tot = wait = 0.0
        wrapped = 0
        for _, rs in by_port.items():
            dt = num(rs[-1]["ts"], 0) - num(rs[0]["ts"], 0)
            # first/last is exact for a monotonic counter even if samples were
            # dropped, but a window this short cannot resolve a rate at all
            if dt < 2:
                continue
            rx = (num(rs[-1]["rcv_bytes"], 0) - num(rs[0]["rcv_bytes"], 0)) / dt / 1e9
            tx = (num(rs[-1]["xmit_bytes"], 0) - num(rs[0]["xmit_bytes"], 0)) / dt / 1e9
            if rx < 0 or tx < 0:      # 32-bit counters wrap at high rates
                wrapped += 1
                continue
            rx_tot += rx; tx_tot += tx
            peak = max(peak, rx, tx)
            wait += max(0.0, num(rs[-1]["xmit_wait"], 0) - num(rs[0]["xmit_wait"], 0))
        if peak:
            out["ib_gbs"] = peak
            out["ib_node_gbs"] = max(rx_tot, tx_tot)
            if ib_cap > 0:
                out["ib_ratio"] = peak / ib_cap
        out["ib_wait"] = wait
        if wrapped:
            out["ib_wrapped"] = wrapped

    nrows = inwin(read_cached(os.path.join(tel, "nvlink.csv")))
    if len(nrows) >= 2:
        by_gpu = {}
        for r in nrows:
            by_gpu.setdefault(r["gpu"], []).append(r)
        rates = []
        for g, rs in by_gpu.items():
            dt = num(rs[-1]["ts"], 0) - num(rs[0]["ts"], 0)
            if dt <= 0:
                continue
            unit = (rs[0].get("unit") or "KiB")
            k = 1048576 if "MiB" in unit else (1073741824 if "GiB" in unit else 1024)
            for f in ("tx_cum", "rx_cum"):
                d = num(rs[-1][f], 0) - num(rs[0][f], 0)
                if d > 0:
                    rates.append(d * k / dt / 1e9)
        if rates:
            out["nvlink_gbs"] = max(rates)
    return out

# ---------------------------------------------------------------- roofline
# Peak HBM bandwidth in GB/s. Only used when --mem-bw is not given; the point is
# to make the common cards work with no arguments, not to be exhaustive.
HBM = [("H200", 4800), ("H100 NVL", 3900), ("H100 PCIE", 2000), ("H100", 3350),
       ("GH200", 4000), ("GB200", 8000), ("B200", 8000), ("B100", 8000),
       ("A100-PCIE-40", 1555), ("A100-SXM4-40", 1555), ("A100 40", 1555),
       ("A100-PCIE-80", 1935), ("A100", 2039),
       ("L40S", 864), ("L40", 864), ("L4", 300), ("A10G", 600), ("A10", 600),
       ("A30", 933), ("A40", 696), ("V100", 900),
       ("RTX 6000 ADA", 960), ("RTX PRO 6000", 1792), ("RTX 5090", 1792),
       ("RTX 4090", 1008), ("RTX 3090", 936), ("1080 TI", 484)]

def hbm_for(name):
    up = (name or "").upper()
    for key, bw in HBM:
        if key in up:
            return bw, key
    return None, None

roof = {"ok": False, "notes": []}

model_dir = argmd or env.get("tokenizer") or ""
weights_bytes = None
if argw:
    weights_bytes = float(argw) * GIB
    roof["notes"].append("weights %.1f GiB (given)" % float(argw))
elif model_dir and os.path.isdir(model_dir):
    # on-disk weight files are the served weights, in the served dtype
    for pat in ("*.safetensors", "*.bin", "*.pth", "*.pt", "*.gguf"):
        files = [f for f in glob.glob(os.path.join(model_dir, pat))
                 if "training_args" not in os.path.basename(f)]
        if files:
            weights_bytes = float(sum(os.path.getsize(f) for f in files))
            roof["notes"].append("weights %.1f GiB (summed %d %s in %s)"
                                 % (weights_bytes / GIB, len(files), pat.lstrip("*."),
                                    os.path.basename(model_dir.rstrip("/"))))
            break

tp = int(argtp) if argtp else None
if tp is None:
    tp = int(num(host.get("gpu_count"), 1) or 1)
    if tp > 1:
        roof["notes"].append("tp=%d assumed from the %d GPUs on the server node -- "
                             "pass --tp if that is wrong" % (tp, tp))

bw_gbs = num(argbw)
if bw_gbs:
    roof["notes"].append("HBM %.0f GB/s (given)" % bw_gbs)
else:
    bw_gbs, matched = hbm_for(host.get("gpu_name"))
    if bw_gbs:
        roof["notes"].append("HBM %.0f GB/s (%s)" % (bw_gbs, host.get("gpu_name", matched)))
    elif host.get("gpu_name"):
        roof["notes"].append("no HBM bandwidth known for '%s' -- pass --mem-bw to enable the roofline"
                             % host["gpu_name"])

# KV traffic per sequence per decode step, straight from the model's own config
kv_per_seq_per_tok = None
if model_dir and os.path.isfile(os.path.join(model_dir, "config.json")):
    try:
        c = json.load(open(os.path.join(model_dir, "config.json")))
        c = c.get("text_config", c)
        L  = c.get("num_hidden_layers")
        H  = c.get("num_attention_heads")
        KH = c.get("num_key_value_heads", H)
        hd = c.get("head_dim") or (c["hidden_size"] // H if H and c.get("hidden_size") else None)
        dt = {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1}.get(
            str(c.get("torch_dtype", "bfloat16")), 2)
        if L and KH and hd:
            kv_per_seq_per_tok = 2.0 * L * KH * hd * dt      # K and V, per token of context
            roof["notes"].append("KV %.1f KiB per token of context (L=%d kv_heads=%d head_dim=%d)"
                                 % (kv_per_seq_per_tok / 1024, L, KH, hd))
    except (ValueError, KeyError, TypeError, OSError):
        pass

if weights_bytes and bw_gbs:
    roof.update(ok=True, weights=weights_bytes, bw=bw_gbs * 1e9, tp=tp,
                kv_tok=kv_per_seq_per_tok)
    if kv_per_seq_per_tok is None:
        roof["notes"].append("no config.json -- KV traffic excluded, so MBU is a lower bound")

def mbu_for(p, eng):
    """Model Bandwidth Utilization: bytes each GPU must move per decode step,
    times steps per second, over peak HBM bandwidth."""
    if not roof["ok"]:
        return None
    itl = num(p.get("itl_avg"))
    if not itl or itl <= 0:
        return None
    # Each worker re-reads its own weights, so the batch that matters for MBU is
    # the per-worker one -- not the fleet total, which would overstate it by the
    # number of workers.
    batch = num(eng.get("running_max")) or num(eng.get("running")) \
            or num(p.get("concurrency")) or 1
    # per point, never the run-wide setting: the shape sweeps change ISL and OSL
    req = requested.get(p["point"], {})
    isl = num(p.get("isl")) or req.get("isl") or cfg.get("isl", 0) or 0
    osl = num(p.get("osl")) or req.get("osl") or cfg.get("osl", 0) or 0
    ctx = isl + osl / 2.0
    kv_bytes = batch * roof["kv_tok"] * ctx if (roof["kv_tok"] and ctx) else 0.0
    per_step = roof["weights"] + kv_bytes
    # How much of that traffic is KV decides what a high MBU *means*. Weight
    # reads are a fixed cost per step that a bigger batch amortizes; KV traffic
    # is not. Stashed on the point so classify() can tell the two regimes apart.
    p["_kv_share"] = kv_bytes / per_step if per_step else None
    return (per_step / roof["tp"]) * (1000.0 / itl) / roof["bw"]

# ---------------------------------------------------------------- per-point verdict
def classify(p, eng, gpu, mbu):
    """Ordered most-conclusive first; each rule needs evidence the others lack."""
    conc    = num(p.get("concurrency"))
    kv      = eng.get("kv")
    queue   = eng.get("queue")
    preempt = eng.get("preempt")
    running = eng.get("running")
    power   = gpu.get("power_ratio")
    hot     = gpu.get("cpu_hot")
    membw   = gpu.get("membw")
    hard    = gpu.get("hard_pct")
    clklow  = gpu.get("clk_low_pct")
    link    = gpu.get("pcie_ratio")
    ib      = gpu.get("ib_ratio")

    # Lost clocks, not lit flags: software power/thermal caps are permanently
    # asserted on many cards while the SMs still run at full speed.
    # An idle GPU parks its clocks; that is not throttling. Only trust the clock
    # ratio when the card is actually working.
    if clklow is not None and clklow > 20 and (power is None or power >= 0.3):
        return ("THROTTLING", "SM clocks below 85%% of maximum in %.0f%% of samples while drawing "
                              "%s of board power -- this number is not reproducible"
                % (clklow, "unknown" if power is None else "%.0f%%" % (power * 100)))
    if hard is not None and hard > 5:
        return ("THROTTLING", "hardware slowdown asserted in %.0f%% of samples" % hard)
    if preempt is not None and preempt > 0 and (kv is None or kv > 0.85):
        return ("KV CACHE", "%.0f preemptions during the point%s -- cache oversubscribed, latency spikes follow"
                % (preempt, "" if kv is None else ", KV at %.0f%%" % (kv * 100)))
    # The roofline settles compute-vs-bandwidth outright, so it outranks the proxies.
    # But a saturated memory system means opposite things depending on what is
    # moving through it, so split on the KV share before calling it bandwidth.
    share = p.get("_kv_share")
    if mbu is not None and mbu >= 0.6 and (share is None or share >= 0.5):
        return ("MEMORY BW", "MBU %.0f%% and %s of that traffic is KV -- more batching would only "
                             "add more; fewer bytes per token is the only lever"
                % (mbu * 100, "most" if share is None else "%.0f%%" % (share * 100)))
    if mbu is not None and mbu >= 0.6:
        detail = ("MBU %.0f%%, but only %.0f%% of it is KV -- the rest is the weights being re-read "
                  "for a batch of %.0f" % (mbu * 100, (share or 0) * 100, conc or 0))
        if queue is not None and queue > 1:
            return ("SCHEDULER CAP", detail + "; %.0f requests are queued, so raising max_num_seqs "
                                              "amortizes those reads over more of them" % queue)
        return ("WEIGHT BOUND", detail + "; a bigger batch spreads that fixed cost -- push "
                                         "concurrency up before touching the model")
    # Either fabric can be the ceiling; name the one that is actually saturated.
    fabric = max([(v, n) for v, n in ((link, "PCIe"), (ib, "InfiniBand")) if v is not None],
                 default=None)
    if fabric and fabric[0] >= 0.7 and (power is None or power < 0.85):
        return ("INTERCONNECT", "%s peaked at %.0f%% of link capacity while the GPU had power headroom"
                % (fabric[1], fabric[0] * 100))
    # A busy tensor pipe is direct evidence; board power is only a proxy for it.
    tensor = gpu.get("dcgm_tensor")
    if tensor is not None and tensor >= 0.5:
        return ("GPU COMPUTE", "tensor pipe active %.0f%% of cycles -- measured, not inferred; "
                               "this really is compute" % (tensor * 100))
    if power is not None and power >= 0.85 and (tensor is None or tensor >= 0.2):
        note = "power at %.0f%% of board limit -- genuinely saturated" % (power * 100)
        if mbu is not None:
            note += " (MBU %.0f%%, so this is compute, not bandwidth)" % (mbu * 100)
        return ("GPU COMPUTE", note)
    if membw is not None and membw >= 70 and (power is None or power < 0.85) and mbu is None:
        return ("MEMORY BW?", "memory controller busy %.0f%% of the time at only %s of power -- "
                              "looks bandwidth bound, but this is a proxy: give the roofline inputs to confirm"
                % (membw, "unknown power" if power is None else "%.0f%%" % (power * 100)))
    if (queue is not None and queue > 1 and (kv is None or kv < 0.6)
            and (power is None or power < 0.8)):
        return ("SCHEDULER CAP", "queue peaked at %.0f while KV%s and the GPU had headroom -- raise max_num_seqs"
                % (queue, " sat at %.0f%%" % (kv * 100) if kv is not None else " was low"))
    if hot is not None and hot >= 95 and (power is None or power < 0.6):
        return ("SERIAL CPU", "one core pegged at %.0f%% with the GPU idle -- HTTP frontend or detokenization" % hot)
    if (running is not None and conc and running < conc * 0.7
            and (queue is None or queue < 1)):
        return ("CLIENT BOUND", "server only ran %.0f of %.0f requested concurrently, with an empty queue -- "
                                "the load generator is not keeping up" % (running, conc))
    if (num(p.get("pass_rate"), 0) >= 0.99 and (queue is None or queue < 1)
            and not preempt):
        detail = "SLA met with no queueing"
        if power is not None:
            detail += ", GPU at %.0f%% of power limit" % (power * 100)
        if mbu is not None:
            detail += ", MBU %.0f%%" % (mbu * 100)
        return ("HEADROOM", detail + " -- push concurrency higher")
    return ("UNCLEAR", "no single resource stands out" + ("" if tel else " (add --telemetry for GPU/CPU evidence)"))

rows = []
for p in points:
    name = p["point"]
    conc = requested.get(name, {}).get("conc")
    if conc:
        p["concurrency"] = "%d" % conc
    else:
        m = re.search(r'(\d+)$', name)      # pre-shape-sweep runs have no such column
        p["concurrency"] = m.group(1) if m else ""
    eng, gpu = engine.get(name, {}), slice_telemetry(name)
    mbu = mbu_for(p, eng)
    kind, why = classify(p, eng, gpu, mbu)
    rows.append((p, eng, gpu, kind, why, mbu))

sweep = [r for r in rows if r[0]["point"].startswith("c")]
sweep.sort(key=lambda r: num(r[0]["concurrency"], 0))

# ---------------------------------------------------------------- report
print("")
print("=" * 100)
print(" bottleneck report -- %s" % run)
print("=" * 100)
if env:
    print(" model: %s" % env.get("model", "?"))
    print(" shape: ISL=%s OSL=%s" % (cfg.get("isl", "?"), cfg.get("osl", "?")))
    print(" SLA:   %s" % (env.get("sla") or "not recorded"))
    if itl_thr:
        print(" -> the gate is per-token: average ITL must stay under %.0f ms%s"
              % (itl_thr, "" if not cfg.get("osl") else
                 ", so a %d-token output has %.1f s of decode budget"
                 % (cfg["osl"], itl_thr * cfg["osl"] / 1000.0)))
    elif ttft_thr and e2e_thr and cfg.get("osl", 0) > 1:
        print(" -> at the TTFT limit that leaves %.1f ms per output token (ITL budget)"
              % ((e2e_thr - ttft_thr) / (cfg["osl"] - 1)))
if not tel:
    print(" ! no --telemetry given: GPU, CPU and interconnect evidence unavailable, verdicts are weaker")
for n in roof["notes"]:
    print(" roofline: %s" % n)
if roof["ok"]:
    print(" roofline: ITL floor %.1f ms/token at tp=%d from the weights alone -- KV traffic pushes"
          " it higher as context and batch grow"
          % (1000.0 * (roof["weights"] / roof["tp"]) / roof["bw"], roof["tp"]))
else:
    print(" roofline: not enough inputs -- compute-vs-bandwidth stays a guess "
          "(need weights + HBM bandwidth; see --weights / --mem-bw)")
for n in align_note:
    print(" ! %s" % n)

print("")
hdr = (" %-8s %9s %6s %6s %6s %7s %6s %5s %6s %7s %6s %6s %6s %6s  %s" %
       ("point", "goodput", "pass", "kv", "queue", "preempt",
        "power", "clk", "dram", "tensor", "mbu", "vram", "pcie", "ib", "verdict"))
print(hdr)
def pct(v, scale=1.0):
    return "-" if v is None else "%.0f%%" % (v * scale)
for p, eng, gpu, kind, _, mbu in sweep:
    clk = gpu.get("clk_low_pct")
    print(" %-8s %9s %6s %6s %6s %7s %6s %5s %6s %7s %6s %6s %6s %6s  %s" % (
        p["point"],
        ("%.2f" % num(p["goodput"], 0)) if p.get("goodput") else "-",
        ("%.2f" % num(p["pass_rate"], 0)) if p.get("pass_rate") else "-",
        pct(eng.get("kv"), 100),
        ("%.0f" % eng["queue"]) if "queue" in eng else "-",
        ("%.0f" % eng["preempt"]) if "preempt" in eng else "-",
        pct(gpu.get("power_ratio"), 100),
        "-" if clk is None else ("ok" if clk <= 20 else "LOW"),
        # DRAM_ACTIVE where DCGM is available, the nvidia-smi proxy otherwise
        pct(gpu.get("dcgm_dram"), 100) if "dcgm_dram" in gpu else pct(gpu.get("membw")),
        pct(gpu.get("dcgm_tensor"), 100),
        pct(mbu, 100),
        pct(gpu.get("vram"), 100),
        pct(gpu.get("pcie_ratio"), 100),
        pct(gpu.get("ib_ratio"), 100),
        kind))
dram_src = "DCGM DRAM_ACTIVE, measured" if any("dcgm_dram" in g for _, _, g, _, _, _ in sweep) \
           else "nvidia-smi utilization.memory, a proxy"
print("   clk = SM clock held at >=85%% of maximum;  dram = %s;  tensor = DCGM tensor pipe;" % dram_src)
print("   mbu = fraction of peak HBM bandwidth the roofline says you are using;"
      "  pcie / ib = peak traffic vs link capacity")

# ---------------------------------------------------------------- headline
print("")
print("-" * 100)
scored = [(num(p["goodput"], 0), p, eng, gpu, kind, why, mbu)
          for p, eng, gpu, kind, why, mbu in sweep if p.get("goodput")]
if not scored:
    print(" no goodput values parsed -- check the run output"); sys.exit(1)

best = max(scored, key=lambda t: t[0])
bg, bp, beng, bgpu, bkind, bwhy, bmbu = best
print(" peak goodput  %.2f req/s at %s (pass rate %s)" % (bg, bp["point"], bp.get("pass_rate", "?")))
osl_at_peak = num(bp.get("osl")) or cfg.get("osl")
if osl_at_peak:
    print(" scored metric %.0f good output tok/s (goodput x OSL %.0f)"
          % (bg * osl_at_peak, osl_at_peak))
gate = num(bp.get("pass_rate"))
if gate is not None and gate < 0.90:
    print(" ! pass rate %.3f is below the 0.90 gate -- this point would not be scored at all"
          % gate)
print(" bottleneck    %s -- %s" % (bkind, bwhy))

if bmbu is not None and bkind not in ("MEMORY BW",):
    print(" roofline      MBU %.0f%% at the peak: %s"
          % (bmbu * 100,
             "the memory system still has room, so the ceiling is elsewhere" if bmbu < 0.6
             else "bandwidth is close to the limit"))
# Two independent answers to the same question. MBU is computed from the model's
# shape and the measured ITL; DRAM_ACTIVE is a hardware counter. They are not the
# same quantity -- the interface can be active without moving peak bytes -- but
# they cannot disagree wildly and both be right.
bdram = bgpu.get("dcgm_dram")
if bmbu is not None and bdram is not None:
    print(" cross-check   roofline MBU %.0f%% vs measured DRAM_ACTIVE %.0f%% of cycles"
          % (bmbu * 100, bdram * 100))
    # Compare the gap, not two absolute thresholds. DRAM_ACTIVE counts cycles the
    # interface was busy and MBU counts bytes against peak, so DRAM_ACTIVE
    # normally sits at or above MBU; it is the size of the disagreement that
    # means something, at any level.
    if bmbu - bdram > 0.25:
        print("               -> the roofline claims %.0f points more of the memory system than the"
              % ((bmbu - bdram) * 100))
        print("                  hardware saw. Check --weights, --tp and --mem-bw.")
    elif bdram - bmbu > 0.35:
        print("               -> the hardware is moving far more than the model shape accounts")
        print("                  for. On an MoE this is the signature of a batch touching many")
        print("                  more experts than the active-parameter count assumes -- re-run")
        print("                  with --weights set to the whole checkpoint for the upper bound.")
    else:
        print("               -> consistent; the bandwidth reading is corroborated by hardware")

btensor = bgpu.get("dcgm_tensor")
if btensor is not None and bmbu is not None and btensor < 0.2 and bmbu >= 0.6:
    print(" cross-check   tensor pipe at %.0f%% while bandwidth is near the limit -- the classic"
          % (btensor * 100))
    print("               decode signature. More FLOPs would change nothing here.")

if bmbu is not None and bmbu > 1.05:
    print(" ! MBU above 100% -- one of the roofline inputs is wrong (weights, --tp, or --mem-bw),")
    print("   most often --tp: the weights are sharded, so the per-GPU footprint is weights/tp")

# Capacity is a separate question from bandwidth: VRAM sitting unused while the
# KV cache is full is pure wasted headroom, and the cheapest fix there is.
kv_peak   = max([e.get("kv", 0) for _, e, _, _, _, _ in sweep] or [0])
vram_peak = max([g.get("vram", 0) for _, _, g, _, _, _ in sweep] or [0])
if vram_peak and kv_peak:
    if kv_peak > 0.9 and vram_peak < 0.85:
        print(" memory        KV cache hit %.0f%% while only %.0f%% of VRAM was in use"
              % (kv_peak * 100, vram_peak * 100))
        print("               -> raise gpu_memory_utilization; you are leaving %.0f%% of the card unused"
              % ((0.95 - vram_peak) * 100))
    elif vram_peak > 0.97:
        print(" memory        VRAM peaked at %.0f%% -- close to OOM, back off gpu_memory_utilization"
              % (vram_peak * 100))

link_peak = max([g.get("pcie_ratio", 0) for _, _, g, _, _, _ in sweep] or [0])
nvl_peak  = max([g.get("nvlink_gbs", 0) for _, _, g, _, _, _ in sweep] or [0])
if link_peak > 0.5:
    print(" interconnect  PCIe peaked at %.0f%% of link capacity -- for TP or PD disaggregation this"
          % (link_peak * 100))
    print("               is a real ceiling; check the topology (nvidia-smi topo -m)")
if nvl_peak:
    print(" interconnect  NVLink peaked at %.1f GB/s" % nvl_peak)

ib_peak  = max([g.get("ib_ratio", 0) for _, _, g, _, _, _ in sweep] or [0])
ib_gbs   = max([g.get("ib_gbs", 0) for _, _, g, _, _, _ in sweep] or [0])
ib_node  = max([g.get("ib_node_gbs", 0) for _, _, g, _, _, _ in sweep] or [0])
ib_wait  = max([g.get("ib_wait", 0) for _, _, g, _, _, _ in sweep] or [0])
if ib_gbs:
    line = " interconnect  InfiniBand peaked at %.1f GB/s per port" % ib_gbs
    if ib_peak:
        line += " (%.0f%% of link)" % (ib_peak * 100)
    if ib_node:
        line += ", %.1f GB/s across the node" % ib_node
    print(line)
    if ib_peak >= 0.5:
        print("               PD disaggregation moves the KV cache over this link -- at this")
        print("               level it is a real constraint on the P:D split and placement")
if ib_wait:
    print(" interconnect  IB xmit_wait grew by %.0f: ports had data to send and no credit." % ib_wait)
    print("               That is congestion or credit starvation, not raw bandwidth --")
    print("               check the routing and the P/D traffic pattern, not the link speed")

def violated(p):
    out = []
    if ttft_thr and num(p.get("ttft_p90")) and num(p["ttft_p90"]) > ttft_thr:
        out.append("TTFT p90 %.0fms > %.0fms" % (num(p["ttft_p90"]), ttft_thr))
    if e2e_thr and num(p.get("e2e_p90")) and num(p["e2e_p90"]) > e2e_thr:
        out.append("e2e p90 %.0fms > %.0fms" % (num(p["e2e_p90"]), e2e_thr))
    # A per-token gate is judged on the per-token number, not end-to-end latency,
    # and on the tail: the gate is applied per request, so p90 is what decides
    # whether a tenth of them fail.
    if itl_thr:
        for field, label in (("itl_p90", "ITL p90"), ("itl_avg", "ITL avg")):
            v = num(p.get(field))
            if v:
                if v > itl_thr:
                    out.append("%s %.1fms > %.0fms" % (label, v, itl_thr))
                break
    return out

after = [t for t in scored if num(t[1]["concurrency"], 0) > num(bp["concurrency"], 0)]
probe = after[0] if after else best
v = violated(probe[1])
if v:
    print(" SLA missed    at %s: %s" % (probe[1]["point"], "; ".join(v)))
    slow_decode = any(("e2e" in x or "ITL" in x) for x in v)
    if any("TTFT" in x for x in v) and not slow_decode:
        print("               -> prefill contention. Try chunked prefill, or more prefill workers.")
    elif slow_decode and not any("TTFT" in x for x in v):
        print("               -> decode is too slow (ITL). Smaller decode batch, more decode")
        print("                  workers, or speculative decoding.")
    else:
        print("               -> both stages are over budget; you are past the knee.")
else:
    # The gate is applied per request; aggregates can clear it while the tail does not.
    pr = num(probe[1].get("pass_rate"))
    if pr is not None and pr < 0.99:
        print(" ! %s clears every aggregate threshold, yet only %.0f%% of its requests passed"
              % (probe[1]["point"], pr * 100))
        print("   the gate is per request, so the distribution's tail is what fails --")
        print("   compare ttft_p99 %s and itl_p99 %s against the thresholds"
              % (probe[1].get("ttft_p99", "?"), probe[1].get("itl_p99", "?")))

if sweep and sweep[-1][0]["point"] == bp["point"] and len(sweep) > 1:
    prev = num(sweep[-2][0].get("goodput"), 0)
    if bg > prev:
        print(" ! goodput was still rising at the largest concurrency tested (%s)."
              % bp["concurrency"])
        print("   Extend --sweep upward; the real peak is past your range.")
lo = [t for t in scored if num(t[1].get("pass_rate"), 1) < 0.99]
if lo:
    first = min(lo, key=lambda t: num(t[1]["concurrency"], 0))
    print(" SLA cliff     pass rate first drops below 0.99 at %s (%.3f)"
          % (first[1]["point"], num(first[1]["pass_rate"], 0)))

# ---------------------------------------------------------------- perturbation
# The concurrency sweep can only correlate a bottleneck with load. These rounds
# vary one shape dimension at a fixed concurrency, so the response either
# matches the mechanism the verdict claims or it falsifies it.
def slope(xs, ys):
    """least-squares dy/dx; None when the x values do not actually vary"""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

def shape_points(prefix):
    out = []
    for p, eng, gpu, kind, why, mbu in rows:
        if not re.match(prefix + r'\d+$', p["point"]):
            continue
        req = requested.get(p["point"], {})
        x = num(p.get(prefix)) or req.get(prefix)      # measured, else requested
        itl, ttft = num(p.get("itl_avg")), num(p.get("ttft_avg"))
        if x and itl:
            out.append({"x": x, "p": p, "itl": itl, "ttft": ttft, "eng": eng, "mbu": mbu})
    out.sort(key=lambda d: d["x"])
    return out

isl_pts, osl_pts = shape_points("isl"), shape_points("osl")
shape_c = cfg.get("shape_c") or cfg.get("anchor_c")

def cell(v, fmt="%.1f"):
    return "-" if v is None else fmt % v

if isl_pts or osl_pts:
    print("")
    print(" perturbation -- one input varied at a time%s"
          % (" at c=%d" % shape_c if shape_c else ""))

if len(isl_pts) >= 2:
    print("")
    print("   ISL sweep (OSL fixed): prefill work and KV context both grow")
    print("     %8s %10s %10s %15s" % ("isl", "ttft_avg", "itl_avg", "prefill tok/s"))
    for d in isl_pts:
        tp_s = d["x"] / (d["ttft"] / 1000.0) if d["ttft"] else None
        print("     %8.0f %10s %10s %15s"
              % (d["x"], cell(d["ttft"], "%.0f"), cell(d["itl"], "%.2f"), cell(tp_s, "%.0f")))

    # Prefill: TTFT = fixed overhead + marginal cost x ISL. Comparing endpoint
    # throughputs conflates the two -- with a large fixed overhead, ISL/TTFT
    # *rises* with ISL and says nothing about linearity. Fit the line instead,
    # and judge linearity from how the marginal cost changes along the sweep.
    b = slope([d["x"] for d in isl_pts], [d["ttft"] for d in isl_pts if d["ttft"]]) \
        if all(d["ttft"] for d in isl_pts) else None
    if b is not None and b > 0:
        n = len(isl_pts)
        fixed = (sum(d["ttft"] for d in isl_pts) / n) - b * (sum(d["x"] for d in isl_pts) / n)
        print("   . prefill      TTFT = %.0f ms fixed + %.3f ms per prompt token (%.0f tok/s marginal)"
              % (fixed, b, 1000.0 / b))
        seg = [(isl_pts[i+1]["ttft"] - isl_pts[i]["ttft"]) / (isl_pts[i+1]["x"] - isl_pts[i]["x"])
               for i in range(n - 1)]
        if len(seg) >= 2 and seg[0] > 0 and seg[-1] > seg[0] * 1.5:
            print("     -> marginal cost per token rose %.1fx along the sweep: prefill is super-linear,"
                  % (seg[-1] / seg[0]))
            print("        suspect attention scaling or prefill chunks contending with decode")
        else:
            print("     -> marginal cost per token is flat across a %.1fx ISL range: prefill is linear"
                  % (isl_pts[-1]["x"] / isl_pts[0]["x"]))
        small = isl_pts[0]
        if small["ttft"] and fixed > 0.5 * small["ttft"]:
            print("     ! %.0f%% of TTFT at ISL %.0f is fixed overhead, not prefill work -- queueing or"
                  % (100 * fixed / small["ttft"], small["x"]))
            print("       client-side cost dominates there, so short prompts measure the harness")

    # Decode: the ITL slope against ISL *is* the KV term of the roofline, so this
    # is the one measurement that can confirm or falsify the MEMORY BW verdict.
    m = slope([d["x"] for d in isl_pts], [d["itl"] for d in isl_pts])
    if m is not None:
        print("   . ITL slope   measured %.2f us per token of context" % (m * 1000))
        if roof["ok"] and roof["kv_tok"] and shape_c:
            ideal = 1000.0 * (shape_c * roof["kv_tok"]) / roof["tp"] / roof["bw"]   # ms per token
            # The engine runs at MBU of peak, so its slope is inflated by the same
            # factor. Dividing it out leaves a test of context scaling alone.
            e = sorted(d["mbu"] for d in isl_pts if d["mbu"])
            eff = e[len(e) // 2] if e else None       # median: robust to one odd point
            pred = ideal / eff if eff and 0.2 <= eff <= 1.3 else ideal
            print("     %sroofline predicts %.2f us at c=%d%s"
                  % (" " * 14, pred * 1000, shape_c,
                     "" if pred == ideal else
                     " (%.2f us at peak bandwidth, scaled by the measured MBU of %.0f%%)"
                     % (ideal * 1000, eff * 100)))
            if pred > 0 and m > 0:
                ratio = m / pred
                if 0.7 <= ratio <= 1.4:
                    print("     -> within %.0f%%: KV traffic explains how ITL grows, so the bandwidth"
                          % (100 * abs(1 - ratio)))
                    print("        model is confirmed by perturbation, not just inferred from power")
                elif ratio > 1.4:
                    print("     -> %.1fx steeper than KV traffic alone can explain: something else scales"
                          % ratio)
                    print("        with context (attention kernel, scheduler, cache thrash)")
                else:
                    print("     -> %.0f%% shallower than predicted: the engine is moving less KV than"
                          % (100 * (1 - ratio)))
                    print("        assumed -- prefix caching, a quantized KV cache, or --tp is wrong")
            elif m <= 0:
                print("     -> ITL does not grow with context at all; the roofline's KV term does not")
                print("        apply here, so re-check --tp and the KV dtype")
        elif not roof["ok"]:
            print("     %s(no roofline inputs, so there is nothing to compare it against)" % (" " * 14))

if len(osl_pts) >= 2:
    print("")
    print("   OSL sweep (ISL fixed): decode steps grow, nothing else does")
    print("     %8s %10s %10s %10s" % ("osl", "ttft_avg", "itl_avg", "e2e_avg"))
    for d in osl_pts:
        print("     %8.0f %10s %10s %10s"
              % (d["x"], cell(d["ttft"], "%.0f"), cell(d["itl"], "%.2f"),
                 cell(num(d["p"].get("e2e_avg")), "%.0f")))
    first, last = osl_pts[0], osl_pts[-1]
    drift = 100 * (last["itl"] - first["itl"]) / first["itl"] if first["itl"] else None
    if drift is not None:
        if abs(drift) <= 15:
            print("   . ITL moved %+.0f%% across a %.1fx OSL range -- each decode step costs the same,"
                  % (drift, last["x"] / first["x"]))
            print("     which is what a steady-state decode bottleneck looks like")
        else:
            print("   ! ITL moved %+.0f%% across a %.1fx OSL range -- the per-step cost is not constant;"
                  % (drift, last["x"] / first["x"]))
            print("     longer outputs grow the KV cache, so this is capacity pressure, not steady state")
    if first["ttft"] and last["ttft"]:
        t_drift = 100 * (last["ttft"] - first["ttft"]) / first["ttft"]
        if abs(t_drift) > 20:
            print("   ! TTFT moved %+.0f%% although the prompt never changed -- prefill is being delayed"
                  % t_drift)
            print("     by the extra decode work; chunked prefill or PD disaggregation would decouple them")

# ---------------------------------------------------------------- validity
warn = []
for p, _, _, _, _, _ in rows:
    req = requested.get(p["point"], {})
    for field, label in (("osl", "OSL"), ("isl", "ISL")):
        got = num(p.get(field))
        target = req.get(field) or cfg.get(field)
        if got and target and abs(got - target) / target > 0.05:
            warn.append("%s produced %s %.0f, not %.0f" % (p["point"], label, got, target))

poll = num(env.get("metrics_interval"))
if engine and poll:
    short = [k for k, d in duration.items()
             if k != "warmup" and k in engine and d < 5 * poll]
    if short:
        warn.append("%d work point(s) lasted under %.0fs, fewer than 5 poll periods at "
                    "--metrics-interval %g -- queue, KV and running are undersampled there "
                    "(%s). Raise --req-per-c, or lower --metrics-interval."
                    % (len(short), 5 * poll, poll, " ".join(sorted(short)[:6])))

pw = [g.get("power_ratio") for _, _, g, _, _, _ in sweep if g.get("power_ratio") is not None]
if pw and max(pw) < 0.15:
    warn.append("the sampled GPUs peaked at %.0f%% of board power across every point -- they were "
                "essentially idle. Either gpu_sample.sh ran on the wrong node, or the work is not "
                "happening on these GPUs; every GPU-side column and verdict above is meaningless."
                % (100 * max(pw)))

wrapped = max([g.get("ib_wrapped", 0) for _, _, g, _, _, _ in sweep] or [0])
if wrapped:
    warn.append("%d InfiniBand port counter(s) wrapped during a point -- those ports are "
                "excluded from the rate. Sample more often, or read the 64-bit counters."
                % wrapped)

users = [u for u in (host.get("compute_app_users", "") or "").split() if u]
if len(users) > 1:
    warn.append("the server node was shared by %d users (%s) -- GPU power, clocks and CPU "
                "are machine-wide, so GPU COMPUTE and SERIAL CPU verdicts are unproven"
                % (len(users), " ".join(users)))
soft_peak = max([g.get("soft_pct", 0) for _, _, g, _, _, _ in sweep] or [0])
clk_peak  = max([g.get("clk_low_pct", 0) for _, _, g, _, _, _ in sweep] or [0])
if soft_peak > 20 and clk_peak <= 20:
    warn.append("software throttle flags were lit in up to %.0f%% of samples, but the SM clocks "
                "held above 85%% of maximum -- ignored, as they should be" % soft_peak)
if warn:
    print("")
    print(" ! measurement validity")
    for w in warn[:8]:
        print("   %s" % w)
    if any("produced" in w for w in warn):
        print("   sequence lengths differ from what was requested -- these numbers are not")
        print("   comparable with anyone else's until that is fixed")

anchor = [r for r in rows if r[0]["point"].startswith("anchor")]
if anchor:
    ac = num(anchor[0][0]["concurrency"], 0)
    twin = [t for t in scored if num(t[1]["concurrency"], 0) == ac]
    if twin:
        a, b = num(anchor[0][0].get("goodput"), 0), twin[0][0]
        if a and b:
            drift = 100 * abs(a - b) / max(a, b)
            note = "stable" if drift < 5 else "UNSTABLE -- do not trust small differences between runs"
            print("")
            print(" repeatability anchor vs c%.0f: %.1f%% apart (%s)" % (ac, drift, note))
print("")
PYEOF
