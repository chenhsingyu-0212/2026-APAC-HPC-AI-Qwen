#!/bin/bash
# Side-channel monitoring for a benchmark run whose command must not be touched.
#
#   ./observe.sh --out ./scored --point c64 --gateway gw01:8000 -- \
#       ./official.sh --host gw01:8000 --model qwen3-pd --concurrency 64
#
# The scored command is fixed by the task rules, so nothing here may be folded
# into it. This wraps it instead: everything below happens *around* the child
# process, which runs exactly as written.
#
# What it collects, all of it from the outside:
#   * the engine's Prometheus metrics, polled continuously *during* the run --
#     queue depth and KV usage are gauges and read zero once load stops, so
#     scraping before and after tells you nothing
#   * the clock skew between this node and the gateway, so telemetry sampled on
#     the server nodes lines up with the run window
#   * start/end timestamps per point, in points.csv
#
# It produces exactly the artifact set diagnose.sh consumes -- env.txt,
# points.csv, metrics.csv, engine_metrics.csv -- so:
#
#   ./gpu_sample.sh --out ./tel        # on each server node, during the run
#   ./observe.sh --out ./scored ... -- ./official.sh ...
#   ./diagnose.sh ./scored --telemetry ./tel --weights <active GiB> --tp <N>
#
# Call it once per declared concurrency into the same --out and the points
# accumulate into one curve.
#
# Defaults match the Qwen3-235B-A22B PD task (ISL 4000, OSL 200, and its SLA
# gate); override them when observing anything else.

set -uo pipefail

die()   { echo "x $*" >&2; exit 1; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; }
need()  { [[ $# -ge 2 ]] || die "$1 requires a value"; }

OUT=""
POINT=""
CONC=""
GATEWAY=""
METRICS_URLS=()
METRICS_INTERVAL=2
NO_METRICS=0
ISL=4000
OSL=200
SLA="time_to_first_token:3000 inter_token_latency:20"
MODEL=""
LABEL=""
CMD=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)              need "$@"; OUT="$2";              shift 2 ;;
        --point)            need "$@"; POINT="$2";            shift 2 ;;
        --concurrency)      need "$@"; CONC="$2";             shift 2 ;;
        --gateway)          need "$@"; GATEWAY="$2";          shift 2 ;;
        --metrics-url)      need "$@"; METRICS_URLS+=("$2"); shift 2 ;;
        --metrics-interval) need "$@"; METRICS_INTERVAL="$2"; shift 2 ;;
        --no-metrics)       NO_METRICS=1; shift ;;
        --isl)              need "$@"; ISL="$2";   shift 2 ;;
        --osl)              need "$@"; OSL="$2";   shift 2 ;;
        --sla)              need "$@"; SLA="$2";   shift 2 ;;
        --model)            need "$@"; MODEL="$2"; shift 2 ;;
        --label)            need "$@"; LABEL="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; CMD=("$@"); break ;;
        *)                  echo "x unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -n "$OUT" ]]        || { usage >&2; die "--out is required"; }
(( ${#CMD[@]} > 0 ))   || die "give me the command to observe, after --"
[[ "$METRICS_INTERVAL" =~ ^[0-9]*\.?[0-9]+$ ]] \
    && awk -v v="$METRICS_INTERVAL" 'BEGIN { exit !(v + 0 >= 0.1) }' \
    || die "--metrics-interval must be a number >= 0.1, got '$METRICS_INTERVAL'"

# The concurrency is worth recovering from the child command when not given:
# it is what names the point and what the roofline uses as the batch size.
if [[ -z "$CONC" ]]; then
    for i in "${!CMD[@]}"; do
        case "${CMD[$i]}" in
            --concurrency|-n) CONC="${CMD[$((i+1))]:-}"; break ;;
        esac
    done
fi
[[ -z "$CONC" || "$CONC" =~ ^[0-9]+$ ]] || die "--concurrency must be an integer, got '$CONC'"
# diagnose.sh treats points named c<N> as the concurrency curve
[[ -n "$POINT" ]] || POINT="c${CONC:-0}"

mkdir -p "$OUT/$POINT" || die "cannot create $OUT/$POINT"
OUT="$(cd "$OUT" && pwd)"

# The child runs from inside the point directory, so a relative ./official.sh
# would no longer resolve. Pin it to an absolute path before that happens.
if [[ "${CMD[0]}" == */* && -e "${CMD[0]}" ]]; then
    CMD[0]="$(cd "$(dirname "${CMD[0]}")" && pwd)/$(basename "${CMD[0]}")"
fi

PY=""
for c in "${AIPERF_VENV:-$(pwd)/aiperf-venv}/bin/python" python3 python; do
    [[ -x "$c" ]] && { PY="$c"; break; }
    command -v "$c" >/dev/null 2>&1 && { PY="$(command -v "$c")"; break; }
done
[[ -n "$PY" ]] || die "no python found"

# ---------------------------------------------------------------- gateway probe
SKEW=0
if [[ -n "$GATEWAY" ]]; then
    GW="${GATEWAY#http://}"; GW="${GW#https://}"; GW="${GW%/}"
    [[ -n "$MODEL" ]] || MODEL="$(curl -sf --max-time 10 "http://$GW/v1/models" 2>/dev/null \
        | "$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)"
    # points.csv is stamped by this node's clock, gpu_sample.sh by the server's;
    # unsynchronised clocks silently slice the wrong window for every point
    DATE_HDR="$(curl -sf -D - -o /dev/null --max-time 10 "http://$GW/v1/models" 2>/dev/null \
                | awk 'tolower($0) ~ /^date: / { sub(/^[^:]*: */, ""); sub(/\r$/, ""); print; exit }')"
    if [[ -n "$DATE_HDR" ]]; then
        SRV="$(date -d "$DATE_HDR" +%s 2>/dev/null)"
        if [[ "$SRV" =~ ^[0-9]+$ ]]; then
            SKEW=$(( SRV - $(date +%s) ))
            (( SKEW > 2 || SKEW < -2 )) \
                && echo "  ! gateway clock is ${SKEW}s off this node's -- recorded, diagnose.sh corrects for it" \
                || echo "  ok clocks agree within ${SKEW}s"
        fi
    fi
    (( ${#METRICS_URLS[@]} )) || (( NO_METRICS )) || METRICS_URLS=("http://$GW/metrics")
fi

# Queue depth, KV usage and preemptions are engine-internal, so in a PD split
# they live on each worker rather than on the gateway -- pass --metrics-url once
# per worker. Each endpoint is scraped into its own file and kept separate:
# summing a cache-usage *fraction* across six workers would read 600%.
LABELS=(); KEPT=()
if (( NO_METRICS == 0 )); then
    for u in "${METRICS_URLS[@]:-}"; do
        [[ -n "$u" ]] || continue
        # captured, not piped into grep -q: pipefail plus an early grep exit would
        # SIGPIPE curl and read as a failed probe on exactly the busiest servers
        PROBE="$(curl -sf --max-time 5 "$u" 2>/dev/null | head -c 65536 || true)"
        if grep -q '^[a-zA-Z]' <<<"$PROBE"; then
            l="${u#*://}"; l="${l%%/*}"; l="${l//[^A-Za-z0-9._-]/_}"
            KEPT+=("$u"); LABELS+=("$l")
        else
            echo "  ! no Prometheus metrics at $u -- skipping it"
        fi
    done
    if (( ${#KEPT[@]} )); then
        echo "  ok engine metrics: ${#KEPT[@]} endpoint(s), polled every ${METRICS_INTERVAL}s during the run"
    # NOT ${#METRICS_URLS[@]:-0}: combining ${#...[@]} with :- is a syntax error,
    # "bad substitution", printed on every run where no endpoint was kept. ${#a[@]}
    # already evaluates to 0 for an unset array, so the default was never needed.
    elif (( ${#METRICS_URLS[@]} )); then
        echo "    (SGLang needs --enable-metrics at launch, per worker)"
    fi
fi

# ---------------------------------------------------------------- fingerprint
[[ -f "$OUT/points.csv" ]] \
    || echo "point,concurrency,requests,isl,osl,start_ts,end_ts,status" > "$OUT/points.csv"
{
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "url=${GATEWAY:-unknown}"
    echo "model=${MODEL:-unknown}"
    echo "tokenizer=Qwen/Qwen3-235B-A22B-Instruct-2507"
    echo "isl=$ISL osl=$OSL"
    echo "sla='$SLA'"
    echo "metrics_url=$( (( ${#KEPT[@]} )) && printf '%s ' "${KEPT[@]}" || echo none) metrics_interval=$METRICS_INTERVAL"
    echo "server_clock_skew_s=$SKEW"
    echo "label=${LABEL:-}"
    echo "observed_command=$(printf '%q ' "${CMD[@]}")"
} > "$OUT/env.txt"
cat "$OUT/env.txt"

# ---------------------------------------------------------------- poll + run
POLL_PIDS=()
poll_stop() {
    for p in "${POLL_PIDS[@]:-}"; do
        [[ -n "$p" ]] || continue
        kill "$p" 2>/dev/null
        wait "$p" 2>/dev/null   # this job only: a bare wait would also block on
    done                        # anything the observed command left running
    POLL_PIDS=()
}
trap poll_stop EXIT INT TERM

for i in "${!KEPT[@]}"; do
    (
        while :; do
            printf '# scrape ts=%s\n' "$(date +%s)"
            curl -sf --max-time 5 "${KEPT[$i]}" 2>/dev/null
            sleep "$METRICS_INTERVAL"
        done
    ) > "$OUT/$POINT/metrics_samples-${LABELS[$i]}.txt" 2>/dev/null &
    POLL_PIDS+=($!)
done

echo "=========================================="
echo " observing point $POINT -> $OUT/$POINT"
printf ' command:'; printf ' %q' "${CMD[@]}"; echo
echo "=========================================="

START=$(date +%s)
( cd "$OUT/$POINT" && "${CMD[@]}" )
RC=$?
END=$(date +%s)
poll_stop

printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$POINT" "${CONC:-}" "" "$ISL" "$OSL" "$START" "$END" \
       "$( (( RC == 0 )) && echo ok || echo failed )" >> "$OUT/points.csv"
echo ""
echo "point $POINT ran for $(( END - START ))s, exit $RC"

# ---------------------------------------------------------------- collect
# The child decides where it writes; find its results rather than dictating them.
"$PY" - "$OUT" "$POINT" <<'PYEOF'
import csv, os, re, sys
dst, point = sys.argv[1], sys.argv[2]

cols = [("Time to First Token (ms)","ttft_avg","avg"), ("Time to First Token (ms)","ttft_p90","p90"),
        ("Time to First Token (ms)","ttft_p99","p99"), ("Inter Token Latency (ms)","itl_avg","avg"), ("Inter Token Latency (ms)","itl_p90","p90"),
        ("Inter Token Latency (ms)","itl_p99","p99"),
        ("Request Latency (ms)","e2e_avg","avg"), ("Request Latency (ms)","e2e_p90","p90"),
        ("Request Latency (ms)","e2e_p99","p99"),
        ("Output Token Throughput (tokens/sec)","out_tok_per_s","avg"),
        ("Request Throughput (requests/sec)","req_per_s","avg"),
        ("Goodput (requests/sec)","goodput","avg"),
        ("Request Count","req_count","avg"), ("Good Request Count","good_count","avg"),
        ("Input Sequence Length (tokens)","isl","avg"),
        ("Output Sequence Length (tokens)","osl","avg")]

def parse(path):
    with open(path, newline="") as fh:
        rd = list(csv.reader(fh))
    if not rd:
        return None
    ix = {h.strip(): i for i, h in enumerate(rd[0])}
    d = {r[0].strip(): r for r in rd[1:] if r}
    row = {"point": point, "pass_rate": ""}
    for metric, key, col in cols:
        r = d.get(metric)
        row[key] = r[ix[col]].strip() if r and col in ix and len(r) > ix[col] else ""
    if row.get("good_count") and row.get("req_count"):
        try:
            row["pass_rate"] = "%.3f" % (float(row["good_count"]) / float(row["req_count"]))
        except (ValueError, ZeroDivisionError):
            pass
    return row

# The export's filename is not stable across AIPerf versions: the dialect the
# rules are written in writes profile_export_aiperf.csv, and 0.11.0 does not.
# Match the family instead of one spelling, preferring the canonical name where
# it exists so older runs keep ranking exactly as they did.
EXPORT_NAMES = ("profile_export_aiperf.csv",)
found = []
for root, _, files in os.walk(os.path.join(dst, point)):
    for f in files:
        if f in EXPORT_NAMES:
            found.append(os.path.join(root, f))
if not found:
    for root, _, files in os.walk(os.path.join(dst, point)):
        for f in files:
            if f.endswith(".csv") and ("aiperf" in f or f.startswith("profile_export")):
                found.append(os.path.join(root, f))
if not found:
    seen = sorted({f for _, _, fs in os.walk(os.path.join(dst, point)) for f in fs})[:8]
    print("! no AIPerf export csv under %s/%s -- did the command fail?" % (dst, point))
    if seen:
        print("  files that are there: %s" % ", ".join(seen))
    sys.exit(0)

def rank(path):
    """AIPerf writes this file three times when --warmup-duration is used: at the
    top of the run directory, under phases/profiling, and under phases/warmup.
    All three share an mtime, so picking "the newest" is a coin flip that can
    silently report warmup numbers as the score. Pick by name, explicitly."""
    n = path.replace(os.sep, "/")
    if "/phases/warmup/" in n:    return 3      # never the scored numbers
    if "/phases/profiling/" in n: return 0      # the scored phase, named outright
    if "/phases/" in n:           return 2      # some other phase
    return 1                                    # top-level summary

found.sort(key=lambda f: (rank(f), -os.path.getmtime(f)))
best = found[0]
if rank(best) == 3:
    print("! only warmup results found under %s -- the profiling phase produced nothing." % point)
    print("  These are NOT scored numbers; treat them as a failed point.")
elif len(found) > 1:
    print("using %s" % os.path.relpath(best, dst))
row = parse(best)
if not row:
    print("! %s is empty" % found[-1]); sys.exit(0)

# merge this point into the table, replacing any earlier attempt at the same one
table = os.path.join(dst, "metrics.csv")
rows = []
if os.path.isfile(table):
    with open(table, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["point"] != point]
rows.append(row)
def key(n):
    m = re.search(r'(\d+)$', n)
    return (0 if re.match(r'c\d+$', n) else 1, int(m.group(1)) if m else 0, n)
rows.sort(key=lambda r: key(r["point"]))
with open(table, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cols and rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("\nresults -> %s (%d point%s)" % (table, len(rows), "" if len(rows) == 1 else "s"))

g, p, o = row.get("goodput"), row.get("pass_rate"), row.get("osl")
if g and p:
    line = "  %s: goodput %s req/s, pass rate %s" % (point, g, p)
    try:
        line += "  ->  %.0f good output tok/s" % (float(g) * float(o))
    except (TypeError, ValueError):
        pass
    print(line)
    try:
        if float(p) < 0.90:
            print("  ! pass rate below the 0.90 gate -- this point would not be scored")
    except ValueError:
        pass
PYEOF

# ---------------------------------------------------------------- engine metrics
if (( ${#KEPT[@]} )); then
"$PY" - "$OUT" <<'PYEOF'
import csv, os, re, sys
dst = sys.argv[1]
KEEP = re.compile(r'running|waiting|queue|pending|preempt|cache_usage|token_usage'
                  r'|kv_cache|cache_hit|prefix_cache|throughput|tokens_total'
                  r'|request_success|num_used_tokens', re.I)
SKIP = re.compile(r'_bucket$|^go_|^process_|^python_')
LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eENaN]+)$')

def parse(path):
    """-> {series: [value per scrape]}, keyed on name+labels.

    Triton/TRT-LLM puts the state in a label where vLLM puts it in the name, so
    collapsing label sets would both hide series and add unrelated ones up.
    """
    series, cur = {}, None
    def flush():
        if cur:
            for k, v in cur.items():
                series.setdefault(k, []).append(v)
    with open(path) as fh:
        for line in fh:
            if line.startswith("# scrape ts="):
                flush(); cur = {}
                continue
            if line.startswith("#") or cur is None:
                continue
            m = LINE.match(line.strip())
            if not m:
                continue
            name, labels, val = m.groups()
            key = name + (labels or "")
            if SKIP.search(name) or not KEEP.search(key):
                continue
            try:
                f = float(val)
            except ValueError:
                continue
            cur[key] = cur.get(key, 0.0) + f
    flush()
    return series

def sortkey(n):
    m = re.search(r'(\d+)$', n)
    return (0 if re.match(r'c\d+$', n) else 1, int(m.group(1)) if m else 0, n)

rows, hi = [], []
for name in sorted(os.listdir(dst), key=sortkey):
    d = os.path.join(dst, name)
    if not os.path.isdir(d):
        continue
    files = sorted(f for f in os.listdir(d) if f.startswith("metrics_samples"))
    series = {}
    for f in files:
        # endpoint label goes into the key, so six workers stay six series and
        # diagnose.sh can decide per metric whether to sum them or take a max
        ep = f[len("metrics_samples"):].lstrip("-").removesuffix(".txt") or "all"
        for k, v in parse(os.path.join(d, f)).items():
            series["%s|%s" % (ep, k)] = v
    if not series:
        continue
    h = {"point": name}
    for metric, vals in sorted(series.items()):
        rows.append({"point": name, "metric": metric, "samples": len(vals),
                     "mean": "%.4g" % (sum(vals) / len(vals)), "peak": "%.4g" % max(vals),
                     "first": "%.4g" % vals[0], "last": "%.4g" % vals[-1],
                     "delta": "%.4g" % (vals[-1] - vals[0])})
        low = metric.lower()
        if "cache_usage" in low or "token_usage" in low:
            h["kv"] = max(h.get("kv", 0), max(vals))
        elif "waiting" in low or "queue" in low:
            h["wait"] = max(h.get("wait", 0), max(vals))
        elif "preempt" in low:
            h["preempt"] = h.get("preempt", 0) + (vals[-1] - vals[0])
    hi.append(h)

if rows:
    with open(os.path.join(dst, "engine_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("engine metrics -> %s/engine_metrics.csv (%d series across %d point(s))"
          % (dst, len(rows), len(hi)))
    print("  %-14s%16s%12s%13s" % ("point", "kv/token usage", "queue peak", "preemptions"))
    for h in hi:
        print("  %-14s%16s%12s%13s"
              % (h["point"], "%.3g" % h["kv"] if "kv" in h else "-",
                 "%.0f" % h["wait"] if "wait" in h else "-",
                 "%.0f" % h["preempt"] if "preempt" in h else "-"))
else:
    print("! metrics were scraped but nothing matched the keyword filter")
    print("  raw scrapes are in <point>/metrics_samples.txt")
PYEOF
fi

echo ""
echo "=========================================="
echo "done -> $OUT"
echo " next: ./diagnose.sh $OUT --telemetry <gpu_sample.sh dir> \\"
echo "         --weights <ACTIVE GiB, not the full MoE checkpoint> --tp <TP degree>"
echo "=========================================="
exit $RC
