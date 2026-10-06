#!/bin/bash
# Portable AIPerf benchmark runner -- produces comparable goodput curves against
# any OpenAI-compatible service.
#
#   ./run.sh --url http://node:8000 --tokenizer /path/to/model
#   ./run.sh --url http://node:8000 --tokenizer /path/to/model \
#            --sweep "8 16 32" --isl 4000 --osl 200 --out ./myrun
#   ./run.sh --url http://node:8000 --tokenizer /path/to/model \
#            --isl-sweep "1000 2000 4000 8000" --osl-sweep "50 200 800"
#
# Why this exists: benchmark numbers are only comparable when the measurement
# method is identical. This pins down the seven things that are easy to get
# wrong -- ask the server for the model id instead of guessing, force
# --streaming, point the tokenizer at a local directory, keep warmup out of the
# results, re-run the team-wide anchor point on its own, use milliseconds for
# SLA units, and assemble metrics.csv at the end.
#
# Teammates running vLLM / SGLang / Dynamo use this same script, so their
# numbers line up directly with ours.
#
# If the server exposes Prometheus metrics, they are polled *during* each work
# point (gauges read zero if you only scrape after the load stops) and rolled up
# into engine_metrics.csv -- KV cache usage, queue depth and preemptions are what
# explain a goodput plateau.
#
# Sweeping concurrency alone can only correlate a bottleneck with load. To test
# a hypothesis you have to perturb the input it depends on and watch the
# response, so --isl-sweep and --osl-sweep vary one shape dimension at a time at
# a fixed concurrency:
#   --isl-sweep  grows the prefill work and the KV context; TTFT and the ITL
#                slope are the response, and that slope is exactly what the
#                roofline predicts -- so it either confirms or falsifies it
#   --osl-sweep  grows the number of decode steps and nothing else; ITL should
#                barely move if decode really is the steady-state cost
# diagnose.sh reads both back and reports whether the response matched.
#
# Exits non-zero if any work point failed or no results could be parsed.
#
# Run ./install.sh first.

set -uo pipefail

# ---------------------------------------------------------------- helpers
die()  { echo "x $*" >&2; exit 1; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; }   # the header block above
need() { [[ $# -ge 2 ]] || die "$1 requires a value"; }
uint() { [[ "$2" =~ ^[0-9]+$ ]] && (( $2 > 0 )) || die "$1 must be a positive integer, got '$2'"; }

# ---------------------------------------------------------------- defaults
URL=""; TOKENIZER=""; MODEL=""
VENV="${AIPERF_VENV:-$(pwd)/aiperf-venv}"
ISL=4000; OSL=200
SWEEP="8 16 32 64 128 256"
ISL_SWEEP=""          # opt-in perturbation dimensions, run at SHAPE_C
OSL_SWEEP=""
SHAPE_C=""            # defaults to ANCHOR_C once that is known
ANCHOR_C=32
SLA="time_to_first_token:1000 request_latency:5000"
REQ_PER_C=10          # requests per work point = concurrency x this value
WARMUP_C=8; WARMUP_N=40
OUT=""
LABEL=""
METRICS_URL=""        # defaults to $URL/metrics once --url is known
METRICS_INTERVAL=2
NO_METRICS=0
POLL_PID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)        need "$@"; URL="$2";        shift 2 ;;
        --tokenizer)  need "$@"; TOKENIZER="$2";  shift 2 ;;
        --model)      need "$@"; MODEL="$2";      shift 2 ;;   # asked from the server if omitted
        --venv)       need "$@"; VENV="$2";       shift 2 ;;
        --isl)        need "$@"; ISL="$2";        shift 2 ;;
        --osl)        need "$@"; OSL="$2";        shift 2 ;;
        --sweep)      need "$@"; SWEEP="$2";      shift 2 ;;
        --isl-sweep)  need "$@"; ISL_SWEEP="$2";  shift 2 ;;
        --osl-sweep)  need "$@"; OSL_SWEEP="$2";  shift 2 ;;
        --shape-c)    need "$@"; SHAPE_C="$2";    shift 2 ;;
        --anchor)     need "$@"; ANCHOR_C="$2";   shift 2 ;;
        --sla)        need "$@"; SLA="$2";        shift 2 ;;
        --req-per-c)  need "$@"; REQ_PER_C="$2";  shift 2 ;;
        --out)        need "$@"; OUT="$2";        shift 2 ;;
        --label)      need "$@"; LABEL="$2";      shift 2 ;;
        --metrics-url)      need "$@"; METRICS_URL="$2";      shift 2 ;;   # default $URL/metrics
        --metrics-interval) need "$@"; METRICS_INTERVAL="$2"; shift 2 ;;
        --no-metrics)       NO_METRICS=1; shift ;;
        --no-warmup)  WARMUP_N=0; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "x unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------- validation
[[ -n "$URL" ]]       || { usage >&2; die "--url is required"; }
[[ -n "$TOKENIZER" ]] || die "--tokenizer is required (point it at a local model directory: no network fetch, and token counts match the server)"
[[ -d "$TOKENIZER" || -f "$TOKENIZER" ]] || die "tokenizer path does not exist: $TOKENIZER"

URL="${URL%/}"                                   # a trailing slash would double up below
[[ "$URL" == http://* || "$URL" == https://* ]] || die "--url must start with http:// or https://, got '$URL'"

uint --isl "$ISL"
uint --osl "$OSL"
uint --anchor "$ANCHOR_C"
uint --req-per-c "$REQ_PER_C"
[[ "$METRICS_INTERVAL" =~ ^[0-9]*\.?[0-9]+$ ]] \
    && awk -v v="$METRICS_INTERVAL" 'BEGIN { exit !(v + 0 >= 0.1) }' \
    || die "--metrics-interval must be a number >= 0.1, got '$METRICS_INTERVAL' (a tight poll loop would perturb the measurement)"
[[ -n "${SWEEP// /}" ]] || die "--sweep is empty"
for C in $SWEEP; do uint --sweep "$C"; done      # unquoted on purpose: split the sweep list

[[ -n "$SHAPE_C" ]] || SHAPE_C="$ANCHOR_C"
uint --shape-c "$SHAPE_C"
for I in $ISL_SWEEP; do uint --isl-sweep "$I"; done
for O in $OSL_SWEEP; do uint --osl-sweep "$O"; done
# One perturbation point is not a slope: the response curve is the whole result.
[[ -z "${ISL_SWEEP// /}" ]] || (( $(wc -w <<<"$ISL_SWEEP") >= 3 )) \
    || die "--isl-sweep needs at least 3 values to be a slope, got: $ISL_SWEEP"
[[ -z "${OSL_SWEEP// /}" ]] || (( $(wc -w <<<"$OSL_SWEEP") >= 3 )) \
    || die "--osl-sweep needs at least 3 values to be a slope, got: $OSL_SWEEP"

AIPERF="$VENV/bin/aiperf"
[[ -x "$AIPERF" ]] || die "$AIPERF not found -- run ./install.sh --venv $VENV first"

[[ -n "$OUT" ]] || OUT="./aiperf-$(date +%Y%m%d-%H%M%S)${LABEL:+-$LABEL}"
mkdir -p "$OUT" || die "cannot create $OUT"
echo "point,concurrency,requests,isl,osl,start_ts,end_ts,status" > "$OUT/points.csv"

# ---------------------------------------------------------------- server check
echo "[check] $URL"
curl -sf --max-time 10 "$URL/health" >/dev/null 2>&1 \
    && echo "  ok /health responded" \
    || echo "  ! /health did not return 200 (some services lack this endpoint, continuing)"

# Always ask for the model id, never guess -- a wrong name wastes the whole run
if [[ -z "$MODEL" ]]; then
    MODEL="$(curl -sf --max-time 10 "$URL/v1/models" 2>/dev/null \
             | "$VENV/bin/python" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)"
    [[ -n "$MODEL" ]] && echo "  ok model id (from server): $MODEL" \
                      || die "could not get a model id from $URL/v1/models, pass --model explicitly
       (the TRT-LLM disagg proxy has no /v1/models -- ask one of the workers instead)"
else
    echo "  . model id (manually specified): $MODEL"
fi

# ---------------------------------------------------------------- clock skew
# points.csv is stamped with this node's clock; gpu_sample.sh stamps its
# telemetry with the server's. If the two disagree, diagnose.sh slices the wrong
# time window for every point and has no way to tell -- so measure the offset
# from the server's own HTTP Date header and record it. (Behind a reverse proxy
# the header may come from the proxy; check by hand if the number looks odd.)
SKEW=0
DATE_HDR="$(curl -sf -D - -o /dev/null --max-time 10 "$URL/v1/models" 2>/dev/null \
            | awk 'tolower($0) ~ /^date: / { sub(/^[^:]*: */, ""); sub(/\r$/, ""); print; exit }')"
if [[ -n "$DATE_HDR" ]]; then
    SRV="$(date -d "$DATE_HDR" +%s 2>/dev/null)"
    if [[ "$SRV" =~ ^[0-9]+$ ]]; then
        SKEW=$(( SRV - $(date +%s) ))
        (( SKEW > 2 || SKEW < -2 )) \
            && echo "  ! server clock is ${SKEW}s off this node's -- recorded, diagnose.sh corrects for it" \
            || echo "  ok clocks agree within ${SKEW}s (telemetry will line up with the work points)"
    fi
else
    echo "  . no Date header from the server -- cannot check clock skew between the nodes"
fi

# ---------------------------------------------------------------- engine metrics probe
# Optional: any Prometheus endpoint works. vLLM serves /metrics by default;
# SGLang needs --enable-metrics; Triton-hosted TRT-LLM uses port 8002.
if (( NO_METRICS == 0 )); then
    [[ -n "$METRICS_URL" ]] || METRICS_URL="$URL/metrics"
    # Captured, not piped into `grep -q`: pipefail is on, and grep exiting early
    # SIGPIPEs curl on a large /metrics page -- which would silently drop the
    # engine metrics on exactly the busy servers that need them most.
    METRICS_PROBE="$(curl -sf --max-time 5 "$METRICS_URL" 2>/dev/null | head -c 65536 || true)"
    if grep -q '^[a-zA-Z]' <<<"$METRICS_PROBE"; then
        echo "  ok engine metrics: $METRICS_URL (polled every ${METRICS_INTERVAL}s during each point)"
    else
        echo "  ! no Prometheus metrics at $METRICS_URL -- continuing without them"
        echo "    (SGLang needs --enable-metrics; Triton/TRT-LLM usually exposes port 8002;"
        echo "     override with --metrics-url, or silence this with --no-metrics)"
        METRICS_URL=""
    fi
else
    METRICS_URL=""
fi

# ---------------------------------------------------------------- flag detection
HELP="$("$AIPERF" profile --help 2>&1)"
# The style is detected once; the values vary per point, because the shape
# sweeps change ISL and OSL between rounds.
if grep -q -- "--synthetic-input-tokens-mean" <<<"$HELP"; then
    LEN_STYLE=long
elif grep -q -- "--isl" <<<"$HELP"; then
    LEN_STYLE=short
else
    die "cannot recognize AIPerf's sequence length flags"
fi
len_args() {  # $1=isl $2=osl -> prints the flags for this AIPerf build
    if [[ "$LEN_STYLE" == long ]]; then
        echo "--synthetic-input-tokens-mean $1 --output-tokens-mean $2"
    else
        echo "--isl $1 --osl $2"
    fi
}
LEN_ARGS=($(len_args "$ISL" "$OSL"))
grep -q -- "--streaming" <<<"$HELP" || die "this version has no --streaming, TTFT/ITL cannot be measured"

# ---------------------------------------------------------------- environment fingerprint
TOTAL_REQ=$(( WARMUP_N + ANCHOR_C * REQ_PER_C ))
for C in $SWEEP; do TOTAL_REQ=$(( TOTAL_REQ + C * REQ_PER_C )); done
for _ in $ISL_SWEEP $OSL_SWEEP; do TOTAL_REQ=$(( TOTAL_REQ + SHAPE_C * REQ_PER_C )); done

{
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "url=$URL"
    echo "model=$MODEL"
    echo "tokenizer=$TOKENIZER"
    echo "aiperf=$("$AIPERF" --version 2>&1 | head -1)"
    echo "python=$("$VENV/bin/python" -V 2>&1 | awk '{print $2}')"
    echo "isl=$ISL osl=$OSL anchor_c=$ANCHOR_C sweep='$SWEEP' req_per_c=$REQ_PER_C"
    echo "shape_c=$SHAPE_C isl_sweep='$ISL_SWEEP' osl_sweep='$OSL_SWEEP'"
    echo "sla='$SLA'"
    echo "len_flags=${LEN_ARGS[*]}"
    echo "warmup_c=$WARMUP_C warmup_n=$WARMUP_N total_requests=$TOTAL_REQ"
    echo "metrics_url=${METRICS_URL:-none} metrics_interval=$METRICS_INTERVAL"
    echo "server_clock_skew_s=$SKEW"
    echo "label=${LABEL:-}"
} > "$OUT/env.txt"
cat "$OUT/env.txt"

echo "=========================================="
echo " work point ISL=$ISL OSL=$OSL | sweep $SWEEP | anchor c=$ANCHOR_C"
[[ -n "${ISL_SWEEP// /}" ]] && echo " ISL sweep $ISL_SWEEP at c=$SHAPE_C"
[[ -n "${OSL_SWEEP// /}" ]] && echo " OSL sweep $OSL_SWEEP at c=$SHAPE_C"
echo " SLA $SLA (milliseconds)"
echo " $TOTAL_REQ requests total"
echo " output $OUT"
echo "=========================================="

# Gauges (queue depth, KV cache usage) are only meaningful while load is applied,
# so scrape throughout the point rather than before/after it.
poll_start() {  # $1=subdirectory
    [[ -n "$METRICS_URL" ]] || return 0
    mkdir -p "$OUT/$1"
    (
        while :; do
            printf '# scrape ts=%s\n' "$(date +%s)"
            curl -sf --max-time 5 "$METRICS_URL" 2>/dev/null
            sleep "$METRICS_INTERVAL"
        done
    ) > "$OUT/$1/metrics_samples.txt" 2>/dev/null &
    POLL_PID=$!
}

poll_stop() {
    [[ -n "$POLL_PID" ]] || return 0
    kill "$POLL_PID" 2>/dev/null
    wait "$POLL_PID" 2>/dev/null
    POLL_PID=""
}
trap poll_stop EXIT INT TERM

run_one() {  # $1=concurrency $2=subdirectory $3=request-count [$4=isl $5=osl]
    local rc=0 start end isl="${4:-$ISL}" osl="${5:-$OSL}"
    local lens=($(len_args "$isl" "$osl"))
    start=$(date +%s)
    poll_start "$2"
    "$AIPERF" profile \
        -m "$MODEL" \
        --url "$URL" \
        --endpoint-type chat --streaming \
        --tokenizer "$TOKENIZER" \
        "${lens[@]}" \
        --concurrency "$1" \
        --request-count "$3" \
        --goodput "$SLA" \
        --artifact-dir "$OUT/$2" || rc=$?
    poll_stop
    end=$(date +%s)
    # the requested shape is recorded per point: the shape sweeps deliberately
    # differ from the run-wide --isl/--osl, so diagnose.sh cannot assume those
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$2" "$1" "$3" "$isl" "$osl" "$start" "$end" \
           "$( (( rc == 0 )) && echo ok || echo failed )" >> "$OUT/points.csv"
    return $rc
}

# ---------------------------------------------------------------- measurement
FAILED=()

if (( WARMUP_N > 0 )); then
    echo ""
    echo "[warmup] cold-start latency runs 30-40% above steady state, this round is **excluded from the results**"
    run_one "$WARMUP_C" warmup "$WARMUP_N" > "$OUT/warmup.log" 2>&1 \
        || echo "  ! warmup failed (see $OUT/warmup.log), continuing"
fi

echo ""
echo "[anchor] team-wide anchor point c=$ANCHOR_C (a separate re-run, used to check measurement stability)"
run_one "$ANCHOR_C" "anchor-c$ANCHOR_C" $(( ANCHOR_C * REQ_PER_C )) 2>&1 | tail -25 \
    || { echo "  ! anchor failed"; FAILED+=("anchor-c$ANCHOR_C"); }

echo ""
echo "[sweep] $SWEEP"
for C in $SWEEP; do
    echo "--- c=$C ---"
    run_one "$C" "c$C" $(( C * REQ_PER_C )) 2>&1 | tail -15 \
        || { echo "  ! c=$C failed, continuing"; FAILED+=("c$C"); }
done

# Perturbation rounds last: they are the extra evidence, and a failure here must
# not cost the concurrency curve that everything else is built on.
if [[ -n "${ISL_SWEEP// /}" ]]; then
    echo ""
    echo "[shape] ISL sweep $ISL_SWEEP at c=$SHAPE_C, OSL fixed at $OSL"
    echo "  grows prefill work and KV context; the ITL slope is what the roofline predicts"
    for I in $ISL_SWEEP; do
        echo "--- isl=$I ---"
        run_one "$SHAPE_C" "isl$I" $(( SHAPE_C * REQ_PER_C )) "$I" "$OSL" 2>&1 | tail -12 \
            || { echo "  ! isl=$I failed, continuing"; FAILED+=("isl$I"); }
    done
fi

if [[ -n "${OSL_SWEEP// /}" ]]; then
    echo ""
    echo "[shape] OSL sweep $OSL_SWEEP at c=$SHAPE_C, ISL fixed at $ISL"
    echo "  adds decode steps and nothing else; ITL should barely move if decode is the cost"
    for O in $OSL_SWEEP; do
        echo "--- osl=$O ---"
        run_one "$SHAPE_C" "osl$O" $(( SHAPE_C * REQ_PER_C )) "$ISL" "$O" 2>&1 | tail -12 \
            || { echo "  ! osl=$O failed, continuing"; FAILED+=("osl$O"); }
    done
fi

# ---------------------------------------------------------------- assemble comparison table
"$VENV/bin/python" - "$OUT" <<'PYEOF'
import csv, os, re, sys
dst = sys.argv[1]
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

def sortkey(n):
    m = re.search(r'(\d+)$', n)
    # concurrency sweep first, then the shape sweeps, then the anchor
    rank = 0 if re.match(r'c\d+$', n) else (1 if n.startswith(('isl', 'osl')) else 2)
    return (rank, n.rstrip('0123456789'), int(m.group(1)) if m else 0)

rows = []
for name in sorted(os.listdir(dst), key=sortkey):
    if name == "warmup":                      # excluded from the results by design
        continue
    f = os.path.join(dst, name, "profile_export_aiperf.csv")
    if not os.path.isfile(f):
        continue
    with open(f, newline="") as fh:
        rd = list(csv.reader(fh))
    if not rd:
        continue
    ix = {h.strip(): i for i, h in enumerate(rd[0])}
    d = {r[0].strip(): r for r in rd[1:] if r}
    row = {"point": name, "pass_rate": ""}
    for metric, key, col in cols:
        r = d.get(metric)
        row[key] = r[ix[col]].strip() if r and col in ix and len(r) > ix[col] else ""
    if row.get("good_count") and row.get("req_count"):
        try:
            row["pass_rate"] = f'{float(row["good_count"])/float(row["req_count"]):.3f}'
        except (ValueError, ZeroDivisionError):
            pass
    rows.append(row)

if not rows:
    print("\n! no parsable results -- check the error messages above")
    sys.exit(1)

with open(os.path.join(dst, "metrics.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\ncomparison table: {dst}/metrics.csv ({len(rows)} work points)")

best = max((r for r in rows if r["point"].startswith("c") and r["goodput"]),
           key=lambda r: float(r["goodput"]), default=None)
if best:
    print(f"best goodput: {float(best['goodput']):.2f} req/s @ {best['point']}"
          f" (pass rate {best['pass_rate']})")
PYEOF
TABLE_RC=$?

# ---------------------------------------------------------------- roll up engine metrics
# Deliberately name-agnostic: vLLM (vllm:*), SGLang (sglang:*), TRT-LLM/Triton
# (nv_trt_llm_*) and Dynamo all name these differently, so select by keyword and
# keep the raw scrapes so nothing is lost.
if [[ -n "$METRICS_URL" ]]; then
"$VENV/bin/python" - "$OUT" <<'PYEOF'
import csv, os, re, sys
dst = sys.argv[1]

KEEP = re.compile(r'running|waiting|queue|pending|preempt|cache_usage|token_usage'
                  r'|kv_cache|cache_hit|prefix_cache|throughput|tokens_total'
                  r'|request_success|num_used_tokens', re.I)
SKIP = re.compile(r'_bucket$|^go_|^process_|^python_')
LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eENaN]+)$')

def sortkey(n):
    m = re.search(r'(\d+)$', n)
    return (0 if n.startswith('c') else 1, int(m.group(1)) if m else 0, n)

def parse(path):
    """-> {series: [value per scrape]}.

    Keyed on name+labels, not name alone: Triton/TRT-LLM puts the state in a
    label (nv_trt_llm_request_metrics{request_type="waiting"}), so collapsing
    label sets would both hide those series and add unrelated ones together.
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

rows, highlights = [], []
for name in sorted(os.listdir(dst), key=sortkey):
    if name == "warmup":
        continue
    f = os.path.join(dst, name, "metrics_samples.txt")
    if not os.path.isfile(f):
        continue
    series = parse(f)
    if not series:
        continue
    hi = {"point": name}
    for metric, vals in sorted(series.items()):
        rows.append({"point": name, "metric": metric, "samples": len(vals),
                     "mean": f"{sum(vals)/len(vals):.4g}", "peak": f"{max(vals):.4g}",
                     "first": f"{vals[0]:.4g}", "last": f"{vals[-1]:.4g}",
                     "delta": f"{vals[-1]-vals[0]:.4g}"})
        # worst case across every matching series, so the headline never depends
        # on which metric name happened to sort first
        low = metric.lower()
        if "cache_usage" in low or "token_usage" in low:
            hi["kv"] = max(hi.get("kv", 0), max(vals))
        elif "waiting" in low or "queue" in low:
            hi["wait"] = max(hi.get("wait", 0), max(vals))
        elif "preempt" in low:
            hi["preempt"] = hi.get("preempt", 0) + (vals[-1] - vals[0])
    highlights.append(hi)

if rows:
    with open(os.path.join(dst, "engine_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nengine metrics: {dst}/engine_metrics.csv "
          f"({len(rows)} series across {len(highlights)} work points)")
    print(f"  {'point':<14}{'kv/token usage':>16}{'queue peak':>12}{'preemptions':>13}")
    for h in highlights:
        kv = f"{h['kv']:.3g}" if "kv" in h else "-"
        wt = f"{h['wait']:.0f}" if "wait" in h else "-"
        pr = f"{h['preempt']:.0f}" if "preempt" in h else "-"
        print(f"  {h['point']:<14}{kv:>16}{wt:>12}{pr:>13}")
    print("  kv usage flat and low -> not memory bound; queue growing -> scheduler cap;")
    print("  preemptions climbing  -> KV cache oversubscribed, expect SLA violations")
else:
    print("\n! engine metrics were scraped but nothing matched the keyword filter")
    print("  raw scrapes are in <point>/metrics_samples.txt")
PYEOF
fi

echo ""
echo "=========================================="
if (( ${#FAILED[@]} > 0 )); then
    echo "! ${#FAILED[@]} work point(s) failed: ${FAILED[*]}"
fi
echo "done -> $OUT"
echo " the per-request raw files (inputs.json / profile_export.jsonl) are large, delete them before committing"
echo "=========================================="

(( TABLE_RC == 0 && ${#FAILED[@]} == 0 )) || exit 1
