#!/bin/bash
# Turn results/<job>/ into rows for docs/results-log.md.
#
#   ./run/results-row.sh results/17610
#   ./run/results-row.sh results/17610 --table B
#   ./run/results-row.sh results/*/            # every run, in order
#
# A log that has to be filled in by hand does not get filled in. This reads what
# the job already wrote and prints markdown.
#
# It prints "?" for anything it cannot find, and says why on stderr. A blank cell
# that silently means "missing" is the same failure this repo keeps hitting --
# an empty engine_metrics.csv looks exactly like a well-behaved engine.

set -uo pipefail

TABLE=A
DIRS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --table) TABLE="${2:-A}"; shift 2 ;;
        -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
        *) DIRS+=("$1"); shift ;;
    esac
done
(( ${#DIRS[@]} )) || { echo "x give me one or more results/<job> directories" >&2; exit 1; }

miss() { echo "  ! $1" >&2; }

# deployment.txt is key=value, but submit.sbatch writes several per line
# ("context_len=8192 mem_fraction=0.90 backend=nixl"), so this must not anchor
# to the start of a line -- the test caught exactly that.
dep() {   # $1=dir $2=key
    local v
    v="$(grep -oP "(?<![a-z_])$2=\K\S+" "$1/deployment.txt" 2>/dev/null | head -1)"
    [[ -n "$v" ]] || { v="?"; miss "$(basename "$1"): no $2 in deployment.txt"; }
    echo "$v"
}

# metrics.csv is one row per concurrency point, written by AIPerf/observe.sh.
# Column names come from its `cols` list: ttft_avg/p90/p99, itl_avg/p90/p99,
# out_tok_per_s, goodput, req_count, good_count, pass_rate, osl.
emit_A() {   # $1=dir
    local d="$1" job eng spec ep mf csv
    job="$(basename "$d")"
    csv="$d/scored/metrics.csv"
    [[ -r "$csv" ]] || { miss "$job: no scored/metrics.csv -- did the run reach the client?"; return 1; }

    eng="$(dep "$d" engine)"; spec="$(dep "$d" pd_spec)"
    ep="$(dep "$d" decode_ep)"; mf="$(dep "$d" mem_fraction)"
    local ib="?"
    grep -q "ib_min_rate=400" "$d/deployment.txt" 2>/dev/null && ib=ndr || ib=all

    python3 - "$csv" "$job" "$eng" "$spec" "$ep" "$mf" "$ib" <<'PY'
import csv, sys
path, job, eng, spec, ep, mf, ib = sys.argv[1:8]
def num(r, k, fmt="%.2f"):
    v = r.get(k, "")
    try: return fmt % float(v)
    except Exception: return "?"
with open(path) as fh:
    for r in csv.DictReader(fh):
        n = r.get("concurrency") or r.get("point") or "?"
        # The ranking metric: good requests/sec x output tokens per request.
        # observe.sh computes the same product for its summary line.
        try:
            good = "%.0f" % (float(r["goodput"]) * float(r["osl"]))
        except Exception:
            good = "?"
        gate = ""
        try:
            if float(r.get("pass_rate", 0)) < 0.90: gate = " **over**"
        except Exception: pass
        print("| | %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s |%s |" % (
            job, eng, spec, n, ep, mf, ib, good,
            num(r, "pass_rate", "%.3f"), num(r, "ttft_p90"), num(r, "itl_p90"), gate))
PY
}

# Table B reads engine_metrics.csv, which observe.sh writes as one row per
# (timestamp, endpoint, series). Aggregate to a max per series -- the peak is
# what tells you whether a queue ever backed up.
emit_B() {   # $1=dir
    local d="$1" job em
    job="$(basename "$d")"
    em="$d/scored/engine_metrics.csv"
    if [[ ! -s "$em" ]]; then
        miss "$job: engine_metrics.csv missing or empty -- checklist 7. On SGLang that"
        miss "  means --enable-metrics did not take; on TensorRT-LLM it means observe.sh"
        miss "  was pointed at /metrics instead of /prometheus/metrics."
        echo "| | $job | ? | ? | ? | ? | ? | ? | ? |"
        return 1
    fi
    python3 - "$em" "$job" "$d" <<'PY'
import csv, sys, os, re
path, job, d = sys.argv[1], sys.argv[2], sys.argv[3]
peak = {}
with open(path) as fh:
    for r in csv.DictReader(fh):
        name = (r.get("metric") or r.get("series") or "").lower()
        try: v = float(r.get("value", ""))
        except Exception: continue
        for key, pat in (("batch", r"num_running_reqs"),
                         ("queue", r"num_queue_reqs"),
                         ("kv_used", r"kv_used_tokens"),
                         ("kv_avail", r"kv_available_tokens"),
                         ("pd_q", r"num_decode_(transfer|prealloc)_queue_reqs"),
                         ("preempt", r"num_(retracted|paused)_reqs")):
            if re.search(pat, name):
                peak[key] = peak.get(key, 0) + v if key in ("pd_q", "preempt") else max(peak.get(key, 0), v)
def g(k, fmt="%.0f"):
    return fmt % peak[k] if k in peak else "?"
kvpct = "?"
if peak.get("kv_avail"):
    kvpct = "%.0f" % (100.0 * peak.get("kv_used", 0) / peak["kv_avail"])
# MBU and the verdict come from diagnose.sh's text report
mbu = verdict = "?"
for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
    if not f.startswith("diagnose-"): continue
    txt = open(os.path.join(d, f), errors="ignore").read()
    m = re.search(r"MBU[^0-9]{0,20}([0-9.]+)\s*%", txt)
    if m: mbu = m.group(1)
    m = re.search(r"(WEIGHT BOUND|MEMORY BW|SCHEDULER CAP)", txt)
    if m: verdict = m.group(1)
print("| | %s | %s | %s | %s | %s | %s | %s | %s |" % (
    job, g("batch"), g("queue"), kvpct, g("pd_q"), g("preempt"), mbu, verdict))
PY
}

case "$TABLE" in
    A) echo "| # | job | engine | PD_SPEC | N | EP | mem_frac | IB | good_out_tok_s | pass_rate | TTFT p90 | ITL p90 | note |"
       echo "|---|-----|--------|---------|---|----|----------|----|----------------|-----------|----------|---------|------|"
       for d in "${DIRS[@]}"; do emit_A "${d%/}"; done ;;
    B) echo "| # | job | run_batch | queue_depth | kv_used % | pd_transfer_q | preempt | MBU % | verdict |"
       echo "|---|-----|-----------|-------------|-----------|---------------|---------|-------|---------|"
       for d in "${DIRS[@]}"; do emit_B "${d%/}"; done ;;
    *) echo "x --table takes A or B" >&2; exit 1 ;;
esac

echo "" >&2
echo "Paste into docs/results-log.md and number the rows. Anything printed as ? was" >&2
echo "not found -- the reason is above, and a ? is a finding about the run, not a gap" >&2
echo "in the table." >&2
