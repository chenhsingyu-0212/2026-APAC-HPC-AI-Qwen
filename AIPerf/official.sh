#!/bin/bash
# The official scored AIPerf command for the Qwen3-235B-A22B PD task, verbatim.
#
#   ./official.sh --host node01:8000 --model my-qwen3 --concurrency 64
#   ./official.sh --print --host H --model M --concurrency N   # show, run nothing
#   ./official.sh --verify                                     # diff against the rules file
#
# Nothing in the invocation below may be tuned. The task rules fix the command
# for the scored re-run, so this script exists to make "we ran exactly that"
# checkable rather than a claim: --verify re-extracts the command from the
# organisers' own markdown and compares it token by token with what runs here.
#
# Three values are the team's to declare, and they are the only substitutions:
#   --concurrency N   the declared concurrency (also becomes --workers-max)
# (--aiperf overrides which aiperf binary runs it, for container images.)
#   --host HOST:PORT  the team's gateway, never a Prefill or Decode worker
#   --model ID        the id the gateway reports at /v1/models
#
# The command has no --artifact-dir, and adding one would modify it, so this
# runs from inside --out instead: AIPerf's default artifact directory is
# 'artifacts', which therefore lands in <out>/artifacts/.
#
# Monitoring lives in observe.sh, deliberately not here. Wrap this script with
# it when you want telemetry -- that polls from the outside and leaves the
# command untouched:
#
#   ./observe.sh --out ./scored-run -- ./official.sh --host H --model M --concurrency N
#
# For exploration, use run.sh instead. The rules allow any duration, seed or
# sweep during your own testing; only the scored re-run is locked.
#
# --venv defaults to ./aiperf-venv when one is there, else the one beside this
# script, so it still resolves when observe.sh runs it from a point directory.

set -uo pipefail

die()   { echo "x $*" >&2; exit 1; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; }
need()  { [[ $# -ge 2 ]] || die "$1 requires a value"; }

N=""; HOST=""; MODEL=""
# observe.sh runs this from inside its point directory, so a cwd-relative
# default would miss the venv. Fall back to the one beside this script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${AIPERF_VENV:-}" ]]; then VENV="$AIPERF_VENV"
elif [[ -x "$(pwd)/aiperf-venv/bin/aiperf" ]]; then VENV="$(pwd)/aiperf-venv"
else VENV="$HERE/aiperf-venv"; fi
AIPERF_BIN=""
OUT=""
PRINT=0
VERIFY=""
DO_VERIFY=0
RULES_DEFAULT="../2026-APAC-HPC-AI/1. Qwen3 AI Task Rules.md"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--concurrency) need "$@"; N="$2";     shift 2 ;;
        --host)           need "$@"; HOST="$2";  shift 2 ;;
        --model)          need "$@"; MODEL="$2"; shift 2 ;;
        --venv)           need "$@"; VENV="$2";  shift 2 ;;
        --aiperf)         need "$@"; AIPERF_BIN="$2"; shift 2 ;;
        --out)            need "$@"; OUT="$2";   shift 2 ;;
        --print)          PRINT=1;  shift ;;
        --verify)         DO_VERIFY=1
                          if [[ $# -ge 2 && "$2" != -* ]]; then VERIFY="$2"; shift 2
                          else shift; fi ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "x unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# On a cluster AIPerf usually comes from a container image rather than a venv,
# so --aiperf takes whatever binary is on PATH in there.
AIPERF="${AIPERF_BIN:-$VENV/bin/aiperf}"

# ---------------------------------------------------------------- THE COMMAND
# Do not edit. If the organisers change the rules, re-run --verify: it will
# fail, and the fix is to update this block to match them, never the reverse.
build_cmd() {   # fills CMD using the current N / HOST / MODEL / AIPERF
    CMD=(
        "$AIPERF" profile
        --url "http://${HOST}"
        --endpoint-type chat
        --endpoint /v1/chat/completions
        --model "${MODEL}"
        --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507
        --streaming
        -H 'Accept: text/event-stream'
        --concurrency "${N}"
        --workers-max "${N}"
        --api-key NOT_USED
        --warmup-duration 60
        --benchmark-duration 120
        --benchmark-grace-period 30
        --synthetic-input-tokens-mean 4000
        --synthetic-input-tokens-stddev 0
        --output-tokens-mean 200
        --output-tokens-stddev 0
        --extra-inputs min_tokens:200
        --extra-inputs max_tokens:200
        --extra-inputs ignore_eos:true
        --extra-inputs temperature:0.0
        --extra-inputs repetition_penalty:1.0
        --num-dataset-entries 12800
        --random-seed 2026
        --cache-bust first_turn_prefix
        --goodput 'time_to_first_token:3000 inter_token_latency:20'
        --ui simple
    )
}

# ---------------------------------------------------------------- verify
# Rebuild the command with the rules file's own placeholders and diff it against
# the fenced block in that file, so a silent divergence is impossible.
if (( DO_VERIFY )); then
    RULES="${VERIFY:-$RULES_DEFAULT}"
    [[ -f "$RULES" ]] || die "rules file not found: $RULES
       pass the path: ./official.sh --verify /path/to/'1. Qwen3 AI Task Rules.md'"
    AIPERF=aiperf; N='${N}'; HOST='${HOST}'; MODEL='${MODEL}'
    build_cmd
    printf '%s\n' "${CMD[@]}" > /tmp/.official_ours.$$
    PY=""
    for c in "$VENV/bin/python" python3 python; do
        [[ -x "$c" ]] && { PY="$c"; break; }
        command -v "$c" >/dev/null 2>&1 && { PY="$(command -v "$c")"; break; }
    done
    [[ -n "$PY" ]] || die "no python found"
    "$PY" - "$RULES" /tmp/.official_ours.$$ <<'PYEOF'
import re, shlex, sys
rules, ours = sys.argv[1], sys.argv[2]
md = open(rules).read()
blocks = [b for b in re.findall(r'```bash\n(.*?)```', md, re.S) if "aiperf profile" in b]
if not blocks:
    print("x no aiperf command found in %s" % rules); sys.exit(2)
if len(blocks) > 1:
    print("! %d aiperf blocks in the rules; comparing against the first" % len(blocks))
cmd = blocks[0][blocks[0].index("aiperf profile"):].replace("\\\n", " ")
theirs = shlex.split(cmd)
mine = [l.rstrip("\n") for l in open(ours)]
if theirs == mine:
    print("ok the command here is identical to the one in the rules (%d tokens)" % len(mine))
    sys.exit(0)
print("x THE COMMAND HAS DIVERGED FROM THE RULES")
import difflib
for line in difflib.unified_diff(theirs, mine, "rules", "official.sh", lineterm="", n=2):
    print("   " + line)
print("\n   the rules are authoritative: edit official.sh to match, never the reverse")
sys.exit(1)
PYEOF
    rc=$?
    rm -f /tmp/.official_ours.$$
    exit $rc
fi

# ---------------------------------------------------------------- run
[[ -n "$N"     ]] || { usage >&2; die "--concurrency is required (the team-declared N)"; }
[[ -n "$HOST"  ]] || die "--host is required (gateway host:port -- never a P or D worker)"
[[ -n "$MODEL" ]] || die "--model is required (the id your gateway reports at /v1/models)"
[[ "$N" =~ ^[0-9]+$ ]] && (( N > 0 )) || die "--concurrency must be a positive integer, got '$N'"
[[ "$HOST" == http://* || "$HOST" == https://* ]] \
    && die "--host takes host:port, not a URL (the command builds http://\$HOST itself)"

build_cmd

if (( PRINT )); then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

command -v "$AIPERF" >/dev/null 2>&1 || [[ -x "$AIPERF" ]] \
    || die "$AIPERF not found -- run ./install.sh --venv $VENV, or pass --aiperf with the binary in your container image"

[[ -n "$OUT" ]] || OUT="./official-$(date +%Y%m%d-%H%M%S)-c$N"
mkdir -p "$OUT" || die "cannot create $OUT"
OUT="$(cd "$OUT" && pwd)"

# Evidence of what was actually run, for the submission. A record, not a change.
{
    echo "# scored run, official command"
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "gateway=$HOST"
    echo "model=$MODEL"
    echo "concurrency=$N"
    echo "aiperf=$("$AIPERF" --version 2>&1 | head -1)"
    echo ""
    printf '%q ' "${CMD[@]}"; echo
} > "$OUT/command.txt"
cat "$OUT/command.txt"

echo "=========================================="
echo " running the official command from $OUT"
echo " (no --artifact-dir: AIPerf writes to \$PWD/artifacts by design)"
echo "=========================================="

cd "$OUT" || die "cannot enter $OUT"
"${CMD[@]}"
rc=$?

echo ""
echo "=========================================="
(( rc == 0 )) && echo "done -> $OUT/artifacts" || echo "! aiperf exited $rc"
echo " gate: good_request_fraction >= 0.90, TTFT <= 3000ms, ITL <= 20ms"
echo " score is good output token/s -- read it off profile_export_aiperf.csv"
echo "=========================================="
exit $rc
