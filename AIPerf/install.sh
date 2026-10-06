#!/bin/bash
# Portable AIPerf installer -- use this same script on a new machine, a new
# cluster, or when handing it to a teammate.
#
#   ./install.sh                          # install into ./aiperf-venv
#   ./install.sh --venv /work/$USER/venv  # install into a given location
#   ./install.sh --pin 0.12.0             # pin the version (recommended for cross-environment comparison)
#   ./install.sh --check-only --venv ...  # only verify an existing install, install nothing
#
# It does four things: find a usable Python -> create the venv -> install aiperf
# -> **verify and print the flags that are actually available**. That last step
# is the point: AIPerf's sequence length flags come in two spellings depending
# on the version, and the benchmark script needs to know which one to use.
#
# Exits non-zero if a flag run.sh depends on is missing, so --check-only works
# as a CI gate.
#
# This deliberately does not depend on the project's 00_env.sh, so it can be
# copied out and used standalone.

set -uo pipefail

# ---------------------------------------------------------------- helpers
ok()    { echo "  ok $*"; }
warn()  { echo "  ! $*"; }
die()   { echo "  x $*" >&2; exit 1; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; }   # the header block above
need()  { [[ $# -ge 2 ]] || die "$1 requires a value"; }

# Flags run.sh depends on; a missing one means the benchmark script needs adjusting
REQUIRED_FLAGS=(--streaming --goodput --tokenizer --concurrency --request-count --artifact-dir --endpoint-type)

# ---------------------------------------------------------------- defaults
VENV="${AIPERF_VENV:-$(pwd)/aiperf-venv}"
PIN=""
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)       need "$@"; VENV="$2"; shift 2 ;;
        --pin)        need "$@"; PIN="==$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "x unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

echo "=========================================="
echo " AIPerf install -> $VENV"
echo "=========================================="

# ---------------------------------------------------------------- 1. find Python
# AIPerf declares requires_python >=3.11,<3.14. Both ends matter: on 3.10 pip
# fails with a resolver error *after* the venv exists, and on 3.14 it refuses
# the same way -- neither says "your interpreter is the problem", so check here.
# On many HPC sites /usr/bin/python3 is still 3.6, so the version has to be
# picked explicitly rather than trusted.
PY_MIN="3.11"; PY_MAX_EXCL="3.14"
if (( CHECK_ONLY == 0 )) && [[ ! -x "$VENV/bin/python" ]]; then
    echo "[1/4] looking for a usable Python (>=$PY_MIN, <$PY_MAX_EXCL)..."
    PY=""; SEEN=""
    for c in python3.13 python3.12 python3.11 \
             /usr/bin/python3.13 /usr/bin/python3.12 /usr/bin/python3.11 \
             python3; do
        p="$(command -v "$c" 2>/dev/null)" || continue
        v="$("$p" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" || continue
        SEEN="${SEEN}${SEEN:+, }$v"
        [[ "$(printf '%s\n%s\n' "$v" "$PY_MIN" | sort -V | head -1)" == "$PY_MIN" ]] || continue
        [[ "$(printf '%s\n%s\n' "$v" "$PY_MAX_EXCL" | sort -V | head -1)" == "$v" \
           && "$v" != "$PY_MAX_EXCL" ]] || continue
        PY="$p"; ok "$p (Python $v)"; break
    done
    [[ -n "$PY" ]] || die "no Python in [$PY_MIN, $PY_MAX_EXCL) found${SEEN:+ (saw: $SEEN)}.
       AIPerf will not install outside that range. HPC sites usually need
       'module load python' first, or use conda."

    echo "[2/4] creating the venv..."
    "$PY" -m venv "$VENV" || die "venv creation failed (the python3-venv package may be missing)"
    ok "$VENV"
else
    echo "[1-2/4] reusing the existing venv: $VENV"
    [[ -x "$VENV/bin/python" ]] || die "no python inside $VENV, drop --check-only"
fi

# ---------------------------------------------------------------- 2. install
if (( CHECK_ONLY == 0 )); then
    echo "[3/4] installing aiperf${PIN}..."
    "$VENV/bin/pip" install -q -U pip 2>/dev/null
    "$VENV/bin/pip" install -q -U "aiperf${PIN}" || die "pip install failed (check outbound network or proxy settings)"
    ok "install complete"
else
    echo "[3/4] skipping install (--check-only)"
fi

AIPERF="$VENV/bin/aiperf"
[[ -x "$AIPERF" ]] || die "$AIPERF does not exist"

# ---------------------------------------------------------------- 3. verify
echo "[4/4] verifying..."
VER="$("$AIPERF" --version 2>&1 | head -1)"
ok "aiperf $VER"
ok "python $("$VENV/bin/python" -V 2>&1 | awk '{print $2}')"

HELP="$("$AIPERF" profile --help 2>&1)"
MISSING=()

# Sequence length flags: two spellings, the benchmark script must use the right one
if grep -q -- "--synthetic-input-tokens-mean" <<<"$HELP"; then
    ok "sequence length flags: --synthetic-input-tokens-mean N --output-tokens-mean M"
elif grep -q -- "--isl" <<<"$HELP"; then
    ok "sequence length flags: --isl N --osl M"
else
    warn "cannot recognize the sequence length flags -- check 'aiperf profile --help' yourself"
    MISSING+=("sequence-length")
fi

for f in "${REQUIRED_FLAGS[@]}"; do
    if grep -q -- "$f" <<<"$HELP"; then
        ok "has $f"
    else
        warn "**missing $f** -- the benchmark script needs adjusting"
        MISSING+=("$f")
    fi
done

# A second aiperf often lives in the home directory, easy to hit by accident when testing by hand
OTHER="$(command -v aiperf 2>/dev/null)"
if [[ -n "$OTHER" && "$OTHER" != "$AIPERF" ]]; then
    warn "another aiperf is on PATH: $OTHER ($("$OTHER" --version 2>&1 | head -1))"
    warn "always use the absolute path $AIPERF, never the bare 'aiperf'"
fi

if (( ${#MISSING[@]} > 0 )); then
    echo ""
    echo "=========================================="
    echo "x ${#MISSING[@]} required flag(s) unavailable: ${MISSING[*]}"
    echo "  run.sh will not work as-is against aiperf $VER"
    echo "=========================================="
    exit 1
fi

cat <<EOF

==========================================
ready to use

  benchmark binary: $AIPERF
  run a sweep:      ./run.sh --url http://<host>:<port> --tokenizer <local model dir>

  to compare numbers across environments, **pin the version**: --pin ${VER}
==========================================
EOF
