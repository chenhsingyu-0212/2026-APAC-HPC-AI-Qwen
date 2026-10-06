#!/bin/bash
# Install a tuned Triton fused-MoE config so SGLang will actually load it.
#
#   run/install-moe-config.sh <tuned.json> <outdir>
#   SGLANG_VENV=... run/install-moe-config.sh results/tune-moe-98820/configs/E=128,...json moe-configs/h100-3.7.1
#   MERGE_BASE=<the config the engine loads today> ... run/install-moe-config.sh ...
#
# MERGE_BASE EXISTS SO AN A/B CHANGES ONE THING. A tuner run narrowed to the batch
# sizes we actually operate at produces a config with one or two keys, and SGLang
# resolves EVERY M to the nearest key it has -- so installing it alone silently
# re-tiles every other batch size too, including the ones CUDA-graph capture walks.
# The result would be a real measurement of something nobody chose. Merging our keys
# over the config the engine loads today leaves all the others exactly as they were.
#
# Then point the engine at it:
#
#   SGLANG_MOE_CONFIG_DIR=<outdir> ...
#
# Two things have to be right, and neither is obvious from either project's docs.
#
# 1. THE FILE VLLM'S TUNER WRITES IS NOT LOADABLE BY SGLANG. benchmark_moe.py dumps
#    `{"triton_version": ..., **configs}`, and SGLang's loader is
#
#        return {int(key): val for key, val in json.load(f).items()}
#
#    so int("triton_version") raises ValueError on the first key. vLLM's own loader
#    skips the key, which is why nobody notices until the other engine reads it.
#    We strip it and keep the value in a sidecar so the provenance is not lost.
#
# 2. SGLANG_MOE_CONFIG_DIR REPLACES THE WHOLE TREE, NOT JUST ONE FILE. The lookup is
#    $DIR/configs/triton_<version>/<name>.json, and when that misses, the fallback
#    enumerates `os.listdir($DIR/configs)` for other triton_* directories. Point it
#    at a directory holding a single file and every OTHER (E,N) lookup -- the down
#    projection, any other expert shape -- finds nothing at all and drops to the
#    heuristic, losing the packaged configs it is using today. So the output is a
#    full symlink mirror of the packaged tree with our file layered on top.
set -uo pipefail

SRC="${1:-}"; OUT="${2:-}"
[[ -n "$SRC" && -n "$OUT" ]] || { echo "usage: ${0##*/} <tuned.json> <outdir>" >&2; exit 2; }
[[ -r "$SRC" ]] || { echo "x cannot read $SRC" >&2; exit 1; }

# Locate the packaged tree to mirror. Honour an explicit venv, else ask python.
PKG=""
if [[ -n "${SGLANG_VENV:-}" ]]; then
    PKG="$SGLANG_VENV/lib/python3.10/site-packages/sglang/srt/layers/moe/moe_runner/triton_utils"
else
    PKG="$(python3 - <<'PY' 2>/dev/null
import os.path
try:
    from sglang.srt.layers.moe.moe_runner import triton_utils as t
    print(os.path.dirname(os.path.realpath(t.__file__)))
except Exception:
    pass
PY
)"
fi
[[ -n "$PKG" && -d "$PKG/configs" ]] || {
    echo "x cannot find sglang's triton_utils/configs -- set SGLANG_VENV" >&2; exit 1; }

# The version directory is named for the Triton that will READ the config, which is
# the one in the serving venv -- not whatever the tuner happened to run under.
TRITON_VER="${TRITON_VERSION:-$(python3 -c 'import triton;print(triton.__version__)' 2>/dev/null)}"
[[ -n "$TRITON_VER" ]] || { echo "x cannot determine the triton version -- set TRITON_VERSION" >&2; exit 1; }
VERDIR="triton_${TRITON_VER//./_}"

mkdir -p "$OUT/configs" || exit 1
# Mirror every packaged entry by symlink, so a miss on our file still finds the
# fallbacks SGLang ships. Skip the version dir we are about to write into.
for d in "$PKG/configs"/*; do
    n="$(basename "$d")"
    [[ "$n" == "$VERDIR" ]] && continue
    [[ -e "$OUT/configs/$n" ]] || ln -s "$d" "$OUT/configs/$n" || exit 1
done
# Our version directory is a REAL directory: symlinks for what ships, our file on top.
mkdir -p "$OUT/configs/$VERDIR" || exit 1
if [[ -d "$PKG/configs/$VERDIR" ]]; then
    for f in "$PKG/configs/$VERDIR"/*; do
        [[ -e "$f" ]] || continue
        [[ -e "$OUT/configs/$VERDIR/$(basename "$f")" ]] || ln -s "$f" "$OUT/configs/$VERDIR/" || exit 1
    done
fi

# 3. AND THE FILENAME HAS TO BE SPELLED SGLANG'S WAY. SGLang builds the name with
#    f",block_shape={block_shape}" where block_shape is a Python list, so str() puts a
#    SPACE after the comma:
#
#      ...,dtype=fp8_w8a8,block_shape=[128, 128].json     <- what sglang opens
#      ...,dtype=fp8_w8a8,block_shape=[128,128].json      <- what vllm's tuner wrote
#
#    A one-character difference, no error, and the config is simply never found --
#    indistinguishable from a tuned config that did nothing. Normalise it by reparsing
#    the list and re-rendering it with Python's own str(), which is what sglang did.
DEST_DIR="$OUT/configs/$VERDIR"
python3 - "$SRC" "$DEST_DIR" <<'PY' || exit 1
import json, os, re, sys, pathlib

src, dest_dir = sys.argv[1], sys.argv[2]
mb = os.environ.get("MERGE_BASE", "")

def load_keys(path, what):
    try:
        d = json.loads(pathlib.Path(path).read_text())
    except FileNotFoundError:
        sys.exit(f"x {what} is not a file: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"x {what} is not valid JSON: {path} ({e})")
    tv = d.pop("triton_version", None)
    bad = [k for k in d if not k.lstrip("-").isdigit()]
    if bad:
        sys.exit(f"x {what} has non-integer keys, sglang's loader would raise: {bad}")
    if not d:
        sys.exit(f"x {what} has no batch-size entries: {path}")
    return d, tv

# Validate the base BEFORE anything else, so a bad base says so rather than tripping a
# later check about whatever filename it happens to have.
base = None
if mb:
    if not pathlib.Path(mb).is_file():
        sys.exit(f"x MERGE_BASE is not a file: {mb}")
    base, _ = load_keys(mb, "MERGE_BASE")

raw, tv = load_keys(src, "the tuned config")

# THE INSTALLED NAME IS A LOOKUP KEY, NOT A LABEL. SGLang opens exactly
# "E=<experts>,N=<intermediate/TP>,device_name=<gpu>,dtype=...,block_shape=....json";
# any other name is a file it will never read, and there is no error. The source here
# had been renamed to h100-tuned-bs32.json on its way back from the tuning cluster,
# which would have installed a config nothing could load and measured a no-op.
# With MERGE_BASE the right name is already known: it is the base's own name, which is
# by definition the one the engine looks up.
name = pathlib.Path(mb).name if mb else pathlib.Path(src).name
# And block_shape has to be spelled the way str(list) spells it, with the space.
name = re.sub(r"block_shape=\[([0-9,\s]*)\]",
              lambda m: "block_shape=" + str([int(x) for x in m.group(1).split(",") if x.strip()]),
              name)
if not (name.startswith("E=") and "device_name=" in name):
    sys.exit(f"x '{name}' is not a name sglang looks up -- it opens\n"
             f"    E=<experts>,N=<intermediate/TP>,device_name=<gpu>,dtype=...,block_shape=....json\n"
             f"  Pass MERGE_BASE pointing at the config the engine loads today, or rename the source.")

base_note = ""
if base is not None:
    overridden = sorted((k for k in raw if k in base), key=int)
    added      = sorted((k for k in raw if k not in base), key=int)
    merged = dict(base); merged.update(raw)
    base_note = (f"  merged over {len(base)} key(s) from {mb}\n"
                 f"  replaced: {overridden or 'none'}   added: {added or 'none'}")
    raw = merged

dest = str(pathlib.Path(dest_dir) / name)
pathlib.Path(dest).write_text(json.dumps({k: raw[k] for k in sorted(raw, key=int)}, indent=4))
pathlib.Path(dest + ".provenance").write_text(
    f"tuned-by-triton: {tv}\nsource: {src}\n" + (f"merge-base: {mb}\n" if mb else ""))
if base_note: print(base_note)
print(f"installed {len(raw)} batch size(s) {sorted(raw, key=int)} -> {dest}")
if tv: print(f"  tuned under triton {tv}; installed for the serving venv's version")
PY

echo "point the engine at it with:"
echo "  SGLANG_MOE_CONFIG_DIR=$(cd "$OUT" && pwd)"
