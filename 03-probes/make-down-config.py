#!/usr/bin/env python3
"""Write a _down.json that differs from what the engine runs today by TMA alone.

    run/make-down-config.py --base <up config the engine loads> --out <dir> [--group-size-m N] [--num-stages N]

SGLang runs two grouped GEMMs per MoE layer and looks each one up separately. When the
down-projection file is absent it borrows the up config and says so:

    Down MoE config file not found at ..._down.json;
    reusing the tuned up-projection config without TMA.

So today's down GEMM already runs the up tiling -- and the ONLY thing a _down.json with
identical values changes is that `USE_TMA: true` becomes available to pop:

    down_tma_requested = down_config is not None and down_config.pop("USE_TMA", False)
    down_moe_use_tma   = _moe_support_tma() and down_tma_requested

That makes a one-variable experiment possible with no tuning at all, which matters
because the separate-GEMM tuner has produced nothing in five attempts.

The reference H200 campaign's tuned down config agrees with ours on BLOCK_SIZE_M,
BLOCK_SIZE_N, BLOCK_SIZE_K and num_warps, and differs on GROUP_SIZE_M (1 vs 32) and
num_stages (2 vs 3). Those two are exposed as options so a second arm can test them, but
they are NOT the default here: changing three things at once is how 216159 ended up
unable to say which one did the damage.
"""
import argparse, json, pathlib, sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the up config the engine loads today")
    ap.add_argument("--out", required=True, help="directory to write <base name>_down.json into")
    ap.add_argument("--group-size-m", type=int, help="override GROUP_SIZE_M (reference down uses 1)")
    ap.add_argument("--num-stages", type=int, help="override num_stages (reference down uses 2)")
    ap.add_argument("--no-tma", action="store_true", help="write the file WITHOUT USE_TMA (a null arm)")
    a = ap.parse_args()

    bp = pathlib.Path(a.base)
    if not bp.is_file():
        print(f"x base config not found: {bp}", file=sys.stderr)
        return 1
    if bp.name.endswith("_down.json"):
        print(f"x --base is already a down config: {bp.name}", file=sys.stderr)
        return 1
    base = json.loads(bp.read_text())
    base.pop("triton_version", None)
    bad = [k for k in base if not k.lstrip("-").isdigit()]
    if bad:
        print(f"x base has non-integer keys, sglang's loader would raise: {bad}", file=sys.stderr)
        return 1
    if not base:
        print(f"x base has no batch-size entries: {bp}", file=sys.stderr)
        return 1

    changed = []
    out = {}
    for k, cfg in base.items():
        c = dict(cfg)
        if not a.no_tma:
            c["USE_TMA"] = True
        if a.group_size_m is not None:
            c["GROUP_SIZE_M"] = a.group_size_m
        if a.num_stages is not None:
            c["num_stages"] = a.num_stages
        out[k] = c
    if not a.no_tma:
        changed.append("USE_TMA=true")
    if a.group_size_m is not None:
        changed.append(f"GROUP_SIZE_M={a.group_size_m}")
    if a.num_stages is not None:
        changed.append(f"num_stages={a.num_stages}")

    od = pathlib.Path(a.out); od.mkdir(parents=True, exist_ok=True)
    dest = od / (bp.name[: -len(".json")] + "_down.json")
    dest.write_text(json.dumps({k: out[k] for k in sorted(out, key=int)}, indent=4))
    print(f"wrote {dest}")
    print(f"  {len(out)} batch size(s), changed vs the base: {', '.join(changed) or 'nothing'}")
    print(f"  base: {bp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
