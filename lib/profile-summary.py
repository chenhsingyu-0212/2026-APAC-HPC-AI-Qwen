#!/usr/bin/env python3
"""Bucket a PyTorch profiler trace into the categories the question is about.

    run/profile-summary.py <trace.json.gz | dir> [--stage prefill|decode]

The question this exists to answer: of the time a prefill forward pass takes, how
much is GEMM and how much is the TP all-reduce? The arithmetic says all-reduce should
be 7-30% of a 310 ms 8192-token batch depending on the bandwidth actually achieved,
and 42 ms versus 95 ms is the difference between ignoring it and attacking it. So it
gets measured.

TWO RULES THIS FOLLOWS, because a categoriser that silently mislabels is worse than
no categoriser:

  * Only GPU kernels are counted. CPU-side operator ranges overlap each other and
    each other's children, so summing them double-counts; kernel durations on a
    stream do not.
  * Anything not matched is reported as `unclassified` WITH its largest names, never
    folded into "other" and forgotten. If that bucket is large the categories are
    wrong and the output says so instead of looking tidy.
"""

import argparse, glob, gzip, json, os, sys
from collections import defaultdict

# Ordered: first match wins, so the specific patterns come before the generic ones.
RULES = [
    ("collective/nccl",  ("nccl", "allreduce", "all_reduce", "alltoall", "all_to_all",
                          "reducescatter", "reduce_scatter", "allgather", "all_gather",
                          "broadcast", "sendrecv", "send_recv")),
    ("moe/routing",      ("topk", "moe_align", "moe_sum", "routing", "expert",
                          "scatter_add", "gather_scatter", "permute")),
    ("gemm/fp8",         ("deepgemm", "deep_gemm", "fp8", "scaled_mm", "w8a8",
                          "block_scale", "blockwise")),
    ("gemm/dense",       ("cutlass", "sm90", "sm80", "gemm", "matmul", "mma", "cublas",
                          "nvjet", "ampere", "hopper")),
    ("attention",        ("flash", "fmha", "attn", "attention", "paged", "rope",
                          "rotary", "softmax_lse")),
    ("norm/elementwise", ("layernorm", "rmsnorm", "rms_norm", "silu", "gelu", "swiglu",
                          "add_", "mul_", "cast", "convert", "copy", "memcpy",
                          "elementwise", "vectorized")),
    ("quantize",         ("quant", "dequant", "per_token", "per_tensor")),
    ("sampling",         ("sample", "argmax", "logits", "penalt", "top_p", "top_k")),
]

def classify(name: str) -> str:
    low = name.lower()
    for label, pats in RULES:
        if any(p in low for p in pats):
            return label
    return "unclassified"

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="trace file, or a directory to search")
    ap.add_argument("--top", type=int, default=12, help="kernels to list per bucket")
    args = ap.parse_args()

    files = []
    if os.path.isdir(args.path):
        for pat in ("*.trace.json.gz", "*.trace.json", "*.pt.trace.json.gz", "*.json.gz", "*.json"):
            files += sorted(glob.glob(os.path.join(args.path, "**", pat), recursive=True))
    else:
        files = [args.path]
    files = [f for f in files if os.path.getsize(f) > 4096]
    if not files:
        sys.exit(f"x no trace found under {args.path}")

    for path in files:
        try:
            tr = load(path)
        except Exception as e:
            print(f"  ! {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue
        evs = tr.get("traceEvents", tr if isinstance(tr, list) else [])

        # GPU kernels only. The profiler tags them cat=kernel; some builds also emit
        # gpu_memcpy / gpu_memset, which belong in the elementwise bucket rather than
        # being dropped, because a prefill batch does move a lot of bytes.
        by_bucket, by_name, total = defaultdict(float), defaultdict(float), 0.0
        nkern = 0
        for e in evs:
            if e.get("ph") != "X":
                continue
            cat = (e.get("cat") or "").lower()
            if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            dur = float(e.get("dur") or 0.0)      # microseconds
            name = e.get("name") or "?"
            b = "norm/elementwise" if cat != "kernel" else classify(name)
            by_bucket[b] += dur
            by_name[(b, name)] += dur
            total += dur
            nkern += 1

        print("=" * 78)
        print(f" {os.path.basename(path)}")
        print(f" {nkern:,} 個 GPU kernel，合計 {total/1000:.1f} ms 的 GPU 時間")
        print("=" * 78)
        if not total:
            print("  沒有 kernel 事件 —— 這個 trace 可能只有 CPU 活動")
            continue
        for b, us in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
            bar = "#" * max(1, round(us / total * 44))
            print(f"  {b:<18} {us/1000:9.2f} ms  {us/total*100:5.1f}%  {bar}")
        # The one that matters for the question asked.
        coll = by_bucket.get("collective/nccl", 0.0)
        print(f"\n  → 集合通訊（all-reduce / all-to-all）佔 {coll/total*100:.1f}%")
        unc = by_bucket.get("unclassified", 0.0)
        if unc / total > 0.05:
            print(f"  ! unclassified 佔 {unc/total*100:.1f}% —— 分類規則不完整，最大的幾個是：")
            for (b, n), us in sorted(by_name.items(), key=lambda kv: -kv[1]):
                if b == "unclassified":
                    print(f"      {us/1000:8.2f} ms  {n[:78]}")
                    if us / total < 0.01:
                        break
        print(f"\n  最花時間的 {args.top} 個 kernel：")
        for (b, n), us in sorted(by_name.items(), key=lambda kv: -kv[1])[: args.top]:
            print(f"    {us/1000:8.2f} ms  {us/total*100:5.1f}%  [{b}] {n[:62]}")
        print()

if __name__ == "__main__":
    main()
