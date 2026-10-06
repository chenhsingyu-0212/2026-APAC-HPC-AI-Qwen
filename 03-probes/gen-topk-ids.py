#!/usr/bin/env python3
"""Generate the topk_ids files tuning_fused_moe_triton_sep.py requires.

    run/gen-topk-ids.py --model <path> --batch-size 32 --out <dir> [--ep-size 1]

The separate-GEMM tuner takes --topk-ids-dir and loads 100 tensors from it:

    load_topk_ids(dir, i) -> f"{dir}/topk_ids_layer{i % 58 + 3}_idx{i // 58}.pt"
      (num_layers = 61, dense_layers = 3 are hardcoded in the tuner)

Those are routing decisions, and the tuner is measuring kernel time rather than model
quality, so upstream fills them with uniform random routing -- which is also what the
reference campaign that reached 4,395-4,475 good tok/s on H200 does. Nothing has to be
served to produce them, which is why a tuning run needs no model weights and no
allocation beyond one GPU.

The layer numbering is cosmetic: 61 layers is DeepSeek-V3's count and Qwen3-235B has
94, but the tuner only uses the number to build the filename it then loads, so the
files have to be named with ITS formula, not the model's.
"""
import argparse, json, pathlib, sys

NUM_LAYERS, DENSE_LAYERS = 61, 3          # as hardcoded in the tuner
MOE_LAYERS = NUM_LAYERS - DENSE_LAYERS    # 58
N_FILES = 100                             # load_topk_ids is called for i in range(100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the checkpoint (reads config.json)")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    cfg_path = pathlib.Path(a.model) / "config.json"
    if not cfg_path.is_file():
        print(f"x no config.json under {a.model}", file=sys.stderr)
        return 1
    cfg = json.loads(cfg_path.read_text())
    experts = cfg.get("num_experts") or cfg.get("n_routed_experts")
    topk = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token")
    if not experts or not topk:
        print(f"x cannot read num_experts / num_experts_per_tok from {cfg_path}", file=sys.stderr)
        return 1
    # With expert parallelism the tuner remaps global ids to local ones itself, so the
    # ids written here stay global.
    n_global = experts
    if topk > n_global:
        print(f"x topk {topk} exceeds {n_global} experts", file=sys.stderr)
        return 1

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed)
    written = []
    for i in range(N_FILES):
        ids = torch.rand(a.batch_size, n_global, generator=gen).topk(topk, dim=1).indices
        name = f"topk_ids_layer{i % MOE_LAYERS + DENSE_LAYERS}_idx{i // MOE_LAYERS}.pt"
        torch.save(ids, out / name)
        written.append(name)
    print(f"wrote {len(written)} files of shape [{a.batch_size}, {topk}] "
          f"over {n_global} experts into {out}")
    print(f"  first {written[0]}   last {written[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
