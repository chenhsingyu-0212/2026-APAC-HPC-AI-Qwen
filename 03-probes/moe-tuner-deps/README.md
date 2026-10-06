# Modules the MoE tuners import by bare name

`tuning_fused_moe_triton_sep.py` does `from common_utils import (...)`, and that module
ships in the same directory of SGLang's source tree. Neither is in the pip wheel.

Kept here rather than in `run/` because `common_utils` is a name generic enough to
collide with something else one day, and because having its own directory makes the
version explicit. `run/tune-moe-sep-inner.sh` puts this directory on `PYTHONPATH`.

| file | source |
|---|---|
| `common_utils.py` | `sgl-project/sglang` v0.5.20, `benchmark/kernels/fused_moe_triton/` |

Pinned at v0.5.20 to match the vendored tuner and the SGLang the configs are keyed on.
