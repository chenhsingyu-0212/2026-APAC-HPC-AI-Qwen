# 2,844 good output tokens/s — 16× H100, SGLang PD, cold-start safe

Submission 04 scores 2,814 and would probably **lose most of that** when the
organisers re-run it on a dedicated account. This is the same deployment with the
cold-start hole closed, and the hole is the reason this directory exists.

| | |
|---|---|
| **good output tokens/s** | **2,844** |
| good_request_fraction | 0.971 |
| concurrency | 80 |
| request throughput | 14.64 req/s |
| TTFT p90 / ITL p90 | 1,899 ms / 19.22 ms |
| measured | NSCC jobs 216202 and 216243, **2,844 both times** |

## The hole in 04

Rules: *"the organisers rebuild the stack and re-run on a dedicated account"*
(Task Rules.md:224). A dedicated account has an empty `$HOME`, and three things in
04 depended on ours not being empty.

### 1. DeepGEMM's MoE grouped GEMM JITs on first contact

04 passes `--moe-runner-backend deep_gemm` on the prefill role — worth +10.7% and
one of the four changes behind the score. The DeepGEMM cache lives in
`$HOME/.cache/sglang/deep_gemm`, which is **shared storage on both clusters**, so
one earlier job populated it and every later job of ours found it warm. Every
number we have, including 2,814 and 2,844, was measured warm.

`sglang.compile_deep_gemm` does **not** close this. That tool compiles the DENSE
FP8 path — job 215218 produced 28 `sm90_fp8_gemm_1d2d` kernels whose locks are
named `_g1`, group of one. The MoE grouped GEMM is keyed on the expert and token
shapes of a real batch, so it compiles when a real batch arrives.

With no prewarm, the first real batch arrives **inside the official
`--warmup-duration 60`**, or worse, inside the scored window.

### 2. The KV-transfer path opens per prefill–decode pair

On a cold PD layout the first request on a pair can stall for minutes while NIXL
sets it up. Both router policies are `round_robin`, so there are four pairs, and
60 s of official warm-up does not reliably open all four.

### 3. Nothing measured how long a cold start takes

04 asks for `walltime=01:00:00`. A cold bring-up is roughly
`720 s model load + ~460 s first-contact compile`, against a budget that also has
to hold the 210 s official window. The teammate's vLLM rehearsal 20633 failed on
exactly this: cold pre-warm ran 460–470 s against a 441 s deadline (warm: 226 s).

## What changed here

**A two-phase connection prewarm before the official command**, and it is the
single fix for both #1 and #2:

```
burst       32 requests at concurrency 32    ISL 4000, OSL 16
sequential   8 requests at concurrency  1    ISL 4000, OSL 16
```

Three properties matter and none is incidental:

* **Same shape as the scored run.** ISL 4000 at concurrency 32 produces
  12,288-token prefill batches — exactly what `--chunked-prefill-size` asks for.
  A prewarm at another ISL compiles other kernels and leaves the real ones cold.
* **Burst then sequential, not interchangeable.** Sending n requests at
  concurrency n warms n *routes*; with both policies on `round_robin` advancing
  together, 4 requests visit p0→d0, p1→d1, p0→d0, p1→d1 — two pairs twice, never
  the cross pairs steady state uses. The burst opens every pair at once, the
  sequential tail gives each a warm second pass.
* **Not the official dataset or seed.** `--random-seed 7`, `OSL 16`. The rules
  allow connection warm-up only on other data (Task Rules Q1), and the official
  `--warmup-duration 60` still runs afterwards, unchanged.

**It fails the job if it collected nothing.** Job 215492's prewarm printed
`All 4 inference request(s) failed. No successful responses were collected` and
the run went on to benchmark anyway, measuring a half-open deployment.

**`walltime=01:40:00`** instead of 01:00:00, so a cold first contact fits.

## What was already right in 04, and is the thing not to break

The Triton and Inductor caches are **per worker on node-local `/tmp`**:

```
TRITON_CACHE_DIR="/tmp/pdw-$role-$port/triton"
TORCHINDUCTOR_CACHE_DIR="/tmp/pdw-$role-$port/inductor"
```

This is what makes us immune to the worst of the teammate's cold-cache failure.
Their four prefill engines shared one Triton cache on BeeGFS and one crashed in
Inductor autotune with
`FileNotFoundError: .../triton/<hash>/triton_red_fused_rms_norm_2.ptx` — two
engines writing the same cache directory at once. A fresh account's
`~/.triton/cache` is on shared `/home`, so the default would put us in the same
position.

It also means **every run of ours already pays cold Triton/Inductor compilation**,
so the 12-minute bring-up we measure is the honest number for that part.

## Still open

**The cold prewarm has not been measured on our stack.** The 460–470 s figure is
the teammate's, on vLLM. Nano5 jobs 372421 and 372456 run with
`$HOME/.cache/sglang/deep_gemm` absent, so their first cell is cold and their
second warm — that pair gives the real number, and this README should carry it
once they report.

**`CELL_OVERHEAD` under-estimates a cold cell.** `run/10-campaign.sbatch` budgets
`HEALTH_TIMEOUT/3 + 180` = 780 s for bring-up plus prewarm; a cold cell needs
roughly 1,180. It is a planning estimate, not a kill timer, so it does not end a
job — but on a tight walltime it will start a cell that cannot finish.
