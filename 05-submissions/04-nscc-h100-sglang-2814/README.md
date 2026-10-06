# 2,814 good output tokens/s at N=80 — 16× H100, SGLang PD

Our best declarable number. 2,834 was measured at N=88 and is **not** declarable --
a re-run of the same configuration on the same nodes returned pass 0.880 there -- so
this folder declares N=80, where both runs cleared the gate. It is submission 01 plus
one change: **the prefill forward batch goes from 8,192 tokens to
12,288**, paid for with memory taken out of a KV pool that was never using it.

| | |
|---|---|
| **good output tokens/s** | **2,814** at the declarable concurrency (2,834 was measured at N=88; see below) |
| declared concurrency | **80** |
| pass (`good_request_fraction`) | **0.960**, and 0.930 on a re-run |
| TTFT p90 / ITL p90 | 1,813 ms / 19.6 ms |
| topology | prefill **2× TP4** + decode **2× TP4**, two nodes |
| measured by | job **216132**, paired against a control in the same allocation |
| platform | NSCC ASPIRE 2A+, 16× H100 80GB HBM3 |

Run it with `qsub qwen3-pd-h100-16gpu.pbs`. `./selftest.sh` checks the 41 things that
can be checked without an allocation; `./verify-basic.sh` checks the run afterwards.

## The techniques used

**Tier 1 — the three from submission 01, unchanged.** They are what took 986 to 2,706
and they still carry most of the number.

1. **No expert parallelism on decode.** SGLang defaults `--ep-size` to the TP degree
   for a PD decode worker. Expert parallelism trades an all-reduce for an all-to-all
   that costs 4.3–5.2 ms per step at this batch size. Omitting it: **2.27×**.
2. **Integer GPU ids, so custom all-reduce can initialise.** PBS hands
   `CUDA_VISIBLE_DEVICES` as GPU UUIDs and SGLang calls `int()` on them, so every TP
   rank fell back to NCCL silently. Translating to each GPU's position in the node's
   enumeration: **+6.3%**.
3. **DeepGEMM on the prefill role only**, with `--disable-prefill-cuda-graph` because
   first *use* of a DeepGEMM cubin does a module load and a module load inside a
   CUDA-graph capture is illegal: **+10.7%**.

**Tier 2 — what this submission adds.**

4. **A 12,288-token prefill forward batch** (`--chunked-prefill-size 12288
   --max-prefill-tokens 12288`), **with `--mem-fraction-static 0.80` on the prefill
   role only**. Three ISL-4000 requests per forward instead of two. Measured **+4.6%**
   on the score and **+6.0%** on what it actually moves, prefill's service rate:
   13.89 → **14.72 req/s**.

   All three parts are load-bearing. Both flags are needed because
   `--max-prefill-tokens` caps the batch independently of `--chunked-prefill-size`. And
   the lower fraction is needed because DeepGEMM asks for `max_m = 2 ×
   max_prefill_tokens` and halves it until it fits: at 0.85 a 12,288 batch halves twice
   to 6,144, at 0.80 it halves once to exactly 12,288. Any one of the three missing
   gives a server that comes up healthy and scores 2,710.

5. **The declared concurrency is part of the configuration: N=80, not 72.** This is not
   tuning for its own sake. At N=72 this configuration scores **2,686**, which is
   *below* submission 01's 2,710 at the same concurrency. A prefill-capacity lever pays
   nothing below the old saturation point, and the saturating concurrency moves up with
   the rate (N\* ≈ 5.1R). Job 216132 measured 2,834 at 88, 2,814 at 80, 2,686 at 72.

   **N=88 is where the throughput peaks and it is not declarable.** Job 216159 re-ran
   this exact configuration on the same two nodes and N=88 returned pass **0.880**
   against 216132's 0.962. Under 0.90 the run scores the reference floor, not 2,834.
   This is not thermal and not cell order — the decode node held 1980 MHz in both
   windows at 52.5 °C and 50.8 °C, and average ITL was 19.0 ms both times. What moved
   was the tail: this configuration sits at **95% of the ITL budget**, and whether the
   tail crosses 20 ms is not stable between runs.

   |  | good tok/s | pass, two runs | ITL headroom |
   |---|---|---|---|
   | N=88 | 2,834 / 2,608 | 0.962 / **0.880** | none — one run under the gate |
   | **N=80** | **2,814 / 2,742** | **0.960 / 0.930** | thin but holding |
   | 01 at N=72 | 2,706 / 2,711 | 0.976 / 0.976 | zero ITL failures, max 19.47 ms |

   If the scored run has to be conservative, submission 01 is the configuration with
   margin; this one is +2.5% with less of it.

## Where the memory came from

The whole of change 4 is one observation. Over ~1,250 Prefill-batch steps on each
worker, job 215960 recorded the prefill KV pool's `token usage` at **0.11–0.13**,
peaking at **0.19** — including the first-ten-seconds burst where every one of our TTFT
failures happens. A disaggregated prefill worker holds each request's KV only until the
decode side pulls it, so the pool is about five times larger than the job needs.

In the same logs, what it was starving:

    Available memory 9.70GB is less than required memory 16.94GB for warmup,
    reducing max_m to 8192 to avoid out of memory

Those two facts sat in the same file. The 7 GB DeepGEMM could not have was in a pool
nothing was using, and that is also why job 215921's first attempt at this batch died
of OOM under a single global mem-fraction: **a decode worker holds 200 output tokens
per in-flight request and wants the pool as big as possible, so one number cannot serve
both roles.**

A calibration worth keeping: `available(f) = 79.65(1−f) − 2.2` predicted 13.73 GB at
f=0.80 and the log reported 13.48.

## The bottleneck this run hit

**Still the TTFT tail, and now the ITL gate as well.** At N=88, all 69 failures out of
1,833 are TTFT and all of them are in the first ten seconds; from t=10 s the run is
clean. That is the same signature as every run before it: a thundering herd of N
simultaneous 4,000-token prefills against a finite prefill rate, with the failure count
tracking `N − 3R`.

What is new is that **ITL has started failing.** `bigpf` at N=72 has 30 ITL failures
with a per-request maximum of 20.54 ms, and at N=80 it has 20 at 20.32 ms. The control,
at the same concurrencies in the same allocation, has none and never exceeds 19.47 ms.
A faster prefill has pushed us onto the second gate for the first time in this project.

In job 216132, N=88 had **zero** ITL failures and the lowest maximum of the three
(19.91 ms), which looked like the opposite of the direction ITL normally moves with
batch size. Job 216159 answered it: on the re-run N=88 had **153**. The first reading
was the lucky end of a distribution sitting on the gate, not a mechanism.

## What to try next, from this run's own numbers

At N=80, TTFT p90 is 1,437 ms against a 3,000 ms gate — **48% of budget** — while ITL
is 19.0 against 20, which is **95%**. The two are paired: in a closed loop at fixed N
the total latency is pinned by N/R, so pushing wait out of decode and into prefill
trades ITL for TTFT. We have TTFT to spend and no ITL to spare, and
`--max-running-requests` is the knob that does it — currently 256 against a decode
batch of about 40, so it is not doing anything at all today.

This also rules something out. A tuned Triton MoE config is **not** the way to buy that
ITL margin: job 216159 measured the H100-native config for batch size 32, merged so
that only that one key differed from the config in use, and average ITL went 19.03 →
**20.32 ms**, over the gate, with 1,317 of 1,790 requests failing on ITL and the score
at 712. That is the second tuned config to be harmful — the H200 transplant was the
first — so the problem is the tuner's output, not which GPU produced it.

## What changed since 2,706, and which bottleneck it answered

Submission 01's bottleneck was named precisely: prefill's service rate, saturated at
13.89 req/s, with every failure a TTFT failure in the opening burst. Two things follow
from a saturated rate, and both were measured rather than assumed:

* **Lowering the concurrency loses.** 12.23 req/s at N=56 and 13.11 at N=64 give back
  less raw rate than the failures they save. N=72 was already the optimum for 01.
* **The transient is a fixed tax.** At the saturating concurrency N ≈ 5.1R and failures
  ≈ N − 3R, so the loss is (5.1R − 3R)/120R ≈ 1.75% whatever R becomes.

Which left exactly one thing that moves the score: **raise R.** Change 4 raises it by
6.0%, and the score follows at 4.6%.

Two attempts in between did not survive, and are worth recording because they were the
other candidates aimed at the same bottleneck:

* **Raising the custom all-reduce ceiling** (`SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=
  131072`), measured at +2.5% on the H200 reference track. Job 215960 ran it paired:
  2,712 against the control's 2,710, with the two cells identical per request to three
  figures. **Refuted at TP4 on H100.**
* **A tuned Triton MoE config transplanted from H200**, which is *worse* than having
  none: pass 0.990 → 0.090, because the 1.43× HBM bandwidth difference changes the
  right tiling.

## Deviations from submission 01

None beyond change 4 and the concurrency. The two deviations 01 documents still apply:
`--disable-radix-cache` stands in for the installed AIPerf's missing `--cache-bust`,
and the engine runs from a venv rather than an enroot image because NSCC provides no
container runtime for this image.
