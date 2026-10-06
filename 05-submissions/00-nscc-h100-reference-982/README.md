# 982 good output tokens/s — the organisers' reference, ported to H100

The floor, measured. Not a result of ours and not a submission: it exists so the
other folders have a baseline to be compared against, and because the 2026-09-25
addendum makes the reference deployment's score the floor a re-run is held to.

| | |
|---|---|
| **good output tokens/s** | **982** at N=40 |
| good_request_fraction | **0.580** — and **no** concurrency clears 0.90 |
| the sweep | 0.580 / 0.341 / 0.289 / 0.107 / 0.094 at N = 40 / 48 / 56 / 64 / 72 |
| ITL avg at N=40 | **19.55 ms** against a 20 ms gate |
| TTFT p90 at N=40 | 1,366 ms against 3,000 |
| measured | job 214966, 2026-10-01, official 60/120/30 window |

The same qualitative result the reference gives on its home platform: the teammate
ran `reference-scripts/07` unmodified on Firmus (job 20222) and it cleared 0.90 at
no swept N either, peaking at 1,447 at N=56 with gf 0.79.

## It cannot be run as written on this platform

`reference-scripts/07` deploys **4 prefill workers at `--tp-size 2`**. The FP8
checkpoint is 221 GB, so TP2 needs 110 GB per GPU and an H100 has 80. The script
does not start here — which also means the walkthrough in the Application Notes,
whose flags are copied verbatim from it and marked "Do not alter them", cannot be
followed literally on NSCC.

**This is worth raising with the organisers**: the addendum's floor is "the
reference deployment's score", and on the H100 platform that deployment does not
load.

## Techniques used

The organisers', copied item by item. Four forced deviations, and nothing else.

| | from 07 | here | why it had to change |
|---|---|---|---|
| 1 | prefill 4× TP2 | prefill **2× TP4** | TP2 is 110 GB/GPU against 80 available. **This is the big one** — it halves the number of parallel prefill streams, and the teammate measured TP2 prefill at +36–45% req/s per GPU over TP4 |
| 2 | enroot image `sglang-harbor.sqsh` | bare-metal venv, SGLang 0.5.20 | the image does not exist on NSCC; the engine version therefore differs by an unknown amount |
| 3 | Slurm `srun --exact` | PBS + ssh launcher | different scheduler |
| 4 | — | `--enable-metrics` added | without it the engines serve no `/metrics` and the run yields no diagnostics. It exposes an endpoint; it does not change serving |

Copied unchanged: `--context-length 8192`, `--mem-fraction-static 0.90`, decode
`--ep-size 4 --max-running-requests 256`, `sglang_router --pd-disaggregation` with
**no policy flags** (so it runs the router's `cache_aware` default, as 07 does), no
chunked prefill, no MoE backend flag, no prefix-cache flag.

Reproduced with our harness rather than a standalone script, because the point was
fidelity to 07's flags and not a deliverable:

```bash
ENGINES=sglang SPECS=4x2:4x2 \
CONCURRENCIES="40 48 56 64 72" \
VARIANTS="ref07:DECODE_EP=4,CONTEXT_LEN=8192,MEM_FRACTION=0.90,MAX_RUNNING=256,PD_PREFILL_POLICY=,PD_DECODE_POLICY=" \
run/submit-site.sh run/10-campaign.sbatch
```

The empty `PD_PREFILL_POLICY=` is deliberate and was added for this run: our default
is `round_robin`, and reproducing 07 means passing neither flag.

## The bottleneck this run hit

**ITL, and it is already at the gate at the lowest concurrency measured.** From the
per-request records, not from the average: at N=40, **431 requests fail on ITL alone
against 7 on TTFT alone**, and 41.3% of all requests exceed the 20 ms gate. (The
average, 19.55 ms, is below the gate and says nothing on its own about what fraction
is above it — a mean cannot be read as a pass rate.) TTFT p90 is 1,366 ms, under half
its budget. It gets worse with concurrency: 70.4% of requests over the ITL gate at
N=56, 89.2% at N=72.

The cause is `--ep-size 4` on the decode workers. For the MoE layer, expert
parallelism replaces the TP-style reduction with dispatch/combine all-to-alls, and at
these batch sizes those measured dearer: 188 collectives per step at a ~25 µs floor
is 4.3–5.2 ms added to every decode step.

**This is the other face of our own largest single gain.** Submission 01 removes
expert parallelism and nothing else changed in that step; the result was 986 → 2,240
good tok/s. The reference deployment carries, by default, exactly the cost we
removed — so the gap between 982 and 2,706 is mostly one flag, measured twice from
opposite directions.

## What changed since the previous run

Nothing: this is a measurement of someone else's configuration, run once. Its value
is as the "before" of submission 01, and as the answer to a question we could not
otherwise settle — whether our 2,706 sits above the floor or on it. It is **2.76×**
the floor we can measure, against 2.2× for the H200 track.

The one caveat to carry into the presentation: 982 is **our port** of the reference,
not the organisers' own number, and deviation 1 above is large and in the
reference's favour to remove.
