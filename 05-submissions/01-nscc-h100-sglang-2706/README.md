`selftest.sh` passes, 30 checks. **Verified end to end: job 215703, exit 0.**

It reproduces the measured run to within 0.2%:

| | job 215675 (the harness) | job 215703 (this script) |
|---|---|---|
| `goodput` | 13.53 req/s → 2,706 tok/s | **13.556 req/s → 2,711 tok/s** |
| `good_request_fraction` | 0.976 | **0.9756** |
| requests / failures | 1,721 / 42 | **1,724 / 42** |
| TTFT p90 | 1,371.4 ms | **1,372.8 ms** |
| ITL p90 | 19.23 ms | **19.224 ms** |
| walltime | — | 19 min 58 s |

The failure count is identically 42, which is also the third independent
confirmation of the `N − 30` burst model. `verify-basic.sh` on its logs: 1,235
prefill batches on the prefill workers, 448 decode batches on the decode workers,
zero of either on the wrong role.

**The first attempt failed, and it is worth recording why.** Job 215695 brought the
whole deployment up — both prefill workers healthy, router up, model id resolved —
and then exited 127, because `AIPERF` defaulted to `$VENV/bin/aiperf` while the
client has its own venv at `envs/client/bin/aiperf`. Fifteen minutes and ~540 SU for
a wrong path, found only after a 221 GB load across sixteen GPUs. `selftest.sh` could
not have caught it: it checks flags and syntax, not paths. The script now verifies
the model's `config.json`, imports `sglang` through the engine venv and runs the
client's `--version` **before** it requests a GPU, and `selftest.sh` asserts that
guard exists and precedes the launch.

# 2,706 good output tokens/s — 16× H100, PD disaggregated

The run this folder reproduces, and nothing else. One PBS script, one concurrency,
the official AIPerf command. Our exploratory harness lives in `../run/`; this is the
deliverable.

| | |
|---|---|
| **good output tokens/s** | **2,706** |
| good_request_fraction | 0.976 (gate ≥ 0.90) |
| TTFT p90 | 1,371 ms (gate 3,000) |
| ITL p90 | 19.23 ms (gate 20) |
| concurrency | 72 |
| window | 60 s warm-up / 120 s benchmark / 30 s grace — official |
| platform | NSCC ASPIRE 2A+, 2× DGX H100 (16 GPUs), PBS Pro |
| engine | SGLang 0.5.20, bare-metal venv |
| measured | job 215675, 2026-10-02 |

`a2ap-dgx029` served prefill, `a2ap-dgx033` decode. Walltime 28 min for four
concurrency points; this script runs one, so budget ~20 min.

```bash
qsub qwen3-pd-h100-16gpu.pbs              # N=72 by default
qsub -v N=64 qwen3-pd-h100-16gpu.pbs      # the safer declared point, 2,562
./verify-basic.sh runs/<jobid>            # checks Basic from the engines' own logs
```

## The shape

```
node 0  a2ap-dgx029                      node 1  a2ap-dgx033
  prefill  TP4  GPU 0-3  :30100            decode  TP4  GPU 0-3  :30200
  prefill  TP4  GPU 4-7  :30101            decode  TP4  GPU 4-7  :30201
  sglang_router :8000  ──► prefill ──NIXL/UCX over 400 Gb/s NDR──► decode
```

TP4 is forced, not chosen. The FP8 checkpoint is 221 GB, so TP2 would need 110 GB
per GPU against the 80 available — which is also why the organisers'
`reference-scripts/07`, whose prefill is 4× TP2, cannot load on this platform at
all.

## Why it is 2.74× where we started

Four changes, measured one at a time, each on the rules' allowed list.

| | change | good tok/s | note |
|---|---|---|---|
| 0 | first 16-GPU PD run | 986 | 20 s window, `4plus12`, EP on |
| 1 | **no expert parallelism** | 2,240 | **2.27×**, and the single largest step |
| 2 | **integer `CUDA_VISIBLE_DEVICES`** | 2,380 | +6.3% at N=56; unlocked N=64 and 72 |
| 3 | **DeepGEMM on prefill only** | 2,706 | +10.7%; the knee moved from N=64 to 72 |

**1 — expert parallelism was on by default and it cost 2.27×.** SGLang defaults
`--ep-size` to the TP degree for a PD decode worker. EP trades an all-reduce for an
all-to-all, and at these batch sizes the all-to-all is dearer: 188 dispatch/combine
collectives per step at a ~25 µs floor is 4.3–5.2 ms of every decode step. Two
independent TP4 groups pay none of it.

**2 — custom all-reduce had never once been enabled on this cluster.** PBS hands out
GPU UUIDs in `CUDA_VISIBLE_DEVICES`; SGLang calls `int()` on them, warns once per
rank, and falls back to NCCL for the life of the job:

```
Setup Custom allreduce failed with invalid literal for int() with base 10:
GPU-3f70e3fb-6c96-fa36-b623-d77b30e3b686
```

Every rank of every worker of every earlier run printed that. Translating the UUIDs
to the GPU's **position** in the node's enumeration is worth 6.3%, all of it on
prefill (11.20 → 11.90 req/s). Position and not the index `nvidia-smi` reports: a
cgroup exposing a subset makes the driver renumber from zero, and then the reported
index is wrong while the position is right either way.

**3 — the MoE kernel wants to be different on each half, and in PD it can be.**
DeepGEMM's FP8 grouped GEMM is 1.23× Triton on 4,008-token prefill batches and
1.3–1.5× *slower* on decode batches of tens. Because the two roles are separate
processes, prefill takes the speed-up and decode never pays the cost. TTFT p90 at
N=72 fell 49%, from 2,669 ms to 1,371.

It did not work at first, and the failure is worth knowing: first **use** of a
DeepGEMM cubin does a module load, a module load inside a CUDA-graph capture is
illegal, and the engine dies with `markCaptureEnd called with no captures in
progress`. Pre-compiling the kernels does **not** fix it — a re-run with a fully
warm cache failed identically — because the trigger is the load, not the build.
`--disable-prefill-cuda-graph` on the prefill role is the fix, and it is how
SGLang's own `compile_deep_gemm` handles the same conflict.

## Techniques used

The settings that were **chosen** -- the ones a reviewer would ask "why that value"
about -- with the tier each belongs to and what it measured. Each row cites the job
pair that isolated it.

Not listed, because they are structural rather than tuned: `--context-length 8192`,
`--disaggregation-mode`, `--disaggregation-transfer-backend nixl`,
`--disaggregation-bootstrap-port`, `--max-running-requests 256`, `--enable-metrics`.
All of them are in the script; the first four are what makes it a PD deployment at
all.

| technique | tier | what it does | measured |
|---|---|---|---|
| **PD disaggregation**, 2× TP4 prefill + 2× TP4 decode | Basic (required) | prefill and decode on separate workers, KV moved by NIXL over 400 Gb/s NDR | TP4 is forced: the FP8 checkpoint is 221 GB, so TP2 would need 110 GB of an 80 GB card |
| **No expert parallelism** | Allowed (parallelism degree) | SGLang defaults `--ep-size` to the TP degree for a PD decode worker; two independent TP4 groups instead | **2.27×**, 986 → 2,240 (214750/214760 → 214830). For the MoE layer EP replaces the TP-style reduction with dispatch/combine all-to-alls, and here those measured dearer: 188 collectives per step at a ~25 µs floor is 4.3–5.2 ms added to every decode step |
| **Integer `CUDA_VISIBLE_DEVICES`** | not a flag — a defect we removed | PBS hands out GPU UUIDs; SGLang calls `int()` on them and drops custom all-reduce to NCCL for the life of the job | **+6.3%** at N=56 (214830 → 215267), all of it prefill: 11.20 → 11.90 req/s. It also unlocked N=64 and N=72 |
| **DeepGEMM MoE on the prefill role only** | Allowed (kernel backend) | FP8 grouped GEMM, 1.23× Triton on 4,008-token batches and 1.3–1.5× slower on decode batches of tens; in PD the roles are separate processes | **+10.7% at matched N=72**: 2,444 → 2,706 (215267 → 215675). Not against the 2,380 row above, which is N=56. TTFT p90 at N=72 fell 49%, 2,669 → 1,371 ms |
| `--disable-prefill-cuda-graph` on prefill | enabler for the row above | first *use* of a DeepGEMM cubin does a module load, which is illegal inside a CUDA-graph capture | without it the engine dies in capture; a pre-warmed cache does **not** help (215459) |
| `--mem-fraction-static 0.85` | enabler for the row above | DeepGEMM's masked-layout workspace wants ~2.75 GiB | 0.90 leaves 2.59 GiB and OOMs on the first batch |
| **`round_robin` on both router policies** | Allowed (routing) | `sglang_router` defaults to `cache_aware` | that default gave one prefill worker 126 requests and the other three one apiece, and stretched a 60 s warm-up to 503 s (job 17682). Every request here carries a different prefix, so cache affinity can buy nothing |
| `--disable-radix-cache` | substitute, declared below | prefix reuse off in the engine because the installed AIPerf has no `--cache-bust` | equivalent state, reached from the server side |
| **N = 72** | the declared point | where throughput peaks | 2,562 / **2,706** / 2,682 at N=64/72/80 — saturated |

Deliberately **not** used: speculative decoding, KV quantisation below bf16,
chunked prefill, torch.compile, any change to the checkpoint.

## The bottleneck this run hit

**Prefill capacity, at about 13.9 requests/s offered.** Not a latency gate.

The per-request records say so, and the percentiles do not — so read the records.
Each row below counts requests, not quantiles, against the gate as the rules define
it (TTFT ≤ 3,000 ms **and** mean ITL ≤ 20 ms, per request):

| N | requests | pass | **TTFT-only failures** | ITL-only | ITL p99 | when the failures happened |
|---|---|---|---|---|---|---|
| 56 | 1,512 | 0.983 | 26 | 0 | 19.09 | all in the first 10 s |
| 64 | 1,621 | 0.977 | 34 | 3 | 19.53 | 34 in the first 10 s, 3 later |
| 72 | 1,721 | 0.976 | **42** | 0 | 19.33 | **all in the first 10 s** |
| 80 | 1,722 | 0.970 | 52 | 0 | 19.41 | all in the first 10 s |

Three things follow.

**ITL is not binding, despite looking like it.** ITL p90 at N=72 is 19.23 ms, which
reads as 96% of the gate consumed. ITL p99 is 19.33, and across all four
concurrencies — 6,576 requests — exactly **three** missed on ITL. A percentile near
a threshold is not the same as requests crossing it.

**Every failure is the opening burst, and its size is predictable.** AIPerf drains
the warm-up and then issues all N requests within 0.1 s. They are served at ~13.9
req/s, so a request queued behind k others waits k/13.9 seconds on top of its own
~1.1 s of steady-state TTFT; that crosses 3,000 ms at k ≈ 26. The measured counts
are 26, 34, 42, 52 at N = 56, 64, 72, 80 — `N − 30`, to within two, exactly as a
fixed queue-depth cutoff predicts.
**Steady-state pass is 1.000 at every N measured.** At this rate the burst alone
would not reach the 0.90 gate until N ≈ 200, so pass rate is not the constraint.

**Throughput saturates because prefill does.** The offered rate goes 13.11 → 13.87
→ 13.82 req/s at N = 64 → 72 → 80 — it stops rising and then falls — while TTFT p90
climbs 1,110 → 1,371 → 2,021 ms over the same range. The queue is growing while the
service rate is flat, which is what a capacity ceiling looks like. (Good-request
rate, the one that becomes the score, goes 12.81 → 13.53 → 13.41.)

**What this rules out.** Anything that speeds up decode. In a closed loop a faster
decode returns requests to the generator sooner, which raises the arrival rate at a
prefill that is already saturated and moves the wait into its queue. That is why
speculative decoding is downloaded on this cluster and unused: it is Basic-legal
under the 2026-09-25 addendum (Q1) and an EAGLE-3 head trained against this exact
checkpoint exists, but on this configuration it would push on the wrong side.

**What it points at.** Make prefill faster, or make prefill do less. Of the levers
not yet spent: raising the prefill share of the 16 GPUs (3× TP4 prefill + 1× TP4
decode — but one decoder then holds ~72 requests, and that is the ITL risk this
configuration has so far avoided), chunked-prefill tuning, and attacking the opening
burst directly, which is worth 2.4% of pass rate at N=72.

## What changed since the previous run

Job 215267 scored 2,444 at N=72 and its bottleneck was the same one, one step lower:
**prefill capacity at about 12.6 req/s offered**, with TTFT p90 reaching 2,669 ms at
that concurrency — its own saturation signature, since the offered rate went 12.59 at
N=64 to 12.57 at N=72 while TTFT rose 1,577 → 2,669 ms. Two changes answered it.

**DeepGEMM on the prefill role.** Chosen because the bottleneck was prefill and
because the cost that had made DeepGEMM look unattractive — decode ITL 1.3–1.5×
worse, measured at 4 GPUs solo — cannot land on decode in a PD deployment, where the
two roles are separate processes. At matched N=72: offered rate 12.57 → 13.87 req/s,
good rate 12.22 → 13.53, TTFT p90 2,669 → 1,371 ms (−49%), and the throughput knee
moved from N=64 to N=72.

**`--disable-prefill-cuda-graph`, because the first attempt crashed.** Jobs 215267
and 215459 both died in `PrefillCudaGraphRunner` with `markCaptureEnd called with no
captures in progress`. The first diagnosis — "the kernels are being JIT-compiled
inside the capture, so pre-compile them" — was **refuted**: 215459 ran with 29
grouped kernels already committed to the cache, compiled nothing new, and failed
identically.

What that establishes is narrower than what it is tempting to say. JIT compilation is
ruled out as the sole trigger, and disabling prefill graph capture fixes it. The
remaining **suspected** trigger is the CUDA module load that first use of a cubin
performs, which is not legal inside a capture — but that is a hypothesis consistent
with the evidence, not something these runs measured, and `markCaptureEnd ...` can
also be a secondary error after an earlier failure in the same capture. Disabling
prefill graphs is how SGLang's own `compile_deep_gemm` resolves the same conflict,
and prefill graphs are worth little at 4,008 tokens per batch.

## Compliance

**Basic tier.** No speculative decoding (`speculative_algorithm: None` on both
roles), no expert parallelism (`ep_size: 1`), no re-quantisation
(`quantization: None`, so the checkpoint's own FP8 is used), KV cache at bf16 —
higher precision than the rules' FP8 floor. Prefill and decode are disaggregated and
strictly disjoint in the engines' own accounting: 5,410 prefill batches on the
prefill workers, 1,899 decode batches on the decode workers, and **zero** of either
on the wrong role. `./verify-basic.sh` re-checks all of that from the logs.

### Deviations to declare

The locked command in the rules is the one in submission 02's `submit.sbatch`. Ours
differs in six ways, and only the last two are choices.

**1. A different AIPerf CLI dialect.** The installed client removed the old flag
names, so the same quantities are expressed differently: `--isl` / `--isl-stddev` /
`--osl` / `--osl-stddev` for `--synthetic-input-tokens-mean` /
`--synthetic-input-tokens-stddev` / `--output-tokens-mean` /
`--output-tokens-stddev`, `--custom-endpoint` for `--endpoint`, `--model-names` for
`--model`, `--ui-type` for `--ui`, `--header` for `-H`. Same semantics; the old
spellings do not exist in this build.

**2. No `--extra-inputs max_tokens:200`.** We pass `--use-legacy-max-tokens`, which
is how this build sends `max_tokens` rather than `max_completion_tokens`. The
intended effect is the same field at the same value. **Not verified on the wire** --
with `min_tokens:200` and `ignore_eos:true` the output length would be 200 either
way, so the artefacts cannot distinguish them.

**3. No `--extra-inputs repetition_penalty:1.0`.** Omitted. 1.0 is the engine's
default and therefore a no-op, but it is an omission from a locked command and is
listed as one rather than argued away.

**4. No `--cache-bust first_turn_prefix`.** This build has no such flag, so prefix
reuse is disabled in the engine instead -- `--disable-radix-cache`. Both reach the
same state, and the engine's version is visible in the server's own log where the
client flag never was. The effect should be nil either way: 12,800 dataset entries
against ~1,700 requests per window makes a repeat unlikely, and the only shared
prefix is the 8-token chat template. **This is the one worth asking the organisers
about: how is the locked command to be satisfied if their AIPerf also lacks the
flag?**

**5. `--artifact-dir` added.** So the per-request records land somewhere collectable.
Every per-request claim in this README comes from that directory.

**6. `--enable-metrics` on the engines.** Not in the reference deployment. Kept
because without it the engines serve no `/metrics` and every judgement about queue
depth, KV usage and batch size would be guesswork. It exposes an endpoint; it does
not change serving.

## Files

| | |
|---|---|
| `qwen3-pd-h100-16gpu.pbs` | the whole run: nodes, GPU id translation, four workers, router, the official AIPerf command, teardown. No includes |
| `verify-basic.sh` | Basic compliance from the engines' logs, not from the flags |
| `selftest.sh` | everything checkable without an allocation: both scripts' syntax, the GPU-id translation against a stub driver, and that the AIPerf block still carries every flag the measured run issued |

## Status

`selftest.sh` passes, 30 checks.

**First end-to-end attempt: job 215695, exit 127.** The deployment worked — both
prefill workers reached healthy, the router came up, the model id resolved — and then
the benchmark step died because `AIPERF` defaulted to `$VENV/bin/aiperf` and the
client lives in its own venv, `envs/client/bin/aiperf`. Fifteen minutes and ~540 SU
to find a wrong path, after a 221 GB load across sixteen GPUs.

Fixed, and the lesson is built in rather than remembered: the script now checks the
model, the engine venv (by importing sglang) and the client binary (by running
`--version`) **before** it asks for a GPU, and `selftest.sh` asserts that guard
exists and precedes the launch. A wrong path now costs two seconds.

**Still not confirmed end to end.** `selftest.sh` checks syntax, flags and paths; it
cannot check whether the four workers come up inside `HEALTH_TIMEOUT` or whether the
number comes back 2,706. That needs one more `qsub`.
