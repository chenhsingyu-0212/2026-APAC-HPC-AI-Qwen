# AIPerf benchmark kit

> 中文版：[README.zh-TW.md](README.zh-TW.md)

Four scripts that produce **comparable** LLM serving numbers across machines, clusters,
and inference engines (vLLM / SGLang / Dynamo / TRT-LLM — anything with an
OpenAI-compatible `/v1/chat/completions`).

| Script | Purpose |
| --- | --- |
| `install.sh` | Create a venv, install `aiperf`, and verify which flags this version actually supports. |
| `run.sh` | Warm up, run the anchor point, sweep concurrency, optionally perturb ISL/OSL, and assemble `metrics.csv`. |
| `gpu_sample.sh` | Run on the **server** node during a sweep: GPU power and clocks, memory, PCIe/NVLink/InfiniBand, CPU, and who else is on the box. |
| `diagnose.sh` | Read a finished run and say **where the bottleneck is** — including a roofline that settles compute vs memory bandwidth. |
| `official.sh` | The 2026 APAC HPC-AI scored command, verbatim, with a `--verify` that diffs it against the organisers' own rules file. |
| `observe.sh` | Monitor a run whose command may not be touched — polls from the outside and produces what `diagnose.sh` reads. |

None of them depend on anything else in this repo — copy the files anywhere and they
still work.

---

## Quick start

```bash
chmod +x *.sh                   # first time only

./install.sh                    # creates ./aiperf-venv
./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct
```

Results land in `./aiperf-<timestamp>/`, with the comparison table at
`./aiperf-<timestamp>/metrics.csv`.

For the scored run of the 2026 APAC HPC-AI task, the command is fixed and monitoring has to
stay outside it:

```bash
./official.sh --verify                       # confirm we match the rules, before anything else
./gpu_sample.sh --out ./tel                  # on each server node
./observe.sh --out ./scored --gateway gw01:8000 -- \
    ./official.sh --host gw01:8000 --model qwen3-pd --concurrency 64
./diagnose.sh ./scored --telemetry ./tel --weights 22 --tp 8
```

---

## 1. `install.sh`

Finds a usable Python → creates the venv → installs `aiperf` → **verifies and prints the
flags that are actually available**.

That last step matters: AIPerf's sequence-length flags come in two spellings depending on
the release (`--synthetic-input-tokens-mean/--output-tokens-mean` vs `--isl/--osl`), and
`run.sh` has to pick the right one at runtime.

```bash
./install.sh                            # install into ./aiperf-venv
./install.sh --venv /work/$USER/venv    # install into a specific location
./install.sh --pin 0.12.0               # pin the version
./install.sh --check-only --venv PATH   # verify an existing install, install nothing
./install.sh -h                         # help
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--venv PATH` | `$AIPERF_VENV`, else `./aiperf-venv` | Relative paths resolve against the **current** directory, not the script's. |
| `--pin VERSION` | unpinned (latest) | Becomes `pip install aiperf==VERSION`. |
| `--check-only` | off | Skips Python discovery and install; only runs the verification step. |

**Requirements:** Python ≥ 3.11 and < 3.14 (what AIPerf declares) and outbound network access for pip. The script probes
`python3.13 … python3.10` before falling back to `python3`; on HPC sites you usually need
`module load python` (or a conda env) first, since `/usr/bin/python3` is often 3.6.

**Pin the version when comparing across environments.** The final banner prints the exact
`--pin` argument to hand your teammates.

**Watch for the PATH warning.** If another `aiperf` already exists on `PATH`, the script
says so. Always invoke the absolute path (`$VENV/bin/aiperf`), never a bare `aiperf` —
`run.sh` already does this for you.

---

## 2. `run.sh`

```bash
./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct

./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct \
         --sweep "8 16 32" --isl 4000 --osl 200 --out ./myrun --label vllm-tp2
```

### Required

| Flag | Notes |
| --- | --- |
| `--url` | Base URL of the service, e.g. `http://node01:8000`. No trailing slash, no `/v1`. |
| `--tokenizer` | **Local** model directory (or file). Keeps the run off the network and makes token counts match the server's. |

### Optional

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | asked from `GET /v1/models` | Only needed when the endpoint has no model list (e.g. the TRT-LLM disagg proxy — ask one of its workers instead). |
| `--venv` | `$AIPERF_VENV`, else `./aiperf-venv` | Must match what you gave `install.sh`. |
| `--isl` / `--osl` | `4000` / `200` | Synthetic input / output sequence length in tokens. |
| `--sweep` | `"8 16 32 64 128 256"` | Space-separated concurrency levels, quoted as one argument. |
| `--isl-sweep` | off | Perturbation dimension: input lengths to try at a fixed concurrency. Needs ≥ 3 values to be a slope. |
| `--osl-sweep` | off | Perturbation dimension: output lengths to try at a fixed concurrency. Needs ≥ 3 values. |
| `--shape-c` | `--anchor` | Concurrency the two shape sweeps run at. |
| `--anchor` | `32` | The team-wide alignment point, measured a second time on its own. |
| `--sla` | `"time_to_first_token:1000 request_latency:5000"` | `metric:threshold` pairs, thresholds in **milliseconds**. |
| `--req-per-c` | `10` | Requests per point = `concurrency × this`. Raise it for tighter percentiles. |
| `--out` | `./aiperf-<timestamp>[-label]` | Output directory. |
| `--label` | none | Appended to the default output directory name and recorded in `env.txt`. |
| `--no-warmup` | warmup on | Skips the warmup round. |
| `--metrics-url` | `<url>/metrics` | Prometheus endpoint to poll. Auto-disabled with a warning if the probe fails. |
| `--metrics-interval` | `2` | Poll period in seconds; must be ≥ 0.1 so polling doesn't perturb the run. |
| `--no-metrics` | metrics on | Don't poll the server at all. |
| `-h`, `--help` | — | Prints the header block. |

### What a run does, in order

0. **Validation** — every flag is checked before anything runs: the URL scheme, that the
   tokenizer path exists, that `--isl` / `--osl` / `--anchor` / `--req-per-c` and each sweep
   entry are positive integers, and that the venv holds an `aiperf`. Bad input fails in the
   first second, not twenty minutes in.
1. **Health check** — `GET /health`. A non-200 is a warning, not a failure; some servers
   don't expose it.
2. **Model id** — read from `GET /v1/models` unless `--model` was given. Never guessed; a
   wrong model name silently wastes the entire run.
3. **Clock skew** — the server's own HTTP `Date` header is compared against this node's
   clock and the offset is recorded as `server_clock_skew_s` in `env.txt`. `points.csv` is
   stamped by the client while `gpu_sample.sh` stamps its telemetry by the server, so
   unsynchronized clocks would silently slice the wrong window for every point.
   `diagnose.sh` corrects for the recorded offset.
4. **Flag detection** — picks the sequence-length spelling this AIPerf build supports, and
   aborts if `--streaming` is missing (without it there is no TTFT or ITL).
5. **Fingerprint** — writes `env.txt` (host, URL, model, tokenizer, versions, every
   parameter, total request count). This is what makes two runs comparable after the fact;
   keep it.
6. **Warmup** — `c=8`, 40 requests, logged to `warmup.log` and **excluded from the
   results** — it gets no row in `metrics.csv`. Cold-start latency runs 30–40 % above
   steady state.
7. **Anchor** — the `--anchor` concurrency, run on its own. Comparing it against the same
   concurrency in the sweep tells you how stable the measurement itself is.
8. **Sweep** — each concurrency in `--sweep`.
9. **Shape sweeps** — `--isl-sweep` and `--osl-sweep`, if given, at `--shape-c`. Last on
   purpose: they are extra evidence, and a failure there must not cost the concurrency
   curve everything else is built on.
10. **Table** — parses every `profile_export_aiperf.csv` into one `metrics.csv`, then prints
   the best goodput point.

Every measured round forces `--streaming` and `--endpoint-type chat`.

### Perturbation sweeps (`--isl-sweep`, `--osl-sweep`)

The concurrency sweep can only ever *correlate* a bottleneck with load. To decide whether a
verdict is real, you have to change the input it depends on and check that the response
matches the mechanism the verdict claims. That is what these two do — one shape dimension
at a time, at a fixed concurrency, so nothing else moves:

```bash
./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct \
         --isl-sweep "1000 2000 4000 8000" --osl-sweep "50 200 800"
```

| Sweep | What it grows | The response that matters |
| --- | --- | --- |
| `--isl-sweep` | prefill work **and** KV context per step | TTFT should grow in proportion to ISL; the **ITL slope** is exactly the KV term of the roofline, so it confirms or falsifies the `MEMORY BW` verdict. |
| `--osl-sweep` | decode steps, and nothing else | ITL should barely move. If it climbs, the per-step cost isn't constant — the growing KV cache is, which is capacity pressure rather than steady state. |

Both are off by default; each value costs `--shape-c × --req-per-c` requests. They write
`isl<N>/` and `osl<N>/` point directories, and `diagnose.sh` reads them back automatically.
Each point's requested ISL/OSL is recorded in `points.csv`, so the shape points are checked
against their own target rather than the run-wide one.

### Engine metrics (`engine_metrics.csv`)

If the server exposes a Prometheus endpoint, `run.sh` polls it **throughout each work
point** and rolls the samples up into `engine_metrics.csv`, plus a headline table:

```
  point           kv/token usage  queue peak  preemptions
  c8                        0.23           0            0
  c16                       0.93          90           88
```

| Signal | Conclusion |
| --- | --- |
| KV/token usage flat and low | Not memory bound — raise `gpu_memory_utilization` or concurrency. |
| Queue depth growing | Requests are waiting, not executing: a scheduler cap (`max_num_seqs`) is the limit. |
| Preemptions climbing | KV cache oversubscribed; the resulting latency spikes are what kill SLA pass rate. |

**Poll during the point, not around it.** Gauges like queue depth and cache usage read
zero once load stops, so before/after scraping tells you nothing — which is why this polls
continuously and reports peaks.

Which endpoint each engine uses:

| Engine | Endpoint | Notes |
| --- | --- | --- |
| vLLM | `/metrics` on the API port | On by default. Metric names differ between the V0 and V1 engines. |
| SGLang | `/metrics` | **Needs `--enable-metrics` at launch**, otherwise there is nothing to scrape. |
| TRT-LLM | Triton: port `8002`; `trtllm-serve`: `/metrics` | Pass `--metrics-url` explicitly for the Triton case. |
| Dynamo | frontend `/metrics` | `dynamo_*` names. |

The rollup is **name-agnostic on purpose** — series are selected by keyword against the
full `name{labels}` identity, because Triton encodes the state in a label
(`nv_trt_llm_request_metrics{request_type="waiting"}`) where vLLM puts it in the name. Raw
scrapes are kept per point in `metrics_samples.txt`, so nothing is lost if the filter
misses a metric you care about.

### Output layout

```
aiperf-20260817-143022-vllm-tp2/
├── env.txt          # environment fingerprint — keep this with the results
├── warmup.log       # warmup output (excluded from metrics.csv)
├── warmup/          # warmup artifacts (skipped by the table builder)
├── anchor-c32/      # anchor point
│   └── metrics_samples.txt   # raw Prometheus scrapes taken during this point
├── c8/ c16/ c32/ …  # one directory per sweep point
├── isl1000/ osl50/ …         # shape sweep points, if requested
├── points.csv       # per-point concurrency, requested ISL/OSL, start/end epoch
├── metrics.csv      # the comparison table
└── engine_metrics.csv        # server-side metrics per point (if available)
```

And on the server node, from `gpu_sample.sh`:

```
telemetry/
├── host.txt         # static config, GPU names, PCIe link, and who else was on the GPUs
├── gpu.csv          # util, memory, power, SM clock vs maximum, temperature, throttle flags
├── cpu.csv          # all-core and busiest-core busy, iowait
├── pcie.csv         # rx/tx MB/s per GPU (where nvidia-smi dmon works)
├── nvlink.csv       # cumulative NVLink counters (NVLink boards only)
├── ib.csv           # per-port IB bytes and xmit_wait (IB/RoCE hosts only)
└── dcgm.csv         # DCGM profiling counters (where an exporter is running)
```

`metrics.csv` columns, one row per work point:

`point`, `pass_rate`, `ttft_avg`, `ttft_p90`, `ttft_p99`, `itl_avg`, `e2e_avg`, `e2e_p90`,
`e2e_p99`, `out_tok_per_s`, `req_per_s`, `goodput`, `req_count`, `good_count`, `isl`, `osl`

- **goodput** — requests/sec that met *every* SLA threshold. This is the number to optimize;
  raw throughput can look great while most requests blow the latency budget.
- **pass_rate** — `good_count / req_count`. Goodput without a pass rate is misleading.
- Latencies are milliseconds; `isl` / `osl` are the *measured* sequence lengths, worth
  checking against what you asked for.

---

## 3. `gpu_sample.sh`

`metrics.csv` tells you *what* the goodput was; this tells you *why* it stopped going up.
Run it on the node hosting the server, while `run.sh` drives load from wherever:

```bash
# on the server node
./gpu_sample.sh --out ./telemetry --duration 900

# later, or on the collected directory
./gpu_sample.sh --summary ./telemetry
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--out DIR` | required | Writes `host.txt`, `gpu.csv`, `cpu.csv`, and `pcie.csv` / `nvlink.csv` where the hardware supports them. |
| `--summary DIR` | — | Re-print the summary for an already-sampled directory; samples nothing. |
| `--interval SEC` | `1` | Sampling period; accepts fractions. |
| `--duration SEC` | until Ctrl-C | Stop after this many seconds, then summarize. |
| `--label NAME` | none | Recorded in `host.txt`. |
| `--ib-root DIR` | `/sys/class/infiniband` | Where to read the IB port counters. Mostly for testing against a fake tree. |
| `--dcgm-url URL` | `http://localhost:9400/metrics` | A DCGM exporter, if the site runs one. Probed once; absent is fine. |
| `--no-dcgm` | probe on | Skip it. |

Needs only `nvidia-smi`, `/proc/stat` and sysfs — no DCGM, no root, and no rdma tooling. It
records per-GPU utilization, memory footprint, **power draw against the board limit**, **SM
clock against the card's maximum**, temperature, throttle reasons split by severity, **PCIe
rx/tx throughput**, NVLink counters where the boards have them, **InfiniBand port
throughput and congestion**, all-core and busiest-single-core CPU busy, and **every compute
process resident on the GPUs** at the start and the end of sampling.

Reading the summary:

| Signal | Conclusion |
| --- | --- |
| Power near the limit, clocks at maximum | GPU genuinely saturated — further gains must come from the model/engine, not more load. |
| Power well under the limit | GPU is **starved**: batch too small, client-bound, or requests are queueing. |
| SM clock below 85 % of maximum | Really throttled — the numbers are not reproducible. Fix cooling or power before tuning anything. |
| Throttle flags lit *but* clocks at maximum | Ignore them. See below. |
| Memory peak far below total | KV cache under-provisioned; raise `gpu_memory_utilization`. |
| PCIe peak near link capacity | Interconnect bound — matters for TP across PCIe and for PD-disaggregated KV transfer. |
| IB near link capacity | The cross-node path is the ceiling. For PD disaggregation this constrains the P:D split and worker placement. |
| IB far from capacity but `xmit_wait` climbing | Congestion or credit starvation, not bandwidth. Look at routing and the traffic pattern, not the link speed. |
| One core pegged, mean low | A serial stage (HTTP frontend, detokenization) is the ceiling, not the GPU. |

**Throttle flags are not evidence.** `SwPowerCap` and `SwThermalSlowdown` sit permanently
asserted on many cards while the SMs still run at full speed — on the box this was
developed against, the flag is lit in ~85 % of samples with clocks steady at 95 % of
maximum. So the sampler records `clocks.max.sm` alongside `clocks.sm`, splits the flags
into hardware (real lost clocks) and software (advisory), and both the summary and
`diagnose.sh` judge on the **clock ratio**. A rule built on the flags alone would report
`THROTTLING` on every point forever and hide the actual bottleneck.

**Where a DCGM exporter exists, take it.** It is the one source here that measures what
everything else infers:

| Field | What it settles |
| --- | --- |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Whether the tensor cores are *actually* busy. High means compute, and it needs no argument from power draw. |
| `DCGM_FI_PROF_DRAM_ACTIVE` | The memory interface, measured rather than inferred — and an independent check on the roofline. |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `SM_ACTIVE` | Real occupancy, which GPU-Util% is not. |

The exporter is a Prometheus endpoint, so this scrapes it like any other and writes
`dcgm.csv`. Nothing requires it: without it the `dram` column falls back to
`utilization.memory` and says so in the legend.

For a PD deployment the tensor figure also checks the split itself — a prefill node that
is not tensor-busy, or a decode node that is, is not doing what you think it is.

**`nvidia-smi`'s GPU-Util% is not a saturation metric** — it only means at least one kernel
was resident during the sample window, so LLM decode reads 95–100 % while the SMs sit
mostly idle. Power draw against the board limit is the better proxy; for real occupancy you
need DCGM (`DCGM_FI_PROF_SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`), which this script
deliberately does not require — the roofline in `diagnose.sh` answers the same question
from the model shape instead.

**Every number here is machine-wide.** On a shared node, someone else's job is in your
power draw, your clocks and your CPU busy. `host.txt` therefore records every compute
process (uuid, pid, user, memory) at the start *and* the end of sampling — a job that
starts halfway through contaminates only the tail, and you can see that it did.
`diagnose.sh` turns a second user into an explicit warning that the `GPU COMPUTE` and
`SERIAL CPU` verdicts are unproven.

**PCIe throughput comes from `nvidia-smi dmon`**, which runs its own sampling loop; each
line is stamped with the epoch as it arrives so `pcie.csv` stays on the same clock as
`gpu.csv`. NVLink counters are cumulative and converted to a rate at summary time.

**InfiniBand comes straight from sysfs** — `/sys/class/infiniband/<dev>/ports/<port>/` —
so nothing from the rdma-core toolchain has to be installed. Three details matter:

- `port_rcv_data` and `port_xmit_data` are reported **in units of four octets**, so the
  sampler multiplies by four and `ib.csv` is in bytes. Getting this wrong understates the
  fabric by 4×.
- Link capacity comes from the port's own `rate` file (`400 Gb/sec (4X NDR)` → 50 GB/s), so
  the percentages are against what the port actually negotiated, not what it should be.
  Ports that are not `ACTIVE` are excluded, so a down link cannot dilute the capacity.
- `port_xmit_wait` counts ticks where a port had data to send and no credit. It separates
  *congestion* from *bandwidth*: a fabric at 40 % of line rate with `xmit_wait` climbing is
  not short of bandwidth, and buying faster links will not help it.

Error counters (`link_downed`, `symbol_error`, `port_rcv_errors`, …) are recorded at the
start and the end rather than sampled — a link that retrained mid-run invalidates whatever
was measured around it, and the summary says so.

---

## 4. `diagnose.sh`

The other scripts produce evidence; this one reaches a verdict.

```bash
./diagnose.sh ./aiperf-20260817-143022
./diagnose.sh ./aiperf-20260817-143022 --telemetry ./telemetry   # much stronger
./diagnose.sh ./run --telemetry ./tel --tp 2 --mem-bw 3350       # override roofline inputs
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--telemetry DIR` | none | The `gpu_sample.sh` output directory. Most verdicts depend on it. |
| `--weights GIB` | summed from the model directory | Size of the whole model's weights. |
| `--tp N` | the GPU count on the server node | Tensor-parallel degree; weights and KV are sharded by it. |
| `--mem-bw GBPS` | looked up from the GPU name | Peak HBM bandwidth **per GPU**. |
| `--model-dir DIR` | the `tokenizer` path from `env.txt` | Where to find `config.json` and the weight files. |

```
 point      goodput   pass     kv  queue preempt  power   clk  membw    mbu   vram   pcie  verdict
 c8            3.90   1.00    12%      0       0    40%    ok    70%    59%    67%     3%  HEADROOM
 c32          12.08   1.00    47%      0       0    60%    ok    70%    77%    67%     3%  MEMORY BW
 c64          17.40   0.97    83%      2       0    86%    ok    70%    84%    67%     3%  MEMORY BW
 c128         12.81   0.61    99%     41     292    94%    ok    70%    70%    67%     3%  KV CACHE

 peak goodput  17.40 req/s at c64 (pass rate 0.972)
 bottleneck    MEMORY BW -- MBU 84% -- the memory system is moving the weights + KV per
               decode step at 84% of peak bandwidth; more SMs cannot help
 memory        KV cache hit 99% while only 67% of VRAM was in use
               -> raise gpu_memory_utilization; you are leaving 28% of the card unused
 SLA missed    at c128: TTFT p90 2400ms > 1000ms; e2e p90 7600ms > 5000ms
 SLA cliff     pass rate first drops below 0.99 at c64 (0.972)
 repeatability anchor vs c32: 1.5% apart (stable)
```

**Why a tool and not eyeballing:** each source on its own is ambiguous. A low GPU power
draw is equally consistent with a starved client, a scheduler cap, and a serial CPU stage
— only the *combination* separates them. The verdicts, most conclusive first:

| Verdict | Evidence | Fix |
| --- | --- | --- |
| `THROTTLING` | SM clocks below 85 % of maximum in > 20 % of samples **while drawing ≥ 30 % of board power**, or hardware slowdown in > 5 % | Power/thermal wall — the numbers aren't reproducible; fix before tuning anything. (An *idle* GPU also parks its clocks; that is not throttling, so the power floor is required.) |
| `KV CACHE` | Preemptions > 0, cache near full | Oversubscribed. Lower concurrency, or raise cache capacity. |
| `MEMORY BW` | **MBU ≥ 60 %** *and* KV is ≥ 50 % of the traffic | Genuinely bandwidth bound. More batching only adds more KV traffic; fewer bytes per token is the lever (quantization, speculative decoding, shorter context). |
| `WEIGHT BOUND` | **MBU ≥ 60 %** but KV is a small share — the bandwidth is going into re-reading the weights for a small batch | Not a bandwidth wall. Weight reads are a fixed cost per decode step; a bigger batch spreads them. Push concurrency up before touching the model. |
| `INTERCONNECT` | PCIe **or InfiniBand** peak ≥ 70 % of link capacity while the GPU has power headroom | TP or PD-disagg traffic is the ceiling; the verdict names which fabric. Check `nvidia-smi topo -m`, the P:D split, and worker placement. |
| `GPU COMPUTE` | **DCGM tensor pipe active ≥ 50 % of cycles**, or power ≥ 85 % of board limit with MBU below 60 % | Genuinely compute saturated — gains must come from the model/engine, not more load. The tensor counter is direct evidence; power is only a proxy for it, and a prefill node can be compute-bound at 59 % of board power. |
| `MEMORY BW?` | Memory controller busy ≥ 70 % while power is low, **and no roofline inputs** | A proxy, not a verdict. Pass `--weights` / `--mem-bw` to settle it. |
| `SCHEDULER CAP` | Queue deep, KV low, GPU with headroom — or a high MBU that is mostly weight reads while requests wait | Raise `max_num_seqs` / `max_num_batched_tokens`. |
| `SERIAL CPU` | One core pegged, GPU idle | HTTP frontend or detokenization is the ceiling. |
| `CLIENT BOUND` | Server ran far fewer than requested, queue empty | The load generator isn't keeping up — move it off the server node. |
| `HEADROOM` | SLA met, nothing queueing | Push concurrency higher. |

### The roofline: compute or bandwidth, measured

"Memory bandwidth bound" is the most common claim in LLM serving and the hardest to check
without a profiler. Rather than guess from power draw, `diagnose.sh` computes it:

```
bytes per decode step (per GPU) = (weights + batch × KV_per_sequence) / tp
MBU = bytes per step × steps per second / peak HBM bandwidth
```

`steps per second` is `1000 / itl_avg`, straight out of `metrics.csv`. The rest comes from
the model and the card, all auto-detected and all overridable:

| Input | Auto-detected from |
| --- | --- |
| weights | sum of the `*.safetensors` in the model directory (`tokenizer` in `env.txt`) |
| KV bytes per token of context | `config.json`: `num_hidden_layers`, `num_key_value_heads`, `head_dim`, `torch_dtype` |
| peak HBM bandwidth | the GPU name recorded in the telemetry's `host.txt` |
| tensor-parallel degree | the GPU count on the server node |

**MBU ≥ 60 %** means the memory system is the ceiling: more SMs cannot help. **MBU well
below 60 % with high power** means the opposite — it really is compute.

But a saturated memory system means two different things depending on *what* is moving
through it, so the report splits on the KV share of the per-step bytes:

| Traffic | Verdict | Why it matters |
| --- | --- | --- |
| KV ≥ 50 % of the bytes | `MEMORY BW` | Batching adds KV traffic proportionally, so it cannot help. Reduce bytes per token. |
| KV a small share — mostly weight reads | `WEIGHT BOUND`, or `SCHEDULER CAP` if requests are queued | Weight reads are a *fixed* cost per decode step. A bigger batch amortizes them, so the fix is more concurrency (or a higher `max_num_seqs`), not a smaller model. |

Reading MBU alone would call both "bandwidth bound" and send you to quantize a model whose
real problem is a batch of eight.

This distinction is invisible from outside, and it is exactly where the previous version
was wrong. In the run above, `c64` draws 86 % of the board power limit; the power rule
alone calls that `GPU COMPUTE`, "genuinely saturated". The roofline shows MBU at 84 %: that
power is going into moving weights and KV, and a card with more FLOPs would change nothing.

### Checking the roofline against the hardware

Where a DCGM exporter is running, MBU stops being the only answer. `DRAM_ACTIVE` measures
the memory interface directly, so the report puts the two side by side:

```
 cross-check   roofline MBU 12% vs measured DRAM_ACTIVE 77% of cycles
               -> the hardware is moving far more than the model shape accounts
                  for. On an MoE this is the signature of a batch touching many
                  more experts than the active-parameter count assumes -- re-run
                  with --weights set to the whole checkpoint for the upper bound.
```

They are not the same quantity — `DRAM_ACTIVE` counts cycles the interface was busy, MBU
counts bytes against peak, so `DRAM_ACTIVE` normally sits at or above MBU. It is the size
of the disagreement that carries information, and it resolves in both directions:

| Gap | Reading |
| --- | --- |
| MBU more than 25 points above `DRAM_ACTIVE` | The roofline is claiming more of the memory system than the hardware saw. Check `--weights`, `--tp`, `--mem-bw`. |
| `DRAM_ACTIVE` more than 35 points above MBU | The hardware is moving more than the model shape accounts for — on an MoE, a batch touching more experts than the active-parameter count assumes. |
| Anything between | Consistent. The bandwidth reading has two independent sources agreeing. |

That middle row is the one that matters for this task. Without it, the MoE question only
has the two-bound answer — run with active weights, run with the whole checkpoint, the
truth is between. `DRAM_ACTIVE` says which end you are actually at.

The header also prints an **ITL floor** — the fastest per-token latency the weights alone
allow at this bandwidth. If your measured ITL is near it, no engine tuning will help. An
MBU above 100 % is impossible and means an input is wrong; the report says so and names the
likely culprit (usually `--tp`).

Without `--telemetry` the roofline still works if you pass `--weights` and `--mem-bw`
yourself — it needs the model and the card, not the sampler.

**Memory is two separate questions**, and the report keeps them apart:

| Column | Means | Says |
| --- | --- | --- |
| `mbu` | achieved bytes/s over peak HBM bandwidth | **Bandwidth, measured.** The one that settles compute-vs-bandwidth. |
| `membw` | `utilization.memory` — % of time the memory controller was busy | A **proxy** for the same thing, used only when the roofline inputs are missing. |
| `vram` | `memory.used / memory.total` | **Capacity.** How much of the card is actually claimed. |
| `kv` | The engine's own KV block usage | How much of *what the engine was given* is in use. |

`kv` and `vram` are the pair that catches wasted hardware: KV cache at 96 % while VRAM sits
at 60 % means `gpu_memory_utilization` was set too low and a third of the card is idle —
usually the cheapest goodput win available. The report calls this out explicitly, and warns
the other way too when VRAM peaks above 97 % (OOM risk).

### Several workers, one number

A PD deployment exposes metrics per worker, and collapsing them needs a different rule per
metric. Summing a cache-usage *fraction* across six workers reads 600 %; taking the maximum
of a queue depth hides the fleet's real backlog. So `observe.sh` keeps the endpoints apart
and `diagnose.sh` combines them by kind:

| Column | Across workers | Why |
| --- | --- | --- |
| `kv` | max | A fraction. One worker at 99 % is the constraint, whatever the others do. |
| `queue` | sum | The backlog only means anything fleet-wide. |
| `preempt` | sum | A count of events. |
| roofline batch | **max**, not sum | Every worker re-reads its own weights, so the batch that sets MBU is the per-worker one. Summing would overstate it by the number of workers. |
| `CLIENT BOUND` check | sum | That one really is asking how many requests are in flight everywhere. |

### Perturbation: does the verdict survive changing the input?

If the run has `--isl-sweep` / `--osl-sweep` points, the report adds a section that tests
the verdict instead of restating it:

```
 perturbation -- one input varied at a time at c=32

   ISL sweep (OSL fixed): prefill work and KV context both grow
          isl   ttft_avg    itl_avg   prefill tok/s
         1000        115      11.88            8696
         2000        230      14.16            8696
         4000        460      18.73            8696
         8000        920      27.88            8696
   . prefill throughput held within 0% across a 8.0x ISL range -- TTFT is linear
   . ITL slope   measured 2.29 us per token of context
                   roofline predicts 2.06 us at c=32
     -> within 11%: KV traffic explains how ITL grows, so the bandwidth
        model is confirmed by perturbation, not just inferred from power

   OSL sweep (ISL fixed): decode steps grow, nothing else does
          osl   ttft_avg    itl_avg    e2e_avg
           50        460      18.56       1370
          200        460      18.73       4188
          800        460      19.42      15975
   . ITL moved +5% across a 16.0x OSL range -- each decode step costs the same,
     which is what a steady-state decode bottleneck looks like
```

The middle block is the point of the whole exercise. The roofline predicts how much ITL
*should* grow per extra token of context — `batch × KV_per_token / (tp × bandwidth)` — from
the model shape alone, with no reference to the measurement. The ISL sweep measures the
same slope empirically. The prediction is divided by the measured MBU first, because an
engine running at 90 % of peak has a slope 1/0.9 times the ideal one; taking that out
leaves a test of *context scaling* alone rather than of engine efficiency. When the two agree, the `MEMORY BW` verdict has been **tested**
rather than inferred, and the reading is causal instead of correlational.

When they disagree, the report says which way and what that implies:

| Measured vs predicted | Reading |
| --- | --- |
| within ~30 % | KV traffic explains ITL growth — the bandwidth model holds. |
| much steeper | Something else scales with context: attention kernel, scheduler, cache thrash. Not bandwidth. |
| much shallower | The engine moves less KV than assumed — prefix caching, a quantized KV cache, or `--tp` is wrong. |
| ITL flat in ISL | The KV term doesn't apply at all; re-check `--tp` and the KV dtype. |

The other two checks are independent of the roofline:

- **prefill cost across the ISL sweep** — TTFT is fitted as `fixed overhead + marginal cost
  × ISL`, and linearity is judged from how the marginal cost changes along the sweep. (The
  obvious statistic, `ISL / TTFT`, is misleading: with a large fixed overhead it *rises*
  with ISL and says nothing about linearity.) When the fixed part dominates at the short
  end, the report says so — short prompts there are measuring the harness, not prefill.
- **ITL across the OSL sweep** — should barely move, because only the number of decode
  steps changed. A large drift means the per-step cost isn't constant, and a growing KV
  cache is the usual reason: capacity pressure, not steady state.

### What it refuses to let you believe

It also flags things that quietly invalidate a run:

- measured ISL/OSL drifting > 5 % from what was requested,
- goodput still rising at the largest concurrency tested (your sweep never reached the peak),
- anchor-vs-sweep drift above 5 % (the measurement itself is too noisy to trust small differences),
- **the two nodes' clocks disagreeing** — if the telemetry doesn't overlap the run at all it
  prints the exact offset and how to apply it, rather than silently reporting empty columns,
- **a second user's job on the server GPUs**, which makes the machine-wide GPU and CPU
  verdicts unproven,
- **GPUs that never woke up** — a peak below 15 % of board power across every point means
  the sampler was on the wrong node, or the work isn't happening on those GPUs; every
  GPU-side column is then meaningless and the report says so instead of ranking them,
- **work points shorter than five poll periods**, where the queue, KV and running figures
  are undersampled and the peaks understate reality,
- **software throttle flags that cost no clocks**, reported as deliberately ignored so you
  don't go chasing them.

Works with `metrics.csv` alone, and says more the more you give it — `engine_metrics.csv`
adds the queue/cache/preemption columns, `--telemetry` adds power, clocks, interconnect and
CPU, and the roofline inputs turn the bandwidth question from a guess into a number.

---

## 5. `official.sh` — the scored command, verbatim

The task rules fix the AIPerf command for the official scored re-run, and modifying it
invalidates the score. This script contains exactly that command and nothing else — no
monitoring, no conveniences, no extra flags.

```bash
./official.sh --host gw01:8000 --model qwen3-pd --concurrency 64
./official.sh --print --host gw01:8000 --model qwen3-pd --concurrency 64   # show it, run nothing
./official.sh --verify                                                     # diff against the rules
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--concurrency N` | required | The team-declared N. Becomes both `--concurrency` and `--workers-max`. |
| `--host HOST:PORT` | required | The team's gateway. Takes `host:port`, not a URL — the command builds `http://$HOST` itself, and a URL here is rejected. |
| `--model ID` | required | The id the gateway reports at `/v1/models`. |
| `--out DIR` | `./official-<timestamp>-c<N>` | The script runs *from inside* this directory. |
| `--venv PATH` | `./aiperf-venv`, else the one beside the script | Resolves either way so `observe.sh` can run this from a point directory. |
| `--print` | — | Print the exact command and exit. |
| `--verify [FILE]` | `../2026-APAC-HPC-AI/1. Qwen3 AI Task Rules.md` | Compare against the rules and exit non-zero on any difference. |

**`--verify` is the reason this file exists.** "We ran the official command" is a claim;
this makes it checkable. It re-extracts the fenced command from the organisers' markdown,
tokenises both, and diffs them — so a typo, a stale copy, or a mid-competition rules update
fails loudly instead of silently invalidating a scored run:

```
x THE COMMAND HAS DIVERGED FROM THE RULES
   --- rules
   +++ official.sh
    --benchmark-duration
   -180
   +120

   the rules are authoritative: edit official.sh to match, never the reverse
```

**There is no `--artifact-dir` in the published command**, and adding one would modify it.
So the script `cd`s into `--out` and lets AIPerf's default artifact directory (`artifacts`)
land there. AIPerf then nests one level further,
`artifacts/<model>-<endpoint>-concurrency<N>/` — which is why `observe.sh` searches for the
results instead of assuming a path.

**Only three values are substituted**, and they are the three the rules leave to the team.
Everything else — the tokenizer id, seed 2026, the 4000/200 shape, `--cache-bust
first_turn_prefix`, the closed-loop concurrency mode, the goodput thresholds — is fixed.

---

## 6. `observe.sh` — monitoring, from the outside

The scored command cannot carry monitoring flags, so monitoring has to happen *around* it.
`observe.sh` runs any command as a child and collects evidence without touching it:

```bash
# on each server node, for the duration of the run
./gpu_sample.sh --out ./tel

# on the client
./observe.sh --out ./scored --gateway gw01:8000 -- \
    ./official.sh --host gw01:8000 --model qwen3-pd --concurrency 64

./diagnose.sh ./scored --telemetry ./tel --weights 22 --tp 8
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--out DIR` | required | Points accumulate here; call it once per declared concurrency to build a curve. |
| `--point NAME` | `c<N>` | `diagnose.sh` treats `c<N>` points as the concurrency curve. |
| `--concurrency N` | read from the child command | Names the point, and becomes the batch size in the roofline. |
| `--gateway HOST:PORT` | none | Enables the model-id lookup, the clock-skew measurement and the default metrics URL. |
| `--metrics-url URL` | `http://<gateway>/metrics` | Prometheus endpoint. **Repeatable — pass it once per worker.** |
| `--metrics-interval` | `2` | Poll period, seconds. |
| `--isl` / `--osl` | `4000` / `200` | Recorded for the roofline; defaults match the task. |
| `--sla` | `"time_to_first_token:3000 inter_token_latency:20"` | Recorded so `diagnose.sh` judges against the real gate. |
| `--no-metrics` | metrics on | Don't poll. |

It collects three things the command itself cannot:

- **Prometheus metrics polled continuously during the run.** Queue depth and KV usage are
  gauges: they read zero once load stops, so scraping before and after tells you nothing.
  Each endpoint is scraped into its own file and kept separate, because how they combine
  depends on the metric — see below.
- **Clock skew against the gateway**, from the server's own HTTP `Date` header, so
  telemetry sampled on the server nodes lines up with the run window.
- **Start and end timestamps per point**, which is what makes that slicing possible.

Then it finds whatever results the child produced — wherever it put them — and assembles
`metrics.csv`, `engine_metrics.csv`, `points.csv` and `env.txt`, the exact artifact set
`diagnose.sh` consumes. Re-running the same point replaces its row rather than duplicating
it.

It also prints the number the task actually scores:

```
  c64: goodput 27.49 req/s, pass rate 1.000  ->  880 good output tok/s
```

---

## Notes for the Qwen3-235B-A22B PD task

### `build/` — reconstructing the stack

The submission asks for a build method that lets someone else rebuild the environment a
measurement was taken in. There are two honest answers, and `build/build.sh` supports both:

```bash
./build/build.sh                 # build from build/Dockerfile* and export .sqsh files
./build/build.sh --record-only   # you used the images the cluster staged: fingerprint them
```

Use `--record-only` if you ran the provided `sglang-harbor.sqsh` unmodified. It writes the
size and SHA-256 of the images you actually ran, which is a checkable claim; shipping a
Dockerfile you did not build from is not.

| File | What it pins |
| --- | --- |
| `build/Dockerfile` | The SGLang serving stack, **by digest**, plus NIXL for the PD transfer backend. |
| `build/Dockerfile.aiperf` | The client: a pinned AIPerf, and the official tokenizer baked in. |

Both are pinned by digest or exact version on purpose. A floating tag means "the stack we
measured" is whatever the registry held that day, which is neither reproducible nor
defensible in a write-up. Both write a `pip freeze` manifest into the image, so a later run
can be diffed against the one that produced a number.

Three decisions in there that came out of actually building them:

- **`UCX_NET_DEVICES` is deliberately *not* baked in.** The right HCA differs per node, so
  it is detected at launch. Baking it in is the classic way to end up with the whole
  fabric's KV traffic on one port.
- **The client pre-fetches the tokenizer and proves it offline at build time.** The official
  command passes `--tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507`, an HF id, so a node with
  no egress fails *after* you have spent the allocation. Baking the cache in changes nothing
  about the command. The build then loads it with the network disabled and fails if that
  does not work — a cache missing `config.json` looks populated and still reaches for the
  network, which is exactly what the first version of this did.
- **`nixl-cu12`, not `nixl`.** The bare meta-package depends on both the cu12 and cu13
  variants and each pulls its own torch.

### Smoke tests first — `run/01`, `02`, `03`

Three escalating checks from the application notes, each a single submission whose exit
code is the whole verdict. Run them before spending an allocation on a PD topology:

| Script | Proves |
| --- | --- |
| `01-smoke-image.sbatch` | The container starts, the GPU is visible inside it, `sglang` imports, `$HOME` is writable. |
| `02-smoke-serve.sbatch` | A single non-disaggregated server loads a model, becomes healthy, and reports the model id it will answer to. |
| `03-smoke-serve-aiperf.sbatch` | The client image, the network path and the server complete a real streaming round trip. |

```bash
sbatch run/01-smoke-image.sbatch
SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/02-smoke-serve.sbatch
SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/03-smoke-serve-aiperf.sbatch
```

**Use a small model for 2 and 3.** They test model loading and the serving path, and a 0.6B
answers that in a minute where the 235B costs a large slice of your allocation to learn the
same thing. **And submit them** — do not reach for `srun --pty`. An interactive session
holds its GPU allocation for as long as it stays open, and schedulers that lower your
priority by recent usage will push your own later runs down the queue for it.

A zero exit code is not by itself a pass in step 3: it also checks that requests were
actually measured, because AIPerf can exit cleanly having written nothing useful.

### `run/sweep-pd.sbatch` — search the Prefill:Decode ratio

The split is the biggest lever the rules leave open, and the application notes' verified
8+8 is a starting point rather than an answer. This searches around it:

```bash
sbatch run/sweep-pd.sbatch
SPECS="2x4:4x2 2x6:4x1" CONCURRENCIES="64 128" sbatch run/sweep-pd.sbatch
```

A topology is one spec, `<prefill-tp>x<count>:<decode-tp>x<count>`:

| Spec | Split | What it is |
| --- | --- | --- |
| `2x4:4x2` | 8 + 8 | The split the application notes verified. |
| `2x6:4x1` | 12 + 4 | Prefill-heavy; the prefill workers spill onto the second node. |
| `2x2:4x3` | 4 + 12 | Decode-heavy. |
| `4x2:8x1` | 8 + 8 | Same GPU split, fatter tensor parallelism. |

Workers are placed by filling one node before spilling to the next, and **no worker
straddles a node** — tensor parallelism across an InfiniBand hop costs far more than any
ratio gains back. `run/pd-lib.sh` does the placement for both this and the scored job, so
a winner transfers as one variable rather than an edit:

```
 winner   2x6:4x1 at c64 -- 41200 good output tok/s (pass 0.981)

 Take it to a scored run with:
   PD_SPEC=2x6:4x1 CONCURRENCY=64 sbatch run/submit.sbatch
```

Three decisions worth keeping if you rewrite it:

- **Deploy once, measure every concurrency, then tear down.** Loading a 235B checkpoint
  dwarfs a 15-second measurement, and comparing ratios at one fixed N would be misleading
  anyway — the best N moves with the ratio.
- **Short benchmark, identical everything else.** The rules leave your own testing
  unconstrained, so the window is 15 s rather than 120. But the 4000/200 shape and the
  goodput thresholds stay exactly as the official command has them, so a pass rate here
  means what it will mean there.
- **A margin under 5 % is not a result.** The summary says so rather than declaring a
  winner a short run cannot actually distinguish.

### `run/submit.sbatch` — deploy and score in one job

The task requires that bringing the serving stack up and invoking the official command are
consecutive steps of *the same* job. `../run/submit.sbatch` is that job, built around the
verified 8+8 split from the Firmus application notes:

| | node 0 (head) | node 1 (tail) |
| --- | --- | --- |
| role | 4× prefill | 2× decode |
| parallelism | `--tp-size 2` | `--tp-size 4 --ep-size 4` |
| ports | 30100–30103 (bootstrap 9000–9003) | 30200–30201 |
| also | router on 8000 | `--max-running-requests 256` |

That is `PD_SPEC=2x4:4x2`, the default. Any placeable spec works — the sweep above prints
the one to use.

It runs in this order, and the order is the point: verify the official command still matches
the rules → start workers → poll every worker's `/health` → start the router → poll it →
start `gpu_sample.sh` on **both** nodes → run the scored command through `observe.sh` →
stop telemetry → diagnose each node.

```bash
sbatch run/submit.sbatch
CONCURRENCY=128 sbatch run/submit.sbatch     # every config value is env-overridable
```

Two details worth keeping if you rewrite it:

- **`--verify` runs before anything else.** A command that has drifted from the rules
  should fail in the first second, not after an allocation and a 210-second measurement.
- **`UCX_NET_DEVICES` is detected per node**, not hardcoded. NIXL needs the right HCA for
  cross-node KV transfer and the two nodes need not enumerate devices identically. If it
  finds no ACTIVE port it says so loudly rather than letting UCX pick silently — and the
  `ib` column will show the consequence either way.

The gateway the client talks to is the router and only the router; pointing the benchmark
at a prefill or decode worker is a rule violation.

### Four things that change how the kit is used

**The gate is per request, and it is a per-token gate.** `TTFT ≤ 3000 ms` and *average*
`ITL ≤ 20 ms`, with `good_request_fraction ≥ 0.90`. `diagnose.sh` understands
`inter_token_latency` in `--sla` and judges it on `itl_p90`, not `itl_avg` — because the
gate is applied per request, an aggregate mean can sit comfortably under the threshold
while a third of the requests fail:

```
 c64  itl_avg 18.49   itl_p90 21.64   pass_rate 0.562
 SLA missed    at c64: ITL p90 21.6ms > 20ms
```

When every aggregate clears the gate but the pass rate does not, the report says so and
points at the percentiles rather than pretending nothing is wrong.

**The score is good output tokens per second**, not good requests per second.
`diagnose.sh` prints it (`goodput × OSL`) alongside the goodput it ranks on.

**Qwen3-235B-A22B is an MoE, and the roofline's weight auto-detection is wrong for it.**
Summing the checkpoint gives ~235 B parameters, but only ~22 B are activated per token.
Left alone, MBU is overstated by roughly an order of magnitude and everything reads
`MEMORY BW`. Pass the **active** weight bytes explicitly:

```bash
./diagnose.sh ./scored --telemetry ./tel --weights 22 --tp 8   # ~22 GiB active at FP8
```

There is a real effect hiding behind that number: at larger batches, different requests
select different experts, so the bytes actually read per decode step climb toward the full
checkpoint. That makes it a testable prediction — MBU should rise with concurrency faster
than the KV term alone explains — which is exactly what the concurrency sweep and
`--isl-sweep` are for.

**InfiniBand is sampled, and for this task it is the link that matters.** PD
disaggregation moves the KV cache between nodes over 8× NDR, and the task's own
optimization guidance calls out cross-node KV transfer latency. `gpu_sample.sh` reads the
port counters from sysfs and `diagnose.sh` reports them as an `ib` column, as a share of
the negotiated link rate:

```
 point      goodput   pass  ...   pcie     ib  verdict
 c32           9.52   1.00  ...     1%    90%  INTERCONNECT

 interconnect  InfiniBand peaked at 45.0 GB/s per port (90% of link), 180.0 GB/s across the node
               PD disaggregation moves the KV cache over this link -- at this
               level it is a real constraint on the P:D split and placement
 interconnect  IB xmit_wait grew by 15000: ports had data to send and no credit.
               That is congestion or credit starvation, not raw bandwidth
```

`gpu_sample.sh` is per node — run it on **both**, and diagnose each side separately.

---

## Comparing runs with teammates

For numbers to line up, everyone must match on:

- the same `--isl` / `--osl`,
- the same `--sla` (same metrics, same thresholds, same units),
- the same `--anchor`, run as its own point,
- warmup excluded,
- the same pinned `aiperf` version,
- a local tokenizer matching the served model,
- the same `--shape-c` if you are comparing perturbation slopes (the ITL slope scales with
  batch, so slopes measured at different concurrencies are not comparable).

`run.sh` handles all of these as long as the flags agree. Diff the `env.txt` files first —
that is where mismatches show up.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `$AIPERF not found` | The `--venv` you passed to `run.sh` differs from the install one, or `install.sh` never ran. |
| `could not get a model id from …/v1/models` | Endpoint has no model list — pass `--model` explicitly. For a TRT-LLM disagg proxy, query one of its workers for the id. |
| `cannot recognize AIPerf's sequence length flags` | Unexpected AIPerf version. Check `aiperf profile --help` and update the detection block in `run.sh`. |
| `this version has no --streaming` | Too old to measure TTFT/ITL. Upgrade, or drop those metrics. |
| `no parsable results` | Every round failed. Read the error output above it, and `warmup.log`. |
| Empty / missing sweep row | That concurrency failed but the run continued by design; its directory has no `profile_export_aiperf.csv`. |
| `metrics.csv` has no rows at all | Check where AIPerf actually wrote its results: `find $OUT -name profile_export_aiperf.csv`. Some builds nest them one level deeper than the table builder looks. |
| `diagnose.sh` prints `-` in every telemetry column | The telemetry doesn't overlap the run window. The report prints the offset and what to add to `env.txt`; usually the two nodes' clocks are not NTP-synced. |
| Every point says `THROTTLING` | Real on a thermally-limited card: check `clk` in the table and the temperature in `host.txt`. If clocks are at maximum this cannot happen — the rule is clock-based, not flag-based. |
| `MEMORY BW?` instead of `MEMORY BW` | The roofline inputs are missing. Pass `--mem-bw` (and `--weights` if the model directory isn't the tokenizer path). |
| Every point reads `MEMORY BW` on an MoE | The weight auto-detection summed the whole checkpoint. Pass the **active** parameter bytes with `--weights`. |
| `official.sh --verify` fails | The rules changed, or this copy drifted. The rules are authoritative — update the command block to match them. |
| `official.sh` can't find the venv | Pass `--venv`, or set `$AIPERF_VENV`. It looks in `./aiperf-venv` then beside the script. |
| `observe.sh` says "no profile_export_aiperf.csv" | The observed command failed, or wrote outside `--out`. Its own output is in `<out>/<point>/`. |
| The official run takes far longer than 210 s | Setup is not free: 12800 synthetic conversations at ISL 4000 is ~220 MB and takes ~30 s to generate before the warmup phase starts. |

---

## Exit codes

Both scripts return 0 only on a fully clean run, so they can be used as CI gates.

| Script | Non-zero when |
| --- | --- |
| `install.sh` | Python/venv/pip step failed, or a flag `run.sh` depends on is unavailable in this AIPerf build (the summary lists which). |
| `run.sh` | A bad argument, a failed precondition, any failed work point, or no parsable results. |
| `official.sh` | A bad argument, a missing venv, a non-zero exit from AIPerf — or, under `--verify`, any difference from the rules file. |
| `observe.sh` | Whatever the observed command returned. Monitoring failures degrade to warnings; they must never fail a scored run. |

A failed sweep point does **not** abort the run — `set -e` is deliberately off so one bad
concurrency doesn't kill the rest. The point is recorded, listed in the closing summary,
and reflected in the exit code. A failed warmup is reported but never fails the run.

---

## Notes and limitations

- **Runs are not atomic.** Check that `metrics.csv` has every point you expected; a missing
  row means that concurrency produced no `profile_export_aiperf.csv`.
- **Raw artifacts are large.** `inputs.json` and `profile_export.jsonl` hold every request;
  delete them before committing results. `env.txt` and `metrics.csv` are the parts worth
  keeping.
- **Default run size** is 5400 requests (40 warmup + 320 anchor + 5040 sweep). Trim with
  `--sweep` / `--req-per-c` when you just want a smoke test. Each shape-sweep value adds
  `--shape-c × --req-per-c` more, and a long `--osl-sweep` value costs far more wall-clock
  per request than the count suggests.
- **Causal testing is opt-in and covers two dimensions.** Without `--isl-sweep` /
  `--osl-sweep` the concurrency curve can only correlate a bottleneck with load. With them,
  ISL and OSL are the only inputs perturbed — batch composition, quantization and
  parallelism strategy still have to be compared across separate runs (same `--anchor`,
  diff the `env.txt` files).
- **The client's own resources are never sampled.** `gpu_sample.sh` runs on the server
  node. `CLIENT BOUND` depends on the engine reporting `num_requests_running`, so a
  client-bound run against a server with no Prometheus endpoint is undetectable. Run the
  load generator on a machine with cores to spare, and watch its CPU yourself.
- **Prefill and decode are separated only by inference.** TTFT and ITL are the evidence;
  the server-side batch composition (`num_batched_tokens`, iteration time) is not sampled.
- **NVLink and InfiniBand parsing have not been exercised on that hardware.** Both were
  written against a box that has neither — IB was verified against a synthetic sysfs tree
  with known counter values, and both degrade cleanly to "absent". Check `nvlink.csv` and
  `ib.csv` hold plausible numbers the first time you run on an SXM or IB machine, and
  sanity-check the IB rate against `perfquery` or `ibstat` once.
- **IB port counters can wrap.** The legacy sysfs counters are 32-bit on some drivers and
  wrap in seconds at NDR rates. A wrap shows up as a negative delta; those ports are
  dropped from the rate rather than reported wrongly, and the run is flagged. Sample more
  often, or read the extended 64-bit counters, if it happens.
- **The roofline assumes the weights on disk are the weights being served.** Load-time
  quantization (fp8, AWQ applied by the engine) makes the auto-detected size wrong; pass
  `--weights` explicitly there.
