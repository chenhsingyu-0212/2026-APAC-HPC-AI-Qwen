#!/bin/bash
# The experiment that could beat 2,844: an FP8 KV cache, against a paired control.
#
# The KV cache is BF16 today. 94 layers x 4 kv heads x 128 head_dim x 2 (K+V) x 2
# bytes = 192,512 B per token, so 770 MB per ISL-4000 request crosses the fabric --
# and prefill_transfer_kv_cache measures 391 ms for it, which is 1.97 GB/s, or 3.9%
# of ONE of the node's ten 400 Gb/s HCAs. That number cannot be bandwidth.
#
# fp8_e4m3 halves two things at once:
#   * the bytes on the wire, 770 MB -> 385 MB
#   * decode's per-step KV read, 1.69 ms -> 0.85 ms, which is ITL headroom
#
# The rules make FP8 KV the DEFAULT and BF16 the "or higher" option (Task Rules Q1),
# so this is plainly legal -- not Bonus, not a grey area.
#
# KV_CACHE_DTYPE has been plumbed to both roles in run/engines/sglang.sh since
# September and has never once been set by a submit script.
#
# Control first, deliberately: if the allocation runs short it is the experiment that
# gets truncated and not the baseline. That ordering is what made 216243 readable.
set -uo pipefail
cd /work/u8924359/hsingyu-hpcai

export PD_SITE=nano5
export SITE_PARTITION="${SITE_PARTITION:-normal}"
export NODES=2 GPUS=8 TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
export ENGINES=sglang SPECS="4x2:4x2"
export CONCURRENCIES="${CONCURRENCIES:-80 88}"
export MEM_FRACTION=0.85 DISABLE_PREFIX_CACHE=1

# The official window, because this one is meant to be comparable with 2,844.
export WARMUP_SECONDS=60 BENCH_SECONDS=120 GRACE_SECONDS=30
export DATASET_ENTRIES=12800 SEED=2026

BIG="MOE_RUNNER_BACKEND_PREFILL=deep_gemm,DISABLE_PREFILL_GRAPH=1,CHUNKED_PREFILL=12288,MAX_PREFILL_TOKENS=12288,MEM_FRACTION_PREFILL=0.80"
# No whitespace inside either assignment -- VARIANTS word-splits, and the campaign
# now refuses a value containing a space rather than silently making four cells.
# EXPERIMENT FIRST NOW. It used to be control first, so that a short allocation
# would truncate the experiment and not the baseline. That reason is spent: the
# baseline is measured twice on this machine, 2,768 and 2,760 at N=80, so the
# slot worth protecting is the unknown one. It also removes the position
# confound that made 372421 and 372456 unreadable -- their second cells both
# failed on port 9100 and I could not tell flag from position until the two
# jobs produced identical numbers under different flags.
export VARIANTS="kv8:$BIG,KV_CACHE_DTYPE=fp8_e4m3+bigpf:$BIG"

run/submit-site.sh run/10-campaign.sbatch
