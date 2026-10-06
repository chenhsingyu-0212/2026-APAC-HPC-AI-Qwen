#!/bin/bash
# The 16,384-token prefill batch, against a paired control at the 12,288 that scored
# 2,844. Four requests per forward instead of three.
#
# WHY NOW: the teammate measured +5.5-8.3% from bt8192 -> bt16384 on vLLM (their B11,
# VM-measured, and their scored path *_dgpf_e3 renders prefill bt16384), so the
# mechanism is confirmed on the other engine. It is also the only lever left that
# raises prefill throughput without touching the KV transfer, and R is what score is.
#
# MEM_FRACTION_PREFILL=0.74, derived rather than guessed. The one measured anchor is
# that at f=0.80 with chunked 12,288, DeepGEMM asked for 2x = 24,576 and halved ONCE
# to exactly 12,288 -- so required(12288) <= available(0.80) < required(24576), where
# available(f) = 79.65(1-f) - 2.2 is calibrated on measurement. Taking required as
# linear in max_m, that brackets the per-token cost and puts the f needed for a
# 16,384 batch between 0.743 (pessimistic end) and 0.857 (optimistic). 0.74 holds
# across the whole bracket; 0.75 is marginal.
#
# The cost is pool size, and it is affordable: f=0.74 leaves roughly 3.7 GB of KV
# pool against a measured prefill-side peak of ~0.6 GB -- job 215960 watched token
# usage sit at 0.11-0.13 and peak 0.19 over ~1,250 prefill steps, burst included.
#
# IF THE GUARD FIRES, the job tells us the answer: pd_assert_prefill_batch fails the
# cell and prints DeepGEMM's own "Available memory X GB is less than required memory
# Y GB" line, which pins the constant exactly for a retry. That is the whole reason
# to run the control first.
set -uo pipefail
cd /work/u8924359/hsingyu-hpcai

export PD_SITE=nano5
export SITE_PARTITION="${SITE_PARTITION:-normal}"
export NODES=2 GPUS=8 TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
export ENGINES=sglang SPECS="4x2:4x2"
export CONCURRENCIES="${CONCURRENCIES:-80 88}"
export MEM_FRACTION=0.85 DISABLE_PREFIX_CACHE=1
export WARMUP_SECONDS=60 BENCH_SECONDS=120 GRACE_SECONDS=30
export DATASET_ENTRIES=12800 SEED=2026

COMMON="MOE_RUNNER_BACKEND_PREFILL=deep_gemm,DISABLE_PREFILL_GRAPH=1"
CTL="$COMMON,CHUNKED_PREFILL=12288,MAX_PREFILL_TOKENS=12288,MEM_FRACTION_PREFILL=0.80"
BIG="$COMMON,CHUNKED_PREFILL=16384,MAX_PREFILL_TOKENS=16384,MEM_FRACTION_PREFILL=0.74"
# Both flags on each arm: --max-prefill-tokens caps the batch independently of
# --chunked-prefill-size, so raising one and not the other leaves the batch put.
# EXPERIMENT FIRST NOW. It used to be control first, so that a short allocation
# would truncate the experiment and not the baseline. That reason is spent: the
# baseline is measured twice on this machine, 2,768 and 2,760 at N=80, so the
# slot worth protecting is the unknown one. It also removes the position
# confound that made 372421 and 372456 unreadable -- their second cells both
# failed on port 9100 and I could not tell flag from position until the two
# jobs produced identical numbers under different flags.
export VARIANTS="bigpf16k:$BIG+bigpf:$CTL"

# afterany, not afterok: if 372421 fails we still want this one to run, because the
# two jobs test different things and 372421 failing says nothing about this.
exec run/submit-site.sh run/10-campaign.sbatch "${DEPEND:+--dependency=$DEPEND}"
