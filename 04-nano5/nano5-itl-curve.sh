#!/bin/bash
# ITL as a function of decode batch, on one node.
#
# WHAT THIS SETTLES. I published "decode's ITL budget supports R ~ 29, so decode has
# two-fold headroom and all the remaining upside is in prefill". That came from a
# specs model -- 55 GB of FP8 weights over H100's 3.35 TB/s is 16.5 ms, plus a KV
# read of 1.69 ms at batch 28 -- which matched ONE measurement to 3.5% and so looked
# settled. Fitting the 23 real H100 measurements we already have says otherwise:
#
#   measured   ITL = 15.88 + 0.128 x batch     (R2 0.78, n=23)
#   my model   ITL = 16.50 + 0.060 x batch
#
# The slope is 2.1x steeper, which puts the 20 ms gate at batch ~32 and R ~16 rather
# than batch 58 and R 29 -- headroom of 10%, not 98%. But those 23 points are
# different CONFIGURATIONS (EP on/off, DeepGEMM, three prefill batch sizes), so the
# slope absorbs configuration effects, and the relationship may well be concave: with
# top-8-of-128 routing, batch 28 already touches nearly every expert while batch 14
# touches maybe two thirds, so the weight read grows with batch until it saturates.
# A concave curve means the linear fit over 18-30 overestimates the slope above 30
# and there is more headroom than 10%.
#
# One configuration, batch varied, is what separates those.
#
# WHY SINGLE-NODE PD AND NOT solo. In solo mode the same engine prefills and decodes,
# and a 785 ms prefill forward is a 785 ms gap in every decoding request's token
# stream -- aiperf's ITL would be dominated by prefill stalls, not by decode. Smaller
# chunks only make the gap 130 ms instead of 785. A real decode worker does only
# decode, so 4x1:4x1 on 8 GPUs (one TP4 prefill, one TP4 decode) measures the thing;
# TP4 also matches the decode worker in the scored deployment, which TP8 would not --
# weights per GPU, and so the intercept, halve at TP8.
#
# The sweep is CONCURRENCY, not --max-running-requests, because the latter is a server
# flag and would cost a 12-minute checkpoint load per value. With one prefill worker R
# caps around 7 req/s, so decode batch caps near 7 x 0.2 x ITL ~ 27 -- which happens
# to be the same per-worker batch the 16-GPU runs sit at, so the range overlaps the
# operating point and covers everything below it. That is enough to tell a line from
# a curve, which is the question.
set -uo pipefail
cd /work/u8924359/hsingyu-hpcai

export PD_SITE=nano5
export SITE_PARTITION="${SITE_PARTITION:-dev}"   # MaxNodes=1 is fine: this IS one node
export NODES=1 GPUS=8 TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
export ENGINES=sglang SPECS="4x1:4x1"
export MEM_FRACTION=0.85 DISABLE_PREFIX_CACHE=1

# Wide and cheap. 20/20/20 windows, ~100 s per point, so eleven points cost about
# 18 minutes against one 12-minute bring-up.
export CONCURRENCIES="${CONCURRENCIES:-4 8 12 16 20 24 28 32 40 48 64}"
export WARMUP_SECONDS=20 BENCH_SECONDS=20 GRACE_SECONDS=20

# The scored configuration's prefill settings, so the prefill side is not a second
# variable. mem_fraction 0.80 on prefill is what lets DeepGEMM keep the 12,288 batch.
export VARIANTS="itlcurve:MOE_RUNNER_BACKEND_PREFILL=deep_gemm,DISABLE_PREFILL_GRAPH=1,CHUNKED_PREFILL=12288,MAX_PREFILL_TOKENS=12288,MEM_FRACTION_PREFILL=0.80"

exec run/submit-site.sh run/10-campaign.sbatch
