#!/bin/bash
# The first run on Nano5. Short window (20/20/20, the campaign's own exploration
# default), one concurrency, our best known configuration -- so that if anything in
# the new platform's plumbing is wrong, it costs one short job and not a scored one.
#
# normal, not dev. dev would be cheaper to queue and its 2-hour cap is plenty, but
# dev has MaxNodes=1 and a PD run needs two nodes -- sbatch refuses with "Node count
# specification invalid", which names neither the partition nor the limit.
# 2 nodes x 8 GPUs x 12 CPUs = billing 192, so an hour is 192 SU; billing is on
# elapsed, not on the reservation, so a 25-minute smoke bills about 80.
set -uo pipefail
cd /work/u8924359/hsingyu-hpcai

export PD_SITE=nano5
export SITE_PARTITION="${SITE_PARTITION:-normal}"
export NODES=2 GPUS=8 TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
export ENGINES=sglang SPECS="4x2:4x2"
export CONCURRENCIES="${CONCURRENCIES:-80}"
export MEM_FRACTION=0.85 DISABLE_PREFIX_CACHE=1

# The 2,844 configuration, verbatim. No new variable on the first run here: the
# thing under test is the platform, not the setting.
BIG="MOE_RUNNER_BACKEND_PREFILL=deep_gemm,DISABLE_PREFILL_GRAPH=1,CHUNKED_PREFILL=12288,MAX_PREFILL_TOKENS=12288,MEM_FRACTION_PREFILL=0.80"
export VARIANTS="${VARIANTS:-bigpf:$BIG}"

run/submit-site.sh run/10-campaign.sbatch
