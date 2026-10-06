#!/bin/bash
# site_gpu_arg(): the whole GPU argument, per site. Nano5 needs --gpus-per-node=N
# and refuses --gres=gpu:N; everyone else keeps --gres.
set -uo pipefail
cd "$(dirname "$0")/.."
pass=0 fail=0
ck() {  # $1=label $2=expected $3=actual
    if [[ "$2" == "$3" ]]; then (( pass++ ))
    else echo "x $1: want '$2' got '$3'" >&2; (( fail++ )); fi
}

# 1-2. the default is the --gres form every other site already used
out="$(unset SITE_GPU_ARG_FMT; . run/site-lib.sh; site_gpu_arg 8)"
ck "default 8"      "--gres=gpu:8" "$out"
out="$(unset SITE_GPU_ARG_FMT; . run/site-lib.sh; site_gpu_arg)"
ck "default no arg" "--gres=gpu:8" "$out"

# 3-5. THE REGRESSION THE FIRST VERSION OF THIS FILE CAUGHT: the default must keep
# honouring SITE_GRES_FMT. NYCU's job_submit plugin demands the type be named --
# "For full GPU: --gres=gpu:gpu:4" -- so a default of plain --gres=gpu:N is refused
# there. The default derives from site_gres() rather than restating the format.
out="$(SITE_GRES_FMT='gpu:gpu:%d'; unset SITE_GPU_ARG_FMT; . run/site-lib.sh; site_gpu_arg 4)"
ck "gres fmt kept"  "--gres=gpu:gpu:4" "$out"
out="$(. run/sites/nycu.sh; . run/site-lib.sh; site_gpu_arg 8)"
ck "nycu.sh typed"  "--gres=gpu:gpu:8" "$out"
out="$(SITE_GPU_ARG_FMT='--gres=gpu:gpu:%d'; . run/site-lib.sh; site_gpu_arg 4)"
ck "explicit fmt"   "--gres=gpu:gpu:4" "$out"

# 5-7. Nano5. THE REGRESSION THIS FILE EXISTS FOR: a format starting with "--"
# makes bash's printf treat it as an option unless printf is given `--` first.
out="$(SITE_GPU_ARG_FMT='--gpus-per-node=%d'; . run/site-lib.sh; site_gpu_arg 8)"
ck "nano5 8"        "--gpus-per-node=8" "$out"
out="$(SITE_GPU_ARG_FMT='--gpus-per-node=%d'; . run/site-lib.sh; site_gpu_arg 1)"
ck "nano5 1"        "--gpus-per-node=1" "$out"
err="$(SITE_GPU_ARG_FMT='--gpus-per-node=%d'; . run/site-lib.sh; site_gpu_arg 8 2>&1 >/dev/null)"
ck "nano5 no stderr" "" "$err"

# 8. the real site file, read as the campaign reads it
out="$(. run/sites/nano5.sh; . run/site-lib.sh; site_gpu_arg "$SITE_GPUS_PER_NODE")"
ck "nano5.sh"       "--gpus-per-node=8" "$out"

# 9-11. the three things Nano5's plugin enforces are all set
acct="$(. run/sites/nano5.sh; echo "$SITE_ACCOUNT")"
ck "account"        "MST114554" "$acct"
cpg="$(. run/sites/nano5.sh; echo "$SITE_CPUS_PER_GPU")"
ck "cpus per gpu"   "12" "$cpg"
via="$(. run/sites/nano5.sh; echo "$SLURM_STEP_VIA")"
ck "step via"       "overlap" "$via"

# 12. a step-level --gres must never be emitted here, because the job already
# holds --gpus-per-node for the same cards
grep -q 'SLURM_STEP_VIA="${SLURM_STEP_VIA:-overlap}"' run/sites/nano5.sh \
    && (( pass++ )) || { echo "x nano5 must default to overlap steps" >&2; (( fail++ )); }

# 13. submit-site.sh must use site_gpu_arg, not --gres directly
if grep -q 'site_gpu_arg "$GPUS"' run/submit-site.sh \
   && ! grep -q -- '--gres="\$(site_gres "\$GPUS")"' run/submit-site.sh; then
    (( pass++ ))
else echo "x submit-site.sh still hardcodes --gres" >&2; (( fail++ )); fi

# 14. and must pass --account when the site names one
grep -q 'ARGS+=(--account="$SITE_ACCOUNT")' run/submit-site.sh \
    && (( pass++ )) || { echo "x submit-site.sh does not pass --account" >&2; (( fail++ )); }

echo "test-gpu-arg: $pass passed, $fail failed"
(( fail == 0 ))
