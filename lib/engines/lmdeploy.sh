#!/bin/bash
# LMDeploy. A fourth reading on the same ruler.
#
# Worth adding because it is a genuinely different implementation rather than
# another wrapper: its PyTorch backend has its own MoE and FP8 paths, and
# Qwen3MoeForCausalLM is in pytorch/models/module_map.py, with
# weight_block_size=[128,128] / fmt=e4m3 handled as a first-class case in
# pytorch/config.py -- which is exactly this checkpoint's quantisation.
#
# It also has a property none of the other three have: PREFIX CACHING IS OFF BY
# DEFAULT. It is opt-in through --enable-prefix-caching. So the server-side stand-in
# for the client's --cache-bust (see run/client-lib.sh) costs nothing here -- we
# simply do not turn it on, and DISABLE_PREFIX_CACHE has nothing to do.
#
# NOT a PD engine as far as we can tell. LMDeploy has no equivalent of SGLang's
# --disaggregation-mode, so eng_pd_worker_cmd is deliberately absent: the campaign
# will refuse a PD spec for this engine rather than silently serve a colocated
# deployment and let it be compared against one that is disaggregated.

# No container anywhere: LMDeploy was only ever installed as a pip venv here.
eng_default_image() { echo "${LMDEPLOY_IMG-}"; }

eng_import_check() {
cat <<SNIP
PY=$(eng_python)
"\$PY" -c "
import lmdeploy, sys
print('  python  ', sys.version.split()[0])
print('  lmdeploy', lmdeploy.__version__)
" || { echo "x lmdeploy did not import" >&2; exit 1; }

# NOT the console script. pip bakes the venv path of the moment into its shebang,
# so envs/lmdeploy/bin/lmdeploy still begins "#!/home/hsingyu/envs/_lmd/bin/python"
# from when that venv had a different name -- and exec'ing it fails with
# "bad interpreter", exit 126, five seconds into a 4-GPU allocation (job 81599).
# "python3 -m lmdeploy" asks the interpreter we already chose and cannot go stale.
"\$PY" -m lmdeploy serve api_server --help > /dev/null 2>&1 \\
    || { echo "x 'python -m lmdeploy serve api_server' does not run" >&2; exit 1; }
echo "  ok python -m lmdeploy serve api_server runs"

# The architecture has to be in the PyTorch engine's map, or it falls back in ways
# that are slower and not comparable.
"\$PY" -c "
from lmdeploy.pytorch.models.module_map import MODULE_MAP
hit = [k for k in MODULE_MAP if 'Qwen3Moe' in k]
print('  Qwen3Moe entries in module_map:', len(hit))
" 2>/dev/null || echo "  !  could not read module_map -- check Qwen3MoeForCausalLM support"
SNIP
}

# ---------------------------------------------------------------- flag names
# Kept as variables for the same reason as trtllm.sh: when a name changes, it
# changes in one place and the failure is a diff rather than an investigation.
LMDEPLOY_BACKEND="${LMDEPLOY_BACKEND:-pytorch}"
LMDEPLOY_EXTRA="${LMDEPLOY_EXTRA:-}"            # whitespace-split, appended verbatim

eng_serve_cmd() {   # model name port tp
    local -a cmd=(
        "$(eng_python)" -m lmdeploy serve api_server "$1"
        --backend "$LMDEPLOY_BACKEND"
        --tp "$4"
        --model-name "$2"
        # The checkpoint is FP8 with block scales; say so rather than letting it be
        # inferred, so a misread shows up as a startup error and not as a slow run.
        --model-format fp8
        --server-name 0.0.0.0 --server-port "$3"
    )
    # Its own names for the knobs the campaign varies.
    #   --session-len            = SGLang's --context-length
    #   --cache-max-entry-count  = the fraction of free HBM given to the KV cache
    #   --max-batch-size         = SGLang's --max-running-requests
    #   --max-prefill-token-num  = SGLang's --chunked-prefill-size
    #   --ep                     = expert parallel degree
    [[ -n "${CONTEXT_LEN:-}" ]]      && cmd+=(--session-len "$CONTEXT_LEN")
    [[ -n "${MEM_FRACTION:-}" ]]     && cmd+=(--cache-max-entry-count "$MEM_FRACTION")
    [[ -n "${MAX_RUNNING_SOLO:-}" ]] && cmd+=(--max-batch-size "$MAX_RUNNING_SOLO")
    [[ -n "${CHUNKED_PREFILL:-}" ]]  && cmd+=(--max-prefill-token-num "$CHUNKED_PREFILL")
    [[ -n "${EP_SIZE:-}" ]]          && cmd+=(--ep "$EP_SIZE")
    # KV_CACHE_DTYPE is deliberately NOT mapped. LMDeploy's --quant-policy takes
    # 0/4/8 for bf16/int4/int8 KV, which is a different quantisation from the fp8
    # the other engines take -- wiring one to the other would make a row labelled
    # "KV fp8" mean two different things depending on the engine.
    if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
        echo "  ! lmdeploy: KV_CACHE_DTYPE=$KV_CACHE_DTYPE ignored;" \
             "--quant-policy is int4/int8, not fp8" >&2
    fi
    # Prefix caching is opt-in here, so the default already matches what
    # DISABLE_PREFIX_CACHE asks for and there is nothing to pass.
    # shellcheck disable=SC2206
    [[ -n "$LMDEPLOY_EXTRA" ]] && cmd+=($LMDEPLOY_EXTRA)
    printf '%s\n' "${cmd[@]}"
}

# Bare metal needs the toolkit and headers a container would have shipped --
# see eng_venv_env() in engine-lib.sh.
eng_pd_env() { eng_venv_env; }

eng_notes() {
cat <<'SNIP'
No PD mode: LMDeploy has no --disaggregation-mode equivalent, so this engine can
only fill the `solo` shape. A PD spec will fail at pd_launch rather than quietly
serve a colocated deployment.
KV_CACHE_DTYPE does not apply: --quant-policy is int4/int8, not fp8.
SNIP
}
