#!/bin/bash
# vLLM. Phase E's control arm -- a second measurement of the same thing on the
# same ruler (see docs/trtllm-plan.md).
#
# It is here in Phase B because the smoke tests cost one GPU and fifteen minutes,
# and having a validated second path is what turns "TRT-LLM's KV transfer will
# not behave on this fabric" from a rewrite under deadline into a switch.
#
# vLLM's /metrics is the most complete of the three engines. Where TRT-LLM cannot
# tell you what the scheduler is doing, vLLM at the same operating point can
# bound it -- see Q4 of the plan.

# See sglang.sh: no colon, so an empty VLLM_IMG means bare metal on purpose.
eng_default_image() { echo "${VLLM_IMG-/scratch/enroot-images/vllm.sqsh}"; }

eng_import_check() {
cat <<SNIP
PY=$(eng_python)
VLLM=$(eng_exe vllm)
"\$PY" -c "
import vllm, sys
print('  python', sys.version.split()[0])
print('  vllm  ', getattr(vllm, '__version__', 'imported'))
" || { echo "x vllm did not import" >&2; exit 1; }

command -v "\$VLLM" >/dev/null 2>&1 || { echo "x no vllm binary at \$VLLM" >&2; exit 1; }
"\$VLLM" serve --help > /dev/null 2>&1 || { echo "x 'vllm serve' does not run" >&2; exit 1; }
echo "  ok vllm serve runs"

# Phase E needs a KV connector for disaggregation. Report it now rather than
# discovering it when the PD job will not come up.
if "\$VLLM" serve --help 2>&1 | grep -qi 'kv-transfer-config\|kv_transfer_config'; then
    echo "  ok kv-transfer-config flag present (PD disaggregation)"
else
    echo "  !  no kv-transfer flag in this build -- PD would need a different route"
fi
"\$PY" -c "import nixl" 2>/dev/null \
    && echo "  ok nixl importable" \
    || echo "  !  nixl not importable -- fine here, needed by the NIXL connector"
SNIP
}

# The three knobs the 3,176-point Firmus recipe turns, expressed in vLLM 0.30's own
# names. All three were read out of the installed wheel, not a changelog:
#   --moe-backend            arg_utils.py:511, values in config/kernel.py's MoEBackend
#   --speculative-config     arg_utils.py:1698 (JSON; speculators-format drafts are
#                            recognised by maybe_override_with_speculators)
#   --no-enable-chunked-prefill   SchedulerConfig.enable_chunked_prefill defaults True
#
# MOE_RUNNER_BACKEND* are SGLang's variable names on purpose: the per-role split
# (deep_gemm on prefill, triton on decode) is a finding about the workload, not about
# an engine, so the same variant name has to select it in both.
vllm_moe_backend() {   # $1=role ("" for solo)
    local moe="${MOE_RUNNER_BACKEND:-}"
    if [[ "$1" == prefill && -n "${MOE_RUNNER_BACKEND_PREFILL:-}" ]]; then
        moe="$MOE_RUNNER_BACKEND_PREFILL"
    elif [[ "$1" == decode && -n "${MOE_RUNNER_BACKEND_DECODE:-}" ]]; then
        moe="$MOE_RUNNER_BACKEND_DECODE"
    fi
    echo "$moe"
}

# EAGLE-3 has to be identical on both halves or the pair never connects: vLLM's NIXL
# compatibility hash (version 6) covers the speculative method, the draft and its KV
# dtype, because the draft's KV is computed on prefill and shipped over NIXL. So this
# takes no role argument -- there is deliberately no way to set it on one side only.
vllm_spec_args() {
    [[ -n "${SPEC_DRAFT_PATH:-}" ]] || return 0
    printf -- '--speculative-config\n{"model": "%s", "num_speculative_tokens": %s}\n' \
        "$SPEC_DRAFT_PATH" "${SPEC_NUM_TOKENS:-3}"
}

eng_serve_cmd() {   # model name port tp
    local -a cmd=(
        "$(eng_exe vllm)" serve "$1"
        --served-model-name "$2" --trust-remote-code
        --tensor-parallel-size "$4"
    )
    # vLLM's names for the same two knobs SGLang calls --context-length and
    # --mem-fraction-static. Set them from the same variables so a cross-engine
    # row compares two engines and not two configurations.
    [[ -n "${CONTEXT_LEN:-}" ]] && cmd+=(--max-model-len "$CONTEXT_LEN")
    [[ -n "${MEM_FRACTION:-}" ]] && cmd+=(--gpu-memory-utilization "$MEM_FRACTION")
    # vLLM's names for the same knobs. NOT verified against --help on this
    # cluster: `vllm serve --help` cannot run on the login node (no driver, so
    # importing triton fails), and these names are the documented ones. A wrong
    # name costs that one variant's cell, which the campaign already tolerates --
    # it does not touch `base`.
    [[ -n "${KV_CACHE_DTYPE:-}" ]] && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "${CHUNKED_PREFILL:-}" ]] && cmd+=(--max-num-batched-tokens "$CHUNKED_PREFILL")
    [[ -n "${MAX_RUNNING_SOLO:-}" ]] && cmd+=(--max-num-seqs "$MAX_RUNNING_SOLO")
    [[ -n "${EP_SIZE:-}" ]] && cmd+=(--enable-expert-parallel)
    # See sglang.sh: the server-side equivalent of --cache-bust.
    [[ "${DISABLE_PREFIX_CACHE:-0}" == 1 ]] && cmd+=(--no-enable-prefix-caching)
    [[ "${DISABLE_CHUNKED_PREFILL:-0}" == 1 ]] && cmd+=(--no-enable-chunked-prefill)
    local moe; moe="$(vllm_moe_backend "")"
    [[ -n "$moe" ]] && cmd+=(--moe-backend "$moe")
    mapfile -t -O "${#cmd[@]}" cmd < <(vllm_spec_args)
    cmd+=(--host 0.0.0.0 --port "$3")
    printf '%s\n' "${cmd[@]}"
}

# Same as SGLang: a pip venv has no CUDA toolkit on PATH and no Python headers.
# Job 79370's vLLM cell spent 170 seconds loading the checkpoint and then died in
# triton with "Python.h: No such file or directory".
eng_pd_env() { eng_venv_env; }

# ---------------------------------------------------------------- PD
# vLLM CAN disaggregate; it just does not ship the gateway. The engine side is a
# KV connector selected through --kv-transfer-config, and NIXL is the same transport
# SGLang uses, so the two engines' PD numbers are comparable rather than measuring
# two different fabrics. What is missing is the router: SGLang has sglang_router,
# TensorRT-LLM has `trtllm-serve disaggregated`, and vLLM's wheel contains a .jpg of
# the workflow. run/engines/vllm-pd-proxy.py is that missing piece.
#
# Two requirements that are easy to miss and both fail at run time rather than at
# startup:
#
#   * the `nixl` package. NixlConnector imports nixl._api lazily, so a venv without
#     it starts a worker perfectly and fails on the first transfer. Note the name:
#     `pip install nixl` (the dispatcher), not nixl-cu13 alone, which installs as
#     `nixl_cu13` and does not provide the module vLLM imports.
#   * VLLM_NIXL_SIDE_CHANNEL_PORT must DIFFER between workers on the same host. It is
#     how a consumer finds a producer's metadata, and two workers sharing a node with
#     the same value silently bind-collide.
VLLM_KV_CONNECTOR="${VLLM_KV_CONNECTOR:-NixlConnector}"

eng_pd_worker_cmd() {   # $1=index into the placement arrays
    local i="$1" role="${PD_ROLE[$1]}" tp="${PD_TP[$1]}"
    # vLLM's own vocabulary for the two halves.
    local kvrole; [[ "$role" == prefill ]] && kvrole=kv_producer || kvrole=kv_consumer
    local -a cmd=(
        "$(eng_python)" -m vllm.entrypoints.openai.api_server
        --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --trust-remote-code
        --max-model-len "$CONTEXT_LEN" --tensor-parallel-size "$tp"
        --gpu-memory-utilization "$MEM_FRACTION"
        --kv-transfer-config "{\"kv_connector\":\"$VLLM_KV_CONNECTOR\",\"kv_role\":\"$kvrole\",\"kv_buffer_device\":\"cuda\",\"engine_id\":\"$(pd_worker_name "$i")\"}"
    )
    # The same knobs the solo path takes, so a variant that wins at four GPUs carries
    # over to the 16-GPU PD run instead of being re-derived there at 128 SU an hour.
    [[ -n "${KV_CACHE_DTYPE:-}" ]]     && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "${CHUNKED_PREFILL:-}" ]]    && cmd+=(--max-num-batched-tokens "$CHUNKED_PREFILL")
    [[ "${DISABLE_PREFIX_CACHE:-0}" == 1 ]] && cmd+=(--no-enable-prefix-caching)
    [[ "${DISABLE_CHUNKED_PREFILL:-0}" == 1 ]] && cmd+=(--no-enable-chunked-prefill)
    if [[ "$role" == decode ]]; then
        cmd+=(--max-num-seqs "${MAX_RUNNING:-256}")
        [[ -n "${DECODE_EP:-}" ]] && cmd+=(--enable-expert-parallel)
    fi
    local moe; moe="$(vllm_moe_backend "$role")"
    [[ -n "$moe" ]] && cmd+=(--moe-backend "$moe")
    mapfile -t -O "${#cmd[@]}" cmd < <(vllm_spec_args)
    cmd+=(--host 0.0.0.0 --port "${PD_PORT[$i]}")
    printf '%s\n' "${cmd[@]}"
}

# The side-channel port has to differ per worker, and pd_write_launcher generates ONE
# launcher shared by all of them -- so it cannot be computed here. It is emitted as an
# expression instead, evaluated inside the launcher where $port is in scope. Two
# workers on one node therefore get different values without this file knowing which
# workers those are. 5600+ keeps it clear of the 30000+ range the workers serve on.
eng_pd_env() {
    eng_venv_env
    # NOT 0.0.0.0. This value is both what the side channel BINDS to and what the
    # worker ADVERTISES to its peer, and job 215686 died on the second meaning:
    #
    #   NIXL transfer failure: handshake_failed | remote_host: '0.0.0.0',
    #   remote_port: 5701, num_remote_blocks: 251
    #
    # The decode worker was handed 0.0.0.0 as the address to fetch KV from, which is
    # not routable from another host, so it waited and the prefill side logged
    # "Releasing expired KV blocks ... retrieved by 0 remote worker(s)". Both legs
    # returned HTTP 200 and the stream carried no tokens, which AIPerf reported as
    # "No responses with actual content were received".
    #
    # Resolved on the node, at launch, because one launcher serves every worker.
    # getent ahostsv4 and not `hostname -i`: a compute node resolving its own name
    # can answer with a link-local IPv6 (the teammate lost a 16-GPU job to exactly
    # that), and `hostname -I` lists docker and libvirt bridges too.
    echo 'VLLM_NIXL_SIDE_CHANNEL_HOST=$(getent ahostsv4 "$(hostname -f)" | head -1 | cut -d" " -f1)'
    echo 'VLLM_NIXL_SIDE_CHANNEL_PORT=$(( 5600 + port % 1000 ))'
    # A TUNED TRITON MoE CONFIG, WITHOUT TOUCHING THE VENV. fused_moe.py checks
    # VLLM_TUNED_CONFIG_FOLDER before its own configs directory and logs which file
    # it used, so this is both reversible and verifiable from the engine's log -- and
    # two cells of one campaign can differ on it, which is the only way to A/B a
    # kernel config without two allocations.
    #
    # The file must be named exactly as get_config_file_name() builds it:
    #   E=<experts>,N=<intermediate/TP>,device_name=<GPU>,dtype=...,block_shape=... .json
    # For this checkpoint at TP4 that is E=128,N=384 on NVIDIA_H100_80GB_HBM3.
    [[ -n "${VLLM_TUNED_CONFIG_FOLDER:-}" ]] \
        && echo "VLLM_TUNED_CONFIG_FOLDER=$VLLM_TUNED_CONFIG_FOLDER"
    return 0
}

eng_pd_gateway_cmd() {   # $1=port
    local i
    local -a cmd=("$(eng_python)" "$ENG_DIR/vllm-pd-proxy.py")
    for i in "${!PD_ROLE[@]}"; do
        # host:port, not a URL -- the proxy builds its own.
        if [[ "${PD_ROLE[$i]}" == prefill ]]; then
            cmd+=(--prefill "${PD_IP[$i]}:${PD_PORT[$i]}")
        else
            cmd+=(--decode "${PD_IP[$i]}:${PD_PORT[$i]}")
        fi
    done
    cmd+=(--host 0.0.0.0 --port "$1")
    printf '%s\n' "${cmd[@]}"
}
