#!/bin/bash
# Speak to whichever AIPerf is installed.
#
# The benchmark client has two incompatible command-line dialects, and which one
# you get depends on the machine rather than on anything you chose:
#
#   OLD (the dialect the task rules are written in, and what the organisers'
#        aiperf container on Firmus speaks)
#       --synthetic-input-tokens-mean --synthetic-input-tokens-stddev
#       --output-tokens-mean --output-tokens-stddev
#       --model --endpoint --ui --cache-bust
#
#   NEW (aiperf 0.11.0 from PyPI, which is what NYCU can install)
#       --isl --isl-stddev --osl --osl-stddev
#       --model-names --custom-endpoint --ui-type       and NO --cache-bust
#
# A wrong flag is not a soft failure. AIPerf exits 1 in about one second with
# "Unknown option: --cache-bust. Did you mean --search-stat?", which on job 79417
# happened at every concurrency of every engine -- after SGLang had spent 395
# seconds coming up healthy. Six points, three cells, nothing measured, entirely
# because of the client's spelling.
#
# So this file asks the installed binary what it accepts and builds the argv from
# that. run/AIPerf/official.sh is deliberately NOT routed through here: the scored
# command is fixed text that has to match the rules verbatim, and translating it
# would be exactly the wrong kind of clever.
#
#   client_init "$AIPERF_BIN"        once, before building any command
#   client_has --cache-bust          is that flag accepted
#   mapfile -t argv < <(client_profile_args ...)
#
# CACHE BUSTING. `--cache-bust first_turn_prefix` exists so prefix caching cannot
# serve part of a request that the benchmark meant to measure in full. The new
# dialect has no equivalent, so when it is missing client_init sets
# CLIENT_NEEDS_SERVER_PREFIX_OFF=1 and the caller disables prefix reuse in the
# ENGINE instead -- --disable-radix-cache, --no-enable-prefix-caching, and
# enable_block_reuse: false. That reaches the same state from the other side, and
# it is visible in the server's own log, which the client flag never was.

CLIENT_FLAGS=""          # newline-separated list of flags the binary accepts
CLIENT_DIALECT=""        # old | new
CLIENT_NEEDS_SERVER_PREFIX_OFF=0

client_init() {   # $1=path to the aiperf binary
    local bin="${1:-aiperf}"
    CLIENT_BIN="$bin"
    CLIENT_FLAGS="$("$bin" profile --help 2>&1 \
        | tr -d '│|' | grep -oE '(^|[[:space:]])--[a-z0-9-]+' | tr -d ' ' | sort -u)"
    if [[ -z "$CLIENT_FLAGS" ]]; then
        echo "x cannot read '$bin profile --help' -- is the client installed?" >&2
        return 1
    fi
    if client_has --synthetic-input-tokens-mean; then CLIENT_DIALECT=old
    elif client_has --isl;                       then CLIENT_DIALECT=new
    else
        echo "x '$bin' speaks neither dialect: no --synthetic-input-tokens-mean, no --isl" >&2
        return 1
    fi
    client_has --cache-bust || CLIENT_NEEDS_SERVER_PREFIX_OFF=1
    return 0
}

client_has() { grep -qx -- "$1" <<< "$CLIENT_FLAGS"; }

# Emit a flag only if this binary accepts it. Used for the handful that are
# genuinely optional -- a missing --goodput changes what is reported, a missing
# --workers-max does not.
client_opt() {   # $1=flag  $2..=values
    client_has "$1" || return 0
    printf '%s\n' "$@"
}

client_banner() {
    echo " client:   $CLIENT_BIN"
    echo " dialect:  $CLIENT_DIALECT   ($(wc -l <<< "$CLIENT_FLAGS") flags)"
    if (( CLIENT_NEEDS_SERVER_PREFIX_OFF )); then
        echo " !  no --cache-bust: disabling prefix reuse in the engine instead"
    fi
}

# The measurement command, one argument per line for `mapfile -t`.
#   $1 artifact dir   $2 gateway host:port   $3 model id      $4 concurrency
# and from the environment: ISL OSL SLA WARMUP_SECONDS BENCH_SECONDS
# GRACE_SECONDS DATASET_ENTRIES SEED.
client_profile_args() {
    local dir="$1" gw="$2" mid="$3" n="$4"
    local isl="${ISL:-4000}" osl="${OSL:-200}"

    printf '%s\n' profile --artifact-dir "$dir" --url "http://$gw" --endpoint-type chat
    if [[ "$CLIENT_DIALECT" == old ]]; then
        printf '%s\n' --endpoint /v1/chat/completions --model "$mid" --ui simple
        printf '%s\n' --synthetic-input-tokens-mean "$isl" --synthetic-input-tokens-stddev 0
        printf '%s\n' --output-tokens-mean "$osl" --output-tokens-stddev 0
        # Every request gets a different first-turn prefix, so prefix caching
        # cannot answer part of it.
        printf '%s\n' --cache-bust first_turn_prefix
    else
        printf '%s\n' --custom-endpoint /v1/chat/completions --model-names "$mid" --ui-type simple
        printf '%s\n' --isl "$isl" --isl-stddev 0
        printf '%s\n' --osl "$osl" --osl-stddev 0
    fi

    printf '%s\n' --tokenizer "${TOKENIZER:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
    printf '%s\n' --streaming --header 'Accept: text/event-stream' --api-key NOT_USED
    printf '%s\n' --concurrency "$n"
    client_opt --workers-max "$n"
    printf '%s\n' --warmup-duration "${WARMUP_SECONDS:-20}" \
                  --benchmark-duration "${BENCH_SECONDS:-20}" \
                  --benchmark-grace-period "${GRACE_SECONDS:-20}"
    # WHICH FIELD NAMES THE OUTPUT LENGTH. aiperf 0.11.0 sends the current OpenAI
    # spelling, max_completion_tokens. TensorRT-LLM 1.2.1's request schema forbids
    # unknown fields outright, so every request comes back
    #
    #   400 {'type': 'extra_forbidden', 'loc': ('body','max_completion_tokens'),
    #        'msg': 'Extra inputs are not permitted', 'input': 200}
    #
    # -- job 79593 served 52,410 of those with a healthy engine behind them.
    # --use-legacy-max-tokens switches to max_tokens, which every engine here
    # accepts. It renames a field and changes no value, so it does not affect
    # comparability; and because --osl then maps onto max_tokens itself, the
    # explicit max_tokens extra-input would be sent twice, so it is dropped.
    local legacy=0
    client_has --use-legacy-max-tokens && { printf '%s\n' --use-legacy-max-tokens; legacy=1; }

    # min_tokens pins the floor so OSL is not a distribution, and ignore_eos stops
    # the model ending early -- both are what makes goodput comparable run to run.
    printf '%s\n' --extra-inputs "min_tokens:$osl"
    (( legacy )) || printf '%s\n' --extra-inputs "max_tokens:$osl"
    printf '%s\n' --extra-inputs ignore_eos:true --extra-inputs temperature:0.0
    client_opt --num-dataset-entries "${DATASET_ENTRIES:-2000}"
    client_opt --random-seed "${SEED:-2026}"
    # ONE argument, not two. --goodput takes space-separated KEY:VALUE pairs
    # inside a single token; unquoted, the second pair becomes a stray positional
    # and AIPerf exits 1 with "Unused Tokens: ['inter_token_latency:20']" -- which
    # it did at every point of job 79449/79450, after the engine was already up.
    client_opt --goodput "${SLA:-time_to_first_token:3000 inter_token_latency:20}"
}

# A short unmeasured pass, to pay the first-request cost with the client we
# measure with rather than with curl -- job 17914 spent 300 seconds proving that
# a curl prewarm can hang while the real client does not.
client_prewarm_args() {   # $1 artifact dir  $2 gateway  $3 model id  $4 requests  [$5 concurrency]
    # CONCURRENCY IS NOT THE REQUEST COUNT. Sending n requests at concurrency n warms n
    # ROUTES, and with both router policies on round_robin and advancing together, four
    # requests visit p0->d0, p1->d1, p0->d0, p1->d1 -- two of the four prefill-decode
    # pairs, twice each, and never the cross pairs the router uses in steady state.
    # Separating the two lets a burst open every pair at once and a sequential tail then
    # give each a warm second pass, which is what the reference H200 campaign does.
    local dir="$1" gw="$2" mid="$3" n="$4" conc="${5:-$4}"
    local isl="${ISL:-4000}"
    printf '%s\n' profile --artifact-dir "$dir" --url "http://$gw" --endpoint-type chat
    if [[ "$CLIENT_DIALECT" == old ]]; then
        printf '%s\n' --endpoint /v1/chat/completions --model "$mid" --ui simple
        printf '%s\n' --synthetic-input-tokens-mean "$isl" --synthetic-input-tokens-stddev 0
        printf '%s\n' --output-tokens-mean 16 --output-tokens-stddev 0
    else
        printf '%s\n' --custom-endpoint /v1/chat/completions --model-names "$mid" --ui-type simple
        printf '%s\n' --isl "$isl" --isl-stddev 0 --osl 16 --osl-stddev 0
    fi
    printf '%s\n' --tokenizer "${TOKENIZER:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
    printf '%s\n' --streaming --header 'Accept: text/event-stream' --api-key NOT_USED
    printf '%s\n' --concurrency "$conc" --request-count "$n"
    local legacy=0
    client_has --use-legacy-max-tokens && { printf '%s\n' --use-legacy-max-tokens; legacy=1; }
    printf '%s\n' --extra-inputs min_tokens:16
    (( legacy )) || printf '%s\n' --extra-inputs max_tokens:16
    printf '%s\n' --extra-inputs ignore_eos:true
}
