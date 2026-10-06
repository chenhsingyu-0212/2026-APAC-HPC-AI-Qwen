#!/usr/bin/env python3
"""The two request bodies the vLLM PD proxy builds.

Job 215492 brought the deployment up correctly -- both prefill workers reached
"Application startup complete", NixlConnector registered, the proxy listened -- and
then every request came back 400 from the prefill leg, 110,306 of them, because the
proxy set stream=False and left aiperf's stream_options in the body. vLLM 0.30 raises
that from a Pydantic validator, so the server log shows a bare "400 Bad Request" with
no reason and the cause is invisible from the engine side.

These are pure functions, so the test reads them out of the source with ast and runs
them. No fastapi, no httpx, no GPU: it would have caught 215492 on a laptop.
"""
import ast
import pathlib
import sys

SRC = pathlib.Path(__file__).with_name("engines") / "vllm-pd-proxy.py"
tree = ast.parse(SRC.read_text())
want = {"prefill_body", "decode_body"}
# The two functions, plus the module-level literal constants they close over (the
# drop list). Taking them from the source rather than restating them here is the
# point: if the list changes, this test sees the change.
picked = [
    n for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name in want)
    or (isinstance(n, ast.Assign)
        and all(isinstance(t, ast.Name) for t in n.targets)
        and isinstance(n.value, (ast.Tuple, ast.List, ast.Constant, ast.Set)))
]
missing = want - {n.name for n in picked if isinstance(n, ast.FunctionDef)}
if missing:
    print("FAIL the proxy no longer defines %s" % ", ".join(sorted(missing)))
    sys.exit(1)
ns: dict = {}
exec(compile(ast.Module(body=picked, type_ignores=[]), str(SRC), "exec"), ns)
prefill_body, decode_body = ns["prefill_body"], ns["decode_body"]

# What aiperf actually sends. Both of the fields that broke a job are in here:
# stream_options (215492) and min_tokens (215674, "min_tokens must be less than or
# equal to max_tokens=1, got 16").
AIPERF = {
    "model": "qwen",
    "messages": [{"role": "user", "content": "x" * 10}],
    "stream": True,
    "stream_options": {"include_usage": True},
    "max_completion_tokens": 200,
    "min_tokens": 16,
    "temperature": 0.0,
    "ignore_eos": True,
}

fail = 0
def ck(name, got, want):
    global fail
    if got == want:
        print("ok   %s" % name)
    else:
        print("FAIL %s: got %r want %r" % (name, got, want))
        fail = 1

p = prefill_body(AIPERF)
ck("stream is off on the prefill leg", p["stream"], False)
ck("stream_options is GONE, not false", "stream_options" in p, False)
ck("max_tokens is 1", p["max_tokens"], 1)
ck("max_completion_tokens is gone", "max_completion_tokens" in p, False)
ck("min_tokens is gone", "min_tokens" in p, False)
ck("the handover is requested", p["kv_transfer_params"], {"do_remote_decode": True})
ck("the prompt survives", p["messages"], AIPERF["messages"])
ck("sampling survives", (p["temperature"], p["ignore_eos"]), (0.0, True))
ck("the caller's body is not mutated", AIPERF["stream"], True)
ck("and still has its options", "stream_options" in AIPERF, True)

# n/best_of would make the KV handover ambiguous.
p2 = prefill_body({**AIPERF, "n": 2, "best_of": 4})
ck("n is dropped", "n" in p2, False)
ck("best_of is dropped", "best_of" in p2, False)

KTP = {"remote_block_ids": [1, 2, 3], "remote_engine_id": "w0",
       "remote_host": "10.0.0.1", "remote_port": 5600}
d = decode_body(AIPERF, KTP)
ck("decode keeps streaming", d["stream"], True)
ck("decode keeps its stream_options", d["stream_options"], {"include_usage": True})
ck("decode keeps the full length", d["max_completion_tokens"], 200)
ck("decode is told where the KV is", d["kv_transfer_params"],
   {**KTP, "do_remote_prefill": True})
ck("decode body does not mutate the caller's", "kv_transfer_params" in AIPERF, False)

# The rule, not the list: nothing left in the prefill body may ask for more than the
# one token it is allowed to emit, and nothing may require streaming. A new aiperf
# release adding another length field should fail HERE rather than on a 16-GPU job.
LENGTH_FIELDS = ("min_tokens", "min_completion_tokens", "max_completion_tokens",
                 "n", "best_of")
noisy = {**AIPERF}
for k in LENGTH_FIELDS:
    noisy.setdefault(k, 7)
p3 = prefill_body(noisy)
ck("no length field survives", sorted(k for k in LENGTH_FIELDS if k in p3), [])
ck("max_tokens is the only length left", p3["max_tokens"], 1)
ck("nothing needing a stream survives", "stream_options" in p3, False)

# decode must keep every one of them, because it generates the real answer.
d3 = decode_body(noisy, KTP)
ck("decode keeps the length fields",
   sorted(k for k in LENGTH_FIELDS if k in d3), sorted(LENGTH_FIELDS))

sys.exit(fail)
