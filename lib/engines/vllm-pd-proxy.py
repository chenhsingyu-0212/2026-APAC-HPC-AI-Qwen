#!/usr/bin/env python3
"""The gateway vLLM does not ship.

SGLang has sglang_router and TensorRT-LLM has `trtllm-serve disaggregated`. vLLM has
a .jpg of the workflow in its wheel and nothing else, so PD on vLLM needs this.

THE HANDSHAKE, from vllm/distributed/kv_transfer/kv_connector/v1/nixl/ rather than
from memory:

  1. POST the request to a PREFILL worker with max_tokens=1 and
     kv_transfer_params={"do_remote_decode": True}. It prefills, keeps the KV, and
     returns kv_transfer_params describing where that KV lives -- remote_block_ids,
     remote_engine_id, remote_host, remote_port, remote_tp_size and friends.
  2. POST the original request to a DECODE worker with those same fields plus
     {"do_remote_prefill": True}. It pulls the KV over NIXL and decodes.

WHY THIS STREAMS, and why that is not a detail. aiperf derives ITL from the gaps
between SSE chunks. A proxy that buffers the decode response and forwards it whole
reports its own latency, not the engine's: TTFT would absorb the entire generation and
every ITL would collapse to near zero. The scored metric would be measuring this file.
So step 2 is forwarded chunk by chunk, unbuffered, and the prefill step -- which is
never streamed, being one token -- is the only place a full read happens.

Round-robin over each pool, because a PD spec has several workers of each role and
pinning every request to worker 0 would measure one GPU group out of sixteen.
"""

import argparse
import asyncio
import itertools
import json
import os
import sys

try:
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
except ImportError as e:      # a missing dep here is a deployment error, not a bug
    sys.stderr.write(f"x vllm-pd-proxy needs fastapi/uvicorn/httpx from the vLLM venv: {e}\n")
    raise

ap = argparse.ArgumentParser()
ap.add_argument("--prefill", action="append", default=[], metavar="HOST:PORT")
ap.add_argument("--decode",  action="append", default=[], metavar="HOST:PORT")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="0.0.0.0")
# The prefill step is one forward pass over ISL tokens; the decode step is the whole
# generation. One timeout for both would either cut prefill's tail off or let a wedged
# decode hold a connection for the rest of the run.
ap.add_argument("--prefill-timeout", type=float, default=600.0)
ap.add_argument("--decode-timeout",  type=float, default=3600.0)
args = ap.parse_args()

if not args.prefill or not args.decode:
    sys.exit("x need at least one --prefill and one --decode")

PREFILL = itertools.cycle(args.prefill)
DECODE  = itertools.cycle(args.decode)
_lock = asyncio.Lock()

app = FastAPI()
# One client for the whole process. Limits sized above the concurrency we sweep to
# (48) with headroom, because a proxy that queues requests internally would show up as
# TTFT and be read as the engine being slow.
_limits = httpx.Limits(max_connections=512, max_keepalive_connections=256)
client = httpx.AsyncClient(limits=_limits, timeout=None)


async def _next(pool):
    async with _lock:
        return next(pool)


@app.get("/health")
async def health():
    return {"status": "ok", "prefill": args.prefill, "decode": args.decode}


@app.get("/v1/models")
async def models():
    # aiperf asks the endpoint what it serves. Answer from a decode worker, which is
    # the one that produces tokens.
    d = await _next(DECODE)
    r = await client.get(f"http://{d}/v1/models", timeout=30.0)
    return JSONResponse(status_code=r.status_code, content=r.json())


# THE PREFILL LEG EMITS EXACTLY ONE TOKEN, so every field that asks for more than
# one -- or that only makes sense while streaming -- has to be REMOVED, not
# overridden. Two jobs were lost one field at a time before this was written as a
# rule instead of a list:
#
#   215492  stream=False with aiperf's stream_options still beside it. vLLM 0.30
#           raises "Stream options can only be defined when `stream=True`" from a
#           Pydantic validator (chat_completion/protocol.py:799), which logs a bare
#           "400 Bad Request" with no reason. 25 minutes, 110,306 requests.
#   215674  min_tokens=16 against our max_tokens=1: "min_tokens must be less than or
#           equal to max_tokens=1, got 16."
#
# max_completion_tokens is in the list for a different reason: the chat API prefers
# it over max_tokens, so leaving it at 200 makes the prefill leg generate the whole
# answer. The teammate's job 18762 shipped exactly that and reported it as PD.
#
# n and best_of are there because more than one sequence has no single KV handover.
_PREFILL_DROP = (
    "stream_options",          # only valid with stream=True
    "min_tokens",              # would exceed max_tokens=1
    "min_completion_tokens",   # the chat-API spelling of the same thing
    "max_completion_tokens",   # preferred over max_tokens, so it would win
    "n",
    "best_of",
)


def prefill_body(body: dict) -> dict:
    """The prefill leg: compute the prompt, emit one token, keep the KV."""
    pbody = dict(body)
    pbody["stream"] = False
    pbody["max_tokens"] = 1
    for k in _PREFILL_DROP:
        pbody.pop(k, None)
    pbody["kv_transfer_params"] = {"do_remote_decode": True}
    return pbody


def decode_body(body: dict, ktp: dict) -> dict:
    """The decode leg: the caller's request as it stood, plus where the KV lives."""
    dbody = dict(body)
    dbody["kv_transfer_params"] = {**ktp, "do_remote_prefill": True}
    return dbody


async def _handle(path: str, body: dict):
    pre = await _next(PREFILL)
    dec = await _next(DECODE)

    # --- step 1: prefill, one token, never streamed
    pbody = prefill_body(body)
    pr = await client.post(f"http://{pre}{path}", json=pbody,
                           timeout=args.prefill_timeout)
    if pr.status_code != 200:
        # LOG IT, do not only return it. 215492 returned this detail faithfully and
        # it still took two jobs to read, because the only place it landed was
        # aiperf's per-request records -- gateway.log held nothing but the startup
        # line. One line here would have named both bugs on the first run.
        print(f"  prefill {pre} -> {pr.status_code}: {pr.text[:300]}", flush=True)
        return JSONResponse(status_code=pr.status_code,
                            content={"error": "prefill failed",
                                     "detail": pr.text[:400], "worker": pre})
    ktp = pr.json().get("kv_transfer_params")
    if not ktp:
        # The worker answered but did not hand over any KV. Saying so is better than
        # letting decode recompute the prompt and reporting the result as PD.
        return JSONResponse(status_code=502, content={
            "error": "prefill returned no kv_transfer_params",
            "hint": "is this worker running with kv_role=kv_producer?",
            "worker": pre})

    # --- step 2: decode, streamed straight through
    dbody = decode_body(body, ktp)
    if not dbody.get("stream"):
        dr = await client.post(f"http://{dec}{path}", json=dbody,
                               timeout=args.decode_timeout)
        return JSONResponse(status_code=dr.status_code, content=dr.json())

    async def relay():
        async with client.stream("POST", f"http://{dec}{path}", json=dbody,
                                 timeout=args.decode_timeout) as dr:
            if dr.status_code != 200:
                detail = (await dr.aread())[:400]
                yield b"data: " + json.dumps(
                    {"error": "decode failed", "detail": detail.decode("utf-8", "replace")}
                ).encode() + b"\n\n"
                return
            # aiter_raw, not aiter_lines: no re-chunking, no buffering, so the gaps
            # the client measures are the gaps the engine produced.
            async for chunk in dr.aiter_raw():
                yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/v1/completions", await request.json())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/v1/chat/completions", await request.json())


if __name__ == "__main__":
    print(f"  vllm-pd-proxy  prefill={args.prefill}  decode={args.decode}"
          f"  listening on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port,
                log_level=os.environ.get("PROXY_LOG_LEVEL", "warning"),
                access_log=False)
