"""Replay one cold text benchmark directly against the patched vLLM API."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4


QUESTION = ("By how much does ALPHA plus BRAVO exceed CHARLIE? "
            "Check the result with the inverse operation and give the final number.")
FILLER = "".join(
    f"Entry {i} records a routine warehouse inspection. The reviewer verified the label and date.\n"
    for i in range(32)
)


def token_prompt(post, model, length):
    def tokens(**fields):
        with post("/tokenize", {"model": model, **fields}) as response:
            return json.load(response)["tokens"]

    marker = "PLACEHOLDER"
    template = tokens(messages=[{"role": "user", "content": marker + "\n" + QUESTION}],
                      chat_template_kwargs={"reasoning_effort": "high"})
    marker_ids = tokens(prompt=marker, add_special_tokens=False)
    anchors = [i for i in range(len(template)) if template[i:i + len(marker_ids)] == marker_ids]
    if len(anchors) != 1:
        raise ValueError("The benchmark requires the Inkling high-reasoning chat template")
    offset = anchors[0]
    header, footer = template[:offset], template[offset + len(marker_ids):]
    notes = [tokens(prompt=f"\nConfirmed balance of {key} is {value} units.\n", add_special_tokens=False)
             for key, value in (("ALPHA", 7319), ("BRAVO", 2864), ("CHARLIE", 9541))]
    filler = tokens(prompt=FILLER, add_special_tokens=False)
    remaining = length - len(header) - len(footer) - sum(map(len, notes))
    if remaining < 0 or not filler:
        raise ValueError("Prompt is too short or filler tokenization failed")

    def pad(n):
        repetitions, extra = divmod(n, len(filler))
        return filler * repetitions + filler[:extra]

    half = remaining // 2
    return header + notes[0] + pad(half) + notes[1] + pad(remaining - half) + notes[2] + footer


def measure(post, model, tokens, seed, max_tokens, endpoint):
    request = {
        "model": model, "prompt": tokens, "stream": True, "return_token_ids": True,
        "stream_options": {"include_usage": True}, "seed": seed, "temperature": 1.0,
        "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "cache_salt": uuid4().hex, "max_tokens": max_tokens,
    }
    events = []
    usage = None
    finished = False
    started = time.monotonic()
    with post("/v1/completions", request) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                finished = True
                break
            frame = json.loads(payload)
            if frame.get("error"):
                raise RuntimeError(str(frame["error"]))
            for choice in frame.get("choices", []):
                ids = choice.get("token_ids")
                if ids:
                    events.append((time.monotonic() - started, len(ids)))
            if frame.get("usage"):
                usage = frame["usage"]

    emitted = sum(count for _, count in events)
    if not finished or not usage or not events or emitted != usage["completion_tokens"]:
        raise RuntimeError("Incomplete stream or missing vLLM token IDs/usage")
    hits = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if hits != 0 or usage["prompt_tokens"] != len(tokens):
        raise RuntimeError(f"Expected {len(tokens)} prompt tokens and zero prefix hits; check API and cache config")
    ttft, first = events[0]
    end_time = events[-1][0]
    seen = 0
    matched = None
    for stamp, count in events:
        seen += count
        if seen >= endpoint:
            if stamp > ttft and endpoint > first:
                matched = (endpoint - first) / (stamp - ttft)
            break
    return {
        "input_tokens": len(tokens), "seed": seed, "cached_tokens": hits,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        "first_chunk_tokens": first, "output_tokens": emitted,
        "matched_endpoint_tokens": endpoint, "matched_decode_tok_s": matched, "ttft_s": ttft,
        "prefill_tok_s": len(tokens) / ttft,
        "full_decode_tok_s": (emitted - first) / (end_time - ttft) if end_time > ttft else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM origin without /v1")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--tokens", type=int, default=32768, help="Exact formatted input length")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--endpoint", type=int, help="Last output token included in matched decode metric")
    parser.add_argument("--output", type=Path, help="Optional new result file")
    args = parser.parse_args()
    endpoint = args.endpoint if args.endpoint is not None else min(128 if args.tokens == 512 else 256, args.max_tokens)
    if not 1 < endpoint <= args.max_tokens or args.tokens <= 0:
        parser.error("Use a positive input length and 1 < endpoint <= max-tokens")
    if args.output and args.output.exists():
        parser.error("Select a new output file")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("INKLING_API_KEY") or os.environ.get("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    def post(route, body):
        req = Request(args.base_url.rstrip("/") + route, json.dumps(body).encode(), headers=headers)
        return urlopen(req, timeout=3600)

    tokens = token_prompt(post, args.model, args.tokens)
    row = measure(post, args.model, tokens, args.seed, args.max_tokens, endpoint)
    content = json.dumps(row, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(content)
    print(content, end="")


if __name__ == "__main__":
    main()
