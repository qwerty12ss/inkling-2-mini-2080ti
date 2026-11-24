"""Measure a cold image prompt and an identical prefix-cache repeat on the 448K profile."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from uuid import uuid4


PHOTO = Path(__file__).resolve().parents[1] / "assets/fixtures/coffee-code.jpg"
WINDOW = 458752
HEADER = "Inventory notes. A separate photograph follows the ledger.\n"
BLOCK = "".join(f"Shelf {i:02d}: the label and the quantity were checked during inventory.\n" for i in range(32))
TASK = ("On the first line give the main subject of the photograph in one word (ASTRONAUT or COFFEE), "
        "a space, and the five-character code printed in the white band.")
TIMERS = {"ttft": "time_to_first_token_seconds", "prefill": "request_prefill_time_seconds",
          "decode": "request_decode_time_seconds"}
COUNTS = [*TIMERS.values(), "request_generation_tokens"]
TEMPLATE = {"reasoning_effort": "low"}


def calibrate(http, model, image_url, target, max_tokens):
    def messages(blocks, padding):
        content = [{"type": "text", "text": HEADER + BLOCK * blocks + "End of ledger." + " check" * padding},
                   {"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": TASK}]
        return [{"role": "user", "content": content}]

    def count(prompt):
        payload = {"model": model, "messages": prompt, "chat_template_kwargs": TEMPLATE}
        with http("/tokenize", payload) as r:
            result = json.load(r)
        if result["max_model_len"] != WINDOW or target + max_tokens > result["max_model_len"]:
            raise ValueError("Select the 448K GPU vision profile and reserve room for the completion")
        return result["count"]

    base = count(messages(0, 0))
    block_size = count(messages(1, 0)) - base
    if target < base or block_size < 1:
        raise ValueError(f"Target too small for the photo and instructions ({base} tokens) or tokenizer mismatch")
    blocks, padding = divmod(target - base, block_size)
    for _ in range(4):
        prompt = messages(blocks, padding)
        actual = count(prompt)
        if actual == target:
            return prompt
        padding += target - actual
        if padding < 0 and blocks:
            blocks -= 1
            padding += block_size
        if padding < 0:
            break
    raise RuntimeError(f"Multimodal tokenizer produced {actual} tokens, target {target}")


def metrics(http):
    names = {f"{name}_{field}" for name in COUNTS for field in ("count", "sum")}
    names.update(("num_requests_running", "num_requests_waiting"))
    values = {}
    with http("/metrics") as response:
        for line in response:
            sample = line.decode().split()
            if len(sample) < 2 or not sample[0].startswith("vllm:"):
                continue
            key = sample[0][5:].partition("{")[0]
            if key in names:
                values[key] = values.get(key, 0.0) + float(sample[1])
    if missing := names.difference(values):
        raise ValueError(f"Missing vLLM metrics: {sorted(missing)}")
    return values


def measure(http, request, target, phase):
    before = metrics(http)
    if before["num_requests_running"] or before["num_requests_waiting"]:
        raise RuntimeError("This timing test needs an idle server with no concurrent requests")
    started = time.monotonic()
    with http("/v1/chat/completions", request) as r:
        result = json.load(r)
    wall = time.monotonic() - started
    usage = result["usage"]
    output = usage["completion_tokens"]
    details = usage["prompt_tokens_details"]
    cached = details["cached_tokens"]
    image = details["multimodal_tokens"]["image"]
    if usage["prompt_tokens"] != target or image != 521 or output < 1 or usage["total_tokens"] != target + output:
        raise ValueError(f"Multimodal input, image or completion count differs: {usage}")
    if (phase == "cold" and cached != 0) or (phase == "repeat" and not 0 < cached <= target):
        raise ValueError(f"Unexpected cached_tokens={cached} in the {phase} request")

    for _ in range(30):
        after = metrics(http)
        if after["request_generation_tokens_count"] > before["request_generation_tokens_count"]:
            break
        time.sleep(0.2)
    if any(after[n + "_count"] - before[n + "_count"] != 1 for n in COUNTS):
        raise RuntimeError("Expected exactly one completed request in each latency and token histogram")
    if after["request_generation_tokens_sum"] - before["request_generation_tokens_sum"] != output:
        raise RuntimeError("Engine and API completion counts differ")
    elapsed = {kind: after[name + "_sum"] - before[name + "_sum"] for kind, name in TIMERS.items()}
    answer = result["choices"][0]["message"].get("content") or ""
    return {"phase": phase, "input_tokens": target, "output_tokens": output, "image_tokens": image,
            "cached_tokens": cached, "computed_tokens": target - cached, "engine_ttft_s": elapsed["ttft"],
            "engine_prefill_s": elapsed["prefill"], "engine_decode_s": elapsed["decode"], "client_wall_s": wall,
            "engine_decode_tok_s": (output - 1) / elapsed["decode"] if elapsed["decode"] > 0 and output > 1 else None,
            "answer": answer, "ocr_ok": bool(re.search(r"\bCOFFEE\b\W+Q7M42\b", answer, re.I))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Direct vLLM origin without /v1")
    parser.add_argument("--model", default="inkling-2-mini-3loop-vision-gpu-448k")
    parser.add_argument("--tokens", type=int, default=8192, help="Exact multimodal input, including image and chat framing")
    parser.add_argument("--max-tokens", type=int, default=256, help="Output ceiling including reasoning")
    parser.add_argument("--tokenize-only", action="store_true", help="Verify the exact multimodal length without generation")
    parser.add_argument("--output", type=Path, help="Optional new JSON file")
    args = parser.parse_args()
    if args.tokens < 4096 or args.max_tokens < 1 or args.tokens + args.max_tokens > WINDOW:
        parser.error("Set input >= 4096 for the cache repeat; input + max-tokens must fit the 458752-token window")
    if args.output and args.output.exists():
        parser.error("Select a new output file")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("INKLING_API_KEY") or os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + key

    def http(route, body=None):
        data = None if body is None else json.dumps(body).encode()
        return urlopen(Request(args.base_url.rstrip("/") + route, data=data, headers=headers), timeout=1800)

    photo = PHOTO.read_bytes()
    url = "data:image/jpeg;base64," + base64.b64encode(photo).decode()
    messages = calibrate(http, args.model, url, args.tokens, args.max_tokens)
    result = {"model": args.model, "input": "synthetic-inventory-and-coffee-v1",
              "image_sha256": hashlib.sha256(photo).hexdigest(), "input_tokens": args.tokens,
              "max_tokens": args.max_tokens, "tokenize_only": args.tokenize_only}
    rows = []
    if not args.tokenize_only:
        payload = {"model": args.model, "messages": messages, "chat_template_kwargs": TEMPLATE,
                   "max_tokens": args.max_tokens, "thinking_token_budget": 32, "seed": 187, "temperature": 0,
                   "cache_salt": uuid4().hex}
        rows = [measure(http, payload, args.tokens, phase) for phase in ("cold", "repeat")]
        result["cases"] = rows
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(rendered)
    print(rendered, end="")
    if not all(row["ocr_ok"] for row in rows):
        raise SystemExit("OCR check failed; see the returned answers")


if __name__ == "__main__":
    main()
