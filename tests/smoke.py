"""End-to-end checks of a running Inkling OpenAI-compatible server; Python 3 stdlib."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "assets" / "fixtures" / "coffee-code.jpg"


def completion(base_url, model, content, salt):
    body = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "seed": 187, "max_tokens": 160, "cache_salt": salt,
        "chat_template_kwargs": {"reasoning_effort": "low"}, "thinking_token_budget": 32,
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("INKLING_API_KEY") or os.environ.get("VLLM_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = Request(base_url.rstrip("/") + "/v1/chat/completions",
                  data=json.dumps(body).encode(), headers=headers)
    start = time.monotonic()
    try:
        with urlopen(req, timeout=900) as r:
            result = json.load(r)
    except HTTPError as error:
        raise RuntimeError(f"Chat Completions returned HTTP {error.code}") from error
    choice = result["choices"][0]
    answer = choice["message"].get("content") or ""
    return {"answer": answer, "usage": result["usage"], "finish": choice["finish_reason"],
            "wall_s": round(time.monotonic() - start, 3)}


def validate_usage(usage):
    prompt, out = usage["prompt_tokens"], usage["completion_tokens"]
    if prompt <= 0 or out <= 0 or usage["total_tokens"] != prompt + out:
        raise ValueError("Invalid input/output/total counts")
    details = usage.get("prompt_tokens_details") or {}
    read = details.get("cached_tokens")
    if read is None or not 0 <= read <= prompt:
        raise ValueError("Missing or inconsistent cached_tokens")
    return read


def run(base_url, model, image, cache):
    question = "Return just the numeric result of 19 plus 23."
    if cache:
        question = ("Warehouse log. Item 84 was counted and signed at the end of the shift.\n" * 420
                    + "\n" + question)
    salt = uuid4().hex
    rows = []
    for phase in (["cold", "repeat"] if cache else ["cold"]):
        row = completion(base_url, model, question, salt)
        read = validate_usage(row["usage"])
        if (phase == "cold" and read != 0) or (phase == "repeat" and read == 0):
            raise ValueError(f"{phase}: unexpected prefix hit of {read} tokens")
        if phase == "repeat" and row["usage"]["prompt_tokens"] != rows[0]["usage"]["prompt_tokens"]:
            raise ValueError("Input token count changed on an identical request")
        if not re.search(r"\b42\b", row["answer"]):
            raise ValueError(f"{phase}: expected 42, got {row['answer']!r}")
        rows.append({"case": phase, **row})
    if image:
        url = "data:image/jpeg;base64," + base64.b64encode(PHOTO.read_bytes()).decode()
        msg = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text":
                "On the first line give the main subject in one word (ASTRONAUT or COFFEE), "
                "a space, and the five-character code printed in the white band."}]
        row = completion(base_url, model, msg, uuid4().hex)
        if validate_usage(row["usage"]) != 0:
            raise ValueError("Expected cold image input")
        image_tokens = (row["usage"]["prompt_tokens_details"].get("multimodal_tokens") or {}).get("image")
        if image_tokens != 521 or not re.search(r"\bCOFFEE\b\W+Q7M42\b", row["answer"], re.I):
            raise ValueError(f"OCR or image token count differs: {row['answer']!r}, {image_tokens}")
        rows.append({"case": "image-ocr", **row})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", help="Defaults to the served name for the selected profile")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    model = args.model or ("inkling-2-mini-3loop-vision-gpu-448k" if args.vision else "inkling-2-mini-3loop")
    try:
        rows = run(args.base_url, model, args.vision, args.cache)
    except (ValueError, OSError, RuntimeError, KeyError, IndexError) as error:
        sys.exit(str(error))
    print(json.dumps({"model": model, "cases": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
