# Inkling 2 Mini on two 22 GiB RTX 2080 Ti GPUs

This is a tested Turing inference configuration for Thinking Machines Lab’s original NVFP4/MXFP8 checkpoint and DSpark draft head. The two GPUs process each layer in tensor parallel across NVLink. Both profiles run one request at a time with the three-pass decoder. The memory budgets assume the **22 GiB modified cards**.

| Profile | Capacity including output | Measured generation | Cold time to first token |
|---|---:|---:|---:|
| Text, original DSpark K5 | 524,288 | 81.36 tokens/s | 242.76 s at 524,032 input tokens |
| Text, drafter disabled | 524,288 | 35.39 tokens/s | 241.07 s on the paired prompt |
| Image encoder on both GPUs, K5 | 458,752 | 80.89 tokens/s | 197.29 s at 458,496 input tokens including a photo |

The 512K text pair uses temperature 1, high reasoning and seed 1729. Cold means a resident service with zero prefix-cache hits. At 128K, K5 reached 114.77 tokens/s on the paired seed and 136.32 tokens/s across three seeds (median). The vision row uses a different prompt. [Per-trial measurements and chart regeneration](benchmarks.md)

## Run and verify

Follow the [pinned environment and build instructions](build.md), then follow the [checkpoint download, validation, packing and serving commands](run.md). From the project root, run one of the following commands in the prepared environment:

```bash
scripts/serve.sh text-512k
# For the serial control, stop the text server and start it again:
scripts/serve.sh text-512k --no-drafter
# For a photo, switch to this profile:
scripts/serve.sh vision-448k
```

The default endpoint is `http://127.0.0.1:8000`. In another terminal:

```bash
make check
python3 tests/smoke.py --cache                 # text profile
python3 tests/smoke.py --cache --vision        # vision profile
python3 benchmarks/run_text.py --model inkling-2-mini-3loop --tokens 32768 --seed 1729
```

The smoke test checks an arithmetic answer, a real prefix hit on a repeated request and, in vision mode, OCR on the published photo. A cold result for a running service excludes initial compilation and checkpoint loading. The historical dataset includes experiments on sparse verification, KV compression and cache-boundary correctness, plus examples where the model answered incorrectly at long context. [Experiments](experiments.md) · [Numerical evidence](experiments-data.json) · [Observed smoke-test output](../tests/observed-smoke.json)

A separate clean build from this recipe ran the public 458,496-token image prompt on the same two cards. Both responses read `COFFEE Q7M42`. Engine TTFT was 201.23 seconds cold and 3.11 seconds on repeat, with 456,704 cached input tokens. Each trial generated 46 output tokens. In the 32K text replay, median decode reached 182.40 tokens/s across three seeds, compared with the historical 182.46. [Clean-build vision trial](../tests/clean-install-448k.json) · [Rebuilt 32K text data](../tests/rebuilt-text-32k.json)
