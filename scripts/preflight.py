"""Check the selected local engine, original model structure, dense cache and CUDA pair."""
import importlib.util
import json
from pathlib import Path
import sys


def main(source, model, dense):
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin or Path(spec.origin).resolve() != source / "vllm" / "__init__.py":
        raise SystemExit(f"Prepare the patched source in {source} and use the matching virtual environment.")
    for name in ("model-00001-of-00001.safetensors", "mtp.safetensors", "tiktoken/tokenizer.model"):
        if not (model / name).is_file():
            raise SystemExit(f"Missing {name} in {model}. Run scripts/fetch-weights.sh.")
    config = json.loads((model / "config.json").read_text())
    if config["text_config"]["loop_config"]["num_passes"] != 3:
        raise SystemExit("The recipe requires the original three-pass checkpoint.")
    if not (dense / "archive-sha256.json").is_file():
        raise SystemExit(f"Run scripts/pack-dense.py to prepare {dense}.")
    import torch
    if torch.cuda.device_count() != 2:
        raise SystemExit("Select two visible CUDA devices.")
    for rank in (0, 1):
        prop = torch.cuda.get_device_properties(rank)
        if (prop.major, prop.minor) != (7, 5) or prop.total_memory < 21 * 1024 ** 3:
            raise SystemExit(f"Rank {rank}: expected an SM75 22 GiB GPU, saw {prop.name} ({prop.total_memory} bytes).")
    if not all(torch.cuda.can_device_access_peer(a, b) for a, b in ((0, 1), (1, 0))):
        raise SystemExit("Peer access must work in both directions for TP2 over NVLink.")


if __name__ == "__main__":
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))
