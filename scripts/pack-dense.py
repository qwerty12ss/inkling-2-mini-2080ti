#!/usr/bin/env python3
"""Pack the original dense projections for the TP2 SM75 kernels on CPU."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn.functional import pad


def digest(tensor):
    data = tensor.contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def fragments(array):
    rows, cols = array.shape
    return (
        array.reshape(rows // 16, 2, 8, cols // 16, 4, 2, 2)
        .transpose(0, 3, 2, 4, 5, 1, 6)
        .reshape(rows // 16, cols // 16, 32, 8)
        .copy()
    )


def pack(weight):
    if weight.dtype != torch.bfloat16 or weight.device.type != "cpu":
        raise ValueError("Dense packing requires the original BF16 matrix on CPU")
    half = weight.half()
    scaled = (half.float() * 8).half().view(torch.uint16).numpy()
    if not np.isfinite(scaled.view(np.float16)).all():
        raise ValueError("Scaled values exceed the FP16 range")

    # Preserve all unusual 16-bit tiles explicitly. Common tiles fit in 12 bits
    # after scaling. Lay them out by the lanes that feed the Turing MMA.
    words = fragments(scaled)
    exceptions = ((words & 0x4007) != 0).any(axis=(-1, -2))
    offsets = np.full(exceptions.shape, -1, dtype=np.int32)
    offsets[exceptions] = np.arange(exceptions.sum(), dtype=np.int32)
    flags = exceptions.any(axis=1).astype(np.uint8)
    escapes = np.ascontiguousarray(words[exceptions])
    if not len(escapes):
        escapes = np.zeros((1, 32, 8), dtype=np.uint16)
    codes = ((words & 0x3ff8) >> 3) | ((words & 0x8000) >> 4)
    low = np.ascontiguousarray(codes, dtype=np.uint8)
    high = codes.astype(np.uint32) >> 8
    upper = np.zeros(high.shape[:-1], dtype=np.uint32)
    recovered = low.astype(np.uint16)
    for pair in range(4):
        upper |= high[..., 2 * pair] << (4 * pair)
        upper |= high[..., 2 * pair + 1] << (16 + 4 * pair)
    for pair in range(4):
        recovered[..., 2 * pair] |= ((upper >> (4 * pair)) & 15).astype(np.uint16) << 8
        recovered[..., 2 * pair + 1] |= ((upper >> (16 + 4 * pair)) & 15).astype(np.uint16) << 8
    recovered = ((recovered & 0x7ff) << 3) | ((recovered & 0x800) << 4)
    recovered[exceptions] = escapes[offsets[exceptions]]
    if not np.array_equal(words, recovered):
        raise ValueError("MMA tile packing changed a bit")
    if not torch.equal((torch.from_numpy(scaled).view(torch.float16).float() * .125).half().view(torch.int16),
                       half.view(torch.int16)):
        raise ValueError("Scaled serving matrix differs from the original FP16 conversion")

    source_words = weight.contiguous().view(torch.uint16).numpy().reshape(-1)
    restored = half.bfloat16().view(torch.uint16).numpy().copy().reshape(-1)
    positions = np.flatnonzero(restored != source_words).astype(np.int32)
    original_bits = source_words[positions]
    restored[positions] = original_bits
    if not np.array_equal(source_words, restored):
        raise ValueError("BF16 recovery failed")
    return [torch.from_numpy(a) for a in (escapes, low, upper, offsets, flags)], [
        torch.from_numpy(positions), torch.from_numpy(original_bits)]


def matrices(source, layer, rank):
    prefix = f"model.llm.layers.{layer}.attn"
    names = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")
    fused = torch.cat([source.get_tensor(f"{prefix}.{name}.weight").chunk(2)[rank] for name in names])
    output = source.get_tensor(f"{prefix}.wo_ud.weight").chunk(2, dim=1)[rank].contiguous()
    if fused.shape != (4300, 2304) or output.shape != (2304, 1536):
        raise ValueError(f"Unexpected shape in layer {layer}, rank {rank}")
    yield "qkv", pad(fused, (0, 0, 0, 4)), 4300
    yield "out", output, 2304


def expected_files():
    names = {f"{i}-{r}-{k}.pt" for i in range(24) for r in (0, 1) for k in ("qkv", "out")}
    return names | {"rank0.json", "rank1.json"}


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(cache):
    manifest = json.loads((cache / "archive-sha256.json").read_text())
    if set(manifest) != expected_files():
        raise ValueError("Incomplete dense archive manifest")
    for name, checksum in manifest.items():
        if file_digest(cache / name) != checksum:
            raise ValueError(f"Dense archive checksum mismatch: {name}")


def prepare(model, cache):
    cache.parent.mkdir(parents=True, exist_ok=True)
    with (cache.parent / (cache.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            verify(cache)
            return
        stage = Path(tempfile.mkdtemp(prefix=cache.name + "-", dir=cache.parent))
        try:
            with safe_open(model / "model-00001-of-00001.safetensors", framework="pt", device="cpu") as source:
                for rank in (0, 1):
                    records = []
                    for layer in range(24):
                        for kind, weight, valid in matrices(source, layer, rank):
                            planes, recovery = pack(weight)
                            meta = dict(layer=layer, rank=rank, kind=kind, valid=valid,
                                        n=weight.shape[0], k=weight.shape[1],
                                        original_bf16=digest(weight[:valid]), serving_half=digest(weight[:valid].half()),
                                        physical_bytes=sum(t.nbytes for t in planes), source_recovery=len(recovery[0]),
                                        recovery_indices=digest(recovery[0]), recovery_bits=digest(recovery[1]))
                            records.append(meta)
                            torch.save(dict(planes=planes, meta=meta, recovery=recovery), stage / f"{layer}-{rank}-{kind}.pt")
                        print(f"rank {rank}, layer {layer + 1}/24", flush=True)
                    (stage / f"rank{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
            sums = {name: file_digest(stage / name) for name in sorted(expected_files())}
            (stage / "archive-sha256.json").write_text(json.dumps(sums, indent=2) + "\n")
            stage.rename(cache)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    work = Path(os.environ.get("WORK_DIR", repo / ".work"))
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, default=Path(os.environ.get("MODEL_DIR", work / "model")))
    cli.add_argument("--cache", type=Path, default=Path(os.environ.get("DENSE_DIR", work / "dense12")))
    cli.add_argument("--check", action="store_true", help="Verify the 98 existing archive hashes")
    args = cli.parse_args()
    torch.set_num_threads(2)
    if args.check:
        verify(args.cache)
    else:
        prepare(args.model, args.cache)
    print(f"Dense banks ready: {args.cache}")
