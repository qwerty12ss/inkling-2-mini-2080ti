#!/usr/bin/env python3
"""Compare GPU machine code against the recorded SM75 modules (CUDA Toolkit cuobjdump)."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

MANIFEST = Path(__file__).resolve().with_name("sm75-code.json")


def instruction_digest(library, cuobjdump):
    """Hash the multiset of .text sections, ignoring compiler-generated file paths in names."""
    digests = []
    with tempfile.TemporaryDirectory(prefix="inkling-cubin-") as tmp:
        subprocess.run([cuobjdump, "--extract-elf", "all", library.resolve()],
                       cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        for cubin in Path(tmp).glob("*.sm_75.cubin"):
            data = cubin.read_bytes()
            if data[:6] != b"\x7fELF\x02\x01":
                raise ValueError(f"Expected little-endian ELF64 cubin in {library}")
            header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
            table_offset, entry_size, count, names_idx = header[6], *header[11:14]
            sections = [struct.unpack_from("<IIQQQQIIQQ", data, table_offset + i * entry_size)
                        for i in range(count)]
            names_offset = sections[names_idx][4]
            for name_idx, _, _, _, offset, size, *_ in sections:
                start = names_offset + name_idx
                name = data[start:data.index(b"\0", start)]
                if name.startswith(b".text.") and size:
                    digests.append(sha256(data[offset:offset + size]).digest())
    if not digests:
        raise ValueError(f"No SM75 .text sections found in {library}")
    return {"sections": len(digests), "sha256": sha256(b"".join(sorted(digests))).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or
                        str(Path(os.getenv("CUDA_HOME", "/usr/local/cuda")) / "bin/cuobjdump"))
    args = parser.parse_args()
    cache_base = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", args.work))
    cache = cache_base / ".cache/flashinfer/0.6.18.post1/75/cached_ops"
    mismatches = []
    for key, expected in json.loads(MANIFEST.read_text()).items():
        if key.startswith("flashinfer/"):
            name = key.split("/", 1)[1]
            path = cache / name / (name + ".so")
        else:
            path = args.work / "vllm" / key
        value = instruction_digest(path, args.cuobjdump)
        if value != expected:
            mismatches.append(key)
        print(f"{'OK' if value == expected else 'DIFF'} {key} ({value['sections']} SM75 text sections)")
    if mismatches:
        raise SystemExit(f"GPU instruction checksums differ for {len(mismatches)} libraries")
