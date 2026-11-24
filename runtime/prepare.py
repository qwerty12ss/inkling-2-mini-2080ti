#!/usr/bin/env python3
"""Reconstruct the Inkling Python tree against an immutable upstream revision."""
import argparse
from hashlib import sha256
from pathlib import Path
import os
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

COMMIT = "fdda4ee7232366f4490f6484e98533de0289ac98"
DIGEST = "d45373a19993e826d1a54176621f66152859986f0c1c2e219bec74e9e30be9a7"
URL = f"https://codeload.github.com/vllm-project/vllm/tar.gz/{COMMIT}"
RUNTIME = Path(__file__).resolve().parent
PATCH = RUNTIME.parent / "patches/vllm-sm75.patch"
OVERLAY = RUNTIME / "overlay"


def source_id():
    identity = sha256(DIGEST.encode())
    for path in (PATCH, *sorted(OVERLAY.rglob("*.py"))):
        identity.update(str(path.relative_to(RUNTIME.parent)).encode())
        identity.update(sha256(path.read_bytes()).digest())
    return identity.hexdigest()


def prepare(work_dir, archive):
    work_dir.mkdir(exist_ok=True, parents=True)
    target = work_dir / "vllm"
    marker = ".inkling-source"
    identity = source_id()
    if target.exists():
        if (target / marker).is_file() and (target / marker).read_text().strip() == identity:
            return target
        raise SystemExit(f"Existing {target} belongs to another source revision. Select another work directory.")
    with tempfile.TemporaryDirectory(prefix=".inkling-", dir=work_dir) as tmp:
        staging = Path(tmp)
        if archive is None:
            archive = staging / "upstream.tar.gz"
            with urlopen(URL, timeout=120) as remote, archive.open("wb") as local:
                shutil.copyfileobj(remote, local)
        if sha256(archive.read_bytes()).hexdigest() != DIGEST:
            raise SystemExit("Upstream source archive SHA-256 mismatch")
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging, filter="data")
        tree = staging / f"vllm-{COMMIT}"
        isolated_git = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tree.parent))
        subprocess.run(["git", "apply", "--whitespace=nowarn", PATCH], cwd=tree, env=isolated_git, check=True)
        check = tree / "vllm/models/inkling2/nvidia/model.py"
        if "from inkling_vision_sm75 import SM75VisionEncoder" not in check.read_text():
            raise RuntimeError("vLLM patch did not apply to the extracted source tree")
        for source in sorted(OVERLAY.rglob("*.py")):
            destination = tree / source.relative_to(OVERLAY)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (tree / marker).write_text(identity + "\n")
        tree.rename(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="use the matching upstream tar.gz offline")
    args = parser.parse_args()
    print(prepare(args.work_dir.resolve(), args.archive))
