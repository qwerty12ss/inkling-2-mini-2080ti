"""Rebuild the six attention modules into the same FlashInfer cache used by serve."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile

from flashinfer.jit.env import FLASHINFER_JIT_DIR
from tvm_ffi import load_module

SOURCES = Path(__file__).resolve().with_name("flashinfer")
SITE = Path(sysconfig.get_paths()["purelib"])
STOCK = SITE / "flashinfer/data/include"
FP32 = {"inkling_global_far_sm75_f32", "inkling_global_q_sm75_f32_v1", "inkling_local_q8_sm75_fp32_v2"}


def copy_changed(src, dst):
    if not dst.exists() or src.read_bytes() != dst.read_bytes():
        shutil.copyfile(src, dst)


def build(directory, work, jobs):
    name = directory.name
    out = work / "build" / name
    generated = out / "generated"
    generated.mkdir(exist_ok=True, parents=True)
    for part in (SOURCES / "common", directory):
        for file in part.iterdir():
            if file.suffix in (".cu", ".inc"):
                copy_changed(file, generated / file.name)
    headers = STOCK
    patch = directory / "headers.patch"
    if patch.exists():
        headers = out / "include"
        # Create private, incrementally reusable header trees without touching the package.
        revision = headers / ".patch"
        if not revision.is_file() or revision.read_bytes() != patch.read_bytes():
            with tempfile.TemporaryDirectory(dir=out) as temp:
                tree = Path(temp) / "include"
                shutil.copytree(STOCK, tree)
                subprocess.run(["git", "apply", "--whitespace=nowarn", patch], cwd=tree, env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(tree.parent)), check=True)
                (tree / ".patch").write_bytes(patch.read_bytes())
                if headers.exists():
                    shutil.rmtree(headers)
                tree.rename(headers)
    variables = {
        "OUT": out,
        "SOURCE": generated,
        "SITE": SITE,
        "PYINC": sysconfig.get_path("include"),
        "CUDA": os.environ["CUDA_HOME"],
        "NAME": name,
        "CXX": os.environ.get("CXX", "g++"),
        "EXTRA_INCLUDE": f"    -isystem {headers} $\n",
        "EXTRA_NVCC": " $\n    --pre-include=" + str(SOURCES / "cp_async_fp32.cuh") if name in FP32 else "",
    }
    ninja = (SOURCES / "build.ninja.in").read_text()
    for key, value in variables.items():
        ninja = ninja.replace(f"@{key}@", str(value))
    (out / "build.ninja").write_text(ninja)
    subprocess.run(["ninja", "-C", str(out), "-j", str(jobs)], check=True)
    library = out / f"{name}.so"
    load_module(str(library))  # validate the FFI registration before publishing
    dest = FLASHINFER_JIT_DIR / name
    dest.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=dest, delete=False) as fp:
        staged = Path(fp.name)
    try:
        shutil.copy2(library, staged)
        staged.replace(dest / library.name)
    finally:
        staged.unlink(missing_ok=True)
    print(dest / library.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if importlib.metadata.version("flashinfer-python") != "0.6.18.post1" or FLASHINFER_JIT_DIR.parent.name != "75":
        raise SystemExit("This recipe requires flashinfer-python 0.6.18.post1 and FLASHINFER_CUDA_ARCH_LIST=7.5.")
    for module in sorted(SOURCES.glob("inkling_*")):
        build(module, args.work.resolve(), args.jobs)
