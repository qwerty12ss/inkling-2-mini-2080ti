#!/usr/bin/env bash
# Build only into the user-selected work directory. The ABI checkout is a read-only input.
set -euo pipefail
work=$(realpath "${1:?usage: runtime/build.sh WORK}")
runtime=$(cd "$(dirname "$0")" && pwd)
tree="$work/vllm"
abi=${INKLING_ABI_SRC:-"$work/abi"}
python=${PYTHON:-"$work/.venv/bin/python"}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CC=${CC:-gcc-13} CXX=${CXX:-g++-13}
export TORCH_CUDA_ARCH_LIST=7.5 FLASHINFER_CUDA_ARCH_LIST=7.5
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"$work"}
export PATH="$(dirname "$python"):$CUDA_HOME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -f "$tree/.inkling-source" ]]; then
    echo "Run runtime/prepare.py --work-dir WORK first." >&2
    exit 1
fi
for path in "$work" "$runtime" "$abi" "$CUDA_HOME" "$FLASHINFER_WORKSPACE_BASE"; do
    if [[ ! $path =~ ^[a-zA-Z0-9_./+-]+$ ]]; then
        echo "Ninja/CUDA build paths must contain only ASCII letters, digits, /, _, +, . and -." >&2
        exit 1
    fi
done
if [[ $(git -C "$abi" rev-parse HEAD) != fe9872d764b54ed6a1ac35804a4b6ebd813b7f56 ]]; then
    echo "Incorrect ABI checkout revision. See docs/build.md." >&2
    exit 1
fi
site=$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export CPLUS_INCLUDE_PATH="$site/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
"$python" - <<'PY'
import sys
import torch
if sys.version_info[:2] != (3, 12) or torch.__version__ != "2.13.0+cu130":
    raise SystemExit(f"Expected CPython 3.12 and torch 2.13.0+cu130, got {sys.version} and {torch.__version__}")
PY
for ext in _C_stable_libtorch _moe_C_stable_libtorch cumem_allocator fs_io_C spinloop; do
    path="$abi/vllm/$ext.abi3.so"
    test -f "$path" || { echo "Missing $path. Finish the ABI source build first." >&2; exit 1; }
    ln -sfn "$path" "$tree/vllm/"
done
test -d "$abi/vllm/third_party/triton_kernels"
ln -sfnT "$abi/vllm/third_party/triton_kernels" "$tree/vllm/third_party/triton_kernels"

mkdir -p "$tree/resident_bank" "$work/build"
scratch=$(mktemp -d "$work/build/.native.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
native="$runtime/native"
nvcc=("$CUDA_HOME/bin/nvcc" -std=c++17 -O3 -gencode arch=compute_75,code=sm_75 --shared -Xcompiler -fPIC)
"${nvcc[@]}" -lineinfo --expt-relaxed-constexpr "$native/skinny/resident.cu" -o "$scratch/skinny.so"
"${nvcc[@]}" "$native/dense12.cu" -o "$scratch/libdense12_integrated.so"
"${nvcc[@]}" "$native/head_int8.cu" -o "$scratch/libinkling_head_int8.so"
"${nvcc[@]}" "$native/stable_routes.cu" -o "$scratch/libinkling_stable_routes.so"
"$CXX" -std=c++17 -O3 -shared -fPIC -I"$CUDA_HOME/include" "$native/tune_lt.cpp" \
    -L"$CUDA_HOME/lib64" -lcublasLt -lcublas -lcudart -o "$scratch/libinkling_tune_lt.so"
"$python" "$native/build_marlin.py" "$scratch" "$abi"
mv "$scratch/skinny.so" "$tree/resident_bank/"
mv "$scratch/inkling_marlin_mx32f32.so" "$tree/vllm/models/inkling2/nvidia/ops/"
mv "$scratch/"lib*.so "$tree/"
"$python" "$runtime/build_flashinfer.py" "$work" --jobs "${MAX_JOBS:-4}"

# Import against exactly the reconstructed tree. No GPU allocations or model download.
PYTHONPATH="$tree" "$python" - <<'PY'
import ctypes
from pathlib import Path

import torch
import vllm
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch
import inkling_vision_sm75
import hybrid_head
import inkling_thinking_budget

root = Path(vllm.__file__).resolve().parents[1]
ops = root / "vllm/models/inkling2/nvidia/ops"
torch.ops.load_library(ops / "inkling_marlin_mx32f32.so")
assert hasattr(torch.ops.inkling_sm75_f32, "mx32_gemm")
for path, symbol in (
    ("resident_bank/skinny.so", "project_skinny"),
    ("libdense12_integrated.so", "dense12_materialize"),
    ("libinkling_stable_routes.so", "inkling_stable_routes"),
    ("libinkling_head_int8.so", "head_projection"),
    ("libinkling_tune_lt.so", "make_plan"),
):
    getattr(ctypes.CDLL(str(root / path)), symbol)
print(f"Runtime source and native ABI loaded from {root}")
PY
