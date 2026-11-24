#!/usr/bin/env bash
# Build the SM75 vLLM ABI against the same cu130 Torch family as the recipe.
set -euo pipefail
usage() { echo 'Usage: runtime/install-base.sh [--sources-only] WORK_DIR'; }
sources_only=0
if [[ ${1:-} == --sources-only ]]; then sources_only=1; shift; fi
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 1 ]] || { usage >&2; exit 2; }
mkdir -p -- "$1"
work=$(cd -- "$1" && pwd)
runtime=$(cd -- "$(dirname -- "$0")" && pwd)

checkout() {
    local path=$1 origin=$2 revision=$3
    if [[ ! -d "$path/.git" ]]; then
        if [[ -e $path ]]; then echo "Existing non-checkout: $path" >&2; return 1; fi
        git init -q "$path"
        git -C "$path" remote add origin "$origin"
    fi
    [[ $(git -C "$path" remote get-url origin) == "$origin" ]] || { echo "Different upstream for $path" >&2; return 1; }
    local head
    head=$(git -C "$path" rev-parse --verify HEAD 2>/dev/null || true)
    if [[ -z $head ]]; then
        git -C "$path" fetch --depth=1 origin "$revision"
        git -C "$path" checkout --quiet --detach "$revision"
    elif [[ $head != "$revision" ]]; then
        echo "Existing checkout at $path is on $head; expected $revision." >&2
        return 1
    fi
}

mkdir -p "$work/deps"
checkout "$work/abi" https://github.com/weicj/vLLM-2080Ti-Definitive.git fe9872d764b54ed6a1ac35804a4b6ebd813b7f56
checkout "$work/deps/cutlass" https://github.com/NVIDIA/cutlass.git cb4247394dd82148787aed73e5dc7cef33cbf862
checkout "$work/deps/triton-source" https://github.com/triton-lang/triton.git 0add68262ab0a2e33b84524346cb27cbb2787356
if (( sources_only )); then exit 0; fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH="$CUDA_HOME" CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$work/.venv/bin:$CUDA_HOME/bin:$PATH"
[[ -x $CUDACXX ]] && "$CUDACXX" --version | grep -q 'release 13\.0,' || { echo 'CUDA toolkit 13.0 with nvcc is required.' >&2; exit 1; }
export CC=${CC:-gcc-13} CXX=${CXX:-g++-13}
for bin in uv "$CC" "$CXX"; do command -v "$bin" >/dev/null || { echo "Missing $bin" >&2; exit 1; }; done
if [[ ! -x "$work/.venv/bin/python" ]]; then
    uv venv --python 3.12 "$work/.venv"
fi
py="$work/.venv/bin/python"
"$py" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
export CPLUS_INCLUDE_PATH="$work/.venv/lib/python3.12/site-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST=7.5
export VLLM_TARGET_DEVICE=cuda VLLM_VERSION_OVERRIDE=0.2.1
export VLLM_CUTLASS_SRC_DIR="$work/deps/cutlass"
export TRITON_KERNELS_SRC_DIR="$work/deps/triton-source/python/triton_kernels/triton_kernels"
export MAX_JOBS=${MAX_JOBS:-4} NVCC_THREADS=${NVCC_THREADS:-1}
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}

install=(uv pip install --python "$py" --constraint "$runtime/constraints-cu130.txt")
"${install[@]}" --torch-backend=cu130 \
    'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' 'torchaudio==2.11.0+cu130' 'torchcodec==0.17.0+cu130'
"${install[@]}" cmake==4.4.4 ninja==1.13.2 packaging==26.3 setuptools==78.1.0 \
    setuptools-scm==10.3.4 setuptools-rust==1.13.0 wheel==0.48.0 jinja2==3.1.6 regex==2026.9.29 build==1.6.1
"${install[@]}" --no-build-isolation --torch-backend=cu130 -e "$work/abi" \
    'numpy==2.3.5' 'transformers==5.20.2' 'huggingface-hub==2.3.0' 'safetensors==0.8.0' \
    'tokenizers==0.23.3' 'orjson==3.13.0' 'pillow==12.3.0'
"$py" "$work/abi/tools/patch_torch_inductor_e8m0.py"
"$py" "$work/abi/tools/check_torch_inductor_e8m0.py"
"$py" -c 'import torch, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; assert torch.__version__ == "2.13.0+cu130"; print("SM75 ABI installed:", torch.__version__)'
