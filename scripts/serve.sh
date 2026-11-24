#!/usr/bin/env bash
# Run one measured three-pass Inkling profile. All paths are local to the checkout or WORK_DIR.
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(dirname "$scripts")
work=$(realpath -m "${WORK_DIR:-$repo/.work}")
model=$(realpath -m "${MODEL_DIR:-$work/model}")
dense=$(realpath -m "${DENSE_DIR:-$work/dense12}")
python=${PYTHON:-"$work/.venv/bin/python"}
port=${PORT:-8000}
usage() {
    echo 'Usage: scripts/serve.sh {text-512k|vision-448k} [--no-drafter] [--dry-run]'
    echo 'Paths: WORK_DIR, MODEL_DIR, DENSE_DIR. Devices: CUDA_VISIBLE_DEVICES. Optional: NUMA_NODE, PORT, PYTHON.'
}
case ${1:-} in
    text-512k) model_names=(inkling-2-mini-3loop inkling-2-mini)
        context=524288; pool=10468982784; modality=(--language-model-only) ;;
    vision-448k) model_names=(inkling-2-mini-3loop-vision-gpu-448k)
        context=458752; pool=9261023232
        modality=(--skip-mm-profiling --enable-auto-tool-choice --tool-call-parser inkling
                  --limit-mm-per-prompt '{"image":458752}') ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
shift
spec=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,1,5]]}')
dry=0
for option in "$@"; do
    case $option in
        --no-drafter) spec=() ;;
        --dry-run) dry=1 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ $port =~ ^[0-9]+$ && $((10#$port)) -ge 1 && $((10#$port)) -le 65535 ]] || { echo "Invalid PORT: $port" >&2; exit 2; }
if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then export CUDA_VISIBLE_DEVICES=0,1; fi
[[ $CUDA_VISIBLE_DEVICES =~ ^[^,[:space:]]+,[^,[:space:]]+$ ]] || { echo 'Select exactly two GPUs in CUDA_VISIBLE_DEVICES' >&2; exit 2; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$work/vllm${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_LEVEL=NVL VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST=7.5 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$work/cache/vllm}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$work/cache/triton}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$work}" FLASHINFER_CUDA_ARCH_LIST=7.5
export VLLM_USE_FLASHINFER_SAMPLER=0 INKLING_GRAPH_KV=1 INKLING_NATIVE_GRAPH_KV=1 INKLING_MX_PREFILL=1
export INKLING_SM75_CONV_TAIL=1 INKLING_DENSE12_CACHE="$dense"
source "$scripts/profile.env"

command=()
if [[ -n ${NUMA_NODE:-} ]]; then
    [[ $NUMA_NODE =~ ^[0-9]+$ ]] || { echo 'NUMA_NODE must be a node number' >&2; exit 2; }
    command+=(numactl "--cpunodebind=$NUMA_NODE" "--membind=$NUMA_NODE")
fi
command+=("$python" -m vllm.entrypoints.openai.api_server
          --model "$model" --served-model-name "${model_names[@]}" --load-format safetensors
          --host 127.0.0.1 --port "$port" --tensor-parallel-size 2 --dtype float16
          --reasoning-parser inkling --reasoning-config '{"reasoning_end_str":"<|end_message|><|message_model|><|content_text|>"}'
          --max-model-len "$context" --max-num-seqs 1 --max-num-batched-tokens "$INKLING_PREFILL_CHUNK"
          --enable-prompt-tokens-details "${modality[@]}" --gpu-memory-utilization 0.96
          --no-enable-extensible-kv-cache --kv-cache-memory-bytes "$pool"
          --prefix-cache-retention-interval "$INKLING_PREFIX_RETENTION"
          --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_num_of_warmups":1,"decode_cudagraph_capture_request_sizes":[1],"max_decode_cudagraph_capture_tokens":8}'
          --kernel-config '{"enable_jit_warmup":false}' "${spec[@]}")
if (( dry )); then
    printf 'CUDA_VISIBLE_DEVICES=%q\nSOURCE=%q\nDENSE=%q\n' "$CUDA_VISIBLE_DEVICES" "$work/vllm" "$dense"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

"$python" "$scripts/preflight.py" "$work/vllm" "$model" "$dense"
# pip distributions of the CUDA runtime supply headers used by the JIT.
cu_include=$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib")+"/nvidia/cu13/include")')
if [[ -d $cu_include ]]; then
    export CPLUS_INCLUDE_PATH="$cu_include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
exec "${command[@]}"
