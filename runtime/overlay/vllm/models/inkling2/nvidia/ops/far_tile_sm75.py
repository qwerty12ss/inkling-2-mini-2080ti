"""Load the optional SM75 FA2 module with 64 query rows and two KV MMA tiles.

Use the stock wrapper's existing 96 MiB float workspace and pinned planning slab.
Our module has no extra TVM-FFI arguments. Replanning after a backend change
preserves the stock scheduler's allocation and physical-page upload sequence.
"""
from functools import lru_cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
STOCK_URI = "inkling_far_q64_kv2_sm75_v1"
REGISTER_URI = "inkling_far_q64_schedules_sm75_qreg2_v1"
PIPE_URI = "inkling_far_q64_schedules_sm75_pipe_v1"


@lru_cache(maxsize=3)
def load_tiled_module(uri):
    import flashinfer
    import tvm_ffi
    from flashinfer.jit.env import FLASHINFER_JIT_DIR
    from flashinfer.prefill import get_batch_prefill_jit_module

    path = FLASHINFER_JIT_DIR / uri / f"{uri}.so"
    if flashinfer.__version__ != "0.6.18.post1" or not path.is_file():
        logger.warning_once("SM75 far tile artifact unavailable for this FlashInfer version")
        return None
    try:
        return get_batch_prefill_jit_module(uri, tvm_ffi.load_module(str(path)))
    except (RuntimeError, AttributeError, OSError, ValueError) as exc:
        logger.warning_once("SM75 far tile module could not load: %s", exc)
        return None


def select_module(wrapper, q, k, v, block_table, pages, softmax_scale, seq_len):
    if (getattr(wrapper, "_sm75_far_tile_failed", False)
            or q.shape[0] not in (1024, 1408, 2048)
            or q.shape[1:] != (12,128) or k.shape[1:] != (64,4,128)
            or v.shape != k.shape or v.stride() != k.stride()
            or q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or q.stride(-1) != 1 or k.stride(-1) != 1
            or block_table.dtype != torch.int32 or block_table.shape[0] != 1
            or block_table.stride(-1) != 1 or block_table.shape[1] < pages
            or pages < 128 or softmax_scale != 128**-0.5
            or torch.cuda.get_device_capability(q.device) != (7,5)
            or torch.cuda.is_current_stream_capturing()):
        return None
    if ((131072 == 0 or seq_len <= 131072) and q.shape[0] in (1024, 2048) and k.stride() == (65536,1024,256,1)
            and k.data_ptr() % 16 == 0 and v.data_ptr() % 16 == 0):
        pipelined = load_tiled_module(PIPE_URI)
        if pipelined is not None:
            logger.info_once("SM75 FP16 streamed K/V Q64/K32 enabled for paged far prefill")
            return pipelined
    candidate = load_tiled_module(REGISTER_URI)
    if candidate is not None:
        logger.info_once("SM75 register-Q Q64/K32 enabled for paged far prefill")
        return candidate
    return load_tiled_module(STOCK_URI) if q.shape[0] in (1024, 1408) else None


def plan_with_selected_module(wrapper, module, args, kwargs, pages):
    wrapper._jit_module = module
    if module is not None:
        try:
            tiled_kwargs = {**kwargs, "fixed_split_size": (pages+1)//2}
            wrapper.plan(*args, **tiled_kwargs)
            if len(wrapper._plan_info) != 15 or wrapper._plan_info[3] != 64:
                raise ValueError("FA2 planner returned an unexpected tile")
            logger.info_once("SM75 paged far prefill using 64-row tiles and two KV partitions")
            return
        except (ValueError, RuntimeError, AttributeError) as exc:
            # plan() can issue async copies before reporting an incompatible
            # workspace. Finish them before reusing the same host staging slab.
            torch.cuda.current_stream(wrapper.device).synchronize()
            wrapper._sm75_far_tile_failed = True
            wrapper._jit_module = None
            logger.warning_once("Falling back to standard SM75 far planning: %s", exc)
    wrapper.plan(*args, **kwargs)
