"""Optional W4/A16 grouped projection that shares Marlin's physical expert banks."""
from pathlib import Path

import torch
from vllm.logger import init_logger

from nv_skinny import _lib, forward as grouped

ROOT = Path(__file__).resolve().with_name("resident_bank")
_observed = set()


def prepare(layer):
    if not (layer.hidden_size == 2304 and layer.intermediate_size == 576 and layer.num_experts == 128
            and layer.gate_cap == 4. and layer.up_cap == 25. and (ROOT / "skinny.so").is_file()):
        return None
    bank = {tag: tuple(getattr(layer, tag + "_" + key) for key in ("weight", "weight_scale", "outer"))
            for tag in ("w13", "w2")}
    if not all(t.is_contiguous() for tensors in bank.values() for t in tensors):
        return None
    if not all(c.dtype == torch.int32 and s.element_size() == 1 and a.dtype == torch.float32
               for c, s, a in bank.values()):
        return None
    try:
        _lib(ROOT)
    except (OSError, AttributeError) as exc:
        init_logger('vllm.models.inkling2.resident_nv_sm75').warning("Resident NVFP4 unavailable: %s", exc)
        return None
    return bank


def forward(layer, x, ids, weights):
    if ids.dtype != torch.int32 or not ids.is_contiguous() or tuple(ids.shape) != (len(x), 8):
        from vllm.models.inkling2.nvidia.ops.moe_sm75 import marlin_forward
        return marlin_forward(layer, x, ids, weights)
    if torch.cuda.is_current_stream_capturing() and len(x) not in _observed:
        _observed.add(len(x))
        init_logger('vllm.models.inkling2.resident_nv_sm75').info("SM75 resident NVFP4 active during CUDA capture: rows=%d", len(x))
    return grouped(layer._resident_nv, x.contiguous(), ids, weights.contiguous(), root=ROOT)
