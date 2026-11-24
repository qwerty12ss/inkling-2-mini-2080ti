"""Experimental dual-projection Turing MMA with stable slot routing and original NVFP4 weights."""
import ctypes
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl

from pathlib import Path

CACHE = Path(__file__).resolve().with_name("resident_bank")


@triton.jit
def _group(ids, perm, gids, counts, S: tl.constexpr, N: tl.constexpr):
    slot = tl.program_id(0)
    e = tl.load(ids + slot)
    rows = tl.arange(0, S)
    same = (rows < N) & (tl.load(ids + rows, rows < N, -2) == e)
    first = tl.min(tl.where(same, rows, S), 0)
    tl.store(gids + slot, tl.maximum(e, 0))
    if e < 0:
        tl.store(counts + slot, -1)
    elif slot == first:
        # top8 is distinct within a token, therefore an expert occurs at most
        # eight times in one q8 verify. Each group owns eight storage slots.
        rank = tl.cumsum(same.to(tl.int32), 0) - 1
        tl.store(perm + slot * 8 + rank, rows, same)
        tl.store(counts + slot, tl.sum(same.to(tl.int32), 0))
    else:
        tl.store(counts + slot, 0)


@lru_cache(maxsize=2)
def _lib(root):
    library = ctypes.CDLL(str(root/'skinny.so'))
    fn = library.project_skinny
    fn.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*6 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return library, fn


def load_packed(tp_rank, index, device, *, root=CACHE):
    path = root / f'layer{index}.rank{tp_rank}'
    return {tag: tuple(torch.from_numpy(np.load(path / (tag+'_'+kind+'.npy'))).to(device)
                       for kind in ('code', 'scale', 'outer')) for tag in ('w13', 'w2')}


def forward(packed, x, ids, gamma, *, tile13=2, tile2=2, split13=8, root=CACHE, mx=False):
    from vllm.models.inkling2.nvidia.ops.moe_sm75 import _activate
    if (ids.dtype != torch.int32 or not ids.is_contiguous() or ids.ndim != 2
            or ids.shape[0] not in (4, 6, 8) or ids.shape[1] != 8 or x.dtype != torch.float16
            or tuple(x.shape) != (ids.shape[0], 2304) or not x.is_contiguous()):
        raise ValueError('q4/q6/q8 dense TP2 input and eight local expert IDs per token required')
    slots = ids.numel()
    perm = ids.new_empty(slots * 8)
    gids = ids.new_empty(slots)
    count = ids.new_empty(slots)
    _group[(slots,)](ids, perm, gids, count, triton.next_power_of_2(slots), slots, num_warps=1)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    def project(tag, a, n, tile, split, slot_major):
        y = x.new_empty((slots, n))
        code, sc, outer = packed[tag]
        elements = 128 * n * a.shape[1]
        if code.nbytes != elements // (1 if mx else 2) or sc.nbytes != elements // (32 if mx else 16):
            raise ValueError('incompatible packed bank')
        fn = _lib(root)[1]
        ptrs = [t.data_ptr() for t in (code, sc, outer, a, y, perm, gids, count)]
        rv = fn(*ptrs, n, a.shape[1], slots, slot_major, split, tile, stream)
        if rv:
            raise RuntimeError('skinny CUDA launch failed: '+str(rv))
        return y
    gu = project('w13', x, 1152, tile13, split13, 0)
    h = x.new_empty((slots, 576))
    _activate[(slots, 3)](gu, gamma, h, 576, 4., 25., 256)
    return project('w2', h, 2304, tile2, 4, 1)
