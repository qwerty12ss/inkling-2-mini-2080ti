"""Stage FP16 KV from host-backed pages once per layer and prefill chunk."""
from contextlib import contextmanager
from functools import lru_cache

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_kv(K, V, Table, Packed, KS0: tl.constexpr, KS1: tl.constexpr,
               KS2: tl.constexpr, VS0: tl.constexpr, VS1: tl.constexpr,
               VS2: tl.constexpr, PAGE: tl.constexpr, HEADS: tl.constexpr,
               DIM: tl.constexpr, TILE: tl.constexpr):
    logical = tl.program_id(0)
    physical = tl.load(Table + logical).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offset = i % DIM
    head = i // DIM % HEADS
    token = i // (HEADS * DIM)
    mask = token < PAGE
    keys = tl.load(K + physical * KS0 + token * KS1 + head * KS2 + offset,
                   mask=mask, other=0)
    values = tl.load(V + physical * VS0 + token * VS1 + head * VS2 + offset,
                     mask=mask, other=0)
    dest = ((logical * PAGE + token) * HEADS + head) * (2 * DIM) + offset
    tl.store(Packed + dest, keys, mask=mask)
    tl.store(Packed + dest + DIM, values, mask=mask)


@lru_cache(maxsize=1)
def _workspace(device, page, heads, dim):
    capacity = triton.cdiv(524288, page)
    packed = torch.empty((capacity, page, heads, 2 * dim), device=device, dtype=torch.float16)
    indices = torch.arange(capacity, dtype=torch.int32, device=device).view(1, -1)
    init_logger('vllm.inkling_kv_stage').info("Staging host FP16 KV for prefill, %.0f MiB on %s",
                                               packed.nbytes / 2**20, device)
    return packed, indices, torch.cuda.Event()


@contextmanager
def stage(k, v, table, length):
    _, page, heads, dim = k.shape
    n = triton.cdiv(length, page)
    if (k.dtype != torch.float16 or k.shape != v.shape or k.stride(-1) != 1
            or v.stride(-1) != 1 or table.shape[0] != 1 or table.stride(-1) != 1
            or n > table.shape[-1] or not 0 < length <= 524288):
        raise ValueError("Invalid single-request FP16 host KV page layout")
    buf, logical, finished = _workspace(k.device, page, heads, dim)
    stream = torch.cuda.current_stream(k.device)
    stream.wait_event(finished)
    tile = 1024
    _gather_kv[(n, triton.cdiv(page * heads * dim, tile))](
        k, v, table, buf, *k.stride()[:3], *v.stride()[:3], page, heads, dim, tile)
    try:
        yield buf[:n, ..., :dim], buf[:n, ..., dim:], logical[:, :n]
    finally:
        finished.record(stream)
