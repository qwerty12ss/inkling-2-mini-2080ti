"""Greedy DSpark proposals from the shared, rank-local LM head on SM75."""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from aux_projection import AuxiliaryHead


@triton.jit
def _correct(base, bias, softcap):
    value = (base.to(tl.float32) + bias.to(tl.float32)).to(tl.float16)
    return (libdevice.tanh(value / softcap) * softcap).to(tl.float16)


@triton.jit
def _shortlist(A, Bias, LocalIds, softcap, N: tl.constexpr, VALID: tl.constexpr,
               COUNT: tl.constexpr, BLOCK: tl.constexpr):
    part = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    column = part * BLOCK + lane
    valid = column < VALID
    base = tl.load(A + column, valid, 0)
    bias = tl.load(Bias + column, valid, 0)
    value = _correct(base, bias, softcap)
    value = tl.where(value == value, value, -float("inf")).to(tl.float16)
    value = tl.where(value == 0., 0., value).to(tl.float16)
    bits = value.to(tl.uint16, bitcast=True).to(tl.uint32)
    ordered = bits ^ tl.where(bits >= 32768, 65535, 32768).to(tl.uint32)
    # The low field deterministically orders ties by the smaller vocabulary ID.
    key = ordered * BLOCK + (BLOCK - 1 - lane).to(tl.uint32)
    key = tl.where(valid, key, 0)
    selected = tl.topk(key, COUNT)
    local = part * BLOCK + BLOCK - 1 - (selected % BLOCK).to(tl.int32)
    tl.store(LocalIds + part * COUNT + tl.arange(0, COUNT), tl.where(selected != 0, local, N))


@triton.jit
def _score_candidates(H, W, Bias, LocalIds, Pairs, softcap,
                      N: tl.constexpr, VALID: tl.constexpr, OFFSET: tl.constexpr,
                      VOCAB: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(0)
    row = tl.load(LocalIds + slot)
    valid = row < VALID
    d = tl.arange(0, BLOCK)
    h = tl.load(H + d, d < WIDTH, 0).to(tl.float32)
    w = tl.load(W + row.to(tl.int64) * WIDTH + d, (d < WIDTH) & valid, 0).to(tl.float32)
    base = tl.sum(h * w, 0).to(tl.float16)
    bias = tl.load(Bias + row, valid, 0.)
    value = _correct(base, bias, softcap).to(tl.float32)
    tl.store(Pairs + 2 * slot, tl.where(valid & (value == value), value, -float("inf")))
    tl.store(Pairs + 2 * slot + 1, tl.where(valid, OFFSET + row, VOCAB).to(tl.float32))


def refined_argmax(hidden, weight, coarse, bias, cap, vocab, rank, world, gather, packed=None):
    from dspark_argmax import _finish
    local_n, width = weight.shape if packed is None else (packed.meta["n"],packed.meta["k"])
    assert world == 2 and len(hidden) == width == 2304
    assert coarse.shape == bias.shape == (1, local_n)
    assert weight.dtype == hidden.dtype == coarse.dtype == bias.dtype == torch.float16
    assert weight.is_contiguous() and hidden.is_contiguous()
    assert coarse.stride(1) == bias.stride(1) == 1
    offset = rank * local_n
    valid = max(0, min(local_n, vocab - offset))
    block, count = 1024, 4
    parts = triton.cdiv(local_n, block)
    slots = parts * count
    ids = coarse.new_empty(slots, dtype=torch.int32)
    pairs = coarse.new_empty((1, slots * 2), dtype=torch.float32)
    _shortlist[(parts,)](coarse, bias, ids, cap, local_n, valid, count, block, num_warps=4)
    if packed is None:
        _score_candidates[(slots,)](hidden, weight, bias, ids, pairs, cap, local_n,
                                   valid, offset, vocab, width, triton.next_power_of_2(width), num_warps=8)
    else:
        packed.finalists(hidden,bias,ids,pairs,cap,valid,offset,vocab)
    combined = gather(pairs)
    result = coarse.new_empty((1,), dtype=torch.int64)
    _finish[(1,)](combined, result, slots * world, vocab, triton.next_power_of_2(slots * world), num_warps=4)
    return result


def choose(aux, hidden, weight, coarse, bias, cap, vocab, rank, world, gather):
    if aux.mode == "coarse":
        from dspark_argmax import sharded_markov_argmax
        return sharded_markov_argmax(coarse, bias, cap, vocab, rank, world, gather)
    return refined_argmax(hidden, weight, coarse, bias, cap, vocab, rank, world, gather,
                         getattr(aux,"original_p12",None))


@torch.no_grad()
def refresh(model, original=None):
    """Keep both TP ranks in the same collective path and preserve pointers during reload."""
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from vllm.logger import init_logger
    log = init_logger('vllm.models.inkling2.dspark_proposal_sm75')
    weight = model.lm_head.weight if original is None else original
    mode = os.environ["INKLING_DSPARK_AUX_HEAD"]
    assert mode in ("coarse", "refine")
    group = get_tp_group()
    assert group.world_size == 2
    active = getattr(model, "_conditional_aux_head", None)
    if active is not None:
        # CUDA graphs already hold the original buffers after the first startup.
        active.refresh(weight)
        ranked=getattr(active,"ranked_correction",None)
        if ranked is not None:ranked.refresh(model.model.markov_head)
        return
    eligible = (tuple(weight.shape) == (100512, 2304) and weight.dtype == torch.float16
                and weight.is_contiguous() and torch.cuda.get_device_capability(weight.device) == (7, 5))
    available = torch.zeros(1, dtype=torch.int32, device=weight.device)
    if eligible:
        try:
            active = AuxiliaryHead(weight)
            active.mode = mode
            if os.environ.get("INKLING_DSPARK_MARKOV_R32") == "1":
                from fold_rank import RankedCorrection
                active.ranked_correction=RankedCorrection(model.model.markov_head,rank=32,count=16)
            available.fill_(1)
        except (torch.OutOfMemoryError, OSError) as exc:
            log.warning("DSpark auxiliary head unavailable on rank %d: %s", group.rank_in_group, exc)
    dist.all_reduce(available, op=dist.ReduceOp.MIN, group=group.device_group)
    if available.item():
        model._conditional_aux_head = active
        byte_count = sum(t.nbytes for t in (active.packed, active.scales, active.q, active.qs))
        ranked=getattr(active,"ranked_correction",None)
        if ranked is not None:
            byte_count+=sum(t.nbytes for t in (ranked.w,ranked.b,ranked.mean,ranked.transform,ranked.projector))
            log.info("SM75 original W16/R256 proposal refinement with R32 shortlist active rank=%d count=16", group.rank_in_group)
        log.info("SM75 DSpark auxiliary head active: mode=%s rank=%d allocated=%d bytes",
                 mode, group.rank_in_group, byte_count)
    else:
        model._conditional_aux_head = None
        del active
        torch.cuda.empty_cache()
        log.warning("Both TP ranks selected the original DSpark head")
