"""Research candidate for TP-sharded FP16 Inkling2 DSpark greedy Markov head."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _local_partials(A, B, Pairs, stride_a, stride_b, softcap,
                    N: tl.constexpr, VOCAB: tl.constexpr, OFFSET: tl.constexpr,
                    PARTS: tl.constexpr, BLOCK: tl.constexpr):
    part, row = tl.program_id(0), tl.program_id(1)
    column = part * BLOCK + tl.arange(0, BLOCK)
    valid = (column < N) & (OFFSET + column < VOCAB)
    a = tl.load(A + row * stride_a + column, valid, 0).to(tl.float32)
    b = tl.load(B + row * stride_b + column, valid, 0).to(tl.float32)
    # Match the explicit FP16 roundings in the deployed _add_softcap_kernel.
    x = (a + b).to(tl.float16)
    x = x / softcap
    x = libdevice.tanh(x)
    x = (x * softcap).to(tl.float16).to(tl.float32)
    x = tl.where(valid & (x == x), x, -float("inf"))
    score = tl.max(x, 0)
    first = tl.min(tl.where(valid & (x == score), OFFSET + column, VOCAB), 0)
    pair = Pairs + (row * PARTS + part) * 2
    tl.store(pair, score)
    tl.store(pair + 1, first.to(tl.float32))


@triton.jit
def _finish(Pairs, Out, PAIRS: tl.constexpr, VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    pairs = Pairs + (row * PAIRS + lane) * 2
    values = tl.load(pairs, lane < PAIRS, -float("inf"))
    indices = tl.load(pairs + 1, lane < PAIRS, VOCAB).to(tl.int32)
    top = tl.max(values, 0)
    first = tl.min(tl.where(values == top, indices, VOCAB), 0)
    tl.store(Out + row, first.to(tl.int64))


def sharded_markov_argmax(a, b, softcap, vocab, rank, tp, all_gather):
    """Return original full-vocab argmax with one collective over block winners.

    The per-rank contiguous vocab slicing follows DSpark.markov_logits(). The
    collective concatenates the last dimension in rank order. Retain the
    original implementation for sampling configurations that need full logits.
    """
    rows, width = a.shape
    assert b.shape == a.shape and a.dtype == b.dtype == torch.float16
    assert a.stride(1) == b.stride(1) == 1 and 0 < vocab < 2 ** 24
    assert width * tp >= vocab and 0 <= rank < tp
    block = 1024
    parts = triton.cdiv(width, block)
    local = a.new_empty((rows, parts * 2), dtype=torch.float32)
    _local_partials[(parts, rows)](a, b, local, a.stride(0), b.stride(0), softcap,
                                   width, vocab, rank * width, parts, block, num_warps=4)
    combined = all_gather(local) if tp > 1 else local
    output = a.new_empty((rows,), dtype=torch.int64)
    total = tp * parts
    _finish[(rows,)](combined, output, total, vocab, triton.next_power_of_2(total), num_warps=4)
    return output
