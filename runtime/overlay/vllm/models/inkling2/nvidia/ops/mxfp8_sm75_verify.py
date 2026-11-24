"""SM75, weight-only MXFP8 verification. Prepare the small FP16 LUT before graph capture."""
import torch
import triton
import triton.language as tl


def make_table(device):
    """All packed byte pairs use the serving kernel's FP32 product -> FP16 rounding."""
    values = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.exp2(torch.arange(256, dtype=torch.float32) - 127)
    return (scales[:, None] * values.float()[None, :]).half().to(device)


@triton.jit
def _project(x, w, scales, ids, routing, lut, out, N: tl.constexpr, K: tl.constexpr,
             REPEAT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             FUSE: tl.constexpr, USE_LUT: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    slot = tl.program_id(0)
    expert = tl.load(ids + slot)
    start = tl.program_id(1) * BN
    cols = start + tl.arange(0, BN)
    if expert < 0:
        if FUSE:
            tl.store(out + slot * (N // 2) + start // 2 + tl.arange(0, BN // 2), 0.)
        else:
            tl.store(out + slot * N + cols, 0.)
        return
    axis = tl.arange(0, BK)
    mask = axis < K
    act = tl.load(x + (slot // REPEAT) * K + axis, mask, 0.).to(tl.float32)
    rows = expert * N + cols
    raw = tl.load(w + rows[:, None] * K + axis[None, :], mask[None, :], 0).to(tl.int32)
    s = tl.load(scales + rows[:, None] * (K // 32) + axis[None, :] // 32,
                mask[None, :], 127).to(tl.int32)
    if USE_LUT:
        val = tl.load(lut + (s << 8) + raw).to(tl.float32)
    else:
        # A normalized E4M3 significand times a power of two changes only the
        # FP32 exponent. Handle E4M3 subnormals (1..7) before rounding to half.
        mag = raw & 127
        normal = (mag << 20) + ((s - 7) << 23)
        sub = mag.to(tl.float32).to(tl.int32, bitcast=True) + ((s - 136) << 23)
        bits = tl.where(mag >= 8, normal, tl.where(mag == 0, 0, sub)) | ((raw & 128) << 24)
        val = bits.to(tl.float32, bitcast=True).to(tl.float16).to(tl.float32)
    acc = tl.sum(act[None, :] * val, 1)
    if FUSE:
        gu = tl.reshape(acc.to(tl.float16).to(tl.float32), (BN // 2, 2))
        gate, up = tl.split(gu)
        weight = tl.load(routing + slot).to(tl.float32)
        clipped_g = GATE_CAP * tl.extra.cuda.libdevice.tanh(gate / GATE_CAP)
        clipped_u = UP_CAP * tl.extra.cuda.libdevice.tanh(up / UP_CAP)
        h = tl.sigmoid(gate) * clipped_g * clipped_u * weight
        tl.store(out + slot * (N // 2) + start // 2 + tl.arange(0, BN // 2), h)
    else:
        tl.store(out + slot * N + cols, acc)


def forward(layer, x, ids, weights, table, *, tiles=(8, 8), warps=(4, 4), mode='lut', fuse=True):
    """Standalone opt-in operator; this module is not imported by the service.

    `int` is restricted to a checkpoint separately audited for finite E4M3 codes
    and E8M0 bytes 113..118. `lut` handles every byte pair, including NaNs.
    """
    from vllm.models.inkling2.nvidia.ops.moe_sm75 import _activate

    if x.dtype != torch.float16 or layer.nvfp4 or ids.shape[1] != 8 or x.shape[0] > 8:
        raise ValueError("expected one SM75 verification block with FP16 inputs and MXFP8 experts")
    if mode not in ('lut', 'int') or (mode == 'int' and not getattr(layer, 'mx_codes_audited', False)):
        raise ValueError("integer conversion requires an audited checkpoint")
    slots = ids.numel()
    x = x.contiguous()
    width = layer.intermediate_size

    def run(tag, a, repeat, tile, nw, fused):
        packed = getattr(layer, tag + '_weight').view(torch.uint8)
        scale = getattr(layer, tag + '_weight_scale').view(torch.uint8)
        _, n, k = packed.shape
        if n % tile or k % 32 or a.shape[1] != k:
            raise ValueError("unsupported expert shard")
        result = x.new_empty((slots, n // 2 if fused else n))
        _project[(slots, n // tile)](
            a, packed, scale, ids, weights, table, result, n, k, repeat, tile,
            triton.next_power_of_2(k), fused, mode == 'lut', layer.gate_cap, layer.up_cap,
            num_warps=nw, num_stages=1)
        return result

    activated = run('w13', x, ids.shape[1], tiles[0], warps[0], fuse)
    if not fuse:
        gu = activated
        activated = x.new_empty((slots, width))
        _activate[(slots, triton.cdiv(width, 256))](
            gu, weights, activated, width, layer.gate_cap, layer.up_cap, 256)
    return run('w2', activated, 1, tiles[1], warps[1], False)


@triton.jit
def _check_codes(raw, invalid, total, SCALES: tl.constexpr, TILE: tl.constexpr):
    idx = tl.program_id(0) * TILE + tl.arange(0, TILE)
    val = tl.load(raw + idx, idx < total, other=114).to(tl.uint32)
    if SCALES:
        bad = (val < 113) | (val > 118)
    else:
        bad = (val & 127) == 127
    if tl.sum(bad.to(tl.int32), 0) != 0:
        tl.atomic_or(invalid, 1)


def prepare_runtime(layer):
    """Check both local native banks once, before capture. Unsupported layers use the old path."""
    if (layer.nvfp4 or layer.hidden_size != 2304 or layer.intermediate_size != 576
            or layer.num_experts != 128 or layer.topk != 8):
        return False
    tensors = [(getattr(layer, tag + suffix).view(torch.uint8), suffix == '_weight_scale')
               for tag in ('w13', 'w2') for suffix in ('_weight', '_weight_scale')]
    if any(not x.is_contiguous() for x, _ in tensors):
        return False
    invalid = torch.zeros(1, dtype=torch.int32, device=tensors[0][0].device)
    for raw, scales in tensors:
        _check_codes[(triton.cdiv(raw.numel(), 4096),)](
            raw, invalid, raw.numel(), scales, 4096, num_warps=8)
    return invalid.item() == 0


def runtime_forward(layer, x, ids, weights):
    # The integer kernel never dereferences the LUT argument.
    return forward(layer, x, ids, weights, x, tiles=(8, 8), warps=(4, 4), mode='int')
