"""Turing mixed-weight GEMM over token/expert assignments, with FP16 inputs."""
import torch
import triton
import triton.language as tl


from .e4m3 import _e4m3_value


@triton.jit
def _e2m1_value(nibble):
    # Map magnitudes 0..7 to twice their E2M1 values.
    shift = (nibble.to(tl.uint32) & 7) * 4
    magnitude = ((0xc8643210 >> shift) & 15).to(tl.float32) * 0.5
    return tl.where((nibble & 8) != 0, -magnitude, magnitude)


@triton.jit
def _project(x, weight, scale, outer, sorted_ids, experts, count, out,
             slots: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
             repeat: tl.constexpr, nvfp4: tl.constexpr,
             bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    block = tl.program_id(0)
    if block * bm >= tl.load(count):
        return
    expert = tl.load(experts + block)
    rows = tl.load(sorted_ids + block * bm + tl.arange(0, bm))
    cols = tl.program_id(1) * bn + tl.arange(0, bn)
    lane = tl.arange(0, bk)
    acc = tl.full((bm, bn), 0, tl.float32)
    for chunk in range(k // bk):
        axis = chunk * bk + lane
        a = tl.load(x + (rows[:, None] // repeat) * k + axis[None, :],
                    rows[:, None] < slots, 0)
        base = expert * n + cols
        if nvfp4:
            raw = tl.load(weight + base[:, None] * (k // 2) + axis[None, :] // 2)
            nib = (raw >> ((axis[None, :] & 1) * 4)) & 15
            val = _e2m1_value(nib)
            s = _e4m3_value(tl.load(scale + base[:, None] * (k // 16) + axis[None, :] // 16))
        else:
            val = _e4m3_value(tl.load(weight + base[:, None] * k + axis[None, :]))
            exponent = tl.load(scale + base[:, None] * (k // 32) + axis[None, :] // 32)
            s = tl.exp2(exponent.to(tl.float32) - 127)
        b = (val * s).to(tl.float16)
        acc += tl.dot(a.to(tl.float16), tl.trans(b))
    if nvfp4:
        acc *= tl.load(outer + expert)
    tl.store(out + rows[:, None] * n + cols[None, :], acc, rows[:, None] < slots)


@triton.jit
def _project_row(x, weight, scale, outer, routes, out,
                 n: tl.constexpr, k: tl.constexpr, repeat: tl.constexpr,
                 nvfp4: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.load(routes + row)
    cols = tl.program_id(1) * bn + tl.arange(0, bn)
    if expert < 0:
        tl.store(out + row * n + cols, tl.full((bn,), 0, tl.float32))
        return
    axis = tl.arange(0, bk)
    valid = axis < k
    a = tl.load(x + row // repeat * k + axis, valid, 0).to(tl.float32)
    base = expert * n + cols
    if nvfp4:
        raw = tl.load(weight + base[:, None] * (k // 2) + axis[None, :] // 2, valid[None, :], 0)
        nib = (raw >> ((axis[None, :] & 1) * 4)) & 15
        value = _e2m1_value(nib)
        s = _e4m3_value(tl.load(scale + base[:, None] * (k // 16) + axis[None, :] // 16, valid[None, :], 0))
    else:
        value = _e4m3_value(tl.load(weight + base[:, None] * k + axis[None, :], valid[None, :], 0))
        exponent = tl.load(scale + base[:, None] * (k // 32) + axis[None, :] // 32, valid[None, :], 127)
        s = tl.exp2(exponent.to(tl.float32) - 127)
    acc = tl.sum((value * s).to(tl.float16).to(tl.float32) * a[None, :], 1)
    if nvfp4:
        acc *= tl.load(outer + expert)
    tl.store(out + row * n + cols, acc)


@triton.jit
def _activate(gu, routes, output, width: tl.constexpr, gate_cap: tl.constexpr,
              up_cap: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    mask = col < width
    pair = gu + (row * width + col) * 2
    gate = tl.load(pair, mask, 0).to(tl.float32)
    up = tl.load(pair + 1, mask, 0).to(tl.float32)
    routing = tl.load(routes + row).to(tl.float32)
    clipped_gate = gate_cap * tl.extra.cuda.libdevice.tanh(gate / gate_cap)
    clipped_up = up_cap * tl.extra.cuda.libdevice.tanh(up / up_cap)
    tl.store(output + row * width + col, tl.sigmoid(gate) * clipped_gate * clipped_up * routing, mask)


def expert_forward(layer, x, ids, routing_weights, *, row_limit=64):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    small = x.shape[0] <= row_limit
    if not small:
        sorted_ids, expert_ids, count = moe_align_block_size(
            ids, 16, layer.num_experts, pad_sorted_ids=True, ignore_invalid_experts=True)
    slots = ids.numel()
    def project(tag, values, repeat):
        weight = getattr(layer, f"{tag}_weight").view(torch.uint8)
        scale = getattr(layer, f"{tag}_weight_scale").view(torch.uint8)
        n = weight.shape[1]
        assert n % 32 == 0 and values.shape[1] % 32 == 0
        outer = getattr(layer, f"{tag}_outer", values)
        if small:
            output = torch.empty((slots, n), device=x.device, dtype=torch.float16)
            width = values.shape[1]
            _project_row[(slots, n // 4)](values.contiguous(), weight, scale, outer, ids, output,
                n, width, repeat, layer.nvfp4, 4, triton.next_power_of_2(width), num_warps=4)
            return output
        output = torch.zeros((slots, n), device=x.device, dtype=torch.float16)
        _project[(expert_ids.numel(), n // 32)](
            values.contiguous(), weight, scale, outer, sorted_ids, expert_ids, count, output,
            slots, n, values.shape[1], repeat, layer.nvfp4, 16, 32, 32,
            num_warps=4, num_stages=1)
        return output
    gu = project("w13", x, ids.shape[1])
    width = gu.shape[1] // 2
    h = torch.empty((slots, width), device=x.device, dtype=torch.float16)
    _activate[(slots, triton.cdiv(width, 256))](gu, routing_weights, h, width,
        layer.gate_cap, layer.up_cap, 256)
    return project("w2", h, 1)


def prepare_marlin(layer):
    """Pack the SM75 NVFP4 tiles once, preserving all represented scales."""
    from types import SimpleNamespace
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

    if layer.hidden_size % 128 or layer.intermediate_size % 64:
        return False
    for scale in (layer.w13_weight_scale, layer.w2_weight_scale):
        byte = scale.view(torch.uint8)
        if ((byte > 126) | ((byte != 0) & (byte < 8))).any().item():
            # Marlin would round these scales to zero or discard their sign.
            return False
    layout = SimpleNamespace(num_experts=layer.num_experts, hidden_size=layer.hidden_size,
        intermediate_size_per_partition=layer.intermediate_size, params_dtype=torch.float16)
    packed = prepare_nvfp4_moe_layer_for_marlin(layout,
        layer.w13_weight, layer.w13_weight_scale, layer.w13_outer,
        layer.w2_weight, layer.w2_weight_scale, layer.w2_outer, is_act_and_mul=True)
    for tag, (weight, scale, outer) in zip(("w13", "w2"), (packed[:3], packed[3:])):
        getattr(layer, tag + "_weight").data = weight
        getattr(layer, tag + "_weight_scale").data = scale
        setattr(layer, tag + "_outer", outer)
    layer.register_buffer("marlin_workspace", marlin_make_workspace_new(layer.w2_weight.device, 4), persistent=False)
    return True


def marlin_forward(layer, x, ids, routing_weights):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.scalar_type import scalar_types

    topk = ids.shape[1]
    slots = ids.numel()
    block = 8 if slots < 8 * layer.num_experts else 16
    sorted_ids, expert_ids, count = moe_align_block_size(ids, block, layer.num_experts)
    width = layer.intermediate_size
    weights = routing_weights.view_as(ids).float().contiguous()

    def project(values, tag, repeat, n):
        # Stable ABI of the installed SM75 fork includes an activation scale
        # argument before the per-expert outer scale. Weight-only uses None.
        return torch.ops._moe_C.moe_wna16_marlin_gemm(
            values, None, getattr(layer, tag + "_weight"), None,
            getattr(layer, tag + "_weight_scale"), None, getattr(layer, tag + "_outer"),
            None, layer.marlin_workspace, sorted_ids, expert_ids, count, weights,
            block, repeat, False, scalar_types.float4_e2m1f.id,
            values.shape[0], n, values.shape[1], False, True, False, -1, -1, -1)

    gu = project(x.contiguous(), "w13", topk, 2 * width)
    h = torch.empty((slots, width), dtype=x.dtype, device=x.device)
    _activate[(slots, triton.cdiv(width, 256))](gu, weights, h, width,
        layer.gate_cap, layer.up_cap, 256)
    return project(h, "w2", 1, layer.hidden_size)
