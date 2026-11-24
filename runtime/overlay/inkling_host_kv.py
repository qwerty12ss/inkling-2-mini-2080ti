"""Split Inkling's layer-outer FP16 KV pool at whole-layer boundaries."""
from dataclasses import replace

import torch

_host_ranges = ()


def is_host_tensor(tensor):
    ptr = tensor.data_ptr()
    return any(begin <= ptr < end for begin, end in _host_ranges)


def lane_plan(config, layout, gpu_lanes):
    tensors = config.kv_cache_tensors
    if layout.name != 'LBNHC' or any(t.host_resident for t in tensors):
        raise ValueError('Mapped Inkling KV requires the layer-outer LBNHC layout')
    (size,) = {t.size for t in tensors}
    (stride,) = {t.layer_stride for t in tensors}
    lanes, rem = divmod(size, stride)
    if rem or lanes != 13 or not 0 < gpu_lanes < lanes:
        raise ValueError(f'Unexpected five-pass KV geometry: {lanes=} {stride=} {size=}')
    for t in tensors:
        if (t.block_stride != 131072 or stride != config.num_blocks * t.block_stride
                or t.offset % stride != 0 or t.offset + len(t.layers)*stride > size):
            raise ValueError(f'Cannot split the packed layout of {t.layers}')
    # The full DSpark layer reads its cache directly during drafting. Keep it
    # on device, along with the first lanes which also back the short windows.
    wide, = [t for t in tensors if len(t.layers) == lanes]
    draft_lane, = [i for i, name in enumerate(wide.layers) if '.layers.1.self_attn.' in name]
    order = [draft_lane] + [i for i in range(lanes) if i != draft_lane]
    return stride, sorted(order[:gpu_lanes]), sorted(order[gpu_lanes:])


def build_views(config, layout, kernel_blocks, lane_buffers):
    from vllm.v1.kv_cache_interface import create_kv_cache_views
    from vllm.v1.worker.utils import layer_kernel_block_size, layer_spec_for_tensor
    views = {}
    for tensor in config.kv_cache_tensors:
        group, spec = layer_spec_for_tensor(config, tensor)
        if not spec.has_layer_views or spec.dtype != torch.float16:
            raise ValueError(f'Expected rank-local FP16 KV for {tensor.layers}')
        for index, name in enumerate(tensor.layers):
            lane = tensor.offset // tensor.layer_stride + index
            single = replace(tensor, offset=0, size=tensor.layer_stride, layers=[name])
            view, = create_kv_cache_views(lane_buffers[lane], spec, config.num_blocks,
                                          layout, single, layer_kernel_block_size(group, kernel_blocks))
            views[name] = view
    return views


def allocate(config, device, layout, kernel_blocks, gpu_lanes):
    from vllm.logger import init_logger
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    global _host_ranges
    stride, resident, mapped = lane_plan(config, layout, gpu_lanes)
    gpu = torch.zeros((len(resident), stride), dtype=torch.int8, device=device)
    with torch.cuda.device(device):
        cpu = torch.zeros((len(mapped), stride), dtype=torch.int8, device='cpu', pin_memory=True)
        uva = get_accelerator_view_from_cpu_tensor(cpu)
    if uva.device != gpu.device or uva.shape != cpu.shape or not cpu.is_pinned():
        raise RuntimeError('CUDA did not create the requested rank-local mapped FP16 backing')
    # from_blob in get_cuda_view_from_cpu_tensor retains the underlying pinned
    # tensor while any of its CUDA views are alive, including captured graphs.
    buffers = dict(zip(resident, gpu.unbind(), strict=True))
    buffers.update(zip(mapped, uva.unbind(), strict=True))
    views = build_views(config, layout, kernel_blocks, buffers)
    _host_ranges = ((uva.data_ptr(), uva.data_ptr() + uva.numel()),)
    init_logger('vllm.inkling_host_kv').info(
        'Inkling FP16 KV on %s: %d blocks, %d device bytes (lanes %s), %d mapped host bytes (lanes %s)',
        device, config.num_blocks, gpu.numel(), resident, cpu.numel(), mapped)
    return views
