"""Inkling's replicated image tower for SM75, with resident original BF16 parameters."""

import json
import logging
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.models.inkling2.nvidia.vision import (
    Inkling2VisionEncoder,
    _CLIP_MEAN,
    _CLIP_STD,
    _PROCESSOR_CLIP_STD,
)

log = logging.getLogger(__name__)


def rounded(x):
    # Preserve the ViT's BF16 activation boundaries. FP16 is the Turing GEMM operand.
    return x.to(torch.bfloat16).to(torch.float16)


def linear(module, x):
    # Keep only the current matrix's temporary FP16 conversion on the GPU.
    bias = module.bias.half() if module.bias is not None else None
    return rounded(F.linear(x, module.weight.half(), bias))


def norm(module, x):
    value = F.layer_norm(x.float(), module.normalized_shape, module.weight.float(),
                         module.bias.float(), module.eps)
    return rounded(value)


def residual(hidden, update, gamma):
    scaled = rounded(update.float() * gamma.float())
    return rounded(hidden.float() + scaled.float())


def rotary_phases(height, width, head_dim, device):
    inverse = torch.tensor([10000.0 ** (-i / head_dim) for i in range(0, head_dim, 4)],
                           dtype=torch.float32, device=device)
    ys, xs = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing="ij")
    # Interleaved pairs rotate by column then row. The first row is CLS at zero.
    phase = torch.cat(((xs.reshape(-1, 1) + 1) * inverse,
                       (ys.reshape(-1, 1) + 1) * inverse), dim=1)
    phase = F.pad(phase, (0, 0, 1, 0)).unsqueeze(1)
    return phase.cos(), phase.sin()


def rotary(x, cos, sin):
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    return rounded(torch.stack((even * cos - odd * sin,
                                even * sin + odd * cos), dim=-1).flatten(-2))


def attention(module, hidden, cos, sin):
    qkv = linear(module.qkv_proj, hidden)
    q, k, v = qkv.view(-1, 3, module.num_heads, module.head_dim).unbind(1)
    q, k = rotary(q, cos, sin), rotary(k, cos, sin)
    q, k, v = (t.transpose(0, 1).contiguous().unsqueeze(0) for t in (q, k, v))
    # The complete image is visible to each query, including CLS. Tiling only
    # queries bounds temporary storage if PyTorch uses the math SDPA backend.
    attended = torch.empty_like(q)
    for start in range(0, q.size(2), 128):
        tile = slice(start, start + 128)
        attended[:, :, tile] = F.scaled_dot_product_attention(q[:, :, tile], k, v)
    value = attended[0].transpose(0, 1).reshape_as(hidden)
    return linear(module.out_proj, rounded(value))


class SM75VisionEncoder(Inkling2VisionEncoder):
    def __init__(self, config, text_hidden_size):
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise ValueError(f"Inkling SM75 vision requires a Turing GPU, got {device}")
        # The standard parameter hierarchy and loader keep the original BF16
        # checkpoint, including its fused Q/K/V slots. No host copy is retained.
        with torch.device("meta"):
            super().__init__(config, text_hidden_size)
        self.to_empty(device=device)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        self.telemetry_dir = os.environ.get("INKLING_VISION_TELEMETRY_DIR")
        self.check_dir = os.environ.get("INKLING_VISION_CHECK_EMBEDS")
        log.warning("Inkling vision GPU rank=%s device=%s model=%s resident_parameter_bytes=%d dtype=%s",
                    self.tp_rank, device, torch.cuda.get_device_name(device),
                    self.parameter_bytes, self.proj_out.weight.dtype)

    def image(self, raw, h, w):
        m, p = self.merge_size, self.patch_size
        height, width = h * m, w * m
        emb = self.backbone.embeddings
        mean = raw.new_tensor(_CLIP_MEAN, dtype=torch.float32)
        clip_std = raw.new_tensor(_PROCESSOR_CLIP_STD, dtype=torch.float32)
        pe_std = raw.new_tensor(_CLIP_STD, dtype=torch.float32)
        clip = rounded((raw.float() / 255.0 - mean) / clip_std)
        pixels = rounded(2.0 * (clip.float() * pe_std + mean) - 1.0)
        pixels = (pixels.reshape(h, w, m, p, m, p, 3).permute(0, 2, 1, 4, 3, 5, 6)
                  .reshape(height * width, -1))
        patches = linear(emb.patch_embedding, pixels)
        hidden_size = patches.shape[-1]
        g = self.posemb_grid_size
        grid = emb.position_embedding.view(1, g, g, hidden_size).permute(0, 3, 1, 2)
        pos = F.interpolate(grid.float(), (height, width), mode="bilinear", align_corners=False)
        pos = rounded(pos[0].permute(1, 2, 0).reshape(-1, hidden_size))
        patches = rounded(patches.float() + pos.float())
        cls = rounded(emb.class_embedding.float() + emb.cls_position_embedding.float())
        hidden = norm(self.backbone.ln_pre, torch.cat((cls[None], patches)))
        cos, sin = rotary_phases(height, width, self.backbone.layers[0].self_attn.head_dim, raw.device)

        for layer in self.backbone.layers:
            update = attention(layer.self_attn, norm(layer.ln_1, hidden), cos, sin)
            hidden = residual(hidden, update, layer.ls_1.gamma)
            up = linear(layer.mlp.c_fc, norm(layer.ln_2, hidden))
            down = linear(layer.mlp.c_proj, rounded(F.gelu(up.float())))
            hidden = residual(hidden, down, layer.ls_2.gamma)

        pooled = (hidden[1:].reshape(h, m, w, m, hidden_size).permute(0, 2, 1, 3, 4)
                  .reshape(h * w, -1))
        return linear(self.proj_out, rounded(F.gelu(linear(self.proj_in, pooled).float())))

    @torch.inference_mode()
    def forward(self, pixels, grid_thw):
        device = self.proj_out.weight.device
        if device.type != "cuda" or pixels.device != device:
            raise RuntimeError(f"Inkling vision requires pixels and weights on {device}, got {pixels.device}")
        trace = self.telemetry_dir and (Path(self.telemetry_dir) / "enabled").is_file()
        if trace:
            torch.cuda.synchronize(device)
            start = time.time_ns()
            free_before = torch.cuda.mem_get_info(device)[0]
        grids = grid_thw.tolist()
        if not grids or any(t != 1 or h < 1 or w < 1 for t, h, w in grids):
            raise ValueError("Expected positive, single-frame Inkling image grids")
        counts = [h * w for _, h, w in grids]
        side = self.patch_size * self.merge_size
        if tuple(pixels.shape) != (sum(counts), side, side, 3):
            raise ValueError("Inkling image grid and pixel shapes differ")
        outputs = [self.image(raw, h, w) for raw, (_, h, w) in zip(pixels.split(counts), grids)]
        result = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        if trace:
            torch.cuda.synchronize(device)
            end = time.time_ns()
            event = dict(start_ns=start, end_ns=end, rank=self.tp_rank,
                         device=str(device), gpu=torch.cuda.get_device_name(device),
                         uuid=str(torch.cuda.get_device_properties(device).uuid),
                         dtype=str(self.proj_out.weight.dtype), compute="fp16_gemm_fp32_norm",
                         parameter_bytes=self.parameter_bytes, input_patches=pixels.size(0),
                         output_shape=list(result.shape), free_before_bytes=free_before,
                         free_after_bytes=torch.cuda.mem_get_info(device)[0])
            folder = Path(self.telemetry_dir)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"encoder-{os.getpid()}.jsonl").open("a") as stream:
                print(json.dumps(event), file=stream)
        # Explicitly armed, one-shot diagnostic. Keep the exact processor output
        # so a CPU reference can use the same patches and padding as the API.
        if self.check_dir:
            check = Path(self.check_dir)
            if (check / "armed").is_file():
                output = check / f"reference-rank{self.tp_rank}.pt"
                temporary = check / f".reference-{os.getpid()}.tmp"
                torch.save(dict(pixels=pixels.cpu(), grids=grids, embeddings=result.cpu()), temporary)
                temporary.replace(output)
                self.check_dir = None
        return result
