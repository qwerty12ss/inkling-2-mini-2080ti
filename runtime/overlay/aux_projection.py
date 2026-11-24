"""SM75 auxiliary proposal scores. The original LM head independently verifies or refines them."""
import ctypes
from pathlib import Path

import torch


class AuxiliaryHead:
    def __init__(self, weight):
        rows, hidden = weight.shape
        assert rows == 100512 and hidden == 2304 and weight.dtype == torch.float16
        assert torch.cuda.get_device_capability(weight.device) == (7, 5)
        self.n = rows
        self.packed = torch.empty((rows // 8,18,2,32,4,4),device=weight.device,dtype=torch.int8)
        self.scales = torch.empty((rows // 8,18,8),device=weight.device,dtype=torch.float32)
        self.q = torch.empty((8, hidden), dtype=torch.int8, device=weight.device)
        self.qs = torch.empty((8, 18), dtype=torch.float32, device=weight.device)
        self.module = ctypes.CDLL(str(Path(__file__).resolve().with_name("libinkling_head_int8.so")))
        self.op = self.module.head_projection
        ptr = ctypes.c_void_p
        self.op.argtypes = [ptr, ctypes.c_int, ptr, ptr, ptr, ptr, ptr, ctypes.c_int, ptr]
        self.refresh(weight)

    @torch.no_grad()
    def refresh(self, weight):
        """Update derived proposal weights in place, including after a target weight reload."""
        assert weight.shape == (self.n, 2304) and weight.dtype == torch.float16
        for lo in range(0, self.n, 512):
            hi = min(lo + 512, self.n)
            block = weight[lo:hi].float().reshape(hi-lo,18,128)
            scale = block.abs().amax(-1).clamp_min(1e-20) / 127
            quant = (block / scale[:,:,None]).round().clamp(-127,127).to(torch.int8)
            self.packed[lo//8:hi//8] = quant.reshape(-1,8,18,2,4,4,4).permute(0,2,3,1,5,4,6).reshape(-1,18,2,32,4,4)
            self.scales[lo//8:hi//8] = scale.reshape(-1,8,18).transpose(1,2)

    def __call__(self, h):
        assert 1 <= len(h) <= 8 and h.is_contiguous()
        out = h.new_empty((len(h), self.n))
        status = self.op(h.data_ptr(),len(h),self.packed.data_ptr(),self.scales.data_ptr(),
                         self.q.data_ptr(),self.qs.data_ptr(),out.data_ptr(),self.n,
                         torch.cuda.current_stream(h.device).cuda_stream)
        if status:
            raise RuntimeError(f"SM75 auxiliary vocabulary projection failed with CUDA status {status}")
        return out
