"""Validated SM75 cuBLASLt plans for the replicated 3-stream loop controller."""
import ctypes as ct
import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_library = None
_plans = {}

def _lib():
    global _library
    if _library is None:
        base = Path(__file__).resolve().parents[5]
        library = ct.CDLL(str(base/"libinkling_tune_lt.so"))
        pointer = ct.c_void_p
        library.make_plan.argtypes = [ct.c_int]*4+[ct.c_bool,ct.c_size_t,ct.POINTER(pointer),ct.POINTER(ct.c_int)]
        library.run_plan.argtypes = [pointer,ct.c_int]+[pointer]*5
        _library = library
    return _library

class DownProjection:
    def __init__(self, device, rows, candidate):
        self.rows = rows
        self.candidate = candidate
        self.device = device
        self.scratch = {}
        self.workspace_size = 8*2**20
        self.library = _lib()
        self.ptr = ct.c_void_p()
        count = ct.c_int()
        status = self.library.make_plan(rows,64,4608,3,False,self.workspace_size,ct.byref(self.ptr),ct.byref(count))
        if status or candidate >= count.value:
            raise RuntimeError(f"SM75 controller heuristic unavailable (status {status}, count {count.value})")

    def run(self, x, w):
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if stream not in self.scratch:
            self.scratch[stream] = torch.empty(self.workspace_size, dtype=torch.uint8, device=self.device)
        output = x.new_empty((3,self.rows,64))
        status = self.library.run_plan(self.ptr,self.candidate,x.data_ptr(),w.data_ptr(),output.data_ptr(),
                                       self.scratch[stream].data_ptr(),stream)
        if status:
            raise RuntimeError(f"SM75 controller cuBLASLt status {status}")
        return output

def validated_down(weight, rows=8):
    device = weight.device
    if weight.shape != (4,64,4608) or weight.dtype != torch.float16 or torch.cuda.get_device_capability(device) != (7,5):
        return None
    key=(device, rows)
    if key not in _plans:
        _plans[key] = DownProjection(device,rows,4)
    plan = _plans[key]
    # Validate every layer, including possible contiguous 3-stream slices, before graph capture.
    generator = torch.Generator(device=device).manual_seed(20427)
    for w in (weight[:3],weight[1:]):
        for _ in range(4):
            x = torch.randn((3,rows,4608),dtype=torch.float16,device=device,generator=generator)
            ref = torch.bmm(x,w.transpose(1,2))
            if not torch.equal(ref,plan.run(x,w)):
                logger.warning("SM75 loop controller retained the original projection for this layer.")
                return None
    logger.info_once("SM75 loop controller accepted cuBLASLt plan after exact weight checks.")
    return plan
