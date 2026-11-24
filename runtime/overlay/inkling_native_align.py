"""Graph-safe adapter for the native, stable counting sort."""
import ctypes
from functools import lru_cache
from pathlib import Path

import torch

@lru_cache(maxsize=1)
def _load():
    lib=ctypes.CDLL(str(Path(__file__).resolve().with_name("libinkling_stable_routes.so")))
    ptr=ctypes.c_void_p
    i=ctypes.c_int
    op=lib.inkling_stable_routes
    op.argtypes=[ptr,i,ptr,i,ptr,ptr,ptr,i,ptr]
    op.restype=i
    return op

def align(ids,block,experts):
    assert ids.ndim==2 and ids.shape[1]==8 and ids.dtype==torch.int32
    assert ids.is_contiguous() and experts==128 and block in (8,16,48)
    slots=ids.numel()
    bound=min(slots+experts*(block-1),slots*block)
    order=ids.new_empty(bound)
    owners=ids.new_empty((bound+block-1)//block)
    count=ids.new_empty(1)
    grid=ids.new_empty(((slots+255)//256,experts)) if slots>1024 else count
    status=_load()(ids.data_ptr(),slots,order.data_ptr(),bound,owners.data_ptr(),count.data_ptr(),
                   grid.data_ptr(),block,torch.cuda.current_stream(ids.device).cuda_stream)
    if status:
        raise RuntimeError(f"Stable route alignment CUDA status {status}")
    return order,owners,count
