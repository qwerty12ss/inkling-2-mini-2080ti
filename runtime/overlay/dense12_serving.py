"""Losslessly stored resident target QKV/O with selected SM75 decode and exact prefill materialization."""
import ctypes
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import re

import torch

log=logging.getLogger('vllm.models.inkling2.dense12')
ROOT=Path(__file__).resolve().parent
REPORTED=set()


@lru_cache(maxsize=1)
def library():
    mod=ctypes.CDLL(str(ROOT/'libdense12_integrated.so'))
    ptr=ctypes.c_void_p;integer=ctypes.c_int
    mod.small_dense.argtypes=[ptr]*8+[integer]*6+[ptr]
    mod.dense12_qkv.argtypes=[ptr]*10+[integer,ptr]
    mod.dense12_materialize.argtypes=[ptr]*6+[integer]*2+[ptr]
    for name in ('small_dense','dense12_qkv','dense12_materialize'):getattr(mod,name).restype=integer
    return mod


class ServingBank:
    def __init__(self,data,dev):
        self.dev=dev;self.meta=data['meta'];self.n,self.k=self.meta['n'],self.meta['k']
        self.planes=tuple(x.to(dev) for x in data['planes'])
        self.ptrs=tuple(x.data_ptr() for x in self.planes)
        self.reported=set()

    def report(self,kind,m):
        key=(self.meta['rank'],self.meta['kind'],kind,m if m<=8 else 'prefill')
        if key not in REPORTED:
            REPORTED.add(key)
            log.info('SM75 dense12 dispatch rank=%s matrix=%s mode=%s rows=%s capture=%s',
                     *key,torch.cuda.is_current_stream_capturing())

    def materialize(self):
        y=torch.empty((self.n,self.k),dtype=torch.float16,device=self.dev)
        stream=torch.cuda.current_stream(self.dev).cuda_stream
        status=library().dense12_materialize(*self.ptrs,y.data_ptr(),self.n,self.k,stream)
        if status:raise RuntimeError(f'dense12 materialize returned {status}')
        return y

    def project(self,x,xk=None,xv=None):
        # Matrix rows not divisible by 16 use the original FP16 materialization path.
        if not 1<=len(x)<=8 or x.shape!=(len(x),self.k) or x.dtype!=torch.float16:
            raise ValueError('Invalid SM75 direct projection shape')
        x=x.contiguous();m=len(x);y=x.new_empty((m,self.n))
        stream=torch.cuda.current_stream(x.device).cuda_stream;funcs=library()
        partial=y.data_ptr() # split=1 never addresses temporary storage.
        if xk is None:
            status=funcs.small_dense(x.data_ptr(),*self.ptrs,partial,y.data_ptr(),m,self.n,self.k,1,8,1,stream)
        else:
            assert self.n==4304 and self.k==2304 and xk.shape==x.shape and xv.shape==x.shape
            xk=xk.contiguous();xv=xv.contiguous()
            status=funcs.dense12_qkv(x.data_ptr(),xk.data_ptr(),xv.data_ptr(),*self.ptrs,partial,y.data_ptr(),m,stream)
        if status:raise RuntimeError(f'dense12 projection returned {status}')
        return y


def init(module):
    existing=getattr(module,'_dense12_banks',None)
    if existing is not None and module.qkvr.weight.numel()==module.wo_ud.weight.numel()==0:
        return existing
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
    dev=module.qkvr.weight.device
    match=re.search(r'(?:^|\.)layers\.(\d+)\.attn$',module.prefix)
    if not match or torch.cuda.get_device_capability(dev)!=(7,5) or get_tensor_model_parallel_world_size()!=2:
        return None
    if get_current_vllm_config().lora_config is not None:
        raise ValueError('This frozen-weight SM75 experiment requires LoRA disabled')
    rank=get_tensor_model_parallel_rank();i=int(match[1]);cache=Path(os.environ['INKLING_DENSE12_CACHE'])
    banks=[]
    for name,owner in (('qkv',module.qkvr),('out',module.wo_ud)):
        raw=owner.weight
        data=torch.load(cache/f'{i}-{rank}-{name}.pt',mmap=True,weights_only=True,map_location='cpu')
        bank=ServingBank(data,dev);meta=bank.meta;valid=meta['valid']
        if raw.shape!=(meta['n'],meta['k']) or raw.dtype!=torch.float16:raise ValueError(f'Unexpected original {name} at {module.prefix}: {raw.shape} {raw.dtype}')
        if (meta['rank'],meta['layer'],meta['kind'])!=(rank,i,name):raise ValueError('TP bank identity mismatch')
        # Both the independent source digest and the CUDA unpack are checked on both
        # ranks before any original serving storage is released. Padding is unsampled.
        observed=hashlib.sha256(raw[:valid].detach().contiguous().cpu().view(torch.uint8).numpy()).hexdigest()
        if observed!=meta['serving_half']:raise ValueError(f'Original serving SHA mismatch at {module.prefix} rank={rank} {name}')
        rebuilt=bank.materialize()
        if not torch.equal(rebuilt[:valid].view(torch.int16),raw[:valid].view(torch.int16)):
            raise ValueError(f'GPU roundtrip mismatch at {module.prefix} rank={rank} {name}')
        del rebuilt
        raw.data=raw.new_empty((0,))
        banks.append(bank)
        log.info('SM75 dense12 resident rank=%d layer=%d matrix=%s source_bf16=%s serving_fp16=%s packed_bytes=%d raw_bytes=%d roundtrip=exact',
                 rank,i,name,meta['original_bf16'],observed,meta['physical_bytes'],meta['n']*meta['k']*2)
    return tuple(banks)
