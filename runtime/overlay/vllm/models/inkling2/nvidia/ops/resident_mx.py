"""One-copy, byte-reversible MXFP8 Marlin resident format for the audited TP2 shapes.

The original E8M0 byte scales stay in original shape and order. For the Marlin
operator only, one reusable 20.25 MiB scratch per stream holds permuted FP16
scales for a single projection. Verify may directly use the original scale bytes.
"""
from pathlib import Path
from logging import getLogger
import os

import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.models.inkling2.nvidia.ops.moe_sm75 import _activate
from vllm.scalar_type import scalar_types

from .packed_mx_decode import run as direct_verify

logger=getLogger(__name__)


@triton.jit
def _pack_bank(raw,packed,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    dst=tl.program_id(0)*B+tl.arange(0,B)
    tile=dst//1024
    lane=dst%1024
    th=lane//32
    warp=(lane//8)%4
    high=(lane//4)%2
    byte=lane%4
    k=tile//(N//64)*16+(th%4)*2+(byte//2)+(byte%2)*8
    n=(tile%(N//64))*64+warp*16+high*8+th//4
    expert=tl.program_id(1)
    val=tl.load(raw+expert*(N*K)+n*K+k)
    tl.store(packed+expert*(N*K)+dst,val)


@triton.jit
def _pack_scales(e8,out,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    p=tl.program_id(0)*B+tl.arange(0,B)
    # Each 64-scale row is transposed as 8x8 in groupwise Marlin layout.
    col=p%N
    raw_col=col//64*64+(col%8)*8+(col%64//8)
    g=p//N
    expert=tl.program_id(1)
    byte=tl.load(e8+expert*N*(K//32)+raw_col*(K//32)+g)
    # The audited exponents 113..118 give normal half values after multiplying
    # their exact scale by 256, the bias shift for Marlin E4M3.
    bits=((byte.to(tl.int32)-104)<<10).to(tl.uint16)
    tl.store(out+expert*N*(K//32)+p,bits.to(tl.float16,bitcast=True))


def pack_weight(raw):
    e,n,k=raw.shape
    assert n%64==0 and k%32==0 and n*k%256==0 and raw.dtype==torch.float8_e4m3fn
    dest=torch.empty((e,k//16,n*4),device=raw.device,dtype=torch.int32)
    _pack_bank[(n*k//256,e)](raw.view(torch.uint8),dest.view(torch.uint8),n,k,256)
    return dest


def wide_scales(raw):
    e,n,g=raw.shape
    wide=torch.empty((e,g,n),device=raw.device,dtype=torch.float16)
    _pack_scales[(n*g//64,e)](raw.view(torch.uint8),wide,n,g*32,64)
    return wide


def prepare(layer,library=None):
    # Keep raw parameters untouched if either unsupported or missing the op.
    from vllm.models.inkling2.nvidia.ops.mxfp8_sm75_verify import prepare_runtime
    if not prepare_runtime(layer):
        return False
    path=Path(library) if library else Path(__file__).with_name("inkling_marlin_mx32f32.so")
    if not path.is_file():
        logger.warning("Keeping original MXFP8 layout: private SM75 module is absent")
        return False
    try:
        torch.ops.load_library(str(path))
        packed={tag:pack_weight(getattr(layer,tag+"_weight")) for tag in ("w13","w2")}
        prepared_scales={}
        if os.environ.get("INKLING_SM75_MX_GROUPED")=="wide":
            prepared_scales={tag:wide_scales(getattr(layer,tag+"_weight_scale")) for tag in ("w13","w2")}
        workspace=marlin_make_workspace_new(layer.w2_weight.device,4)
    except (OSError,RuntimeError,ValueError) as error:
        logger.warning("Keeping original MXFP8 layout: SM75 preparation failed: %s",error)
        return False
    for tag,w in packed.items():
        getattr(layer,tag+"_weight").data=w
    for tag,s in prepared_scales.items():
        getattr(layer,tag+"_weight_scale").data=s
    layer.mx_wide_scales=bool(prepared_scales)
    layer.mx_packed_verify=os.environ.get("INKLING_MX_PACKED_VERIFY","direct")
    layer.register_buffer("mx_marlin_workspace",workspace,persistent=False)
    logger.info("SM75 grouped MXFP8 weights converted with %s scales",
                "expanded reversible" if prepared_scales else "original E8M0")
    return True


_scratch={}


def scale_scratch(like):
    stream=torch.cuda.current_stream(like.device)
    key=(like.device,stream.cuda_stream)
    if key not in _scratch:
        _scratch[key]=torch.empty(128*1152*(2304//32),device=like.device,dtype=torch.float16)
    return _scratch[key]


def project(layer,a,tag,ids,weights,align,repeat,n,config,scratch):
    block,tk,tn,ctas=config
    order,experts,count=align
    k=a.shape[-1]
    s=getattr(layer,tag+"_weight_scale")
    if s.dtype != torch.float16:
        byte=s.view(torch.uint8)
        s=scratch[:byte.numel()].view(128,k//32,n)
        _pack_scales[(n*(k//32)//64,128)](byte,s,n,k,64)
    return torch.ops.inkling_sm75_f32.mx32_gemm(a,None,getattr(layer,tag+"_weight"),
        None,s,None,None,None,layer.mx_marlin_workspace,order,experts,count,weights,
        block,repeat,False,scalar_types.float8_e4m3fn.id,a.shape[0],n,k,
        False,True,False,tk,tn,ctas)


def grouped(layer,x,ids,weights,config=None):
    m=x.shape[0]
    if config is None:
        config=(48,64,128,2) if m>=512 else ((8,-1,-1,-1) if m<=16 else (16,-1,-1,-1))
    ids=ids.contiguous()
    weights=weights.view_as(ids).float().contiguous()
    align=moe_align_block_size(ids,config[0],128)
    ws=None if layer.mx_wide_scales else scale_scratch(x)
    gu=project(layer,x.contiguous(),"w13",ids,weights,align,8,1152,config,ws)
    h=x.new_empty((ids.numel(),576))
    _activate[(ids.numel(),3)](gu,weights,h,576,layer.gate_cap,layer.up_cap,256)
    del gu
    return project(layer,h,"w2",ids,weights,align,1,2304,config,ws)


def dispatch(layer,x,ids,weights):
    if x.shape[0]<=8 and layer.mx_packed_verify=="direct":
        return direct_verify(layer,x,ids,weights,raw_scales=not layer.mx_wide_scales)
    return grouped(layer,x,ids,weights)
