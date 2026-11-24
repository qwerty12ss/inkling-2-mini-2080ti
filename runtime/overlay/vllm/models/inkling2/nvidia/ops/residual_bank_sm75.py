"""Turing prefill AttnRes, preserving the runtime's three 1024-wide reduction groups.

For D=2304 the three source pieces stay in registers through score, mix
and optional output normalization. The original FP32 FMA, reduction and
rounding sequence is held fixed. No persistent state is needed.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from vllm.models.inkling2.nvidia.ops.norm import rsqrt


@triton.jit
def _read(B,L,A,stride,r,d,C:tl.constexpr,N:tl.constexpr,S:tl.constexpr,ADD:tl.constexpr):
    if S<N:
        x=tl.load(B+S*stride+r+d,d<C,0).to(tl.float32)
    else:
        x=tl.load(L+r+d,d<C,0).to(tl.float32)
        if ADD:x+=tl.load(A+r+d,d<C,0).to(tl.float32)
    return x


@triton.jit
def _bank(B,L,A,Q,W,O,stride,T,eps,output_eps,
          N:tl.constexpr,D:tl.constexpr,QJ:tl.constexpr,N2:tl.constexpr,
          ADD:tl.constexpr,NORM:tl.constexpr):
    ri=tl.program_id(1).to(tl.int64);r=ri*D;qi=tl.program_id(0)
    d=tl.arange(0,1024);m=tl.arange(0,N2)
    q0=tl.load(Q+qi*D+d);q1=tl.load(Q+qi*D+d+1024)
    q2=tl.load(Q+qi*D+d+2048,d<256,0)
    xs=()
    invs=tl.full((N2,),0,tl.float32)
    scores=tl.full((N2,),float('-inf'),tl.float32)
    for j in tl.static_range(N+1):
        x0=_read(B,L,A,stride,r,d,D,N,j,ADD)
        x1=_read(B,L,A,stride,r,d+1024,D,N,j,ADD)
        x2=_read(B,L,A,stride,r,d+2048,D,N,j,ADD)
        xs+=((x0,x1,x2),)
        square=tl.fma(x2,x2,tl.fma(x1,x1,x0*x0))
        dots=tl.fma(q2,x2,tl.fma(q1,x1,q0*x0))
        inv=rsqrt(tl.fma(tl.sum(square,0),1./D,eps))
        invs=tl.where(m==j,inv,invs)
        scores=tl.where(m==j,tl.sum(dots,0)*inv,scores)
    prob=libdevice.exp(scores-tl.max(scores,0));iz=tl.div_rn(1.,tl.sum(prob,0))
    prob*=invs
    y0=tl.full((1024,),0,tl.float32);y1=tl.full((1024,),0,tl.float32);y2=tl.full((1024,),0,tl.float32)
    for j in tl.static_range(N+1):
        p=tl.sum(tl.where(m==j,prob,0.),0)
        x0=xs[j][0];x1=xs[j][1];x2=xs[j][2]
        y0=tl.fma(p,x0,y0);y1=tl.fma(p,x1,y1);y2=tl.fma(p,x2,y2)
    y0*=iz;y1*=iz;y2*=iz
    if NORM:
        sq=(tl.sum(y0*y0,0)+tl.sum(y1*y1,0))+tl.sum(y2*y2,0)
        norm=rsqrt(tl.div_rn(sq,D)+output_eps)
        y0=y0*norm*tl.load(W+qi*D+d).to(tl.float32)
        y1=y1*norm*tl.load(W+qi*D+d+1024).to(tl.float32)
        y2=y2*norm*tl.load(W+qi*D+d+2048,d<256,0).to(tl.float32)
    out=O+(qi.to(tl.int64)*T+ri)*D
    tl.store(out+d,y0);tl.store(out+d+1024,y1);tl.store(out+d+2048,y2,d<256)


def run(stack,last,add,queries,norm,eps=1e-6,warps=4):
    t,d=last.shape;s=queries.shape[0];n=len(stack)
    assert d==2304 and last.is_contiguous() and (add is None or add.is_contiguous())
    output=torch.empty((s,t,d),device=last.device,dtype=last.dtype if norm is not None else torch.float32)
    _bank[(s,t)](stack,last,add if add is not None else last,queries,norm if norm is not None else queries,output,
                 stack.stride(0),t,1e-6,eps,N=n,D=d,QJ=s,N2=triton.next_power_of_2(n+1),ADD=add is not None,NORM=norm is not None,num_warps=warps,enable_fp_fusion=False)
    return output
