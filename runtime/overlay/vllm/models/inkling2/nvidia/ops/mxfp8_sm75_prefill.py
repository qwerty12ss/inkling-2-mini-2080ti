"""Use one reusable group-sized FP16 weight scratch and cuBLAS strided-batched GEMM."""
import torch
import triton
from triton import language as tl
from vllm.models.inkling2.nvidia.ops.e4m3 import _e4m3_value
from vllm.models.inkling2.nvidia.ops.moe_sm75 import _activate

@triton.jit
def _unpack_weights(weight,scale,out, BLOCK:tl.constexpr):
    p=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    b=tl.load(weight+p)
    s=tl.load(scale+p//32)
    tl.store(out+p, _e4m3_value(b)*tl.exp2(s.to(tl.float32)-127.))

@triton.jit
def _pack_rows(A, Order, Starts, Counts, Packed, start, REPEAT:tl.constexpr, K:tl.constexpr, CAP:tl.constexpr, B:tl.constexpr):
    row, local=tl.program_id(0),tl.program_id(1)
    expert=start+local
    col=tl.program_id(2)*B+tl.arange(0,B)
    ok=row<tl.load(Counts+expert)
    i=tl.load(Order+tl.load(Starts+expert)+row,ok,other=0)
    val=tl.load(A+(i//REPEAT)*K+col,ok & (col<K),other=0)
    tl.store(Packed+(local*CAP+row)*K+col,val,col<K)

@triton.jit
def _unpack_rows(Packed, Order, Starts, Counts, Out, start, N:tl.constexpr, CAP:tl.constexpr, B:tl.constexpr):
    row, local=tl.program_id(0),tl.program_id(1)
    expert=start+local
    if row<tl.load(Counts+expert):
        col=tl.program_id(2)*B+tl.arange(0,B)
        i=tl.load(Order+tl.load(Starts+expert)+row)
        val=tl.load(Packed+(local*CAP+row)*N+col,col<N,other=0)
        tl.store(Out+i*N+col,val,col<N)

def forward(layer,x,ids,weights,group=16):
    experts=layer.num_experts
    routes=ids.reshape(-1)
    routes=torch.where((routes>=0) & (routes<experts), routes, experts)
    order=routes.argsort()
    counts=torch.bincount(routes.long(),minlength=experts+1)[:experts].int()
    starts=counts.cumsum(0)-counts
    cap=triton.cdiv(int(counts.max().item()),8)*8
    slots=ids.numel()
    if cap == 0:
        return x.new_zeros((slots,layer.hidden_size))
    workspace=x.new_empty(group*max(layer.w13_weight[0].numel(),layer.w2_weight[0].numel()))
    def projection(a,tag,repeat):
        w=getattr(layer,tag+'_weight').view(torch.uint8)
        s=getattr(layer,tag+'_weight_scale').view(torch.uint8)
        e,n,k=w.shape
        packed=x.new_empty((group,cap,k))
        scratch=workspace[:group*n*k].view(group,n,k)
        out=x.new_zeros((slots,n))
        for start in range(0,e,group):
            _pack_rows[(cap,group,triton.cdiv(k,256))](a,order,starts,counts,packed,start,repeat,k,cap,256)
            _unpack_weights[(group*n*k//256,)](w[start],s[start],scratch,256)
            result=torch.bmm(packed,scratch.transpose(1,2))
            _unpack_rows[(cap,group,triton.cdiv(n,256))](result,order,starts,counts,out,start,n,cap,256)
        return out
    gu=projection(x,'w13',ids.shape[1])
    h=x.new_empty((slots,layer.intermediate_size))
    _activate[(slots,triton.cdiv(h.shape[1],256))](gu,weights,h,h.shape[1],layer.gate_cap,layer.up_cap,256)
    return projection(h,'w2',1)
