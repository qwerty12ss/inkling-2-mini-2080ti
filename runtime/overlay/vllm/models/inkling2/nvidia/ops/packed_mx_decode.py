"""Access original E4M3 bytes and E8M0 powers directly from one Marlin bank.

FP16 x eight-column verification matches the existing integer-decoding path.
No second copy of the expert weights is needed. The only persistent expansion
is one extra byte per 32 weights for Marlin's per-group FP16 scale storage.
"""
import torch
import triton
import triton.language as tl


def weight_offset(expert, row, axis, n, k):
    # [16 K x 64 N] tile, each warp packs one 16-column slab.
    tile=(expert*(k//16)+axis//16)*(n//64)+row//64
    thread=4*(row%8)+(axis%8)//2
    return tile*1024+thread*32+(row%64//16)*8+(row%16//8)*4+(axis%2)*2+axis%16//8


def scale_offset(expert,row,axis,n,k):
    return (expert*(k//32)+axis//32)*n+row//64*64+(row%8)*8+(row%64//8)


@triton.jit
def _project(x, packed, scales, ids, routing, out, N:tl.constexpr,K:tl.constexpr,
             REPEAT:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
             FUSE:tl.constexpr,GATE_CAP:tl.constexpr,UP_CAP:tl.constexpr,RAW_SCALE:tl.constexpr):
    slot=tl.program_id(0)
    expert=tl.load(ids+slot)
    start=tl.program_id(1)*BN
    cols=start+tl.arange(0,BN)
    if expert<0:
        if FUSE:tl.store(out+slot*(N//2)+start//2+tl.arange(0,BN//2),0.)
        else:tl.store(out+slot*N+cols,0.)
        return
    axis=tl.arange(0,BK)
    mask=axis<K
    act=tl.load(x+(slot//REPEAT)*K+axis,mask,0.).to(tl.float32)
    col=cols[:,None]
    ax=axis[None,:]
    tile=(expert*(K//16)+ax//16)*(N//64)+col//64
    thread=4*(col%8)+(ax%8)//2
    off=tile*1024+thread*32+(col%64//16)*8+(col%16//8)*4+(ax%2)*2+ax%16//8
    raw=tl.load(packed+off,mask[None,:],0).to(tl.int32)
    if RAW_SCALE:
        soff=(expert*N+col)*(K//32)+ax//32
        s=tl.load(scales+soff,mask[None,:],other=127).to(tl.int32)
    else:
        soff=(expert*(K//32)+ax//32)*N+col//64*64+(col%8)*8+(col%64//8)
        scale=tl.load(scales+soff,mask[None,:],other=0.).to(tl.uint16,bitcast=True)
        s=((scale.to(tl.int32)>>10)&31)+104
    mag=raw&127
    normal=(mag<<20)+((s-7)<<23)
    sub=mag.to(tl.float32).to(tl.int32,bitcast=True)+((s-136)<<23)
    bits=tl.where(mag>=8,normal,tl.where(mag==0,0,sub))|((raw&128)<<24)
    val=bits.to(tl.float32,bitcast=True).to(tl.float16).to(tl.float32)
    acc=tl.sum(act[None,:]*val,1)
    if FUSE:
        gu=tl.reshape(acc.to(tl.float16).to(tl.float32),(BN//2,2))
        gate,up=tl.split(gu)
        w=tl.load(routing+slot)
        clipped_g=GATE_CAP*tl.extra.cuda.libdevice.tanh(gate/GATE_CAP)
        clipped_u=UP_CAP*tl.extra.cuda.libdevice.tanh(up/UP_CAP)
        act_out=tl.sigmoid(gate)*clipped_g*clipped_u*w
        tl.store(out+slot*(N//2)+start//2+tl.arange(0,BN//2),act_out)
    else:
        tl.store(out+slot*N+cols,acc)


def run(layer,x,ids,weights,tiles=(8,16),raw_scales=True):
    assert x.shape[0]<=8 and ids.shape==(x.shape[0],8)
    def project(a,tag,repeat,n,tile,fused):
        k=a.shape[1]
        w=getattr(layer,tag+"_weight").view(torch.uint8)
        s=getattr(layer,tag+"_weight_scale")
        if raw_scales:s=s.view(torch.uint8)
        result=x.new_empty((ids.numel(),n//2 if fused else n))
        _project[(ids.numel(),n//tile)](a,w,s,ids,weights,result,n,k,repeat,tile,
            triton.next_power_of_2(k),fused,layer.gate_cap,layer.up_cap,raw_scales,num_warps=4)
        return result
    h=project(x,"w13",8,2*layer.intermediate_size,tiles[0],True)
    return project(h,"w2",1,layer.hidden_size,tiles[1],False)
