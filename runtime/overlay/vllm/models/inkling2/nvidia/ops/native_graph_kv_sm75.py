"""Experimental fixed-capacity FA2 split plan with request metadata refreshed on GPU.

One physical request. The FlashInfer v0.6.18.post1 FA2 scheduler ABI is checked at
initialization. No private workspace view is used until the initial plan validates.
"""
from math import ceil, lcm

import flashinfer
import torch
import triton
import triton.language as tl
import tvm_ffi

from .attention_sm90 import turing_prefill


@triton.jit
def _write_plan(Length, Table, Indptr, Indices, NearTable, NearLen, FarPages,
                ReqIndices, QoTiles, KvTiles, ChunkSize, MergePtr, OutPtr, Mask,
                PAGE:tl.constexpr, ALIGN:tl.constexpr, EXTENT:tl.constexpr,
                T:tl.constexpr, GRID:tl.constexpr, Q_TILES:tl.constexpr,
                MAX_SHARDS:tl.constexpr, NPAGES:tl.constexpr, NEAR:tl.constexpr,
                B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    length=tl.load(Length)
    far=tl.maximum(length-T-EXTENT,0)//ALIGN*ALIGN
    pages=far//PAGE
    active_pages=tl.maximum(pages,1)
    # Cover a single sentinel physical page when no far pages exist. Its
    # contribution is discarded by _merge; FA2 never sees an empty softmax.
    chunk=tl.maximum(2,tl.cdiv(active_pages,MAX_SHARDS))
    parts=tl.cdiv(active_pages,chunk)
    kv=tl.load(Table+i,mask=i<pages,other=0)
    tl.store(Indices+i,kv,mask=i<NPAGES)
    tail=tl.load(Table+pages+i, mask=(i<NEAR)&((pages+i)*PAGE<length),other=0)
    tl.store(NearTable+i,tail,mask=i<NEAR)
    # Native partial-output stride: number of REAL KV chunks for each Q row.
    tl.store(ReqIndices+i,0,mask=i<GRID)
    tl.store(QoTiles+i,i//parts,mask=i<GRID)
    tl.store(KvTiles+i,i%parts,mask=i<GRID)
    tl.store(Mask+i,i<Q_TILES*parts,mask=i<GRID)
    tl.store(MergePtr+i,i*parts,mask=i<=T)
    if tl.program_id(0)==0:
        tl.store(ChunkSize,chunk*PAGE)
        tl.store(OutPtr,0)
        tl.store(OutPtr+1,T*parts)
        tl.store(Indptr,0)
        tl.store(Indptr+1,active_pages)
        tl.store(NearLen,length-far)
        tl.store(FarPages,pages)


@triton.jit
def _merge(Far,FarLog,Near,NearLog,Pages,H:tl.constexpr,D:tl.constexpr,T:tl.constexpr):
    t,h=tl.program_id(0),tl.program_id(1)
    i=(t*H+h)*D+tl.arange(0,D)
    far=tl.load(Pages)>0
    a=tl.load(FarLog+t*H+h,mask=far,other=-float('inf'))*.6931471805599453
    b=tl.load(NearLog+h*T+t)
    top=tl.maximum(a,b)
    top=tl.where(top == -float('inf'),0.,top)
    wa,wb=tl.exp(a-top),tl.exp(b-top)
    x=tl.load(Far+i,mask=far,other=0.).to(tl.float32)
    y=tl.load(Near+i)
    y=tl.where(wb>0,y,0.)
    zz=wa+wb
    tl.store(Near+i,(x*wa+y*wb)/tl.where(zz>0,zz,1.))
    tl.store(NearLog+h*T+t,top+tl.log(zz))


class NativeGraphAttention:
    """Construct in eager warmup, run repeatedly under capture or replay."""
    def __init__(self,q,k,table,extent,shards=None,near_splits=8,whole=False,wrapper_factory=None):
        t,h,d=q.shape
        page,hk=k.shape[1:3]
        pages=table.shape[1]
        assert t in (1,4,6,8) and d==128 and page==64 and q.dtype==k.dtype==torch.float16
        self.capture_logged=False
        self.extent=extent
        self.align=lcm(page,128)
        self.near=ceil((t+extent+self.align-1)/page)
        self.near_splits=near_splits
        def ints(shape):
            return q.new_empty(shape,dtype=torch.int32)
        self.qptr,self.indptr,self.idx,self.last=ints(2),ints(2),ints(pages),ints(1)
        self.ntab,self.nlen,self.fpages=ints((1,self.near)),ints(1),ints(1)
        self.fout=torch.empty_like(q)
        self.flog=q.new_empty((t,h),dtype=torch.float32)
        self.out=q.new_empty(q.shape,dtype=torch.float32)
        self.log=q.new_empty((h,t),dtype=torch.float32)
        self.whole = whole
        if whole:
            if extent != 1024: raise ValueError("Only target full attention is specialized")
            from .full_fa2_sm75 import create_wrapper
        else:
            create_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        self.wrapper=(wrapper_factory or create_wrapper)(
            q.new_empty((16<<20,),dtype=torch.uint8),kv_layout='NHD',backend='fa2',
            use_cuda_graph=True,qo_indptr_buf=self.qptr,paged_kv_indptr_buf=self.indptr,
            paged_kv_indices_buf=self.idx,paged_kv_last_page_len_buf=self.last)
        cpu=lambda value:torch.tensor(value,dtype=torch.int32,device='cpu')
        with tvm_ffi.use_torch_stream():
            self.wrapper.plan(cpu([0,t]),cpu([0,pages]),torch.zeros(pages,dtype=torch.int32,device='cpu'),
                              cpu([page]),h,hk,d,page,causal=False,q_data_type=q.dtype,
                              kv_data_type=k.dtype,o_data_type=torch.float32 if whole else q.dtype,sm_scale=d**-.5)
        plan=list(self.wrapper._plan_info)
        if not (len(plan)==15 and plan[1]==t and plan[13:]==[1,1]):
            raise ValueError(f'Unsupported FA2 plan layout: {plan}')
        self.grid=int(plan[0])
        self.qtiles=ceil(t*(h//hk)/int(plan[3]))
        self.shards=min(self.grid//self.qtiles,shards or self.grid)
        if self.shards<1: raise ValueError('No capacity for split KV')
        buf=self.wrapper._int_workspace_buffer
        host=self.wrapper._pin_memory_int_workspace_buffer
        def view(i,n,dtype,source=buf):
            offset=int(plan[i]); size=dtype.itemsize*n
            if offset<0 or offset+size>source.nbytes or offset%dtype.itemsize:
                raise ValueError(f'Invalid FA2 plan field {i}')
            return source[offset:offset+size].view(dtype)
        self.req=view(4,self.grid,torch.int32)
        self.qtile=view(5,self.grid,torch.int32)
        self.ktile=view(6,self.grid,torch.int32)
        self.merge=view(7,t+1,torch.int32)
        self.optr=view(8,2,torch.int32)
        self.chunk=view(9,1,torch.int32)
        self.mask=view(12,self.grid,torch.bool)
        # Validate against CPU materialization of the default plan before CUDA graph capture.
        initial_size=int(view(9,1,torch.int32,host)[0])
        if initial_size<page or initial_size%page: raise ValueError('Invalid FA2 initial chunk')
        initial_parts=ceil(pages*page/initial_size)
        active=self.qtiles*initial_parts
        def contents(field,n):return view(field,n,torch.int32,host).tolist()
        if not (active<=self.grid and contents(4,active)==[0]*active and
                contents(5,active)==[i//initial_parts for i in range(active)] and
                contents(6,active)==[i%initial_parts for i in range(active)] and
                contents(7,t+1)==[i*initial_parts for i in range(t+1)] and
                contents(8,2)==[0,t*initial_parts] and
                view(12,self.grid,torch.bool,host).tolist()==[i<active for i in range(self.grid)]):
            raise ValueError('Unexpected FA2 initial scheduler metadata')

    def run(self,q,k,v,rel,table,length,starts):
        if self.whole:
            from .full_fa2_sm75 import run
            return run(self,q,k,v,rel,table,length,starts)
        t,h,d=q.shape
        pages=table.shape[1]
        page=k.shape[1]
        _write_plan[(triton.cdiv(max(pages,self.grid,t+1,self.near),256),)](
            length,table,self.indptr,self.idx,self.ntab,self.nlen,self.fpages,
            self.req,self.qtile,self.ktile,self.chunk,self.merge,self.optr,self.mask,
            page,self.align,self.extent,t,self.grid,self.qtiles,self.shards,pages,self.near,256)
        with tvm_ffi.use_torch_stream():
            self.wrapper.run(q,(k,v),out=self.fout,lse=self.flog,return_lse=True)
        turing_prefill(q,k,v,self.out,self.log,rel,num_splits=self.near_splits,
                       block_table=self.ntab,cache_seqlens=self.nlen,cu_seqlens_q=starts,
                       max_query_len=t,softmax_scale=d**-.5,window=0)
        _merge[(t,h)](self.fout,self.flog,self.out,self.log,self.fpages,h,d,t,num_warps=4)
        return self.out,self.log
