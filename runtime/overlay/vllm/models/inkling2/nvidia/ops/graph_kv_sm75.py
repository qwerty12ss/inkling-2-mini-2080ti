"""SM75 FP16, one-request, capture-safe global attention.

Plan one unsplit FA2 request for each fixed split in advance. On each replay,
the device chooses page boundaries from the current (post-rejection) length.
The final reduction combines their LSEs with Inkling's original near attention.
"""
from math import ceil, lcm
import os

import flashinfer
import torch
import triton
import triton.language as tl
import tvm_ffi

from vllm.logger import init_logger

from .attention_sm90 import turing_prefill

logger = init_logger(__name__)


@triton.jit
def _metadata(Length, Table, FarIndptr, Indices, NearTable, NearLen, FarPages,
              PAGE: tl.constexpr, ALIGN: tl.constexpr, EXTENT: tl.constexpr,
              T: tl.constexpr, S: tl.constexpr, NPAGES: tl.constexpr,
              NEAR: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Length)
    far = tl.maximum(length - T - EXTENT, 0) // ALIGN * ALIGN
    pages = far // PAGE
    # Each shard gets at least one real address. The reducer masks sentinel shards.
    tl.store(FarIndptr + i, i * tl.maximum(pages, S) // S, mask=i <= S)
    ids = tl.load(Table + i, mask=i < pages, other=0)
    tl.store(Indices + i, ids, mask=i < NPAGES)
    near_ids = tl.load(Table + pages + i, mask=(i < NEAR) & ((pages + i)*PAGE < length), other=0)
    tl.store(NearTable + i, near_ids, mask=i < NEAR)
    if tl.program_id(0) == 0:
        tl.store(NearLen, length - far)
        tl.store(FarPages, pages)


@triton.jit
def _reduce(Far, FarLog, Near, NearLog, Out, OutLog, Pages,
            T:tl.constexpr, H:tl.constexpr, D:tl.constexpr, S:tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, D)
    shard = tl.arange(0, S)
    valid = shard < tl.minimum(tl.load(Pages), S)
    index = (shard*T + t)*H + h
    score = tl.load(FarLog + index, mask=valid, other=-float("inf")) * 0.6931471805599453
    near_score = tl.load(NearLog + h*T + t)
    peak = tl.maximum(tl.max(score, 0), near_score)
    peak = tl.where(peak == -float("inf"), 0., peak)
    wf = tl.exp(score - peak)
    wn = tl.exp(near_score - peak)
    denom = tl.sum(wf, 0) + wn
    far = tl.load(Far + index[:,None]*D + col[None,:], mask=valid[:,None], other=0.).to(tl.float32)
    near = tl.load(Near + (t*H+h)*D + col)
    near = tl.where(wn > 0., near, 0.)
    tl.store(Out + (t*H+h)*D + col, (tl.sum(wf[:,None]*far,0) + wn*near)/tl.where(denom>0.,denom,1.))
    tl.store(OutLog + h*T+t, peak+tl.log(denom))


class GraphAttention:
    """Prepare outside capture, then call run with stable shapes and tensor pointers."""
    def __init__(self, q, k, table, extent, splits=64, near_splits=1):
        t, h, d = q.shape
        page, hk = k.shape[1:3]
        pages = table.shape[1]
        assert table.shape[0] == 1 and table.stride(1) == 1
        assert splits >= 1 and splits & (splits-1)==0 and pages >= splits
        assert q.dtype==k.dtype==torch.float16 and t in (1,4,6,8) and d==128
        self.capture_logged = False
        self.s, self.extent = splits, extent
        self.near_splits = near_splits
        self.page, self.align = page, lcm(page,128)
        self.nnear = ceil((t + extent + self.align-1)/page)
        def integer(size):
            return torch.empty(size, device=q.device, dtype=torch.int32)
        self.qptr, self.indptr = integer(splits+1), integer(splits+1)
        self.indices, self.last = integer(pages), integer(splits)
        self.near_table, self.near_len, self.far_pages = integer((1,self.nnear)), integer(1), integer(1)
        # Per-request (not per-batch) FA2 plan is UNSPLIT. Changing real page
        # boundaries between replays cannot change its output row strides.
        self.wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(8<<20,device=q.device,dtype=torch.uint8),kv_layout='NHD',
            use_cuda_graph=True,qo_indptr_buf=self.qptr,paged_kv_indptr_buf=self.indptr,
            paged_kv_indices_buf=self.indices,paged_kv_last_page_len_buf=self.last,backend='fa2')
        offsets=torch.arange(splits+1,dtype=torch.int32,device='cpu')
        with tvm_ffi.use_torch_stream():
            self.wrapper.plan(offsets*t,offsets*pages//splits,torch.zeros(pages,dtype=torch.int32,device='cpu'),
                              torch.full((splits,),page,dtype=torch.int32,device='cpu'),h,hk,d,page,
                              causal=False,q_data_type=q.dtype,kv_data_type=k.dtype,sm_scale=d**-.5,
                              disable_split_kv=True)
        self.qbuf=q.new_empty((splits*t,h,d))
        self.far=q.new_empty(self.qbuf.shape)
        self.flog=q.new_empty((splits*t,h),dtype=torch.float32)
        self.near=q.new_empty(q.shape,dtype=torch.float32)
        self.nlog=q.new_empty((h,t),dtype=torch.float32)
        self.out=torch.empty_like(self.near)
        self.olog=torch.empty_like(self.nlog)

    def run(self, q, k, v, rel, table, length, starts):
        t,h,d=q.shape
        pages=table.shape[1]
        _metadata[(triton.cdiv(max(pages,self.s+1,self.nnear),256),)](
            length,table,self.indptr,self.indices,self.near_table,self.near_len,self.far_pages,
            self.page,self.align,self.extent,t,self.s,pages,self.nnear,256)
        self.qbuf.view(self.s,t,h,d).copy_(q.unsqueeze(0))
        with tvm_ffi.use_torch_stream():
            self.wrapper.run(self.qbuf,(k,v),out=self.far,lse=self.flog,return_lse=True)
        turing_prefill(q,k,v,self.near,self.nlog,rel,num_splits=self.near_splits,
                       block_table=self.near_table,cache_seqlens=self.near_len,
                       cu_seqlens_q=starts,max_query_len=t,softmax_scale=d**-.5,window=0)
        _reduce[(t,h)](self.far,self.flog,self.near,self.nlog,self.out,self.olog,self.far_pages,
                      t,h,d,self.s,num_warps=4)
        return self.out,self.olog


# Per-process worker, one sequence in flight. Workspaces are reused by layers
# in stream order; each replay binds the actual layer's input addresses.
_plans = {}
_native_kv = os.environ.get("INKLING_NATIVE_GRAPH_KV") == "1"


def graph_attention(q, k, v, rel, table, length, starts):
    """Return O and natural-log LSE, or None for uncached shapes in capture."""
    splits = 128
    if table.shape[1] < splits or table.stride(1) != 1:
        return None
    extent = 0 if rel is None else rel.shape[-1] - 256
    key = (q.device, tuple(q.shape), k.shape[1], k.shape[2], table.shape[1], extent)
    plan = _plans.get(key)
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            # A shape without an eager warmup follows the original
            # graph-safe Triton implementation for this capture.
            logger.warning_once("SM75 graph KV missing warmup, falling back for q=%d extent=%d", q.shape[0], extent)
            return None
        if _native_kv and q.shape[0] in (4, 6, 8):
            from .native_graph_kv_sm75 import NativeGraphAttention

            # Private FA2 scheduler ABI. Fall back on an incompatible plan;
            # unrelated CUDA runtime errors must remain visible.
            try:
                if extent == 1024:
                    try:
                        from .dual_fa2_sm75 import DualGlobal
                        if q.shape[0] in (6,8):
                            plan = DualGlobal(q, k, table, torch.float32, 1, native_factory=True)
                        else:
                            plan = NativeGraphAttention(q, k, table, extent, whole=True)
                    except (AttributeError, IndexError, TypeError, ValueError, torch.OutOfMemoryError) as exc:
                        logger.warning("SM75 single-pass global FA2 unavailable, reverting to native split: %s", exc)
                    else:
                        logger.info("SM75 single-pass global FA2 ready: q=%s pages=%s", q.shape[0], table.shape[1])
                if plan is None:
                    plan = NativeGraphAttention(q, k, table, extent)
            except (AttributeError, IndexError, TypeError, ValueError, torch.OutOfMemoryError) as exc:
                logger.warning("SM75 native split unavailable, using replicated Q: %s", exc)
            except RuntimeError as exc:
                if "Buffer overflow" not in str(exc):
                    raise
                logger.warning("SM75 native split workspace exceeded, using replicated Q: %s", exc)
        if plan is None:
            plan = GraphAttention(q, k, table, extent, splits, near_splits=8)
        _plans[key] = plan
        logger.info("SM75 graph KV plan ready: q=%d extent=%d pages=%d shards=%d path=%s",
                    q.shape[0], extent, table.shape[1], getattr(plan, "shards", splits),
                    type(plan).__name__)
    if not plan.capture_logged and torch.cuda.is_current_stream_capturing():
        plan.capture_logged = True
        logger.info("SM75 graph KV dispatch during CUDA capture: q=%d extent=%d path=%s",
                    q.shape[0], extent, type(plan).__name__)
    return plan.run(q, k, v, rel, table, length, starts)
