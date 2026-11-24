"""Pure far FA2 plus learned near FA2 over original FP16 pages, sharing temporary storage."""
from pathlib import Path

import flashinfer
import torch
import triton
import triton.language as tl
import tvm_ffi

URI = "inkling_global_far_sm75_f32"
VARIANT = r"""
struct FarSM75 : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t qo_len, kv_len, window_left;
  float sm_scale_log2;
  template<typename P>
  __device__ __host__ FarSM75(const P& p, uint32_t b, uint8_t*) {
    qo_len = p.get_qo_len(b);
    kv_len = p.get_kv_len(b);
    window_left = kv_len;
    sm_scale_log2 = 0.08838834764831845f * math::log2e;
  }
};
"""


def far_wrapper(*args, **kwargs):
    from flashinfer.jit.env import FLASHINFER_JIT_DIR
    from flashinfer.prefill import get_batch_prefill_jit_module
    path = FLASHINFER_JIT_DIR / URI / (URI + ".so")
    if not path.is_file():
        raise ValueError(f"Missing far SM75 library at {path}")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(*args, **kwargs)
    w._jit_module = get_batch_prefill_jit_module(URI, tvm_ffi.load_module(str(path)))
    w._jit_additional_tensor_names = []
    w._jit_additional_scalar_names = []
    return w


@triton.jit
def _merge(Far, Flog, Near, Nlog, Out, Log, Pages, Seq, Starts, Vscale,
           ROWS: tl.constexpr, HEADS: tl.constexpr, SCALED: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    v = (t * HEADS + h) * 128 + tl.arange(0, 128)
    active = (t < tl.load(Starts + 1) - tl.load(Starts)) & (tl.load(Seq) > 0)
    far = active & (tl.load(Pages) != 0)
    row = t * HEADS + h
    a = tl.load(Flog + row, far, -float("inf"))
    b = tl.load(Nlog + row, active, -float("inf"))
    mx = tl.maximum(a, b)
    mx = tl.where(mx == -float("inf"), 0., mx)
    fa, nb = tl.exp2(a-mx), tl.exp2(b-mx)
    total = fa + nb
    fv = tl.load(Far + v, far, 0.).to(tl.float32)
    if SCALED:
        fv *= tl.load(Vscale + (h//3)*128 + tl.arange(0,128)).to(tl.float32)
    nv = tl.load(Near + v, active, 0.).to(tl.float32)
    result = (fv*fa + nv*nb)/tl.where(total > 0., total, 1.)
    tl.store(Out + v, tl.where(active, result, 0.))
    tl.store(Log + h * ROWS + t, (mx + tl.log2(total))*.6931471805599453)


class DualGlobal:
    def __init__(self, q, k, table, far_dtype, near_chunk, native_factory=False):
        from unittest.mock import patch
        from vllm.models.inkling2.nvidia.ops import full_fa2_sm75 as full
        from vllm.models.inkling2.nvidia.ops.native_graph_kv_sm75 import NativeGraphAttention
        assert far_dtype in (torch.float16, torch.float32)
        self.capture_logged = False
        if far_dtype == torch.float32:
            if native_factory:
                self.far = NativeGraphAttention(q, k, table, 1024, whole=True, wrapper_factory=far_wrapper)
            else:
                with patch.object(full, "create_wrapper", far_wrapper):
                    self.far = NativeGraphAttention(q, k, table, 1024, whole=True)
            self.far_value = self.far.out
        else:
            self.far = NativeGraphAttention(q, k, table, 1024)
            self.far_value = self.far.fout
        t, h, d = q.shape
        self.nvalue = q.new_empty(q.shape, dtype=torch.float32)
        self.nlog = q.new_empty(q.shape[:2], dtype=torch.float32)
        self.result = q.new_empty(q.shape, dtype=torch.float32)
        self.logs = q.new_empty((h, t), dtype=torch.float32)
        self.near = full.create_wrapper(self.far.wrapper._float_workspace_buffer, kv_layout="NHD", backend="fa2")
        host = lambda vals: torch.tensor(vals, dtype=torch.int32, device="cpu")
        idx = self.far.ntab[0]
        with tvm_ffi.use_torch_stream():
            self.near.plan(host([0,t]), host([0,len(idx)]), idx, host([64]), h, k.shape[2], d, 64,
                           causal=False, q_data_type=q.dtype, kv_data_type=k.dtype, o_data_type=torch.float32,
                           fixed_split_size=near_chunk, disable_split_kv=near_chunk is None, sm_scale=d**-.5)
        if self.near._paged_kv_indices_buf.data_ptr() != idx.data_ptr():
            raise ValueError("Near FA2 failed to retain the mutable page list")

    def run(self, q, k, v, rel, table, length, starts):
        from vllm.models.inkling2.nvidia.ops.native_graph_kv_sm75 import _write_plan
        far = self.far
        t,h,_ = q.shape
        cap = table.shape[1]
        _write_plan[(triton.cdiv(max(cap,far.grid,t+1,far.near),256),)](
            length,table,far.indptr,far.idx,far.ntab,far.nlen,far.fpages,
            far.req,far.qtile,far.ktile,far.chunk,far.merge,far.optr,far.mask,
            64,far.align,1024,t,far.grid,far.qtiles,far.shards,cap,far.near,256)
        scaled = hasattr(self, "key_scale")
        if scaled:
            torch.mul(q, self.key_scale, out=self.scaled_query)
        with tvm_ffi.use_torch_stream():
            far.wrapper.run(self.scaled_query if scaled else q,(k,v),out=self.far_value,lse=far.flog,return_lse=True)
            # Subtracting the 128-aligned far prefix from the sequence and all
            # keys preserves the learned shear origin, causal distance and query rank.
            self.near.run(q,(k,v),rel,far.nlen,starts,out=self.nvalue,lse=self.nlog,return_lse=True)
        _merge[(t,h)](self.far_value,far.flog,self.nvalue,self.nlog,self.result,self.logs,
                     far.fpages,length,starts,self.value_scale if scaled else self.result,t,h,scaled,num_warps=4)
        return self.result, self.logs


if __name__ == "__main__":
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from native_fp32 import create_spec

    args = [URI,torch.float16,torch.float16,torch.float32,torch.int32,128,128,[],[],[],[],"FarSM75",VARIANT]
    spec = create_spec(args, URI)
    spec.build()
    print("built",spec.get_library_path(),flush=True)
