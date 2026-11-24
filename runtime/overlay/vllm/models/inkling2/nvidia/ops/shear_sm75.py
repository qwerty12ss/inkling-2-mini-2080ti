"""SM75 tensor-core prefill for one Inkling target request, with its learned sheared bias.

The original paged FP16 K/V stay in place. Global attention can supply only
the near part, as long as the cut is a multiple of the shear period (128).
"""
from math import log

import flashinfer
import torch
import tvm_ffi

from vllm.logger import init_logger

logger = init_logger(__name__)

VARIANT = r"""
struct InklingShear : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t qo_len, kv_len, window_left;
  int q_first, key_origin;
  float sm_scale_log2;
  template <typename Params>
  __device__ __host__ InklingShear(const Params& p, uint32_t b, uint8_t*) {
    qo_len=p.get_qo_len(b);
    kv_len=p.get_kv_len(b);
    window_left=p.window_left>=0 ? p.window_left : kv_len;
    q_first=p.q_indptr[b];
    key_origin=p.past + q_first + int(qo_len) - int(kv_len);
    sm_scale_log2=math::log2e;
  }
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    const int row = q_first + int(qo_idx);
    const int key = key_origin + int(kv_idx);
    const int distance = params.past + row - key;
    float value = float(logits) * 0.08838834764831845f;
    if (qo_idx < qo_len && distance >= 0 && distance < 1024) {
      const int shear_end = min(((row >> 7) + 1) << 7, int(params.rows));
      const int bias_base = ((params.past + shear_end + 127) & ~127) - 1280;
      const int column = key - bias_base;
      value += __half2float(params.shear_bias[(row * 12 + qo_head_idx) * 1280 + column]);
    }
    return value;
  })
  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return kv_idx + qo_len + window_left >= kv_len + qo_idx;
  })
};
"""


def jit_args():
    return [
        'inkling_shear_fa2_sm75_v1', torch.float16, torch.float16, torch.float16,
        torch.int32, 128, 128, ['shear_bias'], ['half'], ['rows', 'past'],
        ['int32_t', 'int32_t'], 'InklingShear', VARIANT,
    ]



class _Plan:
    def __init__(self, q, span, mode, workspace):
        rows, heads, dim = q.shape
        page = 64
        self.past = span - rows
        # Only the page intersecting the oldest visible key and later pages
        # are valid in a sliding-window block table. Earlier IDs may be -1.
        first = max(self.past - 1023, 0) // page if mode == "local" else 0
        self.first, self.end = first, (span + page - 1) // page
        self.pages = q.new_empty(self.end - self.first, dtype=torch.int32)
        self.output = torch.empty_like(q)
        self.log2lse = q.new_empty((rows, heads), dtype=torch.float32)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend="fa2", jit_args=jit_args(),
            jit_kwargs={"use_sliding_window": True},
        )
        host = lambda xs: torch.tensor(xs, dtype=torch.int32, device="cpu")
        with tvm_ffi.use_torch_stream():
            self.wrapper.plan(
                host([0, rows]), host([0, self.end-self.first]), self.pages,
                host([(span-1) % page + 1]), heads, 4, dim, page,
                causal=True, window_left=1023 if mode == "local" else span,
                disable_split_kv=True, q_data_type=q.dtype, kv_data_type=q.dtype,
            )
        if self.wrapper._paged_kv_indices_buf.data_ptr() != self.pages.data_ptr():
            raise RuntimeError("FlashInfer did not bind the provided page index buffer")
        self.ready = torch.cuda.Event()
        self.ready.record()
        self.key = (rows, span)

    def run(self, q,k,v,rel,table,out,lse):
        self.pages.copy_(table[0, self.first:self.end])
        with tvm_ffi.use_torch_stream():
            self.wrapper.run(q, (k,v), rel, q.shape[0], self.past,
                             out=self.output, lse=self.log2lse, return_lse=True)
        out.copy_(self.output)
        torch.mul(self.log2lse.T, log(2), out=lse)


# One bounded plan per mode and worker device, shared by all layer visits.
# Single active request. Both modes use the same stream-ordered float workspace.
_plans = {}
_workspaces = {}
_failed = set()


def run_shear(q,k,v,rel,table,span,out,lse,*,mode):
    rows = q.shape[0]
    if not (128 <= rows <= 2048 and q.shape[1:] == (12,128)
            and rel is not None and rel.shape[0] >= rows
            and rel.stride() == (12*1280,1280,1) and rel.dtype == torch.float16
            and q.dtype == k.dtype == v.dtype == torch.float16
            and k.shape[1:] == (64,4,128) and v.stride() == k.stride()
            and table.shape[0] == 1 and table.stride(1) == 1
            and table.dtype == torch.int32 and span >= rows
            and table.shape[1]*64 >= span):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if mode == "local":
        # Keep coordinates under 1152 past tokens to reuse the same plan
        # for all bulk chunks, while preserving the original 128-row shear.
        skip = max(span-rows-1024,0)//128*128
        span -= skip
        table = table[:,skip//64:]
    cache_key = (q.device,mode)
    if cache_key in _failed:
        return False
    plan = _plans.get(cache_key)
    if plan is None or plan.key != (rows,span):
        if plan is not None:
            # Its pinned host planning slab must finish uploading before reuse.
            plan.ready.synchronize()
            del _plans[cache_key]
            del plan
        try:
            workspace = _workspaces.get(q.device)
            if workspace is None:
                # The unsplit FA2 planner needs no partial score/value workspace.
                workspace = q.new_zeros(1<<20,dtype=torch.uint8)
                _workspaces[q.device] = workspace
            plan = _Plan(q,span,mode,workspace)
        except (RuntimeError, ValueError) as exc:
            _failed.add(cache_key)
            logger.warning_once("SM75 sheared %s prefill planner unavailable: %s",mode,exc)
            return False
        _plans[cache_key] = plan
        logger.info_once("SM75 sheared FA2 prefill running for %s attention",mode)
    plan.run(q,k,v,rel,table,out,lse)
    return True
