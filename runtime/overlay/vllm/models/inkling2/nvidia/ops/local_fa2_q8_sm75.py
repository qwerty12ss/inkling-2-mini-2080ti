"""One graph-stable native FA2 launch over up to 18 local pages during target q8 verify."""
import os
from math import ceil

import flashinfer
import torch
import triton
import triton.language as tl
import tvm_ffi

from vllm.logger import init_logger
from .attention import _sigmoid


VARIANT = r"""
struct InklingLocalVerify : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  float sm_scale_log2;
  uint32_t qo_len, kv_len, window_left;
  int first_key, past, n_real, seq, bias_origin;
  template <typename Params>
  __device__ __host__ InklingLocalVerify(const Params& p, uint32_t b, uint8_t*) {
    qo_len=p.get_qo_len(b);
    kv_len=p.get_kv_len(b);
    window_left=kv_len;
    sm_scale_log2=1.f;
    seq=p.sequence_length[0];
    n_real=p.real_starts[1]-p.real_starts[0];
    past=seq-n_real;
    first_key=max(past - 1023, 0)/64*64;
    bias_origin=((seq+127)&~127)-1280;
  }
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    const int key=first_key+int(kv_idx);
    const int distance=past+int(qo_idx)-key;
    if (int(qo_idx)>=n_real || qo_head_idx>=12 || key>=seq || distance<0 || distance>=1024)
      return -math::inf;
    const int offset=(int(qo_idx)*12+int(qo_head_idx))*1280+(key-bias_origin);
    float bias=__half2float(params.shear_bias[offset]);
    return float(logits)*(0.08838834764831845f * math::log2e) + bias * math::log2e;
  })
};
"""


def _jit_args():
    return [
        "inkling_local_q8_sm75_fp16_v1", torch.float16, torch.float16, torch.float16,
        torch.int32, 128, 128,
        ["shear_bias", "sequence_length", "real_starts"], ["half", "int32_t", "int32_t"],
        [], [], "InklingLocalVerify", VARIANT,
    ]


@triton.jit
def _map_pages(table, seq, starts, pages, max_pages: tl.constexpr):
    offsets = tl.arange(0, 32)
    length = tl.load(seq)
    num_queries = tl.load(starts + 1) - tl.load(starts)
    first_page = tl.maximum(length - num_queries - 1023, 0) // 64
    page = first_page + offsets
    valid = (offsets < max_pages) & (page * 64 < length)
    physical_page = tl.load(table + page, mask=valid, other=0)
    tl.store(pages + offsets, physical_page, mask=offsets < max_pages)


@triton.jit
def _finish(output, log2_lse, gate, sink, result, seq, starts,
            gate_stride: tl.constexpr, sink_stride: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = head * 128 + tl.arange(0, 128)
    index = row * 1536 + col
    valid = (row < tl.load(starts + 1) - tl.load(starts)) & (tl.load(seq) > 0)
    value = tl.load(output + index, mask=valid, other=0.).to(tl.float32)
    gate_value = tl.load(gate + row * gate_stride + col).to(tl.float32)
    sink_value = tl.load(sink + row * sink_stride + head).to(tl.float32)
    lse = tl.load(log2_lse + row * 12 + head, mask=valid, other=-float("inf"))
    value = value * _sigmoid(gate_value) * _sigmoid(lse * 0.6931471805599453 - sink_value)
    tl.store(result + index, tl.where(valid, value, 0.))


def _native_fp32_wrapper(workspace):
    from flashinfer.jit.env import FLASHINFER_JIT_DIR
    from flashinfer.prefill import get_batch_prefill_jit_module

    # This is an explicitly prepared, version-and-SM-specific library. Do not compile
    # it from inside a serving process or alter shared FlashInfer caches in-process.
    name = "inkling_local_q8_sm75_fp32_v2"
    path = FLASHINFER_JIT_DIR / name / (name + ".so")
    if not path.is_file():
        raise ValueError(f"FP32 local FA2 must be prebuilt at {path}")
    module = tvm_ffi.load_module(str(path))
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    wrapper._jit_module = get_batch_prefill_jit_module(name, module)
    wrapper._jit_additional_tensor_names = ["shear_bias", "sequence_length", "real_starts"]
    wrapper._jit_additional_scalar_names = []
    return wrapper


class LocalPlan:
    def __init__(self, q, workspace):
        self.rows = q.shape[0]
        # The union of eight 1024-token windows may begin at slot 63 of a page.
        self.max_pages = ceil((1024 + self.rows - 1 + 63) / 64)
        self.pages = q.new_empty(self.max_pages, dtype=torch.int32)
        self.dtype = torch.float32 if os.environ.get("INKLING_LOCAL_FA2_FP32") == "1" else torch.float16
        self.output = q.new_empty(q.shape, dtype=self.dtype)
        self.log2_lse = q.new_empty((self.rows, 12), dtype=torch.float32)
        if self.dtype == torch.float32:
            self.wrapper = _native_fp32_wrapper(workspace)
        else:
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD", backend="fa2", jit_args=_jit_args(),
            )
        def host(values):
            return torch.tensor(values, dtype=torch.int32, device="cpu")
        with tvm_ffi.use_torch_stream():
            self.wrapper.plan(
                host([0, self.rows]), host([0, self.max_pages]), self.pages, host([64]),
                12, 4, 128, 64, causal=False, q_data_type=q.dtype,
                kv_data_type=q.dtype, o_data_type=self.dtype, fixed_split_size=1,
            )
        if self.wrapper._paged_kv_indices_buf.data_ptr() != self.pages.data_ptr():
            raise ValueError("FlashInfer did not bind the mutable page index buffer")
        # The one-time upload completes in eager warmup before a capture stream
        # can start using it. All mutable request state is updated by _map_pages.
        torch.cuda.current_stream(q.device).synchronize()
        self.capture_logged = False
        self.warmed_signatures = set()

    def run(self, q, k, v, rel, table, length, starts, gate, sinks, out):
        _map_pages[(1,)](table, length, starts, self.pages, self.max_pages, num_warps=1)
        with tvm_ffi.use_torch_stream():
            self.wrapper.run(q, (k, v), rel, length, starts, out=self.output,
                             lse=self.log2_lse, return_lse=True)
        _finish[(self.rows, 12)](self.output, self.log2_lse, gate, sinks, out, length, starts,
                         gate.stride(0), sinks.stride(0), num_warps=4)


logger = init_logger(__name__)
_plans = {}
_workspaces = {}
_failed = set()


def run_local_fa2(q, k, v, rel, gate, sinks, table, length, starts, out):
    """Refresh physical pages on each replay for one serial target request."""
    if not (
        q.ndim == 3 and q.shape[0] in (4, 6, 8) and q.shape[1:] == (12, 128) and q.dtype == k.dtype == v.dtype == rel.dtype == torch.float16
        and k.shape == v.shape and k.ndim == 4 and k.shape[0] > 0 and k.shape[1:] == (64, 4, 128)
        and k.stride() == v.stride() and k.stride(3) == 1 and q.stride()[1:] == (128, 1)
        and rel.ndim == 3 and rel.shape[0] >= q.shape[0] and rel.shape[1:] == (12, 1280)
        and rel.stride() == (12 * 1280, 1280, 1) and gate.shape == (q.shape[0], 1536) and sinks.shape == (q.shape[0], 12)
        and gate.stride(1) == sinks.stride(1) == 1 and out.shape == gate.shape
        and out.is_contiguous() and out.dtype == q.dtype and table.ndim == 2
        and table.shape[0] == 1 and table.shape[1] > 0 and table.stride(1) == 1
        and length.shape == (1,) and starts.shape == (2,) and starts.stride(0) == 1
        and starts.dtype == length.dtype == table.dtype == torch.int32
        and all(x.device == q.device for x in (k, v, rel, gate, sinks, out, table, starts, length))
    ):
        return False
    device = q.device
    key = (device, q.shape[0])
    if key in _failed:
        return False
    capturing = torch.cuda.is_current_stream_capturing()
    plan = _plans.get(key)
    if plan is None:
        if capturing:
            logger.info_once("SM75 local FA2 capture used stock attention while its eager plan was unavailable")
            return False
        try:
            workspace = _workspaces.get(device)
            if workspace is None:
                workspace = q.new_empty(16 << 20, dtype=torch.uint8)
                _workspaces[device] = workspace
            plan = LocalPlan(q, workspace)
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            _failed.add(key)
            logger.warning("SM75 local FA2 q%d initialization unavailable, using stock attention: %s", q.shape[0], exc)
            return False
        _plans[key] = plan
        logger.info("SM75 local FA2 q%d plan ready: split=1 page=64 partial=%s", q.shape[0], plan.dtype)
    # Different merged-QKVR views can yield different gate and sink row strides.
    # This is the only layout specialized by the two Triton kernels above.
    signature = (gate.dtype, gate.stride(0), sinks.dtype, sinks.stride(0))
    if capturing and signature not in plan.warmed_signatures:
        return False
    plan.run(q, k, v, rel, table, length, starts, gate, sinks, out)
    if not capturing:
        plan.warmed_signatures.add(signature)
    elif not plan.capture_logged:
        plan.capture_logged = True
        logger.info("SM75 local FA2 q%d active during CUDA capture: split=1 page=64 partial=%s", q.shape[0], plan.dtype)
    return True
