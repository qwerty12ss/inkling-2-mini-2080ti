"""Single-pass, original-FP16 global target attention for one paged TP2 request."""
from pathlib import Path
import os

import flashinfer
import torch
import triton
import triton.language as tl
import tvm_ffi

URI = "inkling_global_q_sm75_f32_v1"
VARIANT = r"""
struct InklingFullVerify : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  float sm_scale_log2;
  uint32_t qo_len, kv_len, window_left;
  int seq, real, past, bias_origin;
  template<typename P>
  __device__ __host__ InklingFullVerify(const P& p, uint32_t b, uint8_t*) {
    qo_len = p.get_qo_len(b);
    kv_len = p.get_kv_len(b);
    window_left = kv_len;
    sm_scale_log2 = 0.08838834764831845f * math::log2e;
    seq = p.sequence_length[0];
    real = p.real_starts[1] - p.real_starts[0];
    past = seq - real;
    bias_origin = ((seq + 127) & ~127) - 1280;
  }
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    const int k = int(kv_idx), q = int(qo_idx);
    // All far keys have exactly zero learned bias. Scores for inactive query
    // rows are discarded after merge, so they need no branch on the far side.
    if (k + 1024 <= past) return float(logits);
    const int distance = past + q - k;
    if (q >= real || qo_head_idx >= 12 || k >= seq || distance < 0) return -math::inf;
    if (distance >= 1024) return float(logits);
    const int offset = (q * 12 + int(qo_head_idx)) * 1280 + k - bias_origin;
    return float(logits) + __half2float(params.shear_bias[offset]) * 11.313708498984761f;
  })
};
"""


def args():
    return [URI, torch.float16, torch.float16, torch.float32, torch.int32, 128, 128,
            ["shear_bias", "sequence_length", "real_starts"], ["half", "int32_t", "int32_t"],
            [], [], "InklingFullVerify", VARIANT]


def create_wrapper(*args, **kwargs):
    from flashinfer.jit.env import FLASHINFER_JIT_DIR
    from flashinfer.prefill import get_batch_prefill_jit_module
    name = URI + "_tile" if os.environ.get("INKLING_GLOBAL_FAR_TILE") == "1" else URI
    path = FLASHINFER_JIT_DIR / name / (name + ".so")
    if not path.is_file():
        raise ValueError(f"Global FP32 FA2 requires the prebuilt SM75 library at {path}")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(*args, **kwargs)
    w._jit_module = get_batch_prefill_jit_module(name, tvm_ffi.load_module(str(path)))
    w._jit_additional_tensor_names = ["shear_bias", "sequence_length", "real_starts"]
    w._jit_additional_scalar_names = []
    return w


@triton.jit
def _metadata(Seq, Table, Indptr, Idx, Last, Req, QoTile, KvTile, Chunk, Merge,
              Optr, Mask, PAGE: tl.constexpr, ROWS: tl.constexpr, GRID: tl.constexpr,
              QTILES: tl.constexpr, SHARDS: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    actual_seq = tl.load(Seq)
    seq = tl.maximum(actual_seq, 1)
    pages = tl.cdiv(seq, PAGE)
    chunk = tl.maximum(2, tl.cdiv(pages, SHARDS))
    parts = tl.cdiv(pages, chunk)
    # Out-of-context physical indices are never dereferenced by FA2, also after rollback.
    p = tl.load(Table + i, (i < pages) & (actual_seq > 0), 0)
    tl.store(Idx + i, p, i < CAP)
    tl.store(Req + i, 0, i < GRID)
    tl.store(QoTile + i, i // parts, i < GRID)
    tl.store(KvTile + i, i % parts, i < GRID)
    tl.store(Mask + i, i < QTILES * parts, i < GRID)
    tl.store(Merge + i, i * parts, i <= ROWS)
    if tl.program_id(0) == 0:
        tl.store(Chunk, chunk * PAGE)
        tl.store(Indptr, 0)
        tl.store(Indptr + 1, pages)
        tl.store(Optr, 0)
        tl.store(Optr + 1, ROWS * parts)
        tl.store(Last, (seq - 1) % PAGE + 1)


@triton.jit
def _format(Y, Log2, OutLog, Seq, Starts, ROWS: tl.constexpr, HEADS: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    valid = (tl.load(Seq) > 0) & (t < tl.load(Starts + 1) - tl.load(Starts))
    a = tl.load(Log2 + t * HEADS + h, valid, -float("inf"))
    tl.store(OutLog + h * ROWS + t, a * .6931471805599453)
    if not valid:
        tl.store(Y + (t * HEADS + h) * 128 + tl.arange(0, 128), 0.)


def run(plan, q, k, v, rel, table, length, starts):
    rows, heads, _ = q.shape
    cap = table.shape[1]
    _metadata[(triton.cdiv(max(cap, plan.grid, rows + 1), 256),)](
        length, table, plan.indptr, plan.idx, plan.last, plan.req, plan.qtile,
        plan.ktile, plan.chunk, plan.merge, plan.optr, plan.mask, 64, rows, plan.grid,
        plan.qtiles, plan.shards, cap, 256)
    with tvm_ffi.use_torch_stream():
        plan.wrapper.run(q, (k, v), rel, length, starts, out=plan.out, lse=plan.flog, return_lse=True)
    _format[(rows, heads)](plan.out, plan.flog, plan.log, length, starts, rows, heads, num_warps=4)
    return plan.out, plan.log


if __name__ == "__main__":
    import argparse
    import re
    import sys
    from flashinfer.jit import env
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from native_fp32 import create_spec
    p = argparse.ArgumentParser()
    p.add_argument("--tile", action="store_true")
    options = p.parse_args()
    name = URI + "_tile" if options.tile else URI
    spec = create_spec(args(), name)
    if options.tile:
        original = env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/prefill.cuh"
        src = original.read_text()
        start = src.index("__device__ __forceinline__ void logits_transform(")
        end = src.index("  const uint32_t lane_idx = tid.x;", start)
        # The equivalent identity transform applies to an entire warp's far
        # MMA tile. Skip all per-accumulator query/head unpacking in this case.
        text = """  if (int(kv_idx_base) + int(KTraits::NUM_MMA_KV * 16) + 1024 <= variant.past) return;
"""
        src = src[:end] + text + src[end:]
        src = re.sub(r'(#include\s+")([^"\n]+)(")',
                     lambda m: m[1] + str((original.parent / m[2]).resolve()) + m[3]
                     if (original.parent / m[2]).exists() else m[0], src)
        dest = Path("/tmp") / name
        dest.mkdir(exist_ok=True)
        header = dest / "inkling_global_fa2.cuh"
        header.write_text(src)
        spec.extra_cuda_cflags += ["--pre-include=" + str(header)]
    spec.build()
    print("built", spec.get_library_path(), flush=True)
