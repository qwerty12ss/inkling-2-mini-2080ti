#pragma once
#include <flashinfer/page.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/attention/variant_helper.cuh>
#include <type_traits>

#define ADDITIONAL_FUNC_PARAMS , ffi::Tensor shear_bias, int32_t rows, int32_t past
#define ADDITIONAL_PARAMS_SETTER params.shear_bias = static_cast<half*>(shear_bias.data_ptr()); \
params.rows = rows; \
params.past = past;

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant, RaggedParams, PagedParams, ...) \
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \
    constexpr auto use_custom_mask = MASK_MODE == MaskMode::kCustom; \
    using AttentionVariant = InklingShear; \
    __VA_ARGS__(); \
  })

using namespace flashinfer;

using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;
using IdType = int32_t;

constexpr int HEAD_DIM_QK = 128;
constexpr int HEAD_DIM_VO = 128;
constexpr bool USE_FP16_QK_REDUCTION = false;
constexpr auto USE_LOGITS_SOFT_CAP = false;
constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
constexpr auto USE_SLIDING_WINDOW = true;


struct RaggedParams {
  using DTypeQ = DTypeQ;
  using DTypeKV = DTypeKV;
  using DTypeO = DTypeO;
  using IdType = IdType;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  IdType* q_indptr;
  IdType* kv_indptr;
  DTypeO* o;
  float* lse;
  uint_fastdiv group_size;

  half* shear_bias;
int32_t rows;
int32_t past;

  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  uint32_t k_sf_stride_page;
  uint32_t k_sf_stride_n;
  uint32_t k_sf_stride_h;
  uint32_t v_sf_stride_page;
  uint32_t v_sf_stride_n;
  uint32_t v_sf_stride_h;
  int32_t window_left;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }
};

struct PagedParams {
  using DTypeQ = DTypeQ;
  using DTypeKV = DTypeKV;
  using DTypeO = DTypeO;
  using IdType = IdType;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  IdType* q_indptr;
  DTypeO* o;
  float* lse;
  uint_fastdiv group_size;

  half* shear_bias;
int32_t rows;
int32_t past;

  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  uint32_t k_sf_stride_page;
  uint32_t k_sf_stride_n;
  uint32_t k_sf_stride_h;
  uint32_t v_sf_stride_page;
  uint32_t v_sf_stride_n;
  uint32_t v_sf_stride_h;
  int32_t window_left;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  bool* block_valid_mask;
  IdType* kv_chunk_size_ptr;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};


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
