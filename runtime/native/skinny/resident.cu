// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Skinny NVFP4/MXFP4 GEMM and grouped MoE kernels for SM70/SM75, taken from
// dnv2003/v100-skinny (MIT) with the grouped MoE kernel and the MXFP4 scale
// mode added by its contributors. See LICENSE.v100-skinny in this directory
// for the retained MIT notice.
//
// Weights are prepacked once at load into the fragment order of the card's
// tensor-core MMA ([tile N/32][group K/16][lane 32] x 8B codes plus 32 scale
// bytes per tile and scale group): mma.m8n8k4 on SM70, mma.m16n8k8 from SM75
// on (mma_fragments_sm70.cuh). The permutation lives in
// vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py.

#include <cuda_runtime.h>
#include "mma_fragments_sm70.cuh"
namespace {

using vllm::sm70::fp4x8_to_half2x4;
using vllm::sm70::mma_m16n8k8;

// Register order the expert weights were packed in (QpnFragmentOrder in
// skinny_sm70_moe.py).

__device__ __forceinline__ uint32_t original_fragment(uint32_t input, int lane) {
    int src = (lane & ~3) + 2 * (lane & 1);
    unsigned shift = (lane & 2) * 2;
    uint32_t a = __shfl_sync(0xffffffff, input, src) >> shift;
    uint32_t b = __shfl_sync(0xffffffff, input, src + 1) >> shift;
    a = (a & 0x000f000f) | ((a >> 4) & 0x00f000f0);
    b = (b & 0x000f000f) | ((b >> 4) & 0x00f000f0);
    return __byte_perm(a, b, 0x6240);
}

__device__ __forceinline__ half2 marlin_scale_byte(uint32_t b) {
    half scalar = __ushort_as_half(static_cast<unsigned short>((b & 255u) << 7));
    return __halves2half2(scalar,scalar);
}
enum FragmentOrder { ORDER_M8N8K4 = 0, ORDER_M16N8K8 = 1 };

__device__ __forceinline__ half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs =
      __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));  // *256
  return __halves2half2(hs, hs);
}

// MXFP4 block scale: E8M0 is a bare exponent, value = 2^(b - 127). fp16
// stores exponent + 15, so the field is b - 112 and the mantissa stays 0 --
// one shift, no multiply. Callers keep the checkpoint's global factor in
// gm2, which is what holds b - 112 inside fp16's normal range (1..30).
__device__ __forceinline__ half2 e8m0_to_half2(unsigned char b) {
  const unsigned short hb = (unsigned short)(((int)b - 112) << 10);
  const half hs = __ushort_as_half(hb);
  return __halves2half2(hs, hs);
}

// Group-scale layouts a weight tile can carry. NVFP4 ships one fp8-e4m3
// scale per 16 codes; MXFP4 one E8M0 scale per 32. Everything else -- code
// packing, fragment order, the decoder, the MMA -- is identical, so the
// kernels take this as a template argument instead of duplicating them.
enum ScaleMode { SCALE_NVFP4_FP8_16 = 0, SCALE_MXFP4_E8M0_32 = 1 };

// Raw scale byte of group g and its decode, split so a caller can issue the
// load ahead of its use.
template <int MODE>
__device__ __forceinline__ uint8_t group_scale_byte(const uint8_t* sb, int g) {
  if (MODE == SCALE_MXFP4_E8M0_32) return __ldg(sb + (size_t)(g >> 1) * 32);
  return __ldg(sb + (size_t)g * 32);
}

template <int MODE>
__device__ __forceinline__ half2 decode_scale(uint8_t b) {
  if (MODE == SCALE_MXFP4_E8M0_32) return e8m0_to_half2(b);
  return fp8e4m3_to_half2(b);
}

template <int MODE>
__device__ __forceinline__ half2 group_scale(const uint8_t* sb, int g) {
  return decode_scale<MODE>(group_scale_byte<MODE>(sb, g));
}

// Eight e2m1 codes times one scale (the 2^14 decoder re-bias is folded into
// the caller's scale). out[i] holds codes (i, i+4) of the 8-code word.
__device__ __forceinline__ void dequant8_tm(unsigned q, half2 sc2p,
                                            half2 out[4]) {
  fp4x8_to_half2x4(q, out);
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = __hmul2(out[i], sc2p);
}

// ---------------------------------------------------------------------------
template <int SPLITK, int SCALE_MODE, int RB, int NT>
__global__ void __launch_bounds__(32 * SPLITK) skinny_nvfp4_moe_m16n8k8(
    const uint8_t* __restrict__ qcodes, const uint8_t* __restrict__ qscales,
    const float* __restrict__ gscales, const half* __restrict__ x,
    half* __restrict__ y_slots, const int* __restrict__ perm,
    const int* __restrict__ gids, const int* __restrict__ goff, int N, int K,
    int topk, int x_slot_major) {
  constexpr int ROWS = RB * 8;
  // 36-float rows put the C fragment stores of a warp into 32 distinct banks.
  constexpr int CS_ROW = 36;
  __shared__ const half* xrows[ROWS];
  __shared__ int slots[ROWS];
  __shared__ float cs[SPLITK][NT][8 * CS_ROW];

  const int grp = blockIdx.y;
  const int beg = 8 * grp;
  const int cnt = goff[grp];
  if (cnt < 0) {
      for (int nc = threadIdx.x; nc < NT * 32; nc += blockDim.x)
        y_slots[(size_t)grp * N + blockIdx.x * NT * 32 + nc] = __float2half(0.f);
      return;
    }
    if (!cnt) return;
  const int e = gids[grp];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile0 = blockIdx.x * NT;
  const int row_group = lane >> 2;
  const int k_offset = (lane & 3) * 4;
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq, gend = g0 + Gq;
  const int n64 = N >> 6;
  const int tile64 = tile0 >> 1;
  const uint4* marlin_code = reinterpret_cast<const uint4*>(qcodes);
  const uint32_t* marlin_scale = reinterpret_cast<const uint32_t*>(qscales);

  for (int base = 0; base < cnt; base += ROWS) {
    const int rows = min(ROWS, cnt - base);
    const int blocks = (rows + 7) / 8;
    __syncthreads();  // previous pass done with slots/xrows/cs
    if (threadIdx.x < ROWS) {
      const int m = threadIdx.x;
      const int slot = m < rows ? perm[beg + base + m] : 0;
      slots[m] = slot;
      xrows[m] = m < rows ? x + (size_t)(x_slot_major ? slot : slot / topk) * K
                          : nullptr;
    }
    __syncthreads();
    const half* xr[RB];
#pragma unroll
    for (int bl = 0; bl < RB; bl++) xr[bl] = xrows[bl * 8 + row_group];

    float c[NT][RB][2][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int bl = 0; bl < RB; bl++)
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
          for (int i = 0; i < 4; i++) c[nt][bl][t][i] = 0.f;

    uint2 q_next[NT], a_next[RB];
    uint32_t s_next[NT];
    auto load_group = [&](int g) {
      const size_t tile = ((size_t)e * G + g) * n64 + tile64;
      // One coalesced 512-byte transaction per warp's N64 x K16 tile.
      const uint4 loaded = __ldcs(marlin_code + tile * 32 + lane);
      const uint32_t words[4] = {loaded.x,loaded.y,loaded.z,loaded.w};
#pragma unroll
      for (int nt=0; nt<NT; ++nt) {
        const int half_tile = (tile0 + nt) & 1;
        q_next[nt] = make_uint2(original_fragment(words[2*half_tile],lane),
                                original_fragment(words[2*half_tile+1],lane));
        const uint32_t scales = __ldg(marlin_scale + tile * 16 + row_group*2 + half_tile);
        s_next[nt] = __byte_perm(scales, 0, 0x3120);
      }
#pragma unroll
      for (int bl = 0; bl < RB; bl++)
        a_next[bl] =
            xr[bl] ? *reinterpret_cast<const uint2*>(xr[bl] + g * 16 + k_offset)
                   : make_uint2(0, 0);
    };
    load_group(g0);
    for (int g = g0; g < gend; g++) {
      uint2 q[NT], a[RB];
      uint32_t s[NT];
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        q[nt] = q_next[nt];
        s[nt] = s_next[nt];
      }
#pragma unroll
      for (int bl = 0; bl < RB; bl++) a[bl] = a_next[bl];
      if (g + 1 < gend) load_group(g + 1);

#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        // Word t holds A tile t: columns 16t + row_group (scale byte 2t) and
        // 16t + row_group + 8 (scale byte 2t + 1), interleaved per MMA step.
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const half2 s_lo = marlin_scale_byte(s[nt] >> (16*t));
          const half2 s_hi = marlin_scale_byte(s[nt] >> (16*t+8));
          half2 v[4];
          fp4x8_to_half2x4(t ? q[nt].y : q[nt].x, v);
          v[0] = __hmul2(v[0], s_lo);
          v[1] = __hmul2(v[1], s_hi);
          v[2] = __hmul2(v[2], s_lo);
          v[3] = __hmul2(v[3], s_hi);
          const uint32_t* w = reinterpret_cast<const uint32_t*>(v);
#pragma unroll
          for (int bl = 0; bl < RB; bl++) {
            if (bl > 0 && bl >= blocks) break;  // block-uniform
            mma_m16n8k8(c[nt][bl][t], w[0], w[1], a[bl].x);
            mma_m16n8k8(c[nt][bl][t], w[2], w[3], a[bl].y);
          }
        }
      }
    }

#pragma unroll
    for (int bl = 0; bl < RB; bl++) {
      if (bl > 0 && bl >= blocks) break;  // block-uniform: barriers match
      if (bl > 0) __syncthreads();        // previous block done reading cs
      // C of A tile t: column 16t + row_group (+8 for i >= 2), row
      // (lane & 3) * 2 + (i & 1) of this 8-row block.
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const int col = 16 * t + row_group + ((i & 2) ? 8 : 0);
            const int row = (lane & 3) * 2 + (i & 1);
            cs[warp][nt][row * CS_ROW + col] = c[nt][bl][t][i];
          }
      __syncthreads();
      for (int idx = threadIdx.x; idx < NT * 256; idx += blockDim.x) {
        const int nt = idx >> 8, row = (idx >> 5) & 7, col = idx & 31;
        float v = 0.f;
#pragma unroll
        for (int split = 0; split < SPLITK; split++)
          v += cs[split][nt][row * CS_ROW + col];
        const int out_row = bl * 8 + row;
        if (out_row < rows)
          y_slots[(size_t)slots[out_row] * N + (size_t)(tile0 + nt) * 32 +
                  col] = __float2half_rn(v * gscales[e]);
      }
    }
  }
}


}  // namespace
extern "C" int project_skinny(const uint8_t *codes, const uint8_t *scales, const float *global,
                              const half *a, half *out, const int *perm, const int *gids,
                              const int *counts, int n, int k, int slots, int slot_major,
                              int splitk, int tiles, cudaStream_t stream) {
    if (slots > 64 || slots < 8 || n % (tiles * 32) || k % (16 * splitk)) return -1;
    dim3 grid(n / (tiles * 32), slots);
    #define GO(S, T) skinny_nvfp4_moe_m16n8k8<S, SCALE_NVFP4_FP8_16, 1, T> \
        <<<grid, 32*S, 0, stream>>>(codes, scales, global, a, out, perm, gids, counts, n, k, 8, slot_major)
    if (splitk == 4 && tiles == 1) {GO(4, 1);}
    else if (splitk == 4 && tiles == 2) {GO(4, 2);}
    else if (splitk == 8 && tiles == 1) {GO(8, 1);}
    else if (splitk == 8 && tiles == 2) {GO(8, 2);}
    else if (splitk == 16 && tiles == 1) {GO(16, 1);}
    else if (splitk == 16 && tiles == 2) {GO(16, 2);}
    else return -2;
    return int(cudaGetLastError());
}
