// SPDX-License-Identifier: Apache-2.0
// Dense y = x W^T (+ bias): bf16 x [M, K] (row stride ldx), NF12 or bf16t W [N, K],
// bf16 y [M, N] (row stride ldy). Each kernel exists for both weight sources
// with identical arithmetic, so NF12 output equals the bf16t twin's.
#include "linear_core.h"

using namespace nf12;

struct LinearParams {
  ulong ldx, ldy;
  uint m, n, k_blocks;
  uint kb_per_split;  // small M: super-atoms per K split
  uint row_blocks;    // small M: 16-row blocks per threadgroup (RB)
  uint splits;        // small M: power-of-two K splits per row block; RB * KS <= kSmallMaxSimdgroups
  uint has_bias;
  uint reserved;  // was base_x4; exp_base is buffer 6 so frameworks can pass it as data
};

constant constexpr uint kSmallMaxSimdgroups = 16;

inline void store_linear(device ushort* y, device const ushort* bias, constant LinearParams& p,
                         uint m, uint n, float sum) {
  if (p.has_bias) sum += bf16_to_float(bias[n]);
  y[ulong(m) * p.ldy + n] = float_to_bf16(sum);
}

// ---------------------------------------------------------------------------
// Small M. Threadgroup (x, y) covers row blocks [x RB, x RB + RB) and activation
// rows [y MT, y MT + MT); simdgroup sg = rb KS + ks owns row block rb, K split ks.
// Splits are summed in increasing ks order.
template <uint MT, typename Src>
inline void linear_small(Src src, device const ushort* x, device const ushort* bias, device ushort* y,
                         constant LinearParams& p, uint2 tg, uint sg, uint lane, uint tid, uint threads,
                         threadgroup float* part) {
  const uint rb = sg >> ctz(p.splits), ks = sg & (p.splits - 1);
  const uint row_blocks = p.n / 16;
  const uint nb = tg.x * p.row_blocks + rb;
  const uint m0 = tg.y * MT;
  ulong xoff[MT];
  for (uint m = 0; m < MT; ++m) xoff[m] = ulong(min(m0 + m, p.m - 1)) * p.ldx;
  float acc0[MT], acc1[MT];
  for (uint m = 0; m < MT; ++m) acc0[m] = acc1[m] = 0.0f;
  if (nb < row_blocks) {
    const uint kb0 = min(ks * p.kb_per_split, p.k_blocks);
    const uint kb1 = min(kb0 + p.kb_per_split, p.k_blocks);
    gemv_accumulate<MT>(src, ulong(nb) * p.k_blocks, kb0, kb1, x, xoff, lane, acc0, acc1);
  }
  gemv_reduce_quad<MT>(acc0, acc1);
  const uint g = lane >> 2;
  // With one split, each quad already has the final reduction.
  if (p.splits == 1) {
    if ((lane & 3u) == 0u && nb < row_blocks) {
      const uint n = nb * 16 + g;
      for (uint m = 0; m < MT && m0 + m < p.m; ++m) {
        store_linear(y, bias, p, m0 + m, n, acc0[m]);
        store_linear(y, bias, p, m0 + m, n + 8, acc1[m]);
      }
    }
    return;
  }
  if ((lane & 3u) == 0u) {
    threadgroup float* dst = part + (sg * MT) * 16;
    for (uint m = 0; m < MT; ++m) {
      dst[m * 16 + g] = acc0[m];
      dst[m * 16 + g + 8] = acc1[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.row_blocks * MT * 16; i += threads) {
    const uint b = i / (MT * 16), m = (i / 16) % MT, row = i % 16;
    const uint nb2 = tg.x * p.row_blocks + b;
    if (nb2 >= row_blocks || m0 + m >= p.m) continue;
    float s = part[((b * p.splits) * MT + m) * 16 + row];
    for (uint k = 1; k < p.splits; ++k) s += part[((b * p.splits + k) * MT + m) * 16 + row];
    store_linear(y, bias, p, m0 + m, nb2 * 16 + row, s);
  }
}

// ---------------------------------------------------------------------------
// Medium M (register MMA). Same grid and split layout as small M with 8 * MB
// activation rows per threadgroup; RB * KS <= kMmaMaxSimdgroups.
constant constexpr uint kMmaMaxSimdgroups = 8;

template <uint MB, typename Src>
inline void linear_mma(Src src, device const ushort* x, device const ushort* bias, device ushort* y,
                       constant LinearParams& p, uint2 tg, uint sg, uint lane, uint tid, uint threads,
                       threadgroup float* part) {
  constexpr uint TM = 8 * MB;
  const uint rb = sg >> ctz(p.splits), ks = sg & (p.splits - 1);
  const uint row_blocks = p.n / 16;
  const uint nb = tg.x * p.row_blocks + rb;
  const uint m0 = tg.y * TM;
  const ushort2 c = mma_coord(lane);
  ulong xrow[2 * MB];
  for (uint mb = 0; mb < MB; ++mb) {
    xrow[2 * mb] = ulong(min(m0 + 8 * mb + c.y, p.m - 1)) * p.ldx;
    xrow[2 * mb + 1] = ulong(min(m0 + 8 * mb + c.y + 1, p.m - 1)) * p.ldx;
  }
  simdgroup_float8x8 acc[2][MB];
  for (uint h = 0; h < 2; ++h)
    for (uint mb = 0; mb < MB; ++mb) acc[h][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  if (nb < row_blocks) {  // uniform per simdgroup
    const uint kb0 = min(ks * p.kb_per_split, p.k_blocks);
    const uint kb1 = min(kb0 + p.kb_per_split, p.k_blocks);
    mma_accumulate<MB>(src, ulong(nb) * p.k_blocks, kb0, kb1, x, xrow, lane, acc);
  }
  // The accumulator fragment has the same lane coordinates as the inputs.
  // Without split-K, write it directly and skip threadgroup memory.
  if (p.splits == 1) {
    if (nb < row_blocks) {
      for (uint h = 0; h < 2; ++h) {
        const uint n = nb * 16 + 8 * h + c.x;
        for (uint mb = 0; mb < MB; ++mb) {
          const uint m = m0 + 8 * mb + c.y;
          if (m < p.m) store_linear(y, bias, p, m, n, acc[h][mb].thread_elements()[0]);
          if (m + 1 < p.m) store_linear(y, bias, p, m + 1, n, acc[h][mb].thread_elements()[1]);
        }
      }
    }
    return;
  }
  threadgroup float* dst = part + sg * 16 * TM;  // [16 rows][TM tokens]
  for (uint h = 0; h < 2; ++h)
    for (uint mb = 0; mb < MB; ++mb) simdgroup_store(acc[h][mb], dst + 8 * h * TM + 8 * mb, TM);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.row_blocks * 16 * TM; i += threads) {
    const uint b = i / (16 * TM), m = (i / 16) % TM, row = i % 16;
    const uint nb2 = tg.x * p.row_blocks + b;
    if (nb2 >= row_blocks || m0 + m >= p.m) continue;
    float s = part[((b * p.splits) * 16 + row) * TM + m];
    for (uint k = 1; k < p.splits; ++k) s += part[((b * p.splits + k) * 16 + row) * TM + m];
    store_linear(y, bias, p, m0 + m, nb2 * 16 + row, s);
  }
}

// ---------------------------------------------------------------------------
// Keep the two operand slabs in BF16 and the accumulator in FP32.
// The BF16 slab and FP32 output tile share threadgroup memory.
struct LinearGemmSlab {
  bfloat w[kGemmBN * kGemmLd];
  bfloat x[kGemmBM * kGemmLd];
};
union LinearGemmSmem {
  LinearGemmSlab slab;
  float c[kGemmBN * kGemmCLd];
};

template <typename Src>
inline void linear_gemm_slab(Src src, ulong sa, bool valid, device const ushort* x, threadgroup const ulong* xoff,
                            uint kb, threadgroup LinearGemmSlab& slab, thread GemmAcc& acc,
                            uint sg, uint lane, uint tid) {
  const uint row = 16 * sg + (lane >> 2);
  const uint pair = lane & 3u;
  threadgroup uint* w0 = reinterpret_cast<threadgroup uint*>(slab.w + row * kGemmLd) + pair;
  threadgroup uint* w1 = w0 + 8 * kGemmLd / 2;
  uint r[4][4] = {};
  if (valid) src.load(sa, lane, r);
  for (uint j = 0; j < 4; ++j) {
    w0[8 * j] = r[j][0];
    w0[8 * j + 4] = r[j][2];
    w1[8 * j] = r[j][1];
    w1[8 * j + 4] = r[j][3];
  }

  for (uint i = tid; i < kGemmBM * (kSuperAtomCols / 2); i += 32 * kGemmSimdgroups) {
    const uint m = i / (kSuperAtomCols / 2), col = i % (kSuperAtomCols / 2);
    const ulong off = xoff[m];
    uint bits = 0;
    if (off != ~0ul)
      bits = *reinterpret_cast<device const uint*>(x + off + ulong(kb) * kSuperAtomCols + 2 * col);
    reinterpret_cast<threadgroup uint*>(slab.x + m * kGemmLd)[col] = bits;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint kk = 0; kk < kSuperAtomCols; kk += 8) {
    simdgroup_bfloat8x8 a0, a1;
    simdgroup_load(a0, slab.w + (16 * sg) * kGemmLd + kk, kGemmLd);
    simdgroup_load(a1, slab.w + (16 * sg + 8) * kGemmLd + kk, kGemmLd);
    for (uint c = 0; c < kGemmBM / 8; ++c) {
      simdgroup_bfloat8x8 b;
      simdgroup_load(b, slab.x + (8 * c) * kGemmLd + kk, kGemmLd, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[0][c], a0, b, acc[0][c]);
      simdgroup_multiply_accumulate(acc[1][c], a1, b, acc[1][c]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Large M. Threadgroup (x, y) computes y rows [32 y, 32 y + 32) x columns [64 x, 64 x + 64).
template <typename Src>
inline void linear_gemm(Src src, device const ushort* x, device const ushort* bias, device ushort* y,
                        constant LinearParams& p, uint2 tg, uint sg, uint lane, uint tid,
                        threadgroup LinearGemmSmem& sm, threadgroup ulong* xoff) {
  const uint n0 = tg.x * kGemmBN, m0 = tg.y * kGemmBM;
  if (tid < kGemmBM) xoff[tid] = m0 + tid < p.m ? ulong(m0 + tid) * p.ldx : ~0ul;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nb = n0 / 16 + sg;
  const bool valid = nb < p.n / 16;
  GemmAcc acc;
  gemm_zero(acc);
  for (uint kb = 0; kb < p.k_blocks; ++kb)
    linear_gemm_slab(src, ulong(nb) * p.k_blocks + kb, valid, x, xoff, kb, sm.slab, acc, sg, lane, tid);
  for (uint h = 0; h < 2; ++h)
    for (uint q = 0; q < kGemmBM / 8; ++q)
      simdgroup_store(acc[h][q], sm.c + (16 * sg + 8 * h) * kGemmCLd + 8 * q, kGemmCLd);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < kGemmBN * kGemmBM; i += 32 * kGemmSimdgroups) {
    const uint nl = i % kGemmBN, ml = i / kGemmBN;
    const uint n = n0 + nl, m = m0 + ml;
    if (n >= p.n || m >= p.m) continue;
    float s = sm.c[nl * kGemmCLd + ml];
    if (p.has_bias) s += bf16_to_float(bias[n]);
    y[ulong(m) * p.ldy + n] = float_to_bf16(s);
  }
}

// ---------------------------------------------------------------------------
// Entry points. Buffers: 0 x, 1 weight (packed or bf16t), 2 escapes (NF12), 3 bias, 4 y, 5 params,
// 6 exp_base (NF12, uchar[1]).
#define NF12_SMALL_KERNEL(MT)                                                                                      \
  NF12_THREADS(32 * kSmallMaxSimdgroups) kernel void nf12_linear_small_nf12_m##MT(                                                                        \
      device const ushort* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],                              \
      device const uint4* escapes [[buffer(2)]], device const ushort* bias [[buffer(3)]],                          \
      device ushort* y [[buffer(4)]], constant LinearParams& p [[buffer(5)]],                                      \
      device const uchar* exp_base [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                           \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                         \
      uint tid [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {                       \
    threadgroup float part[kSmallMaxSimdgroups * MT * 16];                                                          \
    linear_small<MT>(NF12Source{packed, escapes, exp_base_x4(exp_base[0])}, x, bias, y, p, tg, sg, lane, tid,       \
                     threads.x, part);                                                                             \
  }                                                                                                                \
  NF12_THREADS(32 * kSmallMaxSimdgroups) kernel void nf12_linear_small_bf16t_m##MT(                                                                       \
      device const ushort* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],                              \
      device const ushort* bias [[buffer(3)]], device ushort* y [[buffer(4)]],                                     \
      constant LinearParams& p [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],                           \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                         \
      uint tid [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {                       \
    threadgroup float part[kSmallMaxSimdgroups * MT * 16];                                                           \
    linear_small<MT>(Bf16tSource{packed}, x, bias, y, p, tg, sg, lane, tid, threads.x, part);                        \
  }

NF12_SMALL_KERNEL(1)
NF12_SMALL_KERNEL(2)
NF12_SMALL_KERNEL(4)
NF12_SMALL_KERNEL(8)

NF12_THREADS(32 * kGemmSimdgroups) kernel void nf12_linear_gemm_nf12(device const ushort* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],
                                  device const uint4* escapes [[buffer(2)]], device const ushort* bias [[buffer(3)]],
                                  device ushort* y [[buffer(4)]], constant LinearParams& p [[buffer(5)]],
                                  device const uchar* exp_base [[buffer(6)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup LinearGemmSmem sm;
  threadgroup ulong xoff[kGemmBM];
  linear_gemm(NF12Source{packed, escapes, exp_base_x4(exp_base[0])}, x, bias, y, p, tg, sg, lane, tid, sm, xoff);
}

NF12_THREADS(32 * kGemmSimdgroups) kernel void nf12_linear_gemm_bf16t(device const ushort* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],
                                   device const ushort* bias [[buffer(3)]], device ushort* y [[buffer(4)]],
                                   constant LinearParams& p [[buffer(5)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup LinearGemmSmem sm;
  threadgroup ulong xoff[kGemmBM];
  linear_gemm(Bf16tSource{packed}, x, bias, y, p, tg, sg, lane, tid, sm, xoff);
}

#define NF12_MMA_KERNEL(MB)                                                                                        \
  NF12_THREADS(32 * kMmaMaxSimdgroups) kernel void nf12_linear_mma_nf12_b##MB(                                                                          \
      device const ushort* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],                              \
      device const uint4* escapes [[buffer(2)]], device const ushort* bias [[buffer(3)]],                          \
      device ushort* y [[buffer(4)]], constant LinearParams& p [[buffer(5)]],                                      \
      device const uchar* exp_base [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                           \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                         \
      uint tid [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {                       \
    threadgroup float part[kMmaMaxSimdgroups * 16 * 8 * MB];                                                       \
    linear_mma<MB>(NF12Source{packed, escapes, exp_base_x4(exp_base[0])}, x, bias, y, p, tg, sg, lane, tid,         \
                   threads.x, part);                                                                               \
  }                                                                                                                \
  NF12_THREADS(32 * kMmaMaxSimdgroups) kernel void nf12_linear_mma_bf16t_b##MB(                                                                         \
      device const ushort* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],                              \
      device const ushort* bias [[buffer(3)]], device ushort* y [[buffer(4)]],                                     \
      constant LinearParams& p [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],                           \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                         \
      uint tid [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {                       \
    threadgroup float part[kMmaMaxSimdgroups * 16 * 8 * MB];                                                       \
    linear_mma<MB>(Bf16tSource{packed}, x, bias, y, p, tg, sg, lane, tid, threads.x, part);                        \
  }

NF12_MMA_KERNEL(1)
NF12_MMA_KERNEL(2)
NF12_MMA_KERNEL(4)
