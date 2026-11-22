// SPDX-License-Identifier: Apache-2.0
// Fused decode + multiply cores shared by the dense and MoE linear kernels.
// Every output accumulates in FP32 in an order fixed by the kernel
// configuration and rounds once to BF16, so NF12 and bf16t sources give
// identical bits (SPEC.md section 6).
#pragma once

#include "nf12_metal.h"

namespace nf12 {

// ---------------------------------------------------------------------------
// Small M (GEMV-like). One simdgroup streams the super-atoms (nb, kb0..kb1) of
// a 16-row block. Lane (g, t) accumulates rows g and g + 8 over its 16 columns
// per super-atom for up to MT activation rows, whose element offsets in x are
// xoff[m]. acc0/acc1 hold rows g / g + 8 and are reduced across t afterwards.
template <uint MT, typename Src>
inline void gemv_accumulate(Src src, ulong sa_base, uint kb0, uint kb1, device const ushort* x,
                            const thread ulong (&xoff)[MT], uint lane, thread float (&acc0)[MT],
                            thread float (&acc1)[MT]) {
  const uint col = 2 * (lane & 3u);
  for (uint kb = kb0; kb < kb1; ++kb) {
    uint r[4][4];
    src.load(sa_base + kb, lane, r);
    float w0[16], w1[16];  // rows g / g + 8, columns 16j + col + {0, 1, 8, 9}
    for (uint j = 0; j < 4; ++j) {
      w0[4 * j + 0] = bf16_lo(r[j][0]);
      w0[4 * j + 1] = bf16_hi(r[j][0]);
      w0[4 * j + 2] = bf16_lo(r[j][2]);
      w0[4 * j + 3] = bf16_hi(r[j][2]);
      w1[4 * j + 0] = bf16_lo(r[j][1]);
      w1[4 * j + 1] = bf16_hi(r[j][1]);
      w1[4 * j + 2] = bf16_lo(r[j][3]);
      w1[4 * j + 3] = bf16_hi(r[j][3]);
    }
    const ulong kcol = ulong(kb) * kSuperAtomCols + col;
    for (uint m = 0; m < MT; ++m) {
      device const ushort* xr = x + xoff[m] + kcol;
      float a0 = acc0[m], a1 = acc1[m];
      for (uint j = 0; j < 4; ++j) {
        const uint p01 = *reinterpret_cast<device const uint*>(xr + 16 * j);
        const uint p89 = *reinterpret_cast<device const uint*>(xr + 16 * j + 8);
        const float x0 = bf16_lo(p01), x1 = bf16_hi(p01), x8 = bf16_lo(p89), x9 = bf16_hi(p89);
        a0 = fma(w0[4 * j + 0], x0, a0);
        a0 = fma(w0[4 * j + 1], x1, a0);
        a0 = fma(w0[4 * j + 2], x8, a0);
        a0 = fma(w0[4 * j + 3], x9, a0);
        a1 = fma(w1[4 * j + 0], x0, a1);
        a1 = fma(w1[4 * j + 1], x1, a1);
        a1 = fma(w1[4 * j + 2], x8, a1);
        a1 = fma(w1[4 * j + 3], x9, a1);
      }
      acc0[m] = a0;
      acc1[m] = a1;
    }
  }
}

// Sums the four t lanes of each g in a fixed order; every lane of the quad
// ends with the same bits. Must be called by the whole simdgroup.
template <uint MT>
inline void gemv_reduce_quad(thread float (&acc0)[MT], thread float (&acc1)[MT]) {
  for (uint m = 0; m < MT; ++m) {
    acc0[m] += simd_shuffle_xor(acc0[m], ushort(1));
    acc1[m] += simd_shuffle_xor(acc1[m], ushort(1));
    acc0[m] += simd_shuffle_xor(acc0[m], ushort(2));
    acc1[m] += simd_shuffle_xor(acc1[m], ushort(2));
  }
}

// ---------------------------------------------------------------------------
// Medium M (register MMA). Apple's 8x8 simdgroup_matrix fragment gives lane l
// the elements (fm, fn) and (fm, fn + 1) with
//   fm = (l / 4 & 4) + (l / 2) % 4,  fn = (l / 4 & 2) * 2 + (l % 2) * 2.
// A decoded NF12 register is also a bf16 pair (row g, columns 2t, 2t + 1) of an
// 8x8 quarter of an atom, so the A fragment of W[8h.., 16j + 8q..] is register
// R[h + 2q] of lane 4 fm + fn / 2: one shuffle per fragment, no threadgroup
// staging. B[k][m] = x[m][k] is loaded directly (rows fn, fn + 1 of the token tile).
// One simdgroup streams the super-atoms of a 16-row block for MB 8-token tiles;
// acc[h][mb] holds rows 8h..8h+7 x tokens 8mb..8mb+7.
inline ushort2 mma_coord(uint lane) {  // (fm, fn)
  const uint qid = lane / 4;
  return ushort2((qid & 4u) + (lane / 2) % 4u, (qid & 2u) * 2 + (lane % 2u) * 2);
}

template <uint MB, typename Src>
inline void mma_accumulate(Src src, ulong sa_base, uint kb0, uint kb1, device const ushort* x,
                           const thread ulong (&xrow)[2 * MB], uint lane, thread simdgroup_float8x8 (&acc)[2][MB]) {
  const ushort2 c = mma_coord(lane);
  const ushort src_lane = ushort(4 * c.x + c.y / 2);
  for (uint kb = kb0; kb < kb1; ++kb) {
    uint r[4][4];
    src.load(sa_base + kb, lane, r);
    const ulong kcol = ulong(kb) * kSuperAtomCols + c.x;
    for (uint j = 0; j < 4; ++j) {
      for (uint q = 0; q < 2; ++q) {
        simdgroup_float8x8 b[MB];
        for (uint mb = 0; mb < MB; ++mb) {
          const ulong k = kcol + 16 * j + 8 * q;
          b[mb].thread_elements()[0] = bf16_to_float(x[xrow[2 * mb] + k]);
          b[mb].thread_elements()[1] = bf16_to_float(x[xrow[2 * mb + 1] + k]);
        }
        for (uint h = 0; h < 2; ++h) {
          const uint pair = simd_shuffle(r[j][h + 2 * q], src_lane);
          simdgroup_float8x8 a;
          a.thread_elements()[0] = bf16_lo(pair);
          a.thread_elements()[1] = bf16_hi(pair);
          for (uint mb = 0; mb < MB; ++mb) simdgroup_multiply_accumulate(acc[h][mb], a, b[mb], acc[h][mb]);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Large M (GEMM). A threadgroup of kGemmSimdgroups computes a BN x BM tile of
// C = W x^T: W rows [n0, n0 + BN) and BM activation rows, one 64-column
// super-atom slab at a time. Simdgroup s decodes row block n0/16 + s into
// threadgroup memory as FP32 (exact), the activation slab is converted once,
// and simdgroup_float8x8 products accumulate in FP32.
constant constexpr uint kGemmSimdgroups = 4;
constant constexpr uint kGemmBN = 16 * kGemmSimdgroups;  // 64 weight rows
constant constexpr uint kGemmBM = 32;                    // activation rows
constant constexpr uint kGemmLd = kSuperAtomCols + 4;    // padded threadgroup stride
constant constexpr uint kGemmCLd = kGemmBM + 4;

struct GemmSmem {
  float w[kGemmBN * kGemmLd];
  float x[kGemmBM * kGemmLd];
};

// acc[r][c]: rows 16s + 8r, activation columns 8c of the tile.
using GemmAcc = simdgroup_float8x8[2][kGemmBM / 8];

inline void gemm_zero(thread GemmAcc& acc) {
  for (uint r = 0; r < 2; ++r)
    for (uint c = 0; c < kGemmBM / 8; ++c) acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}

template <typename T>
inline void gemm_multiply(threadgroup const T* w, threadgroup const T* x, thread GemmAcc& acc, uint sg) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint kk = 0; kk < kSuperAtomCols; kk += 8) {
    simdgroup_matrix<T, 8, 8> a0, a1;
    simdgroup_load(a0, w + (16 * sg) * kGemmLd + kk, kGemmLd);
    simdgroup_load(a1, w + (16 * sg + 8) * kGemmLd + kk, kGemmLd);
    for (uint c = 0; c < kGemmBM / 8; ++c) {
      simdgroup_matrix<T, 8, 8> b;  // b[k][m] = x[m][k]
      simdgroup_load(b, x + (8 * c) * kGemmLd + kk, kGemmLd, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[0][c], a0, b, acc[0][c]);
      simdgroup_multiply_accumulate(acc[1][c], a1, b, acc[1][c]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One 64-column slab. `nb_valid` is false for row blocks past N (zeros).
// xoff[i] (i < kGemmBM) are element offsets of the tile's activation rows or
// ~0ul for padding rows. All threads of the threadgroup must call this.
template <typename Src>
inline void gemm_slab(Src src, ulong sa, bool nb_valid, device const ushort* x, threadgroup const ulong* xoff,
                      uint kb, threadgroup GemmSmem& sm, thread GemmAcc& acc, uint sg, uint lane, uint tid) {
  // Decode this simdgroup's super-atom into rows 16 sg + {g, g + 8}.
  {
    const uint g = lane >> 2, col = 2 * (lane & 3u);
    threadgroup float* w0 = sm.w + (16 * sg + g) * kGemmLd + col;
    threadgroup float* w1 = w0 + 8 * kGemmLd;
    if (nb_valid) {
      uint r[4][4];
      src.load(sa, lane, r);
      for (uint j = 0; j < 4; ++j) {
        w0[16 * j + 0] = bf16_lo(r[j][0]);
        w0[16 * j + 1] = bf16_hi(r[j][0]);
        w0[16 * j + 8] = bf16_lo(r[j][2]);
        w0[16 * j + 9] = bf16_hi(r[j][2]);
        w1[16 * j + 0] = bf16_lo(r[j][1]);
        w1[16 * j + 1] = bf16_hi(r[j][1]);
        w1[16 * j + 8] = bf16_lo(r[j][3]);
        w1[16 * j + 9] = bf16_hi(r[j][3]);
      }
    } else {
      for (uint j = 0; j < 4; ++j) {
        w0[16 * j + 0] = w0[16 * j + 1] = w0[16 * j + 8] = w0[16 * j + 9] = 0.0f;
        w1[16 * j + 0] = w1[16 * j + 1] = w1[16 * j + 8] = w1[16 * j + 9] = 0.0f;
      }
    }
  }
  // Activation slab: kGemmBM rows x 64 columns, as bf16 pairs.
  for (uint i = tid; i < kGemmBM * (kSuperAtomCols / 2); i += 32 * kGemmSimdgroups) {
    const uint row = i / (kSuperAtomCols / 2), pair = i % (kSuperAtomCols / 2);
    const ulong off = xoff[row];
    float lo = 0.0f, hi = 0.0f;
    if (off != ~0ul) {
      const uint p = *reinterpret_cast<device const uint*>(x + off + ulong(kb) * kSuperAtomCols + 2 * pair);
      lo = bf16_lo(p);
      hi = bf16_hi(p);
    }
    sm.x[row * kGemmLd + 2 * pair] = lo;
    sm.x[row * kGemmLd + 2 * pair + 1] = hi;
  }
  gemm_multiply(sm.w, sm.x, acc, sg);
}

// Writes the accumulators to threadgroup memory as C[n][m] (kGemmBN x kGemmCLd floats,
// reusing the slab buffers). All threads must call this; read C after it returns.
inline threadgroup float* gemm_stage(thread GemmAcc& acc, threadgroup float* c, uint sg) {
  for (uint r = 0; r < 2; ++r)
    for (uint q = 0; q < kGemmBM / 8; ++q) simdgroup_store(acc[r][q], c + (16 * sg + 8 * r) * kGemmCLd + 8 * q, kGemmCLd);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return c;
}

}  // namespace nf12
