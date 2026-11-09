// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
//
// NF12-AD decode + fused GEMM for AMD CDNA3 (gfx942 / MI300X).
//
// Same tiled NF12 byte stream and the same fused-decode contract as the RDNA4
// port: the packed weight is never materialised in HBM, the decoded super-atom
// is staged once per 16-column weight block and reused by every token tile in
// the CTA, and accumulation is FP32 with a single BF16 rounding.
//
// Selection is trait-based (capability.h), not arch-string based. CDNA3 has no
// v_dot2_f32_bf16 (dot12- and dot13-insts are RDNA4) and no wmma-128b-insts,
// so the RDNA4 kernels cannot be reused. It does have v_mfma_*_bf16
// (gfx90a-insts), a 64-lane matrix core, which is what this file drives.
//
// MFMA 16x16x16 BF16 fragment map, DERIVED ON HARDWARE (one-hot probe on
// gfx942, then asserted against a CPU reference):
//
//   A reg (l,i) = A[ row = l % 16      ][ k   = 4*(l/16) + i ]
//   B reg (l,i) = B[ k   = 4*(l/16)+i ][ col = l % 16          ]
//   D reg (l,i) = D[ row = 4*(l/16)+i  ][ col = l % 16          ]
//
// So with B = W^T and W stored [n][k], the B fragment is
//   B(l,i) = W[ col_base + l%16 ][ kb + 4*(l/16) + i ]
// i.e. the same contiguous 4-BF16 window of a weight row that A takes from an
// activation row. That is why the decoded super-atom is staged row-major in
// shared memory: one 8-byte load serves as B for every token tile.
//
// Reduction order is fixed (K ascending, one v_mfma per 16 columns), so the
// nf12 and bf16t sources produce bitwise-identical outputs through the same
// template. SPEC.md 6.2 depends on that.
// ---------------------------------------------------------------------------
#include "capability.h"
#include "format_hip.cuh"
#include "linear.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstdint>

namespace nf12_hip {
namespace {

// 64-lane wave. The codec kernels use 32-lane atom groups; the matrix core does
// not, so this kernel indexes by the hardware wave and a CTA is 4 waves.
constexpr int kWave = 64;
constexpr int kWavesPerBlock = 1;
constexpr int kThreads = kWave * kWavesPerBlock;

// Token tiles per CTA. Each tile is 16 tokens. Capped at 8 so the accumulators
// (4 FP32 per tile) stay in VGPRs; M above 8 tiles is split across grid.y.
constexpr int kMaxTiles = 8;
constexpr int kMaxM = kMaxTiles * 16;

constexpr int64_t kMaxGridX = 0x7fffffffLL;
constexpr int64_t kMaxGridY = 0xffffLL;

// One 16-column weight block decoded row-major: 16 rows x 64 BF16.
//
// The row stride is padded by 4 BF16 (8 bytes) so the 16 weight-row starts land
// on 16 distinct shared-memory bank pairs. At the unpadded 64 BF16 (128 B) every
// row started on bank 0, so each B-fragment load was a 16-way conflict.
constexpr int kTileRowStride = kSuperAtomCols + 4;

using MfmaA = short __attribute__((ext_vector_type(4)));
using MfmaD = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ MfmaD mfma(MfmaA a, MfmaA b, MfmaD c) {
  // Compile-time ISA query, not an arch string. A target without this matrix
  // core falls through to `c`; the trait-based dispatcher never selects the
  // kernel there, and the symbol still exists so host and device agree.
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
  }
  (void)a;
  (void)b;
  return c;
}

// Decode super-atom `sa` (16 weight rows x 64 K) into a rowwise shared tile.
// Called with the low 32 lanes of a wave; the same codec helpers the RDNA4
// decode kernel uses, so the decoded BF16 matches nf12_hip_decode bit for bit.
template <WeightSource S>
__device__ __forceinline__ void decode_tile_from(const uint4 (&planes)[SourceTraits<S>::kLoads],
                                                 const uint4* __restrict__ escapes, uint32_t base_x4,
                                                 int dlane, __hip_bfloat16* __restrict__ dst) {
  AtomRegs r;
  source_planes_to_atoms<S>(planes, base_x4, escapes, r);
  RowTile x;
  atoms_to_tile(r, x);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int p = 0; p < 2; ++p) quad_transpose(x[h][p], dlane);
  }
  // Local tile origin: row (dlane>>2), chunk group (dlane&3). store_row_tile
  // adds (lane>>2)*ld + 8*(lane&3), which is exactly a 16x64 row-major tile.
  store_row_tile(dst, int64_t(dlane >> 2) * kTileRowStride + 8 * (dlane & 3),
                 kTileRowStride, x);
}

// Load + decode in one call. The dense tile kernel instead prefetches the planes
// a step ahead so the packed-weight latency overlaps the matrix-core work.
template <WeightSource S>
__device__ __forceinline__ void decode_tile(const uint4* __restrict__ packed,
                                            const uint4* __restrict__ escapes, uint32_t base_x4,
                                            int64_t sa, int dlane, __hip_bfloat16* __restrict__ dst) {
  uint4 planes[SourceTraits<S>::kLoads];
  load_source_planes<S>(packed, sa, dlane, planes);
  decode_tile_from<S>(planes, escapes, base_x4, dlane, dst);
}

// One lane's 4-BF16 window of an activation row. row = lane%16 (the M index),
// k = 4*(lane/16) + i, i.e. contiguous in K.
[[maybe_unused]] __device__ __forceinline__ MfmaA load_a_dense(const __hip_bfloat16* __restrict__ x, int64_t ldx,
                                              int64_t row, int64_t k) {
  const int lane = threadIdx.x % kWave;
  const int64_t off = row * ldx + k + 4 * (lane / 16);
  return *reinterpret_cast<const MfmaA*>(x + off);
}

// B fragment from the decoded shared tile: row = lane%16 is the weight row
// (= output column), k = 4*(lane/16) + i is contiguous inside the super-atom.
__device__ __forceinline__ MfmaA load_b_shared(const __hip_bfloat16* __restrict__ sh, int64_t k) {
  const int lane = threadIdx.x % kWave;
  const int64_t off = int64_t(lane % 16) * kTileRowStride + k + 4 * (lane / 16);
  return *reinterpret_cast<const MfmaA*>(sh + off);
}

// Row map for the routed (MoE) launch. Null pointers select the dense address
// form, so both routes share one kernel instead of growing a second GEMM.
struct MfmaRoute {
  const int32_t* gather;
  const int32_t* scatter;
  const float* row_scale;
  const int64_t* expert_offsets;
  const int64_t* escape_offsets;
  const uint8_t* exp_base;
  int64_t gather_div;
  int64_t max_assignments;
  int64_t ldx;
  int64_t ldy;
  int64_t weight_expert_vecs;
  int32_t num_experts;
  int64_t row_base;
};

template <WeightSource S, bool kRouted, int kMt>
__global__ void __launch_bounds__(kThreads)
    mfma_kernel(const __hip_bfloat16* __restrict__ x, const uint4* __restrict__ packed,
                const uint4* __restrict__ escapes, uint32_t base_x4, int64_t n, int64_t k,
                int64_t m_rows, __hip_bfloat16* __restrict__ out,
                const __hip_bfloat16* __restrict__ bias, MfmaRoute route_in) {
  static_assert(!kRouted || kMt == 1, "a routed tile never batches token tiles");
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int lo = lane % 16;   // output column within the 16-wide tile
  const int quad = lane / 16; // 0..3, K group inside a 16-K chunk

  const int64_t n0 = (int64_t(blockIdx.x) * kWavesPerBlock + wave) * kSuperAtomRows;
  if (n0 >= n) return;

  int64_t m = m_rows;
  int64_t m0 = int64_t(blockIdx.y) * (kMt * 16);
  MfmaRoute route = route_in;
  if constexpr (kRouted) {
    // grid.y/z flatten (expert, 16-assignment tile). Counts are device-side, so
    // scan the clamped offsets; surplus CTAs return before touching memory.
    int64_t rem = int64_t(blockIdx.z) * gridDim.y + blockIdx.y;
    int64_t start = 0, rows = 0;
    int64_t l = moe_next_offset(route.expert_offsets[0], 0, route.max_assignments);
    int32_t e = 0;
    for (; e < route.num_experts; ++e) {
      const int64_t hi = moe_next_offset(route.expert_offsets[e + 1], l, route.max_assignments);
      const int64_t cnt = hi > l ? hi - l : 0;
      const int64_t tiles = (cnt + 15) / 16;
      if (rem < tiles) {
        start = l;
        rows = cnt;
        break;
      }
      rem -= tiles;
      l = hi;
    }
    if (e == route.num_experts) return;
    m0 = rem * 16;
    m = rows;
    route.row_base = start;
    packed += int64_t(e) * route.weight_expert_vecs;
    base_x4 = exp_base_x4(route.exp_base[e]);
    if (escapes != nullptr) escapes = escapes + route.escape_offsets[e];
  }

  const int64_t m_local = min(int64_t(kMt * 16), m - m0);
  if (m_local <= 0) return;

  const int64_t k_blocks = k / kSuperAtomCols;
  const float bias_v = (bias != nullptr) ? __bfloat162float(bias[n0 + lo]) : 0.0f;

  // Decoded super-atom staging, one per wave.
  __shared__ __align__(16) __hip_bfloat16 tile[kWavesPerBlock][kSuperAtomRows * kTileRowStride];
  __hip_bfloat16* const sh = &tile[wave][0];

  MfmaD acc[kMt];
#pragma unroll
  for (int t = 0; t < kMt; ++t) acc[t] = MfmaD{0.0f, 0.0f, 0.0f, 0.0f};

  // Decode with the wave's low 32 lanes; both halves then meet at the barrier.
  const int dlane = lane & 31;
  const bool decode_pass = lane < 32;

  auto a_frag = [&](int64_t t, int64_t kk) -> MfmaA {
    if constexpr (kRouted) {
      int64_t pos = route.row_base + t + lo;
      const int64_t end = route.row_base + m;
      if (pos >= end) pos = end - 1;  // clamp; the epilogue drops repeated rows
      const int64_t in = route.gather ? int64_t(route.gather[pos]) / route.gather_div : pos;
      return *reinterpret_cast<const MfmaA*>(x + in * route.ldx + kk + 4 * quad);
    } else {
      // A partial tail tile must not read past `x`, so its 16-row window is
      // clamped to the last full window (m - 16). That recomputes rows an
      // earlier tile already covered; the epilogue below keeps each output
      // written exactly once by filtering on the tile's own disjoint range.
      const int64_t origin = max(int64_t(0), min(t, m - 16));
      return load_a_dense(x, route.ldx, origin + lo, kk);
    }
  };

  for (int64_t kb = 0; kb < k_blocks; ++kb) {
    const int64_t sa = (n0 / kSuperAtomRows) * k_blocks + kb;
    if (decode_pass) decode_tile<S>(packed, escapes, base_x4, sa, dlane, sh);
    __syncthreads();

    const int64_t kbase = kb * kSuperAtomCols;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const MfmaA b = load_b_shared(sh, int64_t(16 * j));
#pragma unroll
      for (int t = 0; t < kMt; ++t) {
        const int64_t tr = m0 + int64_t(t) * 16;
        if (tr >= m) break;
        // Dense: clamp the tile origin so a tail tile cannot read past `x`.
        // Routed: `tr` is the tile's offset inside the expert (m0 + 16*t), and
        // a_frag clamps a tail row per assignment.
        const int64_t a_row = kRouted ? tr : max(int64_t(0), min(tr, m - 16));
        acc[t] = mfma(a_frag(a_row, kbase + 16 * j), b, acc[t]);
      }
    }
    __syncthreads();
  }

  // Epilogue. D(row=4*quad+i, col=lo) relative to the tile; rows outside this
  // tile's disjoint range are dropped so a clamped tail tile is harmless.
#pragma unroll
  for (int t = 0; t < kMt; ++t) {
    const int64_t lo_row = m0 + int64_t(t) * 16;
    const int64_t hi_row = min(lo_row + 16, m);
    if (lo_row >= m) break;
    // Dense output rows are relative to the CLAMPED tile origin that a_frag
    // read (a partial tail tile reuses the last full window), then filtered on
    // this tile's own disjoint range so each row is written exactly once.
    // Routed rows are assignment-local, so the origin is the tile offset inside
    // the expert (a_frag's clamp makes the tail reuse the last row).
    const int64_t origin = kRouted ? lo_row : max(int64_t(0), min(lo_row, m - 16));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t local = 4 * quad + i;
      const int64_t row = origin + local;
      if (row < lo_row || row >= hi_row) continue;
      float v = acc[t][i] + bias_v;
      if constexpr (kRouted) {
        const int64_t pos = route.row_base + row;
        const int64_t o = route.scatter ? int64_t(route.scatter[pos]) : pos;
        if (route.row_scale != nullptr) v *= route.row_scale[o];
        out[o * route.ldy + n0 + lo] = __float2bfloat16(v);
      } else {
        out[row * route.ldy + n0 + lo] = __float2bfloat16(v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Dense fused GEMM.
//
// Two costs dominate on a 304-CU part:
//
//  * Parallelism. One 16-column tile per CTA exposes only n/16 CTAs (256 at
//    n = 4096), and a wave that walks all k/64 super-atoms serially is latency
//    bound. A CTA here owns kCols column tiles (one wave each) and the grid
//    splits K across kSplits CTAs, which together fill the machine.
//
//  * Activation re-reads. A CTA covering one 16-column tile re-reads the whole
//    activation matrix once per tile, so at m = 64 the activations cost 16x the
//    packed weight and the kernel is nowhere near its byte floor. Staging the
//    activation block in shared memory once per K step, coalesced and shared by
//    every wave of the CTA, divides that traffic by kCols.
//
// The packed weight is still never materialised: each wave decodes only its own
// 16-column stripe. With kSplits == 1 the FP32 result is rounded to BF16 once,
// exactly as before; with more splits each CTA writes an FP32 partial and
// mfma_reduce_kernel sums them in split order, so the result is deterministic.
// ---------------------------------------------------------------------------
// Activation staging stride: 16-byte aligned so the coalesced copy is a uint4
// store, and an odd multiple of 8 BF16 so the 16 weight rows of a fragment land
// on alternating bank halves instead of one.
constexpr int kXStride = kSuperAtomCols + 8;

// Token tiles per CTA. Above four the activation staging and the accumulator
// array stop fitting in a CTA that keeps a second resident slot, and the extra
// packed-weight re-read that grid.y splitting costs is smaller than the loss.
constexpr int kFusedTiles = 4;

template <WeightSource S, int kMt, int kCols>
__global__ void __launch_bounds__(kWave * kCols)
    mfma_tile_kernel(const __hip_bfloat16* __restrict__ x, const uint4* __restrict__ packed,
                     const uint4* __restrict__ escapes, uint32_t base_x4, int64_t n, int64_t k,
                     int64_t m_rows, __hip_bfloat16* __restrict__ out,
                     const __hip_bfloat16* __restrict__ bias, int64_t ldx, int64_t ldy,
                     float* __restrict__ partial) {
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int lo = lane % 16;   // output column inside the 16-wide tile
  const int quad = lane / 16; // 0..3, K group inside a 16-K chunk

  const int64_t n0 = int64_t(blockIdx.x) * (kCols * kSuperAtomRows) + wave * kSuperAtomRows;
  if (n0 >= n) return;
  const int64_t m0 = int64_t(blockIdx.y) * (kMt * 16);
  if (m0 >= m_rows) return;

  __shared__ __align__(16) __hip_bfloat16 xs[kMt * kSuperAtomRows * kXStride];
  __shared__ __align__(16) __hip_bfloat16 tile[kCols][kSuperAtomRows * kTileRowStride];

  // Per-tile activation row origin. A partial tail tile reuses the last full
  // 16-row window, so it can never read past `x`; the epilogue drops the rows
  // outside that tile's own disjoint range.
  int64_t a_origin[kMt];
#pragma unroll
  for (int t = 0; t < kMt; ++t)
    a_origin[t] = std::max(int64_t(0), std::min(m0 + int64_t(t) * 16, m_rows - 16));

  const int64_t k_blocks = k / kSuperAtomCols;
  const int64_t per = (k_blocks + gridDim.z - 1) / gridDim.z;
  const int64_t kb_begin = int64_t(blockIdx.z) * per;
  const int64_t kb_end = std::min(k_blocks, kb_begin + per);

  MfmaD acc[kMt];
#pragma unroll
  for (int t = 0; t < kMt; ++t) acc[t] = MfmaD{0.0f, 0.0f, 0.0f, 0.0f};

  const int dlane = lane & 31;
  const bool decode_pass = lane < 32;
  const int nthreads = kWave * kCols;
  const int b_row = (lane % 16) * kTileRowStride + 4 * quad;
  const int a_col = (lane % 16) * kXStride + 4 * quad;
  __hip_bfloat16* const sh = &tile[wave][0];
  const int64_t sa0 = (n0 / kSuperAtomRows) * k_blocks;

  // Prefetch one super-atom's packed planes. The decode is a small amount of
  // arithmetic behind a 1.5 KiB load, so issuing that load a step early keeps
  // the packed-weight latency off the critical path.
  uint4 planes[SourceTraits<S>::kLoads];
  if (decode_pass && kb_begin < kb_end) load_source_planes<S>(packed, sa0 + kb_begin, dlane, planes);

  for (int64_t kb = kb_begin; kb < kb_end; ++kb) {
    // Stage this K block of every token tile once, 16 B per thread and fully
    // coalesced, while this wave decodes its own column stripe.
    // Unrolled over the token tiles: a runtime tile index turns `a_origin[t]`
    // into a dynamically indexed array, and the compiler replaces it with a
    // select chain per staged chunk.
    const int64_t kb_off = kb * kSuperAtomCols;
#pragma unroll
    for (int t = 0; t < kMt; ++t) {
      const __hip_bfloat16* srcb = x + a_origin[t] * ldx + kb_off;
      __hip_bfloat16* dstb = &xs[t * kSuperAtomRows * kXStride];
      for (int i = threadIdx.x; i < kSuperAtomRows * 8; i += nthreads) {
        const int rr = i >> 3;
        const int c = (i & 7) * 8;
        *reinterpret_cast<uint4*>(dstb + rr * kXStride + c) =
            *reinterpret_cast<const uint4*>(srcb + rr * ldx + c);
      }
    }
    if (decode_pass) {
      decode_tile_from<S>(planes, escapes, base_x4, dlane, sh);
      if (kb + 1 < kb_end) load_source_planes<S>(packed, sa0 + kb + 1, dlane, planes);
    }
    __syncthreads();

    // Every fragment for this K block is loaded before the first MFMA: left
    // interleaved the compiler waits on each 8-byte load immediately before the
    // MFMA that consumes it and exposes the full latency sixteen times a step.
    MfmaA bf[4];
    MfmaA af[kMt][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) bf[j] = *reinterpret_cast<const MfmaA*>(sh + b_row + 16 * j);
#pragma unroll
    for (int t = 0; t < kMt; ++t) {
      const __hip_bfloat16* arow = &xs[t * kSuperAtomRows * kXStride + a_col];
#pragma unroll
      for (int j = 0; j < 4; ++j) af[t][j] = *reinterpret_cast<const MfmaA*>(arow + 16 * j);
    }
#pragma unroll
    for (int t = 0; t < kMt; ++t) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[t] = mfma(af[t][j], bf[j], acc[t]);
    }
    __syncthreads();
  }

  // Epilogue. One K split writes the result directly; more than one writes an
  // FP32 partial plane for mfma_reduce_kernel.
  float* const pdst =
      partial != nullptr ? partial + int64_t(blockIdx.z) * (int64_t(m_rows) * n) : nullptr;
  const float bias_v = (bias != nullptr) ? __bfloat162float(bias[n0 + lo]) : 0.0f;
#pragma unroll
  for (int t = 0; t < kMt; ++t) {
    const int64_t lo_row = m0 + int64_t(t) * 16;
    if (lo_row >= m_rows) break;
    const int64_t hi_row = std::min(lo_row + 16, m_rows);
    const int64_t origin = a_origin[t];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t row = origin + 4 * quad + i;
      if (row < lo_row || row >= hi_row) continue;
      if (pdst != nullptr) {
        pdst[row * n + n0 + lo] = acc[t][i];
      } else {
        out[row * ldy + n0 + lo] = __float2bfloat16(acc[t][i] + bias_v);
      }
    }
  }
}

// Sums the per-split FP32 planes in ascending split order and applies bias
// before the single BF16 rounding. Split order, not atomic order, so the result
// is bit-identical from run to run.
__global__ void mfma_reduce_kernel(const float* __restrict__ partial, int64_t total, int splits,
                                   int64_t n, const __hip_bfloat16* __restrict__ bias,
                                   __hip_bfloat16* __restrict__ out, int64_t ldy) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += int64_t(gridDim.x) * blockDim.x) {
    float s = partial[i];
    for (int j = 1; j < splits; ++j) s += partial[int64_t(j) * total + i];
    if (bias != nullptr) s += __bfloat162float(bias[i % n]);
    out[(i / n) * ldy + i % n] = __float2bfloat16(s);
  }
}

// Launch shape for the dense route. A pure function of the problem so the
// workspace query and the launcher can never disagree about the split count.
struct MfmaPlan {
  int kCols;
  int kMt;
  int kSplits;
  int64_t mBlocks;
  int64_t nGroups;
};

inline MfmaPlan mfma_plan(int64_t m, int64_t n, int64_t k) {
  (void)k;
  MfmaPlan p{};
  const int64_t n_tiles = n / kSuperAtomRows;
  const int tiles = static_cast<int>((m + 15) / 16);
  p.kCols = (n_tiles % 4 == 0) ? 4 : 1;
  p.kMt = tiles < kFusedTiles ? tiles : kFusedTiles;
  p.mBlocks = (m + p.kMt * 16 - 1) / (p.kMt * 16);
  p.nGroups = n_tiles / p.kCols;
  // Two CTAs per CU is enough to keep the memory pipe busy; stop there rather
  // than paying for more partial planes than the machine can use.
  constexpr int64_t kTargetCtas = 2 * 304;
  // Enough CTAs per CU to keep the memory pipe busy, without paying for a
  // partial plane bigger than the work it splits. A wide N already supplies
  // the CTAs; a tall tile (kMt > 2) has more MFMA per barrier and needs fewer.
  p.kSplits = (p.nGroups >= 128) ? 4 : (p.kMt <= 2 ? 16 : 8);
  while (p.kSplits > 1 && p.nGroups * p.mBlocks * p.kSplits > 2 * kTargetCtas) p.kSplits /= 2;
  return p;
}

// Dense launcher: [m, n] from [m, k]. M is split across grid.y above one CTA's
// tile budget, exactly like the RDNA4 chunking in backend.hip.
template <WeightSource S>
hipError_t linear_launch(const uint8_t* packed, const uint4* escapes, uint32_t exp_base, const void* x,
                         void* out, int64_t n, int64_t k, int64_t m, const void* bias,
                         int64_t ldx, int64_t ldy, void* workspace, size_t workspace_bytes,
                         hipStream_t stream) {
  if constexpr (S == WeightSource::kNf12) {
    if (exp_base > uint32_t(kMaxExpBase)) return hipErrorInvalidValue;
  }
  if (n <= 0 || k <= 0 || n % kSuperAtomRows != 0 || k % kSuperAtomCols != 0) {
    return hipErrorInvalidValue;
  }
  // A 16-row A tile cannot be formed below 16 tokens without reading past `x`;
  // the caller routes that band to the fused GEMV instead.
  if (m < 16) return hipErrorInvalidValue;
  const int64_t n_tiles = n / kSuperAtomRows;
  const int tiles = static_cast<int>((m + 15) / 16);
  if (n_tiles == 0 || tiles == 0) return hipSuccess;
  const MfmaPlan plan = mfma_plan(m, n, k);
  if (plan.nGroups > kMaxGridX || plan.mBlocks > kMaxGridY || plan.kSplits > kMaxGridY) {
    return hipErrorNotSupported;
  }

  const int64_t total = m * n;
  float* partial = nullptr;
  if (plan.kSplits > 1) {
    if (workspace == nullptr || workspace_bytes < size_t(total) * plan.kSplits * sizeof(float)) {
      return hipErrorInvalidValue;
    }
    partial = static_cast<float*>(workspace);
  }

  const dim3 grid(static_cast<unsigned>(plan.nGroups), static_cast<unsigned>(plan.mBlocks),
                  static_cast<unsigned>(plan.kSplits));
  const auto* xb = reinterpret_cast<const __hip_bfloat16*>(x);
  const auto* pb = reinterpret_cast<const uint4*>(packed);
  auto* yb = reinterpret_cast<__hip_bfloat16*>(out);
  const auto* bb = reinterpret_cast<const __hip_bfloat16*>(bias);
  const uint32_t bx4 = exp_base_x4(exp_base);

#define NF12_MFMA_TILE(MT, KC)                                                             \
  mfma_tile_kernel<S, MT, KC><<<grid, kWave * KC, 0, stream>>>(xb, pb, escapes, bx4, n, k,  \
                                                               m, yb, bb, ldx, ldy, partial)
#define NF12_MFMA_COLS(MT)          \
  do {                              \
    if (plan.kCols == 4)            \
      NF12_MFMA_TILE(MT, 4);        \
    else                            \
      NF12_MFMA_TILE(MT, 1);        \
  } while (0)
  switch (plan.kMt) {
    case 1: NF12_MFMA_COLS(1); break;
    case 2: NF12_MFMA_COLS(2); break;
    case 3: NF12_MFMA_COLS(3); break;
    default: NF12_MFMA_COLS(kFusedTiles); break;
  }
#undef NF12_MFMA_COLS
#undef NF12_MFMA_TILE
  if (plan.kSplits > 1) {
    constexpr int kReduceThreads = 256;
    int64_t blocks = (total + kReduceThreads - 1) / kReduceThreads;
    if (blocks > kMaxGridX) blocks = kMaxGridX;
    mfma_reduce_kernel<<<static_cast<unsigned>(blocks), kReduceThreads, 0, stream>>>(
        partial, total, plan.kSplits, n, bb, yb, ldy);
  }
  return hipGetLastError();
}

// Routed launcher. grid.y/z enumerate the flattened (expert, tile) space, so
// the grid is the worst-case bound and surplus CTAs exit before reading.
template <WeightSource S>
hipError_t moe_launch(const uint8_t* packed, const uint4* escapes, const int64_t* escape_offsets,
                      const uint8_t* exp_base, int32_t num_experts, const void* x, int64_t ldx,
                      const int32_t* gather, int64_t gather_div, const int32_t* scatter,
                      const int64_t* expert_offsets, int64_t max_assignments, const float* row_scale,
                      void* out, int64_t ldy, int64_t n, int64_t k, hipStream_t stream) {
  if (n % kSuperAtomRows != 0 || num_experts < 1 || max_assignments <= 0) return hipErrorInvalidValue;
  const int64_t n_blocks = (n / kSuperAtomRows + kWavesPerBlock - 1) / kWavesPerBlock;
  const int64_t m_tiles = (max_assignments + 15) / 16 + num_experts;
  if (n_blocks == 0) return hipSuccess;
  const int64_t grid_y = std::min(m_tiles, kMaxGridY);
  const int64_t grid_z = (m_tiles + grid_y - 1) / grid_y;
  if (n_blocks > kMaxGridX || grid_z > kMaxGridY) return hipErrorNotSupported;

  MfmaRoute route{};
  route.gather = gather;
  route.scatter = scatter;
  route.row_scale = row_scale;
  route.expert_offsets = expert_offsets;
  route.escape_offsets = escape_offsets;
  route.exp_base = exp_base;
  route.gather_div = gather_div;
  route.max_assignments = max_assignments;
  route.ldx = ldx;
  route.ldy = ldy;
  route.weight_expert_vecs = (n / kSuperAtomRows) * (k / kSuperAtomCols) * SourceTraits<S>::kVecs;
  route.num_experts = num_experts;

  const dim3 grid(static_cast<unsigned>(n_blocks), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  mfma_kernel<S, true, 1><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(x), reinterpret_cast<const uint4*>(packed), escapes, 0u,
      n, k, 0, reinterpret_cast<__hip_bfloat16*>(out), nullptr, route);
  return hipGetLastError();
}

}  // namespace

hipError_t linear_mfma_strided(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                              const void* x, void* out, int64_t n, int64_t k, int64_t m,
                              const void* bias, int64_t ldx, int64_t ldy, void* workspace,
                              size_t workspace_bytes, bool bf16t, hipStream_t stream) {
  if (ldx < k || ldy < n) return hipErrorInvalidValue;
  if (bf16t) {
    return linear_launch<WeightSource::kBf16t>(packed, nullptr, 0, x, out, n, k, m, bias, ldx, ldy,
                                               workspace, workspace_bytes, stream);
  }
  return linear_launch<WeightSource::kNf12>(packed, escapes, exp_base, x, out, n, k, m, bias, ldx,
                                            ldy, workspace, workspace_bytes, stream);
}

// K splits the dense route will use for this shape. The workspace query and the
// launcher both read it, so they cannot disagree about the partial plane.
int mfma_dense_splits(int64_t m, int64_t n, int64_t k) {
  if (m < 16 || n <= 0 || k <= 0 || n % kSuperAtomRows != 0) return 1;
  return mfma_plan(m, n, k).kSplits;
}

hipError_t moe_linear_mfma(const uint8_t* packed, const uint4* escapes, const int64_t* escape_offsets,
                           const uint8_t* exp_base, int32_t num_experts, const void* x, int64_t ldx,
                           const int32_t* gather, int64_t gather_div, const int32_t* scatter,
                           const int64_t* expert_offsets, int64_t max_assignments,
                           const float* row_scale, void* out, int64_t ldy, int64_t n, int64_t k,
                           bool bf16t, hipStream_t stream) {
  if (bf16t) {
    return moe_launch<WeightSource::kBf16t>(packed, nullptr, nullptr, nullptr, num_experts, x, ldx,
                                            gather, gather_div, scatter, expert_offsets,
                                            max_assignments, row_scale, out, ldy, n, k, stream);
  }
  return moe_launch<WeightSource::kNf12>(packed, escapes, escape_offsets, exp_base, num_experts, x,
                                         ldx, gather, gather_div, scatter, expert_offsets,
                                         max_assignments, row_scale, out, ldy, n, k, stream);
}

// Largest M one launch covers before M is split across grid.y.
int mfma_max_m() { return kMaxM; }

}  // namespace nf12_hip
