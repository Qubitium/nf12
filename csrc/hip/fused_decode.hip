// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from GPT-QModel (https://github.com/ModelCloud/GPTQModel) by Qubitium.
// ---------------------------------------------------------------------------
// Single-token fused MoE for AMD RDNA4 (gfx1200 / gfx1201 / gfx1250).
//
// Credits: NyoomFloat12 algorithm by Liberman, Fang, Zhang, Dao and Fu
// (Together AI, 2026), https://www.together.ai/. NF12 format, reference codec
// and CUDA kernels by AlpinDale, https://github.com/dphnAI/nf12 (Apache-2.0).
//
// This is the RDNA4 backend for `nf12::torch_detail::fused_decode`, the seam
// csrc/torch/ops.cpp uses for `nf12._fused_moe` when T == 1. It exists so the
// torch op layer and the C ABI are byte-for-byte the same on both backends
// instead of ROCm silently taking the composed moe_* path.
//
// Why this is a reimplementation, not a transliteration
// -----------------------------------------------------
// csrc/torch/fused_decode.cu is built on Hopper features RDNA4 does not have:
// thread-block clusters (`cudaLaunchAttributeClusterDimension`, DSMEM, and the
// cross-rank reduction the down kernel relies on), programmatic dependent
// launch, and the `small_m_core` bulk weight ring (mbarrier + cp.async). None of
// that maps to gfx12, so the arithmetic is rebuilt on the primitives that do:
// packed BF16 dot products (v_dot2_f32_bf16) with 8 warps splitting K.
//
// The numerics contract is the same one the composed route has (SPEC.md 6.2,
// 9.2): round the gate and up halves to BF16, apply SiLU(gate) * up in FP32
// with one BF16 rounding, then per assignment round `sum * scale` to BF16,
// combine the top-k terms in FP32 in increasing j and round once. That is what
// makes the fused path interchangeable with `moe_align` + `moe_linear` +
// `moe_silu_mul` + `moe_linear` + `moe_combine`; the tests assert it against
// both the reference and the composed path.
//
// Two launches, no workspace, no host synchronisation: the activation plane is
// the caller's `[topk, I]` buffer, and every kernel reads the expert ids on
// device. The library therefore keeps no mutable device state and concurrent
// streams are independent.
// ---------------------------------------------------------------------------
#include "torch/fused_decode.h"

#include "format_hip.cuh"

#include <hip/hip_runtime_api.h>

#include <climits>
#include <cstddef>
#include <cstdint>

namespace nf12::torch_detail {
namespace {

using nf12_hip::AtomRegs;
using nf12_hip::dot_super_atom;
using nf12_hip::exp_base_x4;
using nf12_hip::load_source_planes;
using nf12_hip::source_planes_to_atoms;
using nf12_hip::SourceTraits;
using nf12_hip::WeightSource;

constexpr int kWarps = nf12_hip::kWarpsPerBlock;
constexpr int kThreads = nf12_hip::kThreadsPerBlock;
constexpr int kRows = nf12_hip::kSuperAtomRows;
constexpr int kCols = nf12_hip::kSuperAtomCols;
constexpr int kWarp = nf12_hip::kWarpSize;

// ---------------------------------------------------------------------------
// One 16-row GEMV block, K split across the eight warps of the CTA
// ---------------------------------------------------------------------------
// Same partition as the dense fused GEMV (decode.hip, `linear_kernel`):
// contiguous equal super-atom slices, the last warp taking the remainder. The
// reduction order is fixed, so a shape is deterministic across launches.
//
// A warp-per-column-block variant (each warp streaming one block over the whole
// K, no shared reduction) was measured on gfx1201 and is 4% slower here: the
// MoE launch has eight times fewer blocks, so halving the CTA count costs more
// than the saved barrier. The K split also keeps eight streams per CTA over the
// same 98 KB, which measured slightly better than eight 98 KB-apart streams.
__device__ __forceinline__ void k_slice(int64_t k_blocks, int warp, int64_t& lo, int64_t& hi) {
  const int64_t per = (k_blocks + kWarps - 1) / kWarps;
  lo = int64_t(warp) * per;
  hi = lo + per;
  if (hi > k_blocks) hi = k_blocks;
  if (lo > hi) lo = hi;
}

// Walks `nb`'s super-atoms over this warp's K slice and publishes the two rows
// the lane owns into `part[warp]`. The caller zeroes `part` before the first
// use and calls __syncthreads() before reading it.
template <WeightSource S>
__device__ __forceinline__ void gemv_partials(const uint4* __restrict__ packed,
                                              const uint4* __restrict__ escapes, uint32_t base_x4,
                                              int64_t k, int64_t nb,
                                              const __hip_bfloat16* __restrict__ xrow,
                                              float (*part)[kRows], int lane, int warp) {
  const int t = lane & 3;
  const int g = lane >> 2;
  const int64_t k_blocks = k / kCols;
  int64_t kb, kb_end;
  k_slice(k_blocks, warp, kb, kb_end);

  // One super-atom of lookahead, so the 1536-byte fetch for the next super-atom
  // is in flight while this one is decoded and dotted.
  float lo = 0.0f;
  float hi = 0.0f;
  uint4 planes[SourceTraits<S>::kLoads];
  uint4 next[SourceTraits<S>::kLoads];
  if (kb < kb_end) load_source_planes<S>(packed, nb * k_blocks + kb, lane, planes);
  for (; kb < kb_end; ++kb) {
    if (kb + 1 < kb_end) load_source_planes<S>(packed, nb * k_blocks + kb + 1, lane, next);
    AtomRegs regs;
    source_planes_to_atoms<S>(planes, base_x4, escapes, regs);
    dot_super_atom(regs, reinterpret_cast<const __hip_bfloat162*>(xrow + kb * kCols), t, lo, hi);
    if (kb + 1 < kb_end) {
#pragma unroll
      for (int p = 0; p < SourceTraits<S>::kLoads; ++p) planes[p] = next[p];
    }
  }

  // Collapse the four t lanes of this quad (fixed xor order 1 then 2), exactly
  // as the dense GEMV does.
  lo += __shfl_xor(lo, 1);
  lo += __shfl_xor(lo, 2);
  hi += __shfl_xor(hi, 1);
  hi += __shfl_xor(hi, 2);
  if (t == 0) {
    part[warp][g] = lo;
    part[warp][g + kRows / 2] = hi;
  }
}

// Sum the eight warp partials in increasing warp order so the result does not
// depend on scheduling.
__device__ __forceinline__ float gemv_reduce(const float (*part)[kRows], int row) {
  float sum = part[0][row];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) sum += part[w][row];
  return sum;
}

__device__ __forceinline__ const uint4* expert_packed(const void* base, int64_t vecs, int32_t e) {
  return static_cast<const uint4*>(base) + int64_t{e} * vecs;
}

__device__ __forceinline__ const uint4* expert_escapes(const void* escapes,
                                                       const int64_t* offsets, int32_t e) {
  return (escapes != nullptr) ? static_cast<const uint4*>(escapes) + offsets[e] : nullptr;
}

// ---------------------------------------------------------------------------
// Gate: x -> 2I -> SiLU(gate) * up -> activation[topk, I]
// ---------------------------------------------------------------------------
// One CTA owns one 16-column tile of the I-wide activation for one assignment,
// so it evaluates both the gate block (columns nb) and the matching up block
// (columns nb + I/16) and writes the I-wide result. That is why a CTA reads two
// super-atom rows but produces one: the activation plane is I wide, not 2I.
struct GateParams {
  FusedDecodeArgs a;
  int64_t inter_blocks;
  int64_t vecs;  // uint4s per expert of w13
};

// Round the two halves to BF16, gate in FP32, round the product once. This is
// moe_silu_mul's expression; no fast-math in this build, so / and + are
// correctly rounded and cannot be contracted.
__device__ __forceinline__ __hip_bfloat16 silu_mul(float gate_f32, float up_f32) {
  const float gate = __bfloat162float(__float2bfloat16(gate_f32));
  const float up = __bfloat162float(__float2bfloat16(up_f32));
  return __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
}

template <WeightSource S>
__global__ void __launch_bounds__(kThreads) gate_kernel(GateParams p) {
  __shared__ float part[2][kWarps][kRows];
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int64_t nb = blockIdx.x;
  const int j = static_cast<int>(blockIdx.y);

  for (int64_t i = threadIdx.x; i < 2 * kWarps * kRows; i += kThreads) (&part[0][0][0])[i] = 0.0f;
  __syncthreads();

  auto* const act = static_cast<__hip_bfloat16*>(p.a.activation);
  __hip_bfloat16* const dst = act + int64_t{j} * p.a.ldactivation + nb * kRows;
  const int32_t e = p.a.ids[j];

  // A dropped assignment contributes nothing: the composed route zero-fills its
  // w13 row, so silu(0) * 0 is exactly zero here too.
  if (e < 0 || int64_t{e} >= p.a.w13.experts) {
    if (threadIdx.x < kRows) dst[threadIdx.x] = __float2bfloat16(0.0f);
    return;
  }

  const uint4* const packed = expert_packed(p.a.w13.packed, p.vecs, e);
  const uint4* const esc = expert_escapes(p.a.w13.escapes, p.a.w13.escape_offsets, e);
  const uint32_t base_x4 = exp_base_x4(p.a.w13.exp_base[e]);
  const auto* const x = static_cast<const __hip_bfloat16*>(p.a.x);

  gemv_partials<S>(packed, esc, base_x4, p.a.w13.k, nb, x, part[0], lane, warp);
  gemv_partials<S>(packed, esc, base_x4, p.a.w13.k, nb + p.inter_blocks, x, part[1], lane, warp);
  __syncthreads();

  if (threadIdx.x < kRows) {
    const int r = threadIdx.x;
    dst[r] = silu_mul(gemv_reduce(part[0], r), gemv_reduce(part[1], r));
  }
}

// ---------------------------------------------------------------------------
// Down: activation[topk, I] -> sum_j scale_j * (act_j @ W2[e_j]^T)
// ---------------------------------------------------------------------------
// One CTA owns one 16-column tile of the output and walks the assignments in
// increasing j. Each term is scaled in FP32 and rounded to BF16 on its own, the
// terms are added in FP32 starting from -0.0, and the sum is rounded once --
// `round_assignment` followed by `moe_combine`. Owning the tile for the whole
// loop keeps that order without atomics, which is what the CUDA version gets
// from a thread-block cluster.
struct DownParams {
  FusedDecodeArgs a;
  int64_t vecs;  // uint4s per expert of w2
};

template <WeightSource S>
__global__ void __launch_bounds__(kThreads) down_kernel(DownParams p) {
  __shared__ float part[kWarps][kRows];
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int64_t nb = blockIdx.x;
  const int topk = static_cast<int>(p.a.topk);
  const auto* const act = static_cast<const __hip_bfloat16*>(p.a.activation);

  for (int64_t i = threadIdx.x; i < kWarps * kRows; i += kThreads) (&part[0][0])[i] = 0.0f;
  __syncthreads();

  float comb = -0.0f;
  for (int j = 0; j < topk; ++j) {
    const int32_t e = p.a.ids[j];
    if (e < 0 || int64_t{e} >= p.a.w2.experts) {
      // Never read activation, weights or scales for a dropped assignment, but
      // clear the plane so the previous j's partials are not reused. Adding +0
      // also leaves an all-dropped tile at +0.0, like moe_combine.
      for (int64_t i = threadIdx.x; i < kWarps * kRows; i += kThreads) (&part[0][0])[i] = 0.0f;
    } else {
      const uint4* const packed = expert_packed(p.a.w2.packed, p.vecs, e);
      const uint4* const esc = expert_escapes(p.a.w2.escapes, p.a.w2.escape_offsets, e);
      const uint32_t base_x4 = exp_base_x4(p.a.w2.exp_base[e]);
      const __hip_bfloat16* const xrow = act + int64_t{j} * p.a.ldactivation;
      gemv_partials<S>(packed, esc, base_x4, p.a.w2.k, nb, xrow, part, lane, warp);
    }
    __syncthreads();
    if (threadIdx.x < kRows) {
      // round_assignment: scale in FP32, one BF16 rounding, add in FP32.
      comb += __bfloat162float(__float2bfloat16(gemv_reduce(part, threadIdx.x) * p.a.scales[j]));
    }
    __syncthreads();
  }

  if (threadIdx.x < kRows) {
    auto* const out = static_cast<__hip_bfloat16*>(p.a.out);
    out[nb * kRows + threadIdx.x] = __float2bfloat16(comb);
  }
}

// ---------------------------------------------------------------------------
// Host-side validation
// ---------------------------------------------------------------------------
bool aligned(const void* p, uintptr_t alignment) {
  return (reinterpret_cast<uintptr_t>(p) & (alignment - 1)) == 0;
}

bool fits_span(const void* p, int64_t rows, int64_t stride, int64_t width, int bytes) {
  const int64_t limit = INT64_MAX / bytes;
  if (rows < 1 || width < 1 || width > limit || stride < width ||
      rows - 1 > (limit - width) / stride)
    return false;
  const uint64_t span = static_cast<uint64_t>((rows - 1) * stride + width) * bytes;
  return span <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(p);
}

bool valid_weight(const FusedDecodeWeight& w) {
  if (!w.packed || !aligned(w.packed, 16)) return false;
  if (w.experts < 1 || w.experts > INT_MAX) return false;
  if (w.n < 16 || w.n % 16 || w.k < 64 || w.k % 64) return false;
  // The CUDA original caps n and k at 4096. A generous bound instead keeps the
  // span arithmetic below inside int64 and rejects absurd sizes before any
  // pointer math, without rejecting a legitimate large expert.
  if (w.n > (int64_t{1} << 20) || w.k > (int64_t{1} << 20)) return false;
  if (!w.bf16t) {
    if (!w.exp_base || !w.escape_offsets) return false;
    if (w.escapes && !aligned(w.escapes, 16)) return false;
    if (!aligned(w.escape_offsets, 8)) return false;
  }
  const int64_t vecs = (w.n / 16) * (w.k / 64) *
                       (w.bf16t ? nf12_hip::kSuperAtomVecsBf16t : nf12_hip::kSuperAtomVecs);
  return fits_span(w.packed, w.experts, vecs * 16, vecs * 16, 1);
}

// `activation` and `out` may be null here: ops.cpp preflights the capability
// before it allocates them, and the header documents that.
bool valid_args(const FusedDecodeArgs& a) {
  if (a.topk < 1 || a.topk > 8) return false;
  if (!valid_weight(a.w13) || !valid_weight(a.w2)) return false;
  if (a.w13.experts != a.w2.experts) return false;
  // w13 emits gate rows [0, I) then up rows [I, 2I), and w2 consumes I.
  if (a.w13.n != 2 * a.w2.k || (a.w13.n / 2) % kRows) return false;
  if (!a.x || !a.ids || !a.scales) return false;
  if (!aligned(a.x, 2) || !aligned(a.ids, 4) || !aligned(a.scales, 4)) return false;
  if (!aligned(a.activation, 16) || !aligned(a.out, 2)) return false;
  if (a.ldx < a.w13.k || a.ldids < a.topk || a.ldscales < a.topk) return false;
  if (a.ldactivation < a.w2.k || a.ldout < a.w2.n) return false;
  return fits_span(a.x, 1, a.ldx, a.w13.k, 2) &&
         fits_span(a.ids, 1, a.ldids, a.topk, 4) &&
         fits_span(a.scales, 1, a.ldscales, a.topk, 4) &&
         fits_span(a.activation, a.topk, a.ldactivation, a.w2.k, 2) &&
         fits_span(a.out, 1, a.ldout, a.w2.n, 2);
}

struct Plan {
  int64_t inter_blocks;
  int64_t out_blocks;
  int64_t vecs13;
  int64_t vecs2;
};

bool prepare(const FusedDecodeArgs& a, Plan& plan) {
  if (!valid_args(a)) return false;
  plan.inter_blocks = (a.w13.n / 2) / kRows;
  plan.out_blocks = a.w2.n / kRows;
  plan.vecs13 = (a.w13.n / kRows) * (a.w13.k / kCols) *
                (a.w13.bf16t ? nf12_hip::kSuperAtomVecsBf16t : nf12_hip::kSuperAtomVecs);
  plan.vecs2 = (a.w2.n / kRows) * (a.w2.k / kCols) *
               (a.w2.bf16t ? nf12_hip::kSuperAtomVecsBf16t : nf12_hip::kSuperAtomVecs);
  return plan.inter_blocks > 0 && plan.out_blocks > 0 && plan.inter_blocks <= 0x7fffffffLL &&
         plan.out_blocks <= 0x7fffffffLL;
}

}  // namespace

bool fused_decode_supported(int device, const FusedDecodeArgs& args) {
  // Pure host-side validation: no device query, allocation or synchronisation.
  // The kernels use only packed BF16 dot products, which exist on every arch
  // this library is built for, so there is nothing to probe -- if the module
  // loaded, the kernels are resident.
  (void)device;
  Plan plan{};
  return prepare(args, plan);
}

cudaError_t fused_decode(const FusedDecodeArgs& args, cudaStream_t stream) {
  if (args.activation == nullptr || args.out == nullptr) return hipErrorInvalidValue;
  Plan plan{};
  if (!prepare(args, plan)) return hipErrorNotSupported;

  GateParams gate{args, plan.inter_blocks, plan.vecs13};
  const dim3 gate_grid(static_cast<unsigned>(plan.inter_blocks),
                       static_cast<unsigned>(args.topk));
  if (args.w13.bf16t) {
    gate_kernel<WeightSource::kBf16t><<<gate_grid, kThreads, 0, stream>>>(gate);
  } else {
    gate_kernel<WeightSource::kNf12><<<gate_grid, kThreads, 0, stream>>>(gate);
  }
  cudaError_t status = hipGetLastError();
  if (status != hipSuccess) return status;

  DownParams down{args, plan.vecs2};
  const dim3 down_grid(static_cast<unsigned>(plan.out_blocks));
  if (args.w2.bf16t) {
    down_kernel<WeightSource::kBf16t><<<down_grid, kThreads, 0, stream>>>(down);
  } else {
    down_kernel<WeightSource::kNf12><<<down_grid, kThreads, 0, stream>>>(down);
  }
  return hipGetLastError();
}

}  // namespace nf12::torch_detail
