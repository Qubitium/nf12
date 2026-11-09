// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from GPT-QModel (https://github.com/ModelCloud/GPTQModel) by Qubitium.
// ---------------------------------------------------------------------------
// NF12-AD (NyoomFloat12) format + decode for AMD RDNA4 (gfx1200/gfx1201/gfx1250).
//
// Credits: NyoomFloat12 algorithm by Liberman, Fang, Zhang, Dao and Fu
// (Together AI, 2026), https://www.together.ai/. NF12 format, reference codec
// and CUDA kernels by AlpinDale, https://github.com/dphnAI/nf12 (Apache-2.0).
//
// It decodes the identical NF12-AD tiled byte stream that the CUDA kernels
// produce, so a checkpoint is portable between NVIDIA and AMD hosts. The bit
// layout is specified in docs/SPEC.md sections 2-4.
//
// PORTING NOTES (NVIDIA -> AMD), all verified on gfx1201 hardware:
//
//   * __byte_perm(x, y, sel) -> __builtin_amdgcn_perm(x, y, sel) is NOT a
//     transliteration. NVIDIA numbers a 4-bit nibble per output byte over the
//     concatenation [x.b0..x.b3, y.b0..y.b3]. AMD uses ONE FULL BYTE per
//     output byte, where 0..3 selects src1 (the SECOND argument) and 4..7
//     selects src0 (the FIRST). The helper prmt() below converts NVIDIA
//     nibble selectors to the AMD byte form, so the decode expressions can be
//     copied verbatim from include/nf12/format.cuh.
//   * __any_sync/warp vote -> __builtin_amdgcn_any / wave32 ballot.
//   * __ldcs / __stcs -> plain vector loads on RDNA4; there is no portable
//     cache-streaming hint that matters for these sizes.
//   * uint4 -> uint4 is available via hip_runtime.h, but alignment of the
//     packed buffer is only 16 bytes, so all vector loads stay 16-byte.
//   * __nv_bfloat16 -> __hip_bfloat16 (hip_bf16.h).
//
// The interface reads exactly as `uint4 planes[3]` per lane as the CUDA
// decoder does: lane l of a super-atom holds 16 bytes of each of the three
// 512-byte planes.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

#include "capability.h"
#include "nf12/format.cuh"

namespace nf12_hip {

// SPEC.md section 4 geometry.
inline constexpr int kAtomRows = 16;
inline constexpr int kAtomCols = 16;
inline constexpr int kAtomsPerSuperAtom = 4;
inline constexpr int kSuperAtomRows = 16;
inline constexpr int kSuperAtomCols = 64;
inline constexpr int kSuperAtomBytes = 1536;
inline constexpr int kPlaneBytes = 512;
inline constexpr int kMaxExpBase = 120;

// The CUDA header owns these constants. Assert rather than trust the copy, so a
// format change upstream breaks the HIP build instead of silently diverging.
static_assert(kAtomRows == nf12::kAtomRows);
static_assert(kAtomCols == nf12::kAtomCols);
static_assert(kAtomsPerSuperAtom == nf12::kAtomsPerSuperAtom);
static_assert(kSuperAtomRows == nf12::kSuperAtomRows);
static_assert(kSuperAtomCols == nf12::kSuperAtomCols);
static_assert(kSuperAtomBytes == nf12::kSuperAtomBytes);
static_assert(kPlaneBytes == nf12::kPlaneBytes);
static_assert(kMaxExpBase == nf12::kMaxExpBase);

// Lanes that own one super-atom, and lane-groups per CTA. These are FORMAT and
// scheduling constants, not hardware properties: a wave32 device maps one
// group to a wave, a wave64 device maps two. The CTA is 8 groups = 256 threads
// on both, so the RDNA4 occupancy tuning carries over unchanged and no wave
// barrier is ever split by an early return. See capability.h.
inline constexpr int kWarpSize = kAtomLanes;
inline constexpr int kWarpsPerBlock = kGroupsPerBlock;
inline constexpr int kThreadsPerBlock = kGroupsPerBlock * kAtomLanes;
inline constexpr int kSuperAtomVecs = kSuperAtomBytes / 16;  // 96 uint4
inline constexpr int kPlaneVecs = kPlaneBytes / 16;          // 32 uint4
inline constexpr int kSuperAtomVecsBf16t = 2048 / 16;        // 128 uint4

// Hardware wave width of the device this wave runs on. Only used where the
// width matters (vote masks, cross-half-wave reductions); the codec kernels
// index by the 32-lane atom group above and are width independent.
__device__ __forceinline__ int hw_wave_size() { return static_cast<int>(warpSize); }

// 32-lane ballot, valid on wave32 and wave64 targets. On wave64 the compiler
// exposes only `ballot_w64`; the low or high half of the result is the 32-lane
// group's mask, selected from the lane id. The value is exactly what
// `ballot_w32` would have returned for that group.
__device__ __forceinline__ uint32_t ballot_atom_group(bool predicate) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ballot_w32)) {
    return __builtin_amdgcn_ballot_w32(predicate);
  }
  const uint64_t full = __builtin_amdgcn_ballot_w64(predicate);
  return static_cast<uint32_t>(full >> (32 * ((threadIdx.x & 63) / 32)));
}

// CUDA's MoE schedule clamps offsets and takes their running maximum.
__device__ __forceinline__ int64_t moe_next_offset(int64_t offset, int64_t previous,
                                                   int64_t max_assignments) {
  return max(previous, min(offset, max_assignments));
}

// Interleave warps along K so their row-major stores form contiguous segments.
template <int kPerWarp>
__device__ __forceinline__ int64_t warp_super_atom(int i) {
  return (int64_t(blockIdx.x) * kPerWarp + i) * kWarpsPerBlock + (threadIdx.x / kWarpSize);
}

template <int kPerWarp>
inline int64_t grid_blocks(int64_t num_sa) {
  const int64_t per_block = int64_t(kPerWarp) * kWarpsPerBlock;
  return (num_sa + per_block - 1) / per_block;
}

struct Group {
  uint32_t meta;
  uint32_t lo0;
  uint32_t lo1;
};

// Host + device: the launcher needs it on the host side too.
__host__ __device__ __forceinline__ uint32_t exp_base_x4(uint32_t exp_base) {
  return exp_base * 0x01010101u;
}

// ---------------------------------------------------------------------------
// NVIDIA __byte_perm -> AMD v_perm_b32
// ---------------------------------------------------------------------------
// `sel` uses NVIDIA nibble semantics: nibble i (bits 4i..4i+3) selects the
// source byte for output byte i, where 0..3 is x and 4..7 is y.
//
// AMD v_perm_b32 wants byte i of the selector (bits 8i..8i+7) to name a source
// byte, 0..3 -> src1, 4..7 -> src0. With NVIDIA x -> src0 and y -> src1:
//   nibble 0..3 (x.byte n)  -> AMD byte 4 + n
//   nibble 4..7 (y.byte n)  -> AMD byte n - 4
// This conversion is done at COMPILE TIME so every call site below lowers to a
// single v_perm_b32 with a literal selector.
constexpr uint32_t nv_sel_to_amd(uint32_t sel) {
  uint32_t out = 0;
  for (int i = 0; i < 4; ++i) {
    const uint32_t nib = (sel >> (4 * i)) & 0xFu;
    const uint32_t byte = (nib < 4u) ? (4u + nib) : (nib - 4u);
    out |= byte << (8 * i);
  }
  return out;
}

template <uint32_t kNvSel>
__device__ __forceinline__ uint32_t prmt(uint32_t x, uint32_t y) {
  return __builtin_amdgcn_perm(x, y, nv_sel_to_amd(kNvSel));
}

// ---------------------------------------------------------------------------
// Decode (SPEC.md 3 "Reference decode")
// ---------------------------------------------------------------------------
// Requires meta != 0 and base_x4 == exp_base * 0x01010101.
__device__ __forceinline__ void decode_bulk(const Group& g, uint32_t base_x4, uint32_t (&r)[4]) {
  const uint32_t m = g.meta;
  const uint32_t a = (m & 0x87878787u) + base_x4;
  const uint32_t b = (((m << 1) & 0x80808080u) | ((m >> 3) & 0x07070707u)) + base_x4;
  const uint32_t p0 = prmt<0x5140u>(a, b);  // hi(s0), hi(s1), hi(s2), hi(s3)
  const uint32_t p1 = prmt<0x7362u>(a, b);  // hi(s4) .. hi(s7)
  r[0] = prmt<0x5140u>(g.lo0, p0);
  r[1] = prmt<0x7362u>(g.lo0, p0);
  r[2] = prmt<0x5140u>(g.lo1, p1);
  r[3] = prmt<0x7362u>(g.lo1, p1);
}

// Every lane of the wave must reach this call; the escape branch is
// wave-uniform on the fast path.
__device__ __forceinline__ void decode_group(const Group& g, uint32_t base_x4,
                                             const uint4* __restrict__ escapes,
                                             uint32_t (&r)[4]) {
  decode_bulk(g, base_x4, r);
  if (g.meta == 0u) {
    const uint4 v = escapes[g.lo0];
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  }
}

// ---------------------------------------------------------------------------
// Tiling (SPEC.md 4.1 / 4.2)
// ---------------------------------------------------------------------------
// x[h][p]: 16-byte chunk (t + 4p) of row (g + 8h) of the super-atom.
using RowTile = uint4[2][2];
// r[j][i]: register R_i of the group of atom j held by this lane.
using AtomRegs = uint32_t[4][4];

struct SuperAtom {
  int64_t nb, kb;
};

__device__ __forceinline__ SuperAtom super_atom(int64_t sa, int64_t k_blocks) {
  return SuperAtom{sa / k_blocks, sa % k_blocks};
}

__device__ __forceinline__ int64_t row_tile_offset(SuperAtom s, int64_t ld, int lane) {
  return (s.nb * kSuperAtomRows + (lane >> 2)) * ld + s.kb * kSuperAtomCols + 8 * (lane & 3);
}

__device__ __forceinline__ void load_row_tile(const __hip_bfloat16* __restrict__ w,
                                              int64_t off, int64_t ld, RowTile& x) {
  const uint4* p = reinterpret_cast<const uint4*>(w + off);
  const int64_t row8 = ld;  // 8 rows of uint4
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int q = 0; q < 2; ++q) x[h][q] = p[h * row8 + 4 * q];
  }
}

__device__ __forceinline__ void store_row_tile(__hip_bfloat16* __restrict__ out,
                                               int64_t off, int64_t ld, const RowTile& x) {
  uint4* p = reinterpret_cast<uint4*>(out + off);
  const int64_t row8 = ld;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int q = 0; q < 2; ++q) p[h * row8 + 4 * q] = x[h][q];
  }
}

// Swap lane bits 0-1 with the vector-component index using four shuffles.
// Self-inverse. Wave32 on RDNA4, so xor masks 1 and 2 stay inside the wave.
__device__ __forceinline__ void quad_transpose(uint4& v, int lane) {
  const bool b0 = lane & 1;
  uint32_t s0 = b0 ? v.x : v.y;
  uint32_t s1 = b0 ? v.z : v.w;
  s0 = __shfl_xor(s0, 1);
  s1 = __shfl_xor(s1, 1);
  if (b0) {
    v.x = s0;
    v.z = s1;
  } else {
    v.y = s0;
    v.w = s1;
  }
  const bool b1 = lane & 2;
  uint32_t u0 = b1 ? v.x : v.z;
  uint32_t u1 = b1 ? v.y : v.w;
  u0 = __shfl_xor(u0, 2);
  u1 = __shfl_xor(u1, 2);
  if (b1) {
    v.x = u0;
    v.y = u1;
  } else {
    v.z = u0;
    v.w = u1;
  }
}

// After quad_transpose, component c of x[h][p] is column pair (4*(4p+c)+t) of
// row g+8h. Atom j needs pairs 8j+t and 8j+4+t, i.e. p = j>>1 and
// c = 2*(j&1) + {0,1}.
__device__ __forceinline__ void tile_to_atoms(const RowTile& x, AtomRegs& r) {
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    r[2 * p][0] = x[0][p].x;
    r[2 * p][2] = x[0][p].y;
    r[2 * p][1] = x[1][p].x;
    r[2 * p][3] = x[1][p].y;
    r[2 * p + 1][0] = x[0][p].z;
    r[2 * p + 1][2] = x[0][p].w;
    r[2 * p + 1][1] = x[1][p].z;
    r[2 * p + 1][3] = x[1][p].w;
  }
}

__device__ __forceinline__ void atoms_to_tile(const AtomRegs& r, RowTile& x) {
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    x[0][p] = make_uint4(r[2 * p][0], r[2 * p][2], r[2 * p + 1][0], r[2 * p + 1][2]);
    x[1][p] = make_uint4(r[2 * p][1], r[2 * p][3], r[2 * p + 1][1], r[2 * p + 1][3]);
  }
}

// ---------------------------------------------------------------------------
// Weight source
// ---------------------------------------------------------------------------
// The GEMM kernels are written once and parameterised on where the BF16
// super-atom comes from, mirroring `WeightTraits` in csrc/linear/common.cuh:
//
//   * kNf12  - 1536-byte packed super-atom (3 planes); decode reconstructs the
//              atom registers.
//   * kBf16t - 2048-byte uncompressed twin (SPEC.md 4.4, `pack_bf16t`); one
//              plane per atom already IS the atom register, so the "decode" is
//              a load. Used for exactness comparisons.
//
// Both yield the same AtomRegs for the same weights, so the two sources produce
// matching outputs through one code path instead of two kernels.
enum class WeightSource { kNf12, kBf16t };

template <WeightSource S>
struct SourceTraits;

template <>
struct SourceTraits<WeightSource::kNf12> {
  static constexpr int kLoads = 3;
  static constexpr int kVecs = kSuperAtomVecs;
};

template <>
struct SourceTraits<WeightSource::kBf16t> {
  static constexpr int kLoads = 4;
  static constexpr int kVecs = kSuperAtomVecsBf16t;
};

// One lane's 16-byte chunks of super-atom `sa`.
template <WeightSource S>
__device__ __forceinline__ void load_source_planes(const uint4* __restrict__ packed, int64_t sa,
                                                   int lane,
                                                   uint4 (&p)[SourceTraits<S>::kLoads]) {
  const uint4* src = packed + sa * SourceTraits<S>::kVecs + lane;
#pragma unroll
  for (int i = 0; i < SourceTraits<S>::kLoads; ++i) p[i] = src[i * kPlaneVecs];
}

// Decode (kNf12) or reinterpret (kBf16t) the lane's planes as atom registers.
// kNf12 requires the whole wave to reach this call (the escape test is a vote).
template <WeightSource S>
__device__ __forceinline__ void source_planes_to_atoms(
    const uint4 (&p)[SourceTraits<S>::kLoads], uint32_t base_x4,
    const uint4* __restrict__ escapes, AtomRegs& r) {
  if constexpr (S == WeightSource::kNf12) {
    decode_group(Group{p[0].x, p[1].x, p[1].y}, base_x4, escapes, r[0]);
    decode_group(Group{p[0].y, p[1].z, p[1].w}, base_x4, escapes, r[1]);
    decode_group(Group{p[0].z, p[2].x, p[2].y}, base_x4, escapes, r[2]);
    decode_group(Group{p[0].w, p[2].z, p[2].w}, base_x4, escapes, r[3]);
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      r[j][0] = p[j].x;
      r[j][1] = p[j].y;
      r[j][2] = p[j].z;
      r[j][3] = p[j].w;
    }
  }
}

// One token row dotted against one decoded 16x64 super-atom.
//
// `regs` is this lane's AtomRegs for the super-atom and `xrow` points at the
// 64-column window of the token row as packed bf16 pairs. `t` is the lane's
// quad position (lane & 3). The two sums this lane contributes are returned in
// `lo` (row g = lane >> 2) and `hi` (row g + 8); the caller collapses the four
// t lanes of the quad with a fixed xor order, exactly as `linear_kernel` does.
//
// regs[j][i] = R_i of atom j (SPEC 4.1):
//   R0 -> row g,   cols 2t, 2t+1      R1 -> row g+8, cols 2t, 2t+1
//   R2 -> row g,   cols 2t+8, 2t+9    R3 -> row g+8, cols 2t+8, 2t+9
//
// Packed BF16 dot: one v_dot2_f32_bf16 per two columns where the ISA has it
// (RDNA4 `dot12-insts`), otherwise two scalar FMAs into the same accumulator.
// The scalar form is bit-identical for a single product pair (fma of the two
// products is not contracted here: no fast-math in this build) and keeps the
// exact reduction order, so the twin-equality contract holds on CDNA too.
//
// Selection is a compile-time query of the target ISA (`is_invocable`), not an
// arch string and not a `__gfx*__` guard, so a host/device disagreement is
// impossible.
// `kNegate` is a template parameter because the RDNA4 intrinsic requires a
// constant integer there; every call site in this port passes false.
template <bool kNegate = false>
__device__ __forceinline__ float dot2_bf16(__hip_bfloat162 a, __hip_bfloat162 b, float c) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fdot2_f32_bf16)) {
    return __builtin_amdgcn_fdot2_f32_bf16(a, b, c, kNegate);
  }
  const float p = __bfloat162float(a.x) * __bfloat162float(b.x) +
                  __bfloat162float(a.y) * __bfloat162float(b.y);
  return kNegate ? c - p : c + p;
}

__device__ __forceinline__ void dot_super_atom(const AtomRegs& regs,
                                               const __hip_bfloat162* __restrict__ xrow, int t,
                                               float& lo, float& hi) {
  const __hip_bfloat162* r = reinterpret_cast<const __hip_bfloat162*>(&regs[0][0]);
  const __hip_bfloat162* xp = xrow + t;
  const __hip_bfloat162 c0 = xp[0];
  const __hip_bfloat162 c8 = xp[4];
  const __hip_bfloat162 c16 = xp[8];
  const __hip_bfloat162 c24 = xp[12];
  const __hip_bfloat162 c32 = xp[16];
  const __hip_bfloat162 c40 = xp[20];
  const __hip_bfloat162 c48 = xp[24];
  const __hip_bfloat162 c56 = xp[28];

  // r[4*j + i] is R_i of atom j: row g uses i = 0 and 2, row g+8 uses 1 and 3.
  lo = dot2_bf16(c0, r[0], lo);
  lo = dot2_bf16(c8, r[2], lo);
  hi = dot2_bf16(c0, r[1], hi);
  hi = dot2_bf16(c8, r[3], hi);
  lo = dot2_bf16(c16, r[4], lo);
  lo = dot2_bf16(c24, r[6], lo);
  hi = dot2_bf16(c16, r[5], hi);
  hi = dot2_bf16(c24, r[7], hi);
  lo = dot2_bf16(c32, r[8], lo);
  lo = dot2_bf16(c40, r[10], lo);
  hi = dot2_bf16(c32, r[9], hi);
  hi = dot2_bf16(c40, r[11], hi);
  lo = dot2_bf16(c48, r[12], lo);
  lo = dot2_bf16(c56, r[14], lo);
  hi = dot2_bf16(c48, r[13], hi);
  hi = dot2_bf16(c56, r[15], hi);
}


// ---------------------------------------------------------------------------
// Encode (SPEC.md 2, "Reference encode")
// ---------------------------------------------------------------------------
// These mirror codec/tile.cuh's encoder half. The encoder walks the same
// super-atoms as the decoder but in the opposite direction: it loads row-major
// BF16, quad-transposes into the group registers, then packs the meta/low
// words that the decoder unpacks.

// Lane (g,t) = (l>>2, l&3) owns 16-byte chunks t and t+4 in rows g and g+8.
__device__ __forceinline__ void load_atoms(const __hip_bfloat16* __restrict__ w, int64_t ldw,
                                           SuperAtom s, int lane, AtomRegs& r) {
  RowTile x;
  load_row_tile(w, row_tile_offset(s, ldw, lane), ldw, x);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int p = 0; p < 2; ++p) quad_transpose(x[h][p], lane);
  }
  tile_to_atoms(x, r);
}

// High bytes of the group's slots, split by pair position: byte j of `a` is
// hi(s(2j)), byte j of `b` is hi(s(2j+1)), matching the meta byte order.
struct GroupHigh {
  uint32_t a, b;
};

__device__ __forceinline__ GroupHigh group_high(const uint32_t (&r)[4]) {
  const uint32_t p01 = prmt<0x7531u>(r[0], r[1]);
  const uint32_t p23 = prmt<0x7531u>(r[2], r[3]);
  return GroupHigh{prmt<0x6420u>(p01, p23), prmt<0x7531u>(p01, p23)};
}

// Setting each byte's high bit prevents borrowing when subtracting B <= 120.
// In-window bytes become [0x80, 0x87], so bits 3-6 detect out-of-window values.
// A zero meta word requires an escape even when every value is in-window.
__device__ __forceinline__ bool encode_group_fast(const uint32_t (&r)[4], uint32_t base_x4,
                                                  Group& out) {
  const GroupHigh hi = group_high(r);
  const uint32_t da = ((hi.a & 0x7f7f7f7fu) | 0x80808080u) - base_x4;
  const uint32_t db = ((hi.b & 0x7f7f7f7fu) | 0x80808080u) - base_x4;
  out.meta = (hi.a & 0x80808080u) | ((hi.b & 0x80808080u) >> 1) | ((db & 0x07070707u) << 3) |
             (da & 0x07070707u);
  out.lo0 = prmt<0x6420u>(r[0], r[1]);
  out.lo1 = prmt<0x6420u>(r[2], r[3]);
  return ((da | db) & 0x78787878u) == 0u && out.meta != 0u;
}

// With h = e>>1, all values fit for B in [hmax-7, hmin], clamped to [0,120].
// A sentinel collision requires B = hmin = hmax and all sign bits clear.
struct GroupStats {
  uint32_t hmin, hmax;
  bool all_positive;
};

// RDNA4 has no __vminu4/__vmaxu4 in the HIP headers, so the byte-wise
// min/max reduction is spelled out. The compiler turns each pass into
// v_min3_u32 / v_max3_u32 over packed bytes.
__device__ __forceinline__ uint32_t vminu4(uint32_t a, uint32_t b) {
  uint32_t r = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t x = (a >> (8 * i)) & 0xffu;
    const uint32_t y = (b >> (8 * i)) & 0xffu;
    r |= (x < y ? x : y) << (8 * i);
  }
  return r;
}

__device__ __forceinline__ uint32_t vmaxu4(uint32_t a, uint32_t b) {
  uint32_t r = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t x = (a >> (8 * i)) & 0xffu;
    const uint32_t y = (b >> (8 * i)) & 0xffu;
    r |= (x > y ? x : y) << (8 * i);
  }
  return r;
}

__device__ __forceinline__ GroupStats group_stats(const uint32_t (&r)[4]) {
  const GroupHigh hi = group_high(r);
  const uint32_t ha = hi.a & 0x7f7f7f7fu;
  const uint32_t hb = hi.b & 0x7f7f7f7fu;
  uint32_t mn = vminu4(ha, hb);
  uint32_t mx = vmaxu4(ha, hb);
  mn = vminu4(mn, mn >> 16);
  mx = vmaxu4(mx, mx >> 16);
  mn = vminu4(mn, mn >> 8);
  mx = vmaxu4(mx, mx >> 8);
  return GroupStats{mn & 0xffu, mx & 0xffu, ((hi.a | hi.b) & 0x80808080u) == 0u};
}

// ---------------------------------------------------------------------------
// WMMA (gfx1200/gfx1201 only; gfx1250 has no wmma-128b-insts)
// ---------------------------------------------------------------------------
// `v_wmma_f32_16x16x16_bf16` with wave32: each lane holds 8 A, 8 B and 8 FP32
// accumulators. The fragment map below was determined empirically on gfx1201
// with one-hot probes and then asserted against a CPU reference (see
// gptqmodel_ext/nf12_hip/README.md, "WMMA fragment layout").
//
//   A reg (l, i) = A[ l % 16 ][ kb + i + 8 * (l / 16) ]
//   B reg (l, i) = B[ i + 8 * (l / 16) ][ colbase + l % 16 ]
//   D reg (l, i) = D[ i + 8 * (l / 16) ][ colbase + l % 16 ]
//
// so a 16x16 A tile and a 16x16 B tile both load with `rowfrag`, and the 16x16
// D tile scatters with the same (row,col) map. Every operand is BF16.
// NOTE: `__gfx1201__` is defined ONLY during the device compilation pass. Any
// guard that changes a *declaration* between the host and device passes makes
// the two emit different mangled names, and the launch then fails with
// "Cannot find Symbol". So the fragment types and helpers are declared
// unconditionally (they are plain vector types), and `NF12_AD_HAS_WMMA` gates
// only executable bodies.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define NF12_AD_HAS_WMMA 1
#else
#define NF12_AD_HAS_WMMA 0
#endif

using WmmaA = short __attribute__((ext_vector_type(8)));
using WmmaD = float __attribute__((ext_vector_type(8)));

// Element (row = lane % 16, col = kb + i + 8 * (lane / 16)) of `base`.
__device__ __forceinline__ WmmaA wmma_row_frag(const __hip_bfloat16* __restrict__ base,
                                               int64_t ld, int lane) {
  const int64_t off = int64_t(lane & 15) * ld + 8 * (lane >> 4);
  return *reinterpret_cast<const WmmaA*>(base + off);
}


}  // namespace nf12_hip
