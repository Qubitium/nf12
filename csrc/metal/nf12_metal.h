// SPDX-License-Identifier: Apache-2.0
// Metal Shading Language helpers for the NF12 format (docs/SPEC.md).
//
// A simdgroup (32 lanes) plays the role of a CUDA warp: lane l = 4g + t owns
// group l of each atom (SPEC.md 4.1), i.e. the bf16 pairs
//   R0 = (row g,     cols 2t, 2t+1)    R2 = (row g,     cols 2t+8, 2t+9)
//   R1 = (row g + 8, cols 2t, 2t+1)    R3 = (row g + 8, cols 2t+8, 2t+9)
// of atom j (columns 16j..16j+15) of a 16 x 64 super-atom.
// Kernels are compiled with fast math disabled (SPEC.md section 6).
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace nf12 {

constant constexpr uint kSuperAtomRows = 16;
constant constexpr uint kSuperAtomCols = 64;
constant constexpr uint kSuperAtomBytes = 1536;
constant constexpr uint kSuperAtomBytesBf16t = 2048;
constant constexpr uint kPlaneBytes = 512;
constant constexpr uint kMaxExpBase = 120;

// Each kernel declares the largest threadgroup launch.h dispatches it with, so the
// compiler budgets registers for it and every GPU's pipeline accepts that size
// (launch.h also checks maxTotalThreadsPerThreadgroup before each dispatch).
#define NF12_THREADS(n) [[max_total_threads_per_threadgroup(n)]]

struct LinearParams {
  ulong ldx, ldy;
  uint m, n, k_blocks;
  uint kb_per_split;
  uint row_blocks;
  uint splits;
  uint has_bias;
  uint reserved;
};

struct Group {
  uint meta, lo0, lo1;
};

// ---------------------------------------------------------------------------
// BF16 <-> FP32. bf16 -> f32 is exact; f32 -> bf16 rounds to nearest even.
inline float bf16_lo(uint pair) { return as_type<float>(pair << 16); }
inline float bf16_hi(uint pair) { return as_type<float>(pair & 0xffff0000u); }
inline float bf16_to_float(ushort v) { return as_type<float>(uint(v) << 16); }

inline ushort float_to_bf16(float f) {
  const uint u = as_type<uint>(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x0040u);  // quiet NaN, keep sign/payload
  return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline uint pack_bf16x2(float lo, float hi) { return uint(float_to_bf16(lo)) | (uint(float_to_bf16(hi)) << 16); }

inline void store_linear(device ushort* y, device const ushort* bias, constant LinearParams& p,
                         uint m, uint n, float sum) {
  if (p.has_bias) sum += bf16_to_float(bias[n]);
  y[ulong(m) * p.ldy + n] = float_to_bf16(sum);
}

// ---------------------------------------------------------------------------
// Group decode (SPEC.md section 3). Byte j of a/b is hi(s(2j)) / hi(s(2j+1)).
inline uint exp_base_x4(uint exp_base) { return exp_base * 0x01010101u; }

inline void decode_bulk(Group g, uint base_x4, thread uint (&r)[4]) {
  const uint m = g.meta;
  const uchar4 a = as_type<uchar4>((m & 0x87878787u) + base_x4);
  const uchar4 b = as_type<uchar4>((((m << 1) & 0x80808080u) | ((m >> 3) & 0x07070707u)) + base_x4);
  const uchar4 l0 = as_type<uchar4>(g.lo0);
  const uchar4 l1 = as_type<uchar4>(g.lo1);
  r[0] = as_type<uint>(uchar4(l0.x, a.x, l0.y, b.x));
  r[1] = as_type<uint>(uchar4(l0.z, a.y, l0.w, b.y));
  r[2] = as_type<uint>(uchar4(l1.x, a.z, l1.y, b.z));
  r[3] = as_type<uint>(uchar4(l1.z, a.w, l1.w, b.w));
}

// Escapes need no simdgroup-uniform control flow on Metal: each lane reads its own entry.
inline void decode_group(Group g, uint base_x4, device const uint4* escapes, thread uint (&r)[4]) {
  if (g.meta == 0u) {
    const uint4 v = escapes[g.lo0];
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  } else {
    decode_bulk(g, base_x4, r);
  }
}

// ---------------------------------------------------------------------------
// Group encode. Returns false (escape) if a slot is out of the window or the
// meta word would be the M == 0 sentinel.
inline bool encode_group(const thread uint (&r)[4], uint exp_base, thread Group& out) {
  uint meta = 0;
  bool ok = true;
  for (uint j = 0; j < 4; ++j) {
    const uint va = r[j] & 0xffffu, vb = r[j] >> 16;
    const uint ha = (va >> 8) & 0x7fu, hb = (vb >> 8) & 0x7fu;  // exponent >> 1
    const uint oa = ha - exp_base, ob = hb - exp_base;              // wraps when below the window
    ok = ok && oa <= 7u && ob <= 7u;
    const uint byte = ((va >> 15) << 7) | ((vb >> 15) << 6) | ((ob & 7u) << 3) | (oa & 7u);
    meta |= byte << (8 * j);
  }
  out.meta = meta;
  out.lo0 = (r[0] & 0xffu) | ((r[0] >> 8) & 0xff00u) | ((r[1] & 0xffu) << 16) | ((r[1] & 0xff0000u) << 8);
  out.lo1 = (r[2] & 0xffu) | ((r[2] >> 8) & 0xff00u) | ((r[3] & 0xffu) << 16) | ((r[3] & 0xff0000u) << 8);
  return ok && meta != 0u;
}

// ---------------------------------------------------------------------------
// Super-atom addressing. sa = nb * k_blocks + kb (SPEC.md 4.3).
struct SuperAtom {
  uint nb, kb;
};
inline SuperAtom super_atom(ulong sa, uint k_blocks) { return SuperAtom{uint(sa / k_blocks), uint(sa % k_blocks)}; }

// Element offset of register R0 of atom 0 for this lane within a row-major matrix.
inline ulong lane_offset(SuperAtom s, ulong ld, uint lane) {
  return (ulong(s.nb) * kSuperAtomRows + (lane >> 2)) * ld + ulong(s.kb) * kSuperAtomCols + 2 * (lane & 3);
}

// Atom j's registers in a row-major bf16 matrix viewed as uint pairs (ld even).
inline void load_atom(device const ushort* w, ulong off, ulong ld, uint j, thread uint (&r)[4]) {
  device const ushort* p = w + off + 16 * j;
  r[0] = *reinterpret_cast<device const uint*>(p);
  r[1] = *reinterpret_cast<device const uint*>(p + 8 * ld);
  r[2] = *reinterpret_cast<device const uint*>(p + 8);
  r[3] = *reinterpret_cast<device const uint*>(p + 8 * ld + 8);
}

inline void store_atom(device ushort* w, ulong off, ulong ld, uint j, const thread uint (&r)[4]) {
  device ushort* p = w + off + 16 * j;
  *reinterpret_cast<device uint*>(p) = r[0];
  *reinterpret_cast<device uint*>(p + 8 * ld) = r[1];
  *reinterpret_cast<device uint*>(p + 8) = r[2];
  *reinterpret_cast<device uint*>(p + 8 * ld + 8) = r[3];
}

// Stage a decoded 16 x 64 tile in BF16 pairs, with a stride of Ld elements.
template <uint Ld>
inline void stage_bf16_tile(threadgroup uint* tile, uint lane, const thread uint (&r)[4][4]) {
  threadgroup uint* w0 = tile + (lane >> 2) * (Ld / 2) + (lane & 3u);
  threadgroup uint* w1 = w0 + 8 * (Ld / 2);
  #pragma unroll
  for (uint j = 0; j < 4; ++j) {
    w0[8 * j] = r[j][0];
    w0[8 * j + 4] = r[j][2];
    w1[8 * j] = r[j][1];
    w1[8 * j + 4] = r[j][3];
  }
}

// ---------------------------------------------------------------------------
// Weight sources. Both yield the same registers for the same weights, so a
// kernel templated on the source gives identical bits for NF12 and bf16t
// (SPEC.md section 6, item 2).
struct NF12Source {
  device const uint4* packed;   // super-atoms, 96 uint4 each
  device const uint4* escapes;  // may be unused when the weight has none
  uint base_x4;

  // All four atoms of super-atom sa for this lane.
  inline void load(ulong sa, uint lane, thread uint (&r)[4][4]) const {
    device const uint4* src = packed + sa * (kSuperAtomBytes / 16) + lane;
    const uint4 m = src[0];
    const uint4 l01 = src[kPlaneBytes / 16];
    const uint4 l23 = src[2 * kPlaneBytes / 16];
    decode_group(Group{m.x, l01.x, l01.y}, base_x4, escapes, r[0]);
    decode_group(Group{m.y, l01.z, l01.w}, base_x4, escapes, r[1]);
    decode_group(Group{m.z, l23.x, l23.y}, base_x4, escapes, r[2]);
    decode_group(Group{m.w, l23.z, l23.w}, base_x4, escapes, r[3]);
  }
};

struct Bf16tSource {
  device const uint4* packed;  // super-atoms, 128 uint4 each

  inline void load(ulong sa, uint lane, thread uint (&r)[4][4]) const {
    device const uint4* src = packed + sa * (kSuperAtomBytesBf16t / 16) + lane;
    for (uint j = 0; j < 4; ++j) {
      const uint4 v = src[j * (kPlaneBytes / 16)];
      r[j][0] = v.x;
      r[j][1] = v.y;
      r[j][2] = v.z;
      r[j][3] = v.w;
    }
  }
};

}  // namespace nf12
