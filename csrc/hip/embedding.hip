// SPDX-License-Identifier: Apache-2.0
//
// NF12 -> BF16 embedding lookup (SPEC.md 10) for RDNA4.
// Port of csrc/codec/embedding.cu: decode only the requested rows, never the
// whole table. Row r occupies lanes 4*(r%8)+t of each super-atom; the group
// also holds the paired row r^8, so one decode serves both halves.
//
// Adapted from GPT-QModel (https://github.com/ModelCloud/GPTQModel) by Qubitium.

#include "format_hip.cuh"

namespace nf12_hip {
namespace {

struct EmbeddingParams {
  const uint4* packed;
  const uint4* escapes;
  uint32_t num_escapes;
  uint32_t base_x4;
  int64_t n;
  int64_t k_blocks;
  int64_t num_ids;
  __hip_bfloat16* out;
  int64_t ldo;
  uint32_t ids_per_block;
  uint32_t kb_span;
  uint32_t kb_chunks;
};

// upper selects slots s2/s3/s6/s7, otherwise s0/s1/s4/s5.
__device__ __forceinline__ uint2 decode_row_half(const Group& g, uint32_t base_x4,
                                                 const uint4* __restrict__ escapes,
                                                 uint32_t num_escapes, bool upper) {
  uint32_t r[4];
  decode_bulk(g, base_x4, r);
  if (g.meta == 0u && g.lo0 < num_escapes) {
    const uint4 v = escapes[g.lo0];
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  }
  return upper ? make_uint2(r[1], r[3]) : make_uint2(r[0], r[2]);
}

template <int Quads, typename Id>
__global__ void __launch_bounds__(4 * Quads) embedding_kernel(EmbeddingParams p,
                                                              const Id* __restrict__ ids) {
  const int lane = threadIdx.x % kWarpSize;
  const uint32_t quad = threadIdx.x >> 2;
  const uint32_t id_block = blockIdx.x / p.kb_chunks;
  const uint32_t chunk = blockIdx.x - id_block * p.kb_chunks;
  const uint32_t local = quad / p.kb_span;
  const int64_t kb = int64_t(chunk) * Quads + (quad - local * p.kb_span);
  const int64_t i = int64_t(id_block) * p.ids_per_block + local;
  const bool active = local < p.ids_per_block && kb < p.k_blocks && i < p.num_ids;

  uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
  uint4 v1 = v0;
  if (active) {
    const int64_t row = static_cast<int64_t>(ids[i]);
    if (row >= 0 && row < p.n) {
      const int g = static_cast<int>(row & 7);
      const bool upper = (row & 8) != 0;
      const uint4* src = p.packed + ((row >> 4) * p.k_blocks + kb) * kSuperAtomVecs +
                         4 * g + (lane & 3);
      const uint4 m = src[0];
      const uint4 l01 = src[kPlaneVecs];
      const uint4 l23 = src[2 * kPlaneVecs];
      const uint2 a0 = decode_row_half(Group{m.x, l01.x, l01.y}, p.base_x4, p.escapes,
                                       p.num_escapes, upper);
      const uint2 a1 = decode_row_half(Group{m.y, l01.z, l01.w}, p.base_x4, p.escapes,
                                       p.num_escapes, upper);
      const uint2 a2 = decode_row_half(Group{m.z, l23.x, l23.y}, p.base_x4, p.escapes,
                                       p.num_escapes, upper);
      const uint2 a3 = decode_row_half(Group{m.w, l23.z, l23.w}, p.base_x4, p.escapes,
                                       p.num_escapes, upper);
      v0 = make_uint4(a0.x, a0.y, a1.x, a1.y);
      v1 = make_uint4(a2.x, a2.y, a3.x, a3.y);
    }
  }
  // All lanes, including inactive quads, must participate in the shuffles.
  quad_transpose(v0, lane);
  quad_transpose(v1, lane);
  if (active) {
    uint4* dst = reinterpret_cast<uint4*>(p.out + i * p.ldo + kb * kSuperAtomCols) + (lane & 3);
    dst[0] = v0;
    dst[4] = v1;
  }
}

template <int Quads, typename Id>
hipError_t launch_embedding(EmbeddingParams p, const Id* ids, hipStream_t stream) {
  const int64_t kb_chunks = (p.k_blocks + Quads - 1) / Quads;
  p.ids_per_block = p.k_blocks < Quads ? static_cast<uint32_t>(Quads / p.k_blocks) : 1u;
  p.kb_span = static_cast<uint32_t>(p.k_blocks < Quads ? p.k_blocks : Quads);
  const int64_t id_blocks = (p.num_ids + p.ids_per_block - 1) / p.ids_per_block;
  if (id_blocks <= 0 || kb_chunks > 0x7fffffff / id_blocks) return hipErrorInvalidValue;
  p.kb_chunks = static_cast<uint32_t>(kb_chunks);
  const dim3 grid(static_cast<unsigned>(id_blocks * kb_chunks));
  embedding_kernel<Quads><<<grid, 4 * Quads, 0, stream>>>(p, ids);
  return hipGetLastError();
}

}  // namespace

hipError_t embedding(const void* packed, const uint4* escapes, uint32_t num_escapes,
                     uint32_t exp_base, int64_t n, int64_t k, const int32_t* ids32,
                     const int64_t* ids64, int64_t num_ids, void* out, int64_t ldo,
                     hipStream_t stream) {
  if (exp_base > uint32_t(kMaxExpBase) || (ids32 == nullptr) == (ids64 == nullptr)) {
    return hipErrorInvalidValue;
  }
  if (num_ids <= 0) return hipSuccess;
  EmbeddingParams p{};
  p.packed = static_cast<const uint4*>(packed);
  p.escapes = escapes;
  p.num_escapes = escapes != nullptr ? num_escapes : 0;
  p.base_x4 = exp_base_x4(exp_base);
  p.n = n;
  p.k_blocks = k / kSuperAtomCols;
  p.num_ids = num_ids;
  p.out = static_cast<__hip_bfloat16*>(out);
  p.ldo = ldo;
  return ids32 != nullptr ? launch_embedding<8>(p, ids32, stream)
                          : launch_embedding<8>(p, ids64, stream);
}

}  // namespace nf12_hip
