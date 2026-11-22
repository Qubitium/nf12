// SPDX-License-Identifier: Apache-2.0
// Mixture-of-experts kernels (SPEC.md section 9). Per-expert counts stay on the
// device: the host sizes grids from max_assignments (A) and kernels read the
// schedule written by nf12_moe_schedule.
#include "linear_core.h"

using namespace nf12;

constant constexpr uint kAlignChunk = 256;  // assignments per counting chunk
constant constexpr uint kMaxExperts = 1024;

// ---------------------------------------------------------------------------
// moe_align (SPEC.md 9.2): stable sort of assignment ids by expert, dropping ids
// outside [0, E). Pass 1 counts each chunk's experts, pass 2 turns the counts
// into each chunk's first sorted position per expert (and expert_offsets),
// pass 3 places every assignment at its chunk base plus its rank among earlier
// same-expert assignments in the chunk. Deterministic: no order depends on atomics.
NF12_THREADS(kAlignChunk) kernel void nf12_moe_align_count(device const int* ids [[buffer(0)]],
                                 constant uint& num_assignments [[buffer(1)]],
                                 constant uint& num_experts [[buffer(2)]],
                                 device uint* chunk_counts [[buffer(3)]],  // [chunks, E]
                                 uint chunk [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint counts[kMaxExperts];
  for (uint e = tid; e < num_experts; e += kAlignChunk) atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint a = chunk * kAlignChunk + tid;
  if (a < num_assignments) {
    const int e = ids[a];
    if (e >= 0 && uint(e) < num_experts) atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < num_experts; e += kAlignChunk)
    chunk_counts[ulong(chunk) * num_experts + e] = atomic_load_explicit(&counts[e], memory_order_relaxed);
}

// One threadgroup of 1024 threads.
NF12_THREADS(1024) kernel void nf12_moe_align_offsets(constant uint& num_chunks [[buffer(0)]],
                                   constant uint& num_experts [[buffer(1)]],
                                   device uint* chunk_counts [[buffer(2)]],  // in: counts, out: bases
                                   device long* expert_offsets [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint totals[kMaxExperts];
  threadgroup uint sg_sums[32];
  // Per expert: exclusive prefix over chunks (relative to the expert's start).
  for (uint e = tid; e < num_experts; e += 1024) {
    uint run = 0;
    for (uint c = 0; c < num_chunks; ++c) {
      device uint* v = chunk_counts + ulong(c) * num_experts + e;
      const uint count = *v;
      *v = run;
      run += count;
    }
    totals[e] = run;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Exclusive scan of the totals over experts (E <= 1024: one value per thread).
  const uint v = tid < num_experts ? totals[tid] : 0u;
  const uint in_sg = simd_prefix_exclusive_sum(v);
  if (lane == 31) sg_sums[sg] = in_sg + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) sg_sums[lane] = simd_prefix_exclusive_sum(sg_sums[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint start = sg_sums[sg] + in_sg;
  if (tid < num_experts) {
    expert_offsets[tid] = long(start);
    if (tid == num_experts - 1) expert_offsets[num_experts] = long(start + v);
    totals[tid] = start;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < num_experts; e += 1024) {
    const uint base = totals[e];
    for (uint c = 0; c < num_chunks; ++c) chunk_counts[ulong(c) * num_experts + e] += base;
  }
}

NF12_THREADS(kAlignChunk) kernel void nf12_moe_align_scatter(device const int* ids [[buffer(0)]],
                                   constant uint& num_assignments [[buffer(1)]],
                                   constant uint& num_experts [[buffer(2)]],
                                   device const uint* chunk_bases [[buffer(3)]],
                                   device int* sorted [[buffer(4)]],
                                   uint chunk [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup int local[kAlignChunk];
  const uint a = chunk * kAlignChunk + tid;
  const int e = a < num_assignments ? ids[a] : -1;
  local[tid] = e;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e < 0 || uint(e) >= num_experts) return;
  uint rank = 0;
  for (uint i = 0; i < tid; ++i) rank += local[i] == e ? 1u : 0u;
  sorted[chunk_bases[ulong(chunk) * num_experts + uint(e)] + rank] = int(a);
}

// ---------------------------------------------------------------------------
// Tile schedule. Offsets are clamped to [0, A] and made monotone (running max),
// so malformed offsets cannot duplicate or overflow positions: expert e owns
// sorted positions [start[e], start[e + 1]). Its positions form consecutive
// tiles of at most `bm` rows, numbered expert-major. Layout: int count, three
// pad ints, then int4 (expert, row0, rows, 0) tiles. One threadgroup of 1024.
NF12_THREADS(1024) kernel void nf12_moe_schedule(device const long* expert_offsets [[buffer(0)]],
                              constant uint& num_experts [[buffer(1)]],
                              constant uint& max_assignments [[buffer(2)]],
                              constant uint& bm [[buffer(3)]],
                              device int4* schedule [[buffer(4)]],
                              uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint start[kMaxExperts + 1];
  threadgroup uint first_tile[kMaxExperts + 1];
  if (tid == 0) {
    uint run = 0, tiles = 0;
    for (uint e = 0; e <= num_experts; ++e) {
      const long v = expert_offsets[e];
      run = max(run, uint(clamp(v, 0l, long(max_assignments))));
      start[e] = run;
    }
    for (uint e = 0; e < num_experts; ++e) {
      first_tile[e] = tiles;
      const uint count = start[e + 1] - start[e];
      tiles += count / bm + (count % bm != 0u ? 1u : 0u);
    }
    first_tile[num_experts] = tiles;
    schedule[0] = int4(int(tiles), 0, 0, 0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < num_experts; e += 1024) {
    const uint s = start[e], end = start[e + 1];
    uint t = first_tile[e];
    for (uint row0 = s; row0 < end; row0 += bm, ++t) schedule[1 + t] = int4(int(e), int(row0), int(min(bm, end - row0)), 0);
  }
}

// ---------------------------------------------------------------------------
// moe_linear: y[out(p)] = x[in(p)] . W_e^T (* row_scale[out(p)]) for the valid
// sorted positions p of each tile; in(p) = gather[p] / gather_div (else p),
// out(p) = scatter[p] (else p).
struct MoeParams {
  ulong ldx, ldy;
  ulong expert_super_atoms;  // super-atoms per expert: (N / 16) * (K / 64)
  uint n, k_blocks, num_experts;
  uint kb_per_split, row_blocks, splits;  // small-M configuration (as LinearParams)
  uint gather_div;
  uint has_gather, has_scatter, has_scale;
};

struct MoeRows {
  device const int* gather;
  device const int* scatter;
  device const float* scale;
  constant MoeParams& p;
  inline ulong in_row(uint pos) const { return p.has_gather ? ulong(uint(gather[pos]) / p.gather_div) : ulong(pos); }
  inline ulong out_row(uint pos) const { return p.has_scatter ? ulong(uint(scatter[pos])) : ulong(pos); }
};

struct ExpertNF12 {
  device const uint4* packed;
  device const uint4* escapes;
  constant long* escape_offsets;
  device const uchar* exp_base;
  inline NF12Source source(uint e, ulong expert_super_atoms) const {
    return NF12Source{packed + ulong(e) * expert_super_atoms * (kSuperAtomBytes / 16), escapes + escape_offsets[e],
                      exp_base_x4(exp_base[e])};
  }
};

struct ExpertBf16t {
  device const uint4* packed;
  inline Bf16tSource source(uint e, ulong expert_super_atoms) const {
    return Bf16tSource{packed + ulong(e) * expert_super_atoms * (kSuperAtomBytesBf16t / 16)};
  }
};

constant constexpr uint kMoeMaxSimdgroups = 16;
constant constexpr uint kMoeMmaMaxSimdgroups = 8;

// Small: grid (row-block groups, tiles); tiles hold <= kMoeSmallMT positions.
template <uint kMoeSmallMT, typename Experts>
inline void moe_small(Experts experts, device const ushort* x, MoeRows rows, device ushort* y,
                      device const int4* schedule, constant MoeParams& p, uint2 tg, uint sg, uint lane, uint tid,
                      uint threads, threadgroup float* part) {
  if (int(tg.y) >= schedule[0].x) return;  // uniform across the threadgroup
  const int4 tile = schedule[1 + tg.y];
  const uint e = uint(tile.x), row0 = uint(tile.y), count = uint(tile.z);
  const auto src = experts.source(e, p.expert_super_atoms);
  const uint rb = sg / p.splits, ks = sg % p.splits;
  const uint row_blocks = p.n / 16;
  const uint nb = tg.x * p.row_blocks + rb;
  ulong xoff[kMoeSmallMT];
  for (uint m = 0; m < kMoeSmallMT; ++m) xoff[m] = rows.in_row(row0 + min(m, count - 1)) * p.ldx;
  float acc0[kMoeSmallMT], acc1[kMoeSmallMT];
  for (uint m = 0; m < kMoeSmallMT; ++m) acc0[m] = acc1[m] = 0.0f;
  if (nb < row_blocks) {
    const uint kb0 = min(ks * p.kb_per_split, p.k_blocks);
    const uint kb1 = min(kb0 + p.kb_per_split, p.k_blocks);
    gemv_accumulate<kMoeSmallMT>(src, ulong(nb) * p.k_blocks, kb0, kb1, x, xoff, lane, acc0, acc1);
  }
  gemv_reduce_quad<kMoeSmallMT>(acc0, acc1);
  if ((lane & 3u) == 0u) {
    const uint g = lane >> 2;
    threadgroup float* dst = part + (sg * kMoeSmallMT) * 16;
    for (uint m = 0; m < kMoeSmallMT; ++m) {
      dst[m * 16 + g] = acc0[m];
      dst[m * 16 + g + 8] = acc1[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.row_blocks * kMoeSmallMT * 16; i += threads) {
    const uint b = i / (kMoeSmallMT * 16), m = (i / 16) % kMoeSmallMT, row = i % 16;
    const uint nb2 = tg.x * p.row_blocks + b;
    if (nb2 >= row_blocks || m >= count) continue;
    float s = part[((b * p.splits) * kMoeSmallMT + m) * 16 + row];
    for (uint k = 1; k < p.splits; ++k) s += part[((b * p.splits + k) * kMoeSmallMT + m) * 16 + row];
    const ulong out = rows.out_row(row0 + m);
    if (p.has_scale) s *= rows.scale[out];
    y[out * p.ldy + nb2 * 16 + row] = float_to_bf16(s);
  }
}

// Medium (register MMA): grid (row-block groups, tiles); tiles hold <= 8 MB positions.
template <uint MB, typename Experts>
inline void moe_mma(Experts experts, device const ushort* x, MoeRows rows, device ushort* y,
                    device const int4* schedule, constant MoeParams& p, uint2 tg, uint sg, uint lane, uint tid,
                    uint threads, threadgroup float* part) {
  constexpr uint TM = 8 * MB;
  if (int(tg.y) >= schedule[0].x) return;  // uniform across the threadgroup
  const int4 tile = schedule[1 + tg.y];
  const uint e = uint(tile.x), row0 = uint(tile.y), count = uint(tile.z);
  const auto src = experts.source(e, p.expert_super_atoms);
  const uint rb = sg / p.splits, ks = sg % p.splits;
  const uint row_blocks = p.n / 16;
  const uint nb = tg.x * p.row_blocks + rb;
  const ushort2 c = mma_coord(lane);
  ulong xrow[2 * MB];
  for (uint mb = 0; mb < MB; ++mb) {
    xrow[2 * mb] = rows.in_row(row0 + min(8 * mb + c.y, count - 1)) * p.ldx;
    xrow[2 * mb + 1] = rows.in_row(row0 + min(8 * mb + c.y + 1, count - 1)) * p.ldx;
  }
  simdgroup_float8x8 acc[2][MB];
  for (uint h = 0; h < 2; ++h)
    for (uint mb = 0; mb < MB; ++mb) acc[h][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  if (nb < row_blocks) {  // uniform per simdgroup
    const uint kb0 = min(ks * p.kb_per_split, p.k_blocks);
    const uint kb1 = min(kb0 + p.kb_per_split, p.k_blocks);
    mma_accumulate<MB>(src, ulong(nb) * p.k_blocks, kb0, kb1, x, xrow, lane, acc);
  }
  threadgroup float* dst = part + sg * 16 * TM;
  for (uint h = 0; h < 2; ++h)
    for (uint mb = 0; mb < MB; ++mb) simdgroup_store(acc[h][mb], dst + 8 * h * TM + 8 * mb, TM);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.row_blocks * 16 * TM; i += threads) {
    const uint b = i / (16 * TM), m = i % TM, row = (i / TM) % 16;
    const uint nb2 = tg.x * p.row_blocks + b;
    if (nb2 >= row_blocks || m >= count) continue;
    float s = part[((b * p.splits) * 16 + row) * TM + m];
    for (uint k = 1; k < p.splits; ++k) s += part[((b * p.splits + k) * 16 + row) * TM + m];
    const ulong out = rows.out_row(row0 + m);
    if (p.has_scale) s *= rows.scale[out];
    y[out * p.ldy + nb2 * 16 + row] = float_to_bf16(s);
  }
}

// Large: grid (N / 64, tiles); tiles hold <= 32 positions.
template <typename Experts>
inline void moe_gemm(Experts experts, device const ushort* x, MoeRows rows, device ushort* y,
                     device const int4* schedule, constant MoeParams& p, uint2 tg, uint sg, uint lane, uint tid,
                     threadgroup GemmSmem& sm, threadgroup ulong* xoff) {
  if (int(tg.y) >= schedule[0].x) return;  // uniform across the threadgroup
  const int4 tile = schedule[1 + tg.y];
  const uint e = uint(tile.x), row0 = uint(tile.y), count = uint(tile.z);
  const auto src = experts.source(e, p.expert_super_atoms);
  const uint n0 = tg.x * kGemmBN;
  if (tid < kGemmBM) xoff[tid] = tid < count ? rows.in_row(row0 + tid) * p.ldx : ~0ul;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nb = n0 / 16 + sg;
  const bool valid = nb < p.n / 16;
  GemmAcc acc;
  gemm_zero(acc);
  for (uint kb = 0; kb < p.k_blocks; ++kb)
    gemm_slab(src, ulong(nb) * p.k_blocks + kb, valid, x, xoff, kb, sm, acc, sg, lane, tid);
  threadgroup const float* c = gemm_stage(acc, sm.w, sg);
  for (uint i = tid; i < kGemmBN * kGemmBM; i += 32 * kGemmSimdgroups) {
    const uint nl = i % kGemmBN, ml = i / kGemmBN;
    const uint n = n0 + nl;
    if (n >= p.n || ml >= count) continue;
    float s = c[nl * kGemmCLd + ml];
    const ulong out = rows.out_row(row0 + ml);
    if (p.has_scale) s *= rows.scale[out];
    y[out * p.ldy + n] = float_to_bf16(s);
  }
}

// Buffers: 0 x, 1 weight, 2 escapes, 3 escape_offsets, 4 exp_base, 5 gather, 6 scatter,
// 7 row_scale, 8 schedule, 9 y, 10 params.
#define NF12_MOE_ARGS                                                                                            \
  device const ushort *x [[buffer(0)]], device const uint4 *packed [[buffer(1)]],                                \
      device const int *gather [[buffer(5)]], device const int *scatter [[buffer(6)]],                           \
      device const float *row_scale [[buffer(7)]], device const int4 *schedule [[buffer(8)]],                    \
      device ushort *y [[buffer(9)]], constant MoeParams &p [[buffer(10)]],                                      \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]
#define NF12_MOE_NF12_ARGS                                                                                       \
  device const uint4 *escapes [[buffer(2)]], constant long *escape_offsets [[buffer(3)]],                        \
      device const uchar *exp_base [[buffer(4)]]

#define NF12_MOE_SMALL(MT)                                                                                       \
  NF12_THREADS(32 * kMoeMaxSimdgroups) kernel void nf12_moe_small_nf12_m##MT(NF12_MOE_ARGS, NF12_MOE_NF12_ARGS,                                       \
                                        uint2 threads [[threads_per_threadgroup]]) {                             \
    threadgroup float part[kMoeMaxSimdgroups * MT * 16];                                                         \
    moe_small<MT>(ExpertNF12{packed, escapes, escape_offsets, exp_base}, x, MoeRows{gather, scatter, row_scale, p}, \
                  y, schedule, p, tg, sg, lane, tid, threads.x, part);                                           \
  }                                                                                                              \
  NF12_THREADS(32 * kMoeMaxSimdgroups) kernel void nf12_moe_small_bf16t_m##MT(NF12_MOE_ARGS, uint2 threads [[threads_per_threadgroup]]) {             \
    threadgroup float part[kMoeMaxSimdgroups * MT * 16];                                                         \
    moe_small<MT>(ExpertBf16t{packed}, x, MoeRows{gather, scatter, row_scale, p}, y, schedule, p, tg, sg, lane,  \
                  tid, threads.x, part);                                                                         \
  }
NF12_MOE_SMALL(1)
NF12_MOE_SMALL(2)
NF12_MOE_SMALL(4)
NF12_MOE_SMALL(8)

#define NF12_MOE_MMA(MB)                                                                                         \
  NF12_THREADS(32 * kMoeMmaMaxSimdgroups) kernel void nf12_moe_mma_nf12_b##MB(NF12_MOE_ARGS, NF12_MOE_NF12_ARGS,                                         \
                                      uint2 threads [[threads_per_threadgroup]]) {                               \
    threadgroup float part[kMoeMmaMaxSimdgroups * 16 * 8 * MB];                                                  \
    moe_mma<MB>(ExpertNF12{packed, escapes, escape_offsets, exp_base}, x, MoeRows{gather, scatter, row_scale, p}, \
                y, schedule, p, tg, sg, lane, tid, threads.x, part);                                             \
  }                                                                                                              \
  NF12_THREADS(32 * kMoeMmaMaxSimdgroups) kernel void nf12_moe_mma_bf16t_b##MB(NF12_MOE_ARGS, uint2 threads [[threads_per_threadgroup]]) {               \
    threadgroup float part[kMoeMmaMaxSimdgroups * 16 * 8 * MB];                                                  \
    moe_mma<MB>(ExpertBf16t{packed}, x, MoeRows{gather, scatter, row_scale, p}, y, schedule, p, tg, sg, lane,    \
                tid, threads.x, part);                                                                           \
  }
NF12_MOE_MMA(1)
NF12_MOE_MMA(2)
NF12_MOE_MMA(4)

NF12_THREADS(32 * kGemmSimdgroups) kernel void nf12_moe_gemm_nf12(NF12_MOE_ARGS, NF12_MOE_NF12_ARGS) {
  threadgroup GemmSmem sm;
  threadgroup ulong xoff[kGemmBM];
  moe_gemm(ExpertNF12{packed, escapes, escape_offsets, exp_base}, x, MoeRows{gather, scatter, row_scale, p}, y,
           schedule, p, tg, sg, lane, tid, sm, xoff);
}

NF12_THREADS(32 * kGemmSimdgroups) kernel void nf12_moe_gemm_bf16t(NF12_MOE_ARGS) {
  threadgroup GemmSmem sm;
  threadgroup ulong xoff[kGemmBM];
  moe_gemm(ExpertBf16t{packed}, x, MoeRows{gather, scatter, row_scale, p}, y, schedule, p, tg, sg, lane, tid, sm,
           xoff);
}

// ---------------------------------------------------------------------------
// moe_combine: out[t] = sum_j y[t * topk + j] in FP32, increasing j, one rounding.
NF12_THREADS(256) kernel void nf12_moe_combine(device const ushort* y [[buffer(0)]],
                             constant ulong& ldy [[buffer(1)]],
                             constant uint& topk [[buffer(2)]],
                             constant uint& n [[buffer(3)]],
                             device ushort* out [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= n) return;
  const ulong t = gid.y;
  // Start from the first term so a leading -0.0 survives (0.0f + -0.0f == +0.0f).
  float s = bf16_to_float(y[(t * topk) * ldy + gid.x]);
  for (uint j = 1; j < topk; ++j) s += bf16_to_float(y[(t * topk + j) * ldy + gid.x]);
  out[t * n + gid.x] = float_to_bf16(s);
}

// moe_silu_mul: out[r, i] = bf16(s * u), s = g / (1 + exp(-g)), all FP32 (SPEC.md 9.2).
NF12_THREADS(256) kernel void nf12_moe_silu_mul(device const ushort* h [[buffer(0)]],
                              constant ulong& ldh [[buffer(1)]],
                              constant uint& inter [[buffer(2)]],
                              device ushort* out [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inter) return;
  const ulong r = gid.y;
  const float g = bf16_to_float(h[r * ldh + gid.x]);
  const float u = bf16_to_float(h[r * ldh + inter + gid.x]);
  const float s = g / (1.0f + precise::exp(-g));
  out[r * inter + gid.x] = float_to_bf16(s * u);
}
