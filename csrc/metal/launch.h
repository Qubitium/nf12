// SPDX-License-Identifier: Apache-2.0
// Host ABI of the NF12 Metal kernels, shared by the PyTorch MPS backend
// (ops_mps.mm) and the MLX backend (csrc/mlx/ops.cpp). Each launch_* function
// is the single source of truth for one kernel family: its name, buffer
// indices, parameter structs and grid geometry. Backends drive it through an
// encoder adapter `E`:
//
//   E::tensor                          backend array type (at::Tensor, mx::array)
//   E::out_t                           output handle type (a reference)
//   void pipeline(const std::string&)  select a kernel from the Metal library
//   void in(const tensor&, int)        bind an input buffer (with its offset)
//   void out(out_t, int)               bind an output buffer
//   template <class T> void bytes(const T&, int)  inline constant bytes
//   void dispatch(gx, gy, gz, tx, ty, tz)          threadgroups; empty grids are skipped, and a
//                                      threadgroup over the pipeline's limit throws (ThreadgroupLimit)
//   void barrier()                     order the next dispatch after the previous ones
//
// Pure C++: depends on neither torch, MLX nor Metal. Validation and allocation
// stay in the backends. Invalid arguments throw std::invalid_argument; a GPU that
// cannot run a kernel's threadgroup throws std::runtime_error.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace nf12::metal {

constexpr int64_t kMaxExpBase = 120;
constexpr uint32_t kCodecSimdgroups = 8;  // codec.metal kSimdgroups
constexpr uint32_t kAnalyzeSuperAtomsPerSimdgroup = 16;
constexpr uint32_t kHistWords = 3 * 128;     // codec.metal 3 * kHistBins
constexpr uint32_t kSmallMaxSimdgroups = 16;  // linear.metal kSmallMaxSimdgroups
constexpr uint32_t kMmaMaxSimdgroups = 8;     // linear.metal kMmaMaxSimdgroups
constexpr int64_t kGemmBN = 64, kGemmBM = 32;  // linear_core.h
constexpr uint32_t kGemmThreads = 128;
constexpr int64_t kMoeMaxExperts = 1024;  // moe.metal kMaxExperts
constexpr uint32_t kAlignChunk = 256;     // moe.metal kAlignChunk

inline uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
inline uint64_t super_atoms(int64_t n, int64_t k) { return uint64_t(n / 16) * uint64_t(k / 64); }

// Each GPU sets a per-pipeline threadgroup limit, and a dispatch above it does
// not run (its outputs keep their previous contents). Kernels declare their
// launch size (NF12_THREADS in nf12_metal.h), so the limit should never be
// lower; encoders still check it before every dispatch so a GPU that disagrees
// fails loudly instead of returning stale or zero-filled data.
struct ThreadgroupLimit {
  char kernel[64] = {};
  uint64_t max_threads = 0;
  void set(const std::string& name, uint64_t limit) {
    std::snprintf(kernel, sizeof(kernel), "%s", name.c_str());
    max_threads = limit;
  }
  void check(uint64_t threads) const {
    if (threads > max_threads)
      throw std::runtime_error(std::string("nf12: kernel ") + kernel + " needs " + std::to_string(threads) +
                               " threads per threadgroup but this GPU allows " + std::to_string(max_threads));
  }
};

template <class E>
using out_ptr = std::remove_reference_t<typename E::out_t>*;

// ---------------------------------------------------------------------------
// Parameter structs (must match the .metal declarations).

struct LinearParams {  // linear.metal LinearParams
  uint64_t ldx, ldy;
  uint32_t m, n, k_blocks, kb_per_split, row_blocks, splits, has_bias, reserved;
};
static_assert(sizeof(LinearParams) == 48, "LinearParams layout must match linear.metal");

struct MoeParams {  // moe.metal MoeParams
  uint64_t ldx, ldy, expert_super_atoms;
  uint32_t n, k_blocks, num_experts, kb_per_split, row_blocks, splits, gather_div, has_gather, has_scatter, has_scale;
};
static_assert(sizeof(MoeParams) == 64, "MoeParams layout must match moe.metal");

// ---------------------------------------------------------------------------
// Routing. Routes depend only on shapes (SPEC.md 6.5, 9.3).

enum class Route { kSmall, kMma, kGemm, kDecode };

// NF12_MPS_LINEAR=small|mma|gemm|decode forces a route (decode needs NF12) for testing.
inline Route linear_route(int64_t m, bool nf12, bool prefer_fused_gemm = false) {
  if (const char* env = std::getenv("NF12_MPS_LINEAR"); env != nullptr && *env != '\0') {
    const std::string s(env);
    if (s == "small") return Route::kSmall;
    if (s == "mma") return Route::kMma;
    if (s == "gemm") return Route::kGemm;
    if (s == "decode") return nf12 ? Route::kDecode : Route::kGemm;
    throw std::invalid_argument("nf12: NF12_MPS_LINEAR must be small, mma, gemm or decode, got " + s);
  }
  // Measured on an M5 Max (14336 x 4096 and 4096 x 14336): the scalar GEMV
  // streams NF12 at DRAM bandwidth to M = 4, the register-MMA kernel to M = 16;
  // beyond that matmul is compute bound and BF16 matmul on decoded slices wins.
  if (m <= 4) return Route::kSmall;
  if (m <= 16) return Route::kMma;
  return nf12 && !prefer_fused_gemm ? Route::kDecode : Route::kGemm;
}

inline uint32_t small_mt(int64_t m) { return m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : 8; }
inline uint32_t mma_tiles(int64_t m) { return m <= 8 ? 1 : m <= 16 ? 2 : 4; }

// Split K until there are enough simdgroups in flight to stream the weight at
// full bandwidth, keeping at least two super-atoms per split. `simdgroups`
// bounds row_blocks * splits per threadgroup.
struct SmallConfig {
  uint32_t kb_per_split, row_blocks, splits;
};
inline SmallConfig small_config(int64_t n, int64_t k, uint32_t simdgroups = 8) {
  const uint32_t row_blocks = uint32_t(n / 16), k_blocks = uint32_t(k / 64);
  uint32_t splits = 1;
  while (splits < simdgroups && uint64_t(row_blocks) * splits < 2048 && k_blocks >= 4 * splits) splits *= 2;
  return {uint32_t(ceil_div(k_blocks, splits)), std::max<uint32_t>(1, std::min<uint32_t>(simdgroups / splits, row_blocks)),
          splits};
}

// Decode-then-GEMM slice size: decoded BF16 slices are capped at 64 MiB
// by default (NF12_MPS_DECODE_SLICE_MB overrides).
inline int64_t decode_slice_bytes() {
  if (const char* env = std::getenv("NF12_MPS_DECODE_SLICE_MB"); env != nullptr && *env != '\0')
    return std::max<int64_t>(1, std::atoll(env)) << 20;
  return int64_t(64) << 20;
}
inline int64_t decode_slice_rows(int64_t n, int64_t k) {
  return std::min(n, std::max<int64_t>(16, (decode_slice_bytes() / (k * 2)) / 16 * 16));
}

// Upper bound on sum_e ceil(c_e / bm) with sum_e c_e <= A (as moe/schedule.cuh).
inline int64_t moe_max_tiles(int64_t a, int64_t e, int64_t bm) {
  if (a <= 0) return 0;
  const int64_t nonempty = std::min(e, a);
  return std::min(a, (a + bm - 1) / bm + nonempty - 1);
}

inline double expected_rows_per_expert(int64_t a, int64_t e) {
  const double A = double(a), E = double(e);
  return A / (E * (1.0 - std::pow(1.0 - 1.0 / E, A)));
}

// Tiles match the expected rows per expert. NF12_MPS_MOE=small|mma|gemm forces
// a kernel family for testing.
struct MoeRoute {
  enum Kind { kSmall, kMma, kGemm } kind;
  uint32_t width;  // small: rows per tile (MT); mma: 8-row tiles (MB)
  uint32_t bm;     // schedule tile rows
};

inline MoeRoute moe_route(int64_t a, int64_t e) {
  const double rows = expected_rows_per_expert(a, e);
  // Measured on an M5 Max (128 experts, top-8, K = 2048, I = 768).
  MoeRoute::Kind kind = rows <= 4.0 ? MoeRoute::kSmall : rows <= 8.0 ? MoeRoute::kMma : MoeRoute::kGemm;
  if (const char* env = std::getenv("NF12_MPS_MOE"); env != nullptr && *env != '\0') {
    const std::string s(env);
    if (s != "small" && s != "mma" && s != "gemm")
      throw std::invalid_argument("nf12: NF12_MPS_MOE must be small, mma or gemm, got " + s);
    kind = s == "small" ? MoeRoute::kSmall : s == "mma" ? MoeRoute::kMma : MoeRoute::kGemm;
  }
  const int64_t want = int64_t(std::ceil(rows));
  if (kind == MoeRoute::kSmall) {
    const uint32_t mt = small_mt(want);
    return {kind, mt, mt};
  }
  if (kind == MoeRoute::kMma) {
    const uint32_t mb = mma_tiles(want);
    return {kind, mb, 8 * mb};
  }
  return {kind, 0, uint32_t(kGemmBM)};
}

// Exponent base with the fewest escaped groups (ties: smallest B) from the
// nf12_encode_analyze histograms; returns {exp_base, escape count}.
inline std::pair<uint32_t, uint64_t> best_exp_base(const uint32_t* hist, uint64_t groups) {
  uint64_t covered = 0, best = ~0ull;
  uint32_t best_b = 0;
  for (uint32_t b = 0; b <= kMaxExpBase; ++b) {
    covered += hist[b];
    const uint64_t esc = groups - covered + hist[256 + b];
    if (esc < best) {
      best = esc;
      best_b = b;
    }
    covered -= hist[128 + b];
  }
  return {best_b, best};
}

// ---------------------------------------------------------------------------
// Launches.

// A dense NF12 weight's exponent base, read by the kernels from a uchar buffer:
// either a uint8 array (MLX passes the module's array, so the base is graph
// data and stays correct when mx.compile reuses a trace after weights change)
// or a value bound as inline bytes (byte 0 of a little-endian uint32).
template <class E>
struct ExpBase {
  const typename E::tensor* array = nullptr;  // uint8, element 0 is the base
  uint32_t value = 0;
};

template <class E>
void bind_exp_base(E& e, const ExpBase<E>& b, int index) {
  if (b.array != nullptr) {
    e.in(*b.array, index);
  } else {
    const uint32_t v = b.value;
    e.bytes(v, index);
  }
}

// Fills `words` 32-bit words of `buf` with `value`.
template <class E>
void launch_fill(E& e, typename E::out_t buf, uint64_t words, uint32_t value = 0) {
  if (words == 0) return;
  if (ceil_div(words, 256) > 0xffffffffull) throw std::invalid_argument("nf12: fill too large for one dispatch");
  e.pipeline("nf12_fill_u32");
  e.out(buf, 0);
  e.bytes(words, 1);
  e.bytes(value, 2);
  e.dispatch(ceil_div(words, 256), 1, 1, 256, 1, 1);
}

// Encode pass 1: hist (uint32 [kHistWords]) is zeroed, then accumulated.
template <class E>
void launch_encode_analyze(E& e, const typename E::tensor& w, uint64_t ldw, int64_t n, int64_t k,
                           typename E::out_t hist) {
  launch_fill(e, hist, kHistWords);
  e.barrier();
  const uint64_t num_sa = super_atoms(n, k);
  const uint32_t k_blocks = uint32_t(k / 64);
  e.pipeline("nf12_encode_analyze");
  e.in(w, 0);
  e.bytes(ldw, 1);
  e.bytes(num_sa, 2);
  e.bytes(k_blocks, 3);
  e.out(hist, 4);
  e.dispatch(ceil_div(num_sa, uint64_t(kAnalyzeSuperAtomsPerSimdgroup) * kCodecSimdgroups), 1, 1,
             32 * kCodecSimdgroups, 1, 1);
}

// Encode pass 2: escapes per super-atom (int32 [num_sa]).
template <class E>
void launch_encode_count(E& e, const typename E::tensor& w, uint64_t ldw, int64_t n, int64_t k, uint32_t exp_base,
                         typename E::out_t counts) {
  const uint64_t num_sa = super_atoms(n, k);
  const uint32_t k_blocks = uint32_t(k / 64);
  e.pipeline("nf12_encode_count");
  e.in(w, 0);
  e.bytes(ldw, 1);
  e.bytes(num_sa, 2);
  e.bytes(k_blocks, 3);
  e.bytes(exp_base, 4);
  e.out(counts, 5);
  e.dispatch(ceil_div(num_sa, kCodecSimdgroups), 1, 1, 32 * kCodecSimdgroups, 1, 1);
}

// Encode pass 3: packed groups and escapes at the exclusive escape offsets
// (int64 [num_sa]). `escapes` is null when there are none.
template <class E>
void launch_encode_write(E& e, const typename E::tensor& w, uint64_t ldw, int64_t n, int64_t k, uint32_t exp_base,
                         const typename E::tensor& offsets, typename E::out_t packed, out_ptr<E> escapes) {
  const uint64_t num_sa = super_atoms(n, k);
  const uint32_t k_blocks = uint32_t(k / 64);
  e.pipeline("nf12_encode_write");
  e.in(w, 0);
  e.bytes(ldw, 1);
  e.bytes(num_sa, 2);
  e.bytes(k_blocks, 3);
  e.bytes(exp_base, 4);
  e.in(offsets, 5);
  e.out(packed, 6);
  if (escapes != nullptr) {
    e.out(*escapes, 7);
  } else {
    e.out(packed, 7);
  }
  e.dispatch(ceil_div(num_sa, kCodecSimdgroups), 1, 1, 32 * kCodecSimdgroups, 1, 1);
}

// Decodes `experts` stacked weights of n rows into out [experts * n, k] (row
// stride ldo). Stacked experts pass escape_offsets (int64 [E + 1]) and an
// exp_base array (uint8 [E]); a dense weight passes null offsets and its base.
// `escapes` is null when there are none.
template <class E>
void launch_decode(E& e, const typename E::tensor& packed, const typename E::tensor* escapes,
                   const typename E::tensor* escape_offsets, const ExpBase<E>& exp_base, int64_t experts, int64_t n,
                   int64_t k, typename E::out_t out, uint64_t ldo) {
  const uint64_t num_sa = super_atoms(n, k);
  const uint32_t k_blocks = uint32_t(k / 64);
  const uint64_t rows = uint64_t(n);
  e.pipeline("nf12_decode");
  e.in(packed, 0);
  e.in(escapes != nullptr ? *escapes : packed, 1);
  if (escape_offsets != nullptr) {
    e.in(*escape_offsets, 2);
  } else {
    const int64_t zero = 0;
    e.bytes(zero, 2);
  }
  bind_exp_base(e, exp_base, 3);
  e.bytes(num_sa, 4);
  e.bytes(k_blocks, 5);
  e.bytes(ldo, 6);
  e.bytes(rows, 7);
  e.out(out, 8);
  e.dispatch(ceil_div(num_sa, kCodecSimdgroups), uint64_t(experts), 1, 32 * kCodecSimdgroups, 1, 1);
}

template <class E>
void launch_pack_bf16t(E& e, const typename E::tensor& w, uint64_t ldw, int64_t n, int64_t k, typename E::out_t out) {
  const uint64_t num_sa = super_atoms(n, k);
  const uint32_t k_blocks = uint32_t(k / 64);
  e.pipeline("nf12_pack_bf16t");
  e.in(w, 0);
  e.bytes(ldw, 1);
  e.bytes(num_sa, 2);
  e.bytes(k_blocks, 3);
  e.out(out, 4);
  e.dispatch(ceil_div(num_sa, kCodecSimdgroups), 1, 1, 32 * kCodecSimdgroups, 1, 1);
}

// ids: contiguous int32 (ids64 = false) or int64; out [num_ids, k] bf16.
template <class E>
void launch_embedding(E& e, const typename E::tensor& packed, const typename E::tensor* escapes, const ExpBase<E>& exp_base,
                      const typename E::tensor& ids, bool ids64, uint64_t num_ids, int64_t n, int64_t k,
                      typename E::out_t out) {
  if (num_ids > 0xffffffffull) throw std::invalid_argument("nf12.embedding: too many ids");
  const uint64_t vocab = uint64_t(n), ldo = uint64_t(k);
  const uint32_t k_blocks = uint32_t(k / 64);
  // 64 threads per threadgroup along x covers 4 super-atom columns.
  const uint64_t x_threads = uint64_t(k_blocks) * 16;
  const uint64_t tx = std::min<uint64_t>(64, x_threads);
  e.pipeline(ids64 ? "nf12_embedding_i64" : "nf12_embedding_i32");
  e.in(packed, 0);
  e.in(escapes != nullptr ? *escapes : packed, 1);
  bind_exp_base(e, exp_base, 2);
  e.in(ids, 3);
  e.bytes(vocab, 4);
  e.bytes(k_blocks, 5);
  e.out(out, 6);
  e.bytes(ldo, 7);
  e.dispatch(ceil_div(x_threads, tx), num_ids, 1, tx, 1, 1);
}

// Fused linear (route kSmall, kMma or kGemm): y [m, n] (row stride n) =
// x [m, k] (row stride ldx, even, 4-byte aligned) @ W.T (+ bias).
// NF12 weights pass `nf12 = true`, their escapes (null when none) and exp_base;
// bf16t twins pass `nf12 = false`.
template <class E>
void launch_linear(E& e, Route route, const typename E::tensor& x, uint64_t ldx, const typename E::tensor& packed,
                   bool nf12, const typename E::tensor* escapes, const ExpBase<E>& exp_base, const typename E::tensor* bias,
                   typename E::out_t y, int64_t m, int64_t n, int64_t k) {
  if (route == Route::kDecode) throw std::invalid_argument("nf12: launch_linear does not run decode-then-GEMM");
  LinearParams p{};
  p.ldx = ldx;
  p.ldy = uint64_t(n);
  p.m = uint32_t(m);
  p.n = uint32_t(n);
  p.k_blocks = uint32_t(k / 64);
  p.has_bias = bias != nullptr ? 1u : 0u;
  const std::string src = nf12 ? "nf12" : "bf16t";
  std::string kernel;
  uint64_t gx, gy, threads;
  if (route == Route::kSmall || route == Route::kMma) {
    const SmallConfig c = small_config(n, k);
    p.kb_per_split = c.kb_per_split;
    p.row_blocks = c.row_blocks;
    p.splits = c.splits;
    // Each kernel's threadgroup partials are sized for its simdgroup maximum.
    if (p.row_blocks * p.splits > (route == Route::kSmall ? kSmallMaxSimdgroups : kMmaMaxSimdgroups))
      throw std::invalid_argument("nf12: invalid small-M configuration");
    gx = ceil_div(uint64_t(n / 16), p.row_blocks);
    threads = 32 * p.row_blocks * p.splits;
    if (route == Route::kSmall) {
      const uint32_t mt = small_mt(m);
      kernel = "nf12_linear_small_" + src + "_m" + std::to_string(mt);
      gy = ceil_div(uint64_t(m), mt);
    } else {
      const uint32_t mb = mma_tiles(m);
      kernel = "nf12_linear_mma_" + src + "_b" + std::to_string(mb);
      gy = ceil_div(uint64_t(m), 8 * mb);
    }
  } else {
    kernel = "nf12_linear_gemm_" + src;
    gx = ceil_div(uint64_t(n), kGemmBN);
    gy = ceil_div(uint64_t(m), kGemmBM);
    threads = kGemmThreads;
  }
  e.pipeline(kernel);
  e.in(x, 0);
  e.in(packed, 1);
  if (nf12) e.in(escapes != nullptr ? *escapes : packed, 2);
  e.in(bias != nullptr ? *bias : x, 3);
  e.out(y, 4);
  e.bytes(p, 5);
  if (nf12) bind_exp_base(e, exp_base, 6);
  e.dispatch(gx, gy, 1, threads, 1, 1);
}

// moe_align for E <= kMoeMaxExperts: ids int32 [A] -> sorted int32 [A],
// offsets int64 [E + 1]. `bases` is scratch of uint32 [max(1, chunks * E)].
template <class E>
void launch_moe_align(E& e, const typename E::tensor& ids, uint32_t a, uint32_t num_experts,
                      typename E::out_t bases, typename E::out_t sorted, typename E::out_t offsets) {
  if (num_experts > kMoeMaxExperts) throw std::invalid_argument("nf12: launch_moe_align supports at most 1024 experts");
  const uint32_t chunks = uint32_t(ceil_div(a, kAlignChunk));
  e.pipeline("nf12_moe_align_count");
  e.in(ids, 0);
  e.bytes(a, 1);
  e.bytes(num_experts, 2);
  e.out(bases, 3);
  e.dispatch(chunks, 1, 1, kAlignChunk, 1, 1);
  e.barrier();
  e.pipeline("nf12_moe_align_offsets");
  e.bytes(chunks, 0);
  e.bytes(num_experts, 1);
  e.out(bases, 2);
  e.out(offsets, 3);
  e.dispatch(1, 1, 1, 1024, 1, 1);
  e.barrier();
  e.pipeline("nf12_moe_align_scatter");
  e.in(ids, 0);
  e.bytes(a, 1);
  e.bytes(num_experts, 2);
  e.out(bases, 3);  // read only; bound as output to keep the dependency on pass 2
  e.out(sorted, 4);
  e.dispatch(chunks, 1, 1, kAlignChunk, 1, 1);
}

// Scratch for launch_moe_align: one uint32 per (chunk, expert). The count
// exceeds 32 bits for large A and E (chunks < 2^23, E <= 1024), so it is 64-bit;
// the kernels index it with 64-bit arithmetic.
inline uint64_t align_chunks(uint32_t a) { return ceil_div(a, kAlignChunk); }
inline uint64_t align_scratch_words(uint32_t a, uint32_t num_experts) {
  return std::max<uint64_t>(1, align_chunks(a) * num_experts);
}

// Tile schedule for moe_linear: int32 [4 * (1 + max_tiles)].
template <class E>
void launch_moe_schedule(E& e, const typename E::tensor& expert_offsets, uint32_t num_experts, uint32_t a, uint32_t bm,
                         typename E::out_t schedule) {
  e.pipeline("nf12_moe_schedule");
  e.in(expert_offsets, 0);
  e.bytes(num_experts, 1);
  e.bytes(a, 2);
  e.bytes(bm, 3);
  e.out(schedule, 4);
  e.dispatch(1, 1, 1, 1024, 1, 1);
}

inline int64_t moe_schedule_ints(int64_t a, int64_t e, uint32_t bm) { return 4 * (1 + moe_max_tiles(a, e, bm)); }

// An MoE weight operand: NF12 (escapes, escape_offsets, exp_base) or a bf16t twin.
template <class E>
struct MoeWeightArgs {
  const typename E::tensor& packed;
  bool nf12;
  const typename E::tensor* escapes;         // NF12; null when there are none
  const typename E::tensor* escape_offsets;  // NF12
  const typename E::tensor* exp_base;        // NF12
  int64_t experts, n, k;
};

// Grouped linear y [a, n] over the schedule from launch_moe_schedule (same
// route). zero_fill zeroes y first (rows no tile covers stay zero).
template <class E>
void launch_moe_linear(E& e, const MoeRoute& route, const typename E::tensor& x, uint64_t ldx,
                       const MoeWeightArgs<E>& w, const typename E::tensor* gather, uint32_t gather_div,
                       const typename E::tensor* scatter, const typename E::tensor* row_scale,
                       const typename E::tensor& schedule, typename E::out_t y, int64_t a, bool zero_fill) {
  if (zero_fill) {
    launch_fill(e, y, uint64_t(a) * uint64_t(w.n) / 2);
    e.barrier();
  }
  MoeParams p{};
  p.ldx = ldx;
  p.ldy = uint64_t(w.n);
  p.expert_super_atoms = super_atoms(w.n, w.k);
  p.n = uint32_t(w.n);
  p.k_blocks = uint32_t(w.k / 64);
  p.num_experts = uint32_t(w.experts);
  p.gather_div = gather_div;
  p.has_gather = gather != nullptr ? 1u : 0u;
  p.has_scatter = scatter != nullptr ? 1u : 0u;
  p.has_scale = row_scale != nullptr ? 1u : 0u;
  const uint64_t max_tiles = uint64_t(moe_max_tiles(a, w.experts, route.bm));
  const std::string src = w.nf12 ? "nf12" : "bf16t";
  std::string kernel;
  uint64_t gx, threads;
  if (route.kind != MoeRoute::kGemm) {
    const bool mma = route.kind == MoeRoute::kMma;
    kernel = std::string(mma ? "nf12_moe_mma_" : "nf12_moe_small_") + src + (mma ? "_b" : "_m") +
             std::to_string(route.width);
    const SmallConfig c = small_config(w.n, w.k, mma ? 8 : 16);
    p.kb_per_split = c.kb_per_split;
    p.row_blocks = c.row_blocks;
    p.splits = c.splits;
    gx = ceil_div(uint64_t(w.n / 16), p.row_blocks);
    threads = 32 * p.row_blocks * p.splits;
  } else {
    kernel = "nf12_moe_gemm_" + src;
    gx = ceil_div(uint64_t(w.n), kGemmBN);
    threads = kGemmThreads;
  }
  e.pipeline(kernel);
  e.in(x, 0);
  e.in(w.packed, 1);
  if (w.nf12) {
    e.in(w.escapes != nullptr ? *w.escapes : w.packed, 2);
    e.in(*w.escape_offsets, 3);
    e.in(*w.exp_base, 4);
  }
  e.in(gather != nullptr ? *gather : x, 5);
  e.in(scatter != nullptr ? *scatter : x, 6);
  e.in(row_scale != nullptr ? *row_scale : x, 7);
  e.in(schedule, 8);
  e.out(y, 9);
  e.bytes(p, 10);
  e.dispatch(gx, max_tiles, 1, threads, 1, 1);
}

// out [t, n] = sum over topk consecutive rows of y (row stride ldy); n % 8 == 0.
template <class E>
void launch_moe_combine(E& e, const typename E::tensor& y, uint64_t ldy, uint32_t topk, int64_t t, int64_t n,
                        typename E::out_t out) {
  const uint32_t n32 = uint32_t(n);
  const uint64_t tx = std::min<uint64_t>(256, uint64_t(n));
  e.pipeline("nf12_moe_combine");
  e.in(y, 0);
  e.bytes(ldy, 1);
  e.bytes(topk, 2);
  e.bytes(n32, 3);
  e.out(out, 4);
  e.dispatch(ceil_div(uint64_t(n), tx), uint64_t(t), 1, tx, 1, 1);
}

// out [rows, inter] = silu(h[:, :inter]) * h[:, inter:] (h row stride ldh).
template <class E>
void launch_moe_silu_mul(E& e, const typename E::tensor& h, uint64_t ldh, int64_t rows, int64_t inter,
                         typename E::out_t out) {
  const uint32_t i32 = uint32_t(inter);
  const uint64_t tx = std::min<uint64_t>(256, uint64_t(inter));
  e.pipeline("nf12_moe_silu_mul");
  e.in(h, 0);
  e.bytes(ldh, 1);
  e.bytes(i32, 2);
  e.out(out, 3);
  e.dispatch(ceil_div(uint64_t(inter), tx), uint64_t(rows), 1, tx, 1, 1);
}

}  // namespace nf12::metal
