// SPDX-License-Identifier: Apache-2.0
// Standalone codec test and benchmark driver.
//
//   codec_test            correctness: round trip, matching bytes against a
//                         host encoder built on nf12::encode_group_in_window,
//                         bf16t identity, B optimality by brute force
//   codec_test --bench    bandwidth of decode / pack_bf16t / analyze / encode
//                         against a BF16 device-to-device copy
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <nf12/format.cuh>

#include "codec/codec.h"
#include "internal.h"

#define CUDA_CHECK(x)                                                                           \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(2);                                                                             \
    }                                                                                           \
  } while (0)

namespace {

using u16 = uint16_t;

// Host reference, written directly from SPEC.md with the canonical scalar
// group encoder.

struct Matrix {
  int64_t n, k, ld;
  std::vector<u16> v;  // n * ld, padding columns hold garbage on purpose
  u16 at(int64_t r, int64_t c) const { return v[r * ld + c]; }
};

// Slots of group (sa, j, lane) in flat group id order (SPEC.md 4.1-4.3).
void group_slots(const Matrix& w, int64_t sa, int j, int lane, u16 (&s)[8]) {
  const int64_t kb_count = w.k / 64;
  const int64_t nb = sa / kb_count, kb = sa % kb_count;
  for (int i = 0; i < 8; ++i)
    s[i] = w.at(nb * 16 + nf12::slot_row(lane, i), kb * 64 + 16 * j + nf12::slot_col(lane, i));
}

int64_t num_groups(const Matrix& w) { return w.n * w.k / 8; }

// Enumerate every B whose window can hold the input exponents.
std::vector<int64_t> host_escape_counts(const Matrix& w) {
  const int64_t g_total = num_groups(w);
  const int threads = omp_get_max_threads();
  std::vector<std::vector<int64_t>> ok(threads, std::vector<int64_t>(nf12::kMaxExpBase + 1, 0));
#pragma omp parallel for schedule(static)
  for (int64_t gid = 0; gid < g_total; ++gid) {
    u16 s[8];
    group_slots(w, gid / 128, int(gid / 32 % 4), int(gid % 32), s);
    int emin = 255, emax = 0;
    for (u16 v : s) {
      const int e = (v >> 7) & 0xff;
      emin = std::min(emin, e);
      emax = std::max(emax, e);
    }
    auto& mine = ok[omp_get_thread_num()];
    for (int b = 0; b <= nf12::kMaxExpBase; ++b) {
      if (2 * b > emin || 2 * b + 15 < emax) continue;
      nf12::Group g;
      if (nf12::encode_group_in_window(s, uint32_t(b), g)) ++mine[b];
    }
  }
  std::vector<int64_t> esc(nf12::kMaxExpBase + 1, g_total);
  for (auto& t : ok)
    for (int b = 0; b <= nf12::kMaxExpBase; ++b) esc[b] -= t[b];
  return esc;
}

struct Encoded {
  int exp_base = 0;
  std::vector<uint8_t> packed;
  std::vector<uint4> escapes;
};

Encoded host_encode(const Matrix& w, int exp_base) {
  Encoded out;
  out.exp_base = exp_base;
  const int64_t g_total = num_groups(w);
  std::vector<nf12::Group> groups(g_total);
  std::vector<uint8_t> esc(g_total);
#pragma omp parallel for schedule(static)
  for (int64_t gid = 0; gid < g_total; ++gid) {
    u16 s[8];
    group_slots(w, gid / 128, int(gid / 32 % 4), int(gid % 32), s);
    esc[gid] = !nf12::encode_group_in_window(s, uint32_t(exp_base), groups[gid]);
  }
  for (int64_t gid = 0; gid < g_total; ++gid) {
    if (!esc[gid]) continue;
    u16 s[8];
    group_slots(w, gid / 128, int(gid / 32 % 4), int(gid % 32), s);
    uint4 e;
    std::memcpy(&e, s, 16);
    groups[gid] = nf12::Group{0u, uint32_t(out.escapes.size()), 0u};
    out.escapes.push_back(e);
  }
  out.packed.assign(size_t(w.n) * w.k * 3 / 2, 0);
  const int64_t num_sa = g_total / 128;
#pragma omp parallel for schedule(static)
  for (int64_t sa = 0; sa < num_sa; ++sa) {
    uint8_t* base = out.packed.data() + sa * 1536;
    for (int j = 0; j < 4; ++j)
      for (int l = 0; l < 32; ++l) {
        const nf12::Group& g = groups[(sa * 4 + j) * 32 + l];
        std::memcpy(base + 16 * l + 4 * j, &g.meta, 4);
        uint8_t* lo = base + 512 * (1 + j / 2) + 16 * l + 8 * (j % 2);
        std::memcpy(lo, &g.lo0, 4);
        std::memcpy(lo + 4, &g.lo1, 4);
      }
  }
  return out;
}

std::vector<uint8_t> host_bf16t(const Matrix& w) {
  const int64_t num_sa = num_groups(w) / 128;
  std::vector<uint8_t> out(size_t(w.n) * w.k * 2);
#pragma omp parallel for schedule(static)
  for (int64_t sa = 0; sa < num_sa; ++sa)
    for (int j = 0; j < 4; ++j)
      for (int l = 0; l < 32; ++l) {
        u16 s[8];
        group_slots(w, sa, j, l, s);
        std::memcpy(out.data() + sa * 2048 + 512 * j + 16 * l, s, 16);
      }
  return out;
}

u16 f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u16((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

using Gen = std::function<u16(std::mt19937_64&, int64_t r, int64_t c)>;

Matrix make_matrix(int64_t n, int64_t k, int64_t ld, uint64_t seed, const Gen& gen) {
  Matrix w{n, k, ld, std::vector<u16>(size_t(n) * ld)};
  const int threads = omp_get_max_threads();
#pragma omp parallel for schedule(static)
  for (int t = 0; t < threads; ++t) {
    std::mt19937_64 rng(seed * 1000003u + t);
    for (int64_t r = t; r < n; r += threads)
      for (int64_t c = 0; c < ld; ++c) w.v[r * ld + c] = c < k ? gen(rng, r, c) : u16(0xdead);
  }
  return w;
}

u16 gen_random_bits(std::mt19937_64& g, int64_t, int64_t) { return u16(g()); }

u16 gen_gaussian(std::mt19937_64& g, int64_t, int64_t) {
  std::normal_distribution<float> d(0.f, 0.02f);
  return f2bf(d(g));
}

// Realistic weights plus ~0.1% outliers and exact zeros.
u16 gen_gaussian_outliers(std::mt19937_64& g, int64_t r, int64_t c) {
  const uint64_t x = g();
  if (x % 2000 == 0) return f2bf(float(int(x >> 32) % 64) * 0.5f);
  if (x % 2000 == 1) return (x >> 20) & 1 ? 0x8000 : 0x0000;
  return gen_gaussian(g, r, c);
}

// Every class of special bit pattern, densely mixed with ordinary values.
u16 gen_specials(std::mt19937_64& g, int64_t r, int64_t c) {
  const uint64_t x = g();
  const u16 sign = u16((x >> 40) & 1) << 15;
  switch (x % 12) {
    case 0: return sign;                                       // +-0
    case 1: return sign | u16(1 + (x >> 8) % 127);             // subnormal
    case 2: return sign | 0x7f80;                              // +-Inf
    case 3: return sign | u16(0x7f80 | (1 + (x >> 8) % 127));  // NaN payloads
    case 4: return sign | 0x7f7f;                              // max finite
    case 5: return sign | 0x0080;                              // min normal
    default: return gen_gaussian(g, r, c);
  }
}

// Exponents spread over ~120 binades: most groups escape for every B.
u16 gen_wide(std::mt19937_64& g, int64_t, int64_t) {
  const uint64_t x = g();
  const int e = 67 + int(x % 120);
  return u16(((x >> 20) & 1) << 15 | e << 7 | ((x >> 32) & 0x7f));
}

u16 gen_ones(std::mt19937_64&, int64_t, int64_t) { return 0x3f80; }      // all groups collide at B = 63
u16 gen_zeros(std::mt19937_64&, int64_t, int64_t) { return 0x0000; }     // all collide at B = 0
u16 gen_neg_zeros(std::mt19937_64&, int64_t, int64_t) { return 0x8000; } // sign set: no collision

struct DeviceRun {
  int exp_base;
  uint32_t num_escapes;
  std::vector<uint8_t> packed;
  std::vector<uint4> escapes;
  std::vector<u16> decoded;  // n * ldo
  std::vector<uint8_t> bf16t;
};

constexpr u16 kOutFill = 0xabcd;

DeviceRun device_run(const Matrix& w, int64_t ldo) {
  const int64_t n = w.n, k = w.k;
  DeviceRun out;
  __nv_bfloat16 *d_w, *d_out;
  uint8_t *d_packed, *d_bf16t;
  uint4* d_esc;
  void* d_ws;
  const size_t ws = nf12::detail::encode_workspace_bytes(n, k);
  CUDA_CHECK(cudaMalloc(&d_w, w.v.size() * 2));
  CUDA_CHECK(cudaMalloc(&d_ws, ws));
  CUDA_CHECK(cudaMemcpy(d_w, w.v.data(), w.v.size() * 2, cudaMemcpyHostToDevice));
  uint8_t b;
  CUDA_CHECK(nf12::detail::encode_analyze(d_w, n, k, w.ld, d_ws, ws, &b, &out.num_escapes, 0));
  out.exp_base = b;
  const size_t packed_bytes = size_t(n) * k * 3 / 2;
  CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
  CUDA_CHECK(cudaMalloc(&d_esc, std::max<size_t>(16, size_t(out.num_escapes) * 16)));
  CUDA_CHECK(nf12::detail::encode(d_w, n, k, w.ld, b, d_packed, d_esc, out.num_escapes, d_ws, ws, 0));
  out.packed.resize(packed_bytes);
  out.escapes.resize(out.num_escapes);
  CUDA_CHECK(cudaMemcpy(out.packed.data(), d_packed, packed_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.escapes.data(), d_esc, size_t(out.num_escapes) * 16, cudaMemcpyDeviceToHost));

  out.decoded.assign(size_t(n) * ldo, kOutFill);
  CUDA_CHECK(cudaMalloc(&d_out, out.decoded.size() * 2));
  CUDA_CHECK(cudaMemcpy(d_out, out.decoded.data(), out.decoded.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(nf12::detail::decode(d_packed, d_esc, b, n, k, d_out, ldo, 0));
  CUDA_CHECK(cudaMemcpy(out.decoded.data(), d_out, out.decoded.size() * 2, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMalloc(&d_bf16t, size_t(n) * k * 2));
  CUDA_CHECK(nf12::detail::pack_bf16t(d_w, n, k, w.ld, d_bf16t, 0));
  out.bf16t.resize(size_t(n) * k * 2);
  CUDA_CHECK(cudaMemcpy(out.bf16t.data(), d_bf16t, out.bf16t.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_out), static_cast<void*>(d_packed),
                  static_cast<void*>(d_bf16t), static_cast<void*>(d_esc), d_ws})
    CUDA_CHECK(cudaFree(p));
  return out;
}

int g_failures = 0;

#define EXPECT(cond, ...)                         \
  do {                                            \
    if (!(cond)) {                                \
      std::printf("    FAIL %s: ", #cond);        \
      std::printf(__VA_ARGS__);                   \
      std::printf("\n");                          \
      ok = false;                                 \
    }                                             \
  } while (0)

// expect_b: required exp_base, or -1. expect_min_esc_frac: lower bound on
// the escaped-group fraction (checks the case really is adversarial).
void run_case(const char* name, int64_t n, int64_t k, int64_t ldw, int64_t ldo, uint64_t seed, const Gen& gen,
              bool brute_force, int expect_b = -1, double expect_min_esc_frac = 0.0,
              const std::function<void(Matrix&)>& mutate = nullptr,
              const std::function<bool(const DeviceRun&)>& post = nullptr) {
  Matrix w = make_matrix(n, k, ldw, seed, gen);
  if (mutate) mutate(w);
  const DeviceRun d = device_run(w, ldo);
  bool ok = true;

  int ref_b;
  int64_t ref_e;
  if (brute_force) {
    const std::vector<int64_t> esc = host_escape_counts(w);
    ref_b = int(std::min_element(esc.begin(), esc.end()) - esc.begin());
    ref_e = esc[ref_b];
  } else {
    ref_b = d.exp_base;  // large shapes: check the count at the device's choice only
    ref_e = -1;
  }
  const Encoded ref = host_encode(w, ref_b);
  if (ref_e < 0) ref_e = int64_t(ref.escapes.size());
  EXPECT(d.exp_base == ref_b, "exp_base %d vs %d", d.exp_base, ref_b);
  EXPECT(int64_t(d.num_escapes) == ref_e, "num_escapes %u vs %lld", d.num_escapes, (long long)ref_e);
  if (expect_b >= 0) EXPECT(d.exp_base == expect_b, "exp_base %d, expected %d", d.exp_base, expect_b);
  const double frac = double(d.num_escapes) / double(num_groups(w));
  EXPECT(frac >= expect_min_esc_frac, "escape fraction %.4f < %.4f", frac, expect_min_esc_frac);
  if (ok) {
    EXPECT(ref.escapes.size() == d.escapes.size(), "escape array size");
    EXPECT(ref.packed == d.packed, "packed bytes differ (first at %lld)",
           (long long)(std::mismatch(ref.packed.begin(), ref.packed.end(), d.packed.begin()).first -
                       ref.packed.begin()));
    EXPECT(ref.escapes.size() == d.escapes.size() &&
               std::memcmp(ref.escapes.data(), d.escapes.data(), ref.escapes.size() * 16) == 0,
           "escapes differ");
  }
  if (post) EXPECT(post(d), "case-specific check");
  int64_t bad = 0, pad_bad = 0;
  for (int64_t r = 0; r < n; ++r)
    for (int64_t c = 0; c < ldo; ++c) {
      const u16 got = d.decoded[r * ldo + c];
      if (c < k)
        bad += got != w.at(r, c);
      else
        pad_bad += got != kOutFill;
    }
  EXPECT(bad == 0, "%lld decoded values differ", (long long)bad);
  EXPECT(pad_bad == 0, "%lld padding values overwritten", (long long)pad_bad);
  EXPECT(host_bf16t(w) == d.bf16t, "bf16t bytes differ");
  std::printf("  %-4s %-28s %6lldx%-6lld ldw=%-6lld ldo=%-6lld B=%3d esc=%-9u (%.4f%%)\n", ok ? "ok" : "FAIL", name,
              (long long)n, (long long)k, (long long)ldw, (long long)ldo, d.exp_base, d.num_escapes, 100.0 * frac);
  if (!ok) ++g_failures;
}

// Argument contract: escapes_capacity below the real count and exp_base > 120
// return cudaErrorInvalidValue without touching the outputs. Capacity equal
// to the count succeeds and writes exactly that many entries.
void run_contract_case() {
  const int64_t n = 64, k = 512;
  const Matrix w = make_matrix(n, k, k, 99, gen_gaussian_outliers);
  const size_t packed_bytes = size_t(n) * k * 3 / 2, ws = nf12::detail::encode_workspace_bytes(n, k);
  constexpr uint8_t kPackedFill = 0x5a, kEscFill = 0xee;
  constexpr uint32_t kGuard = 4;  // extra escape entries that must stay untouched
  __nv_bfloat16* d_w;
  uint8_t* d_packed;
  uint4* d_esc;
  void* d_ws;
  CUDA_CHECK(cudaMalloc(&d_w, w.v.size() * 2));
  CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
  CUDA_CHECK(cudaMalloc(&d_ws, ws));
  CUDA_CHECK(cudaMemcpy(d_w, w.v.data(), w.v.size() * 2, cudaMemcpyHostToDevice));
  uint8_t b;
  uint32_t e;
  CUDA_CHECK(nf12::detail::encode_analyze(d_w, n, k, k, d_ws, ws, &b, &e, 0));
  CUDA_CHECK(cudaMalloc(&d_esc, size_t(e + kGuard) * 16));
  auto reset = [&] {
    CUDA_CHECK(cudaMemset(d_packed, kPackedFill, packed_bytes));
    CUDA_CHECK(cudaMemset(d_esc, kEscFill, size_t(e + kGuard) * 16));
  };
  // Bytes of the escape buffer from entry `from` on, and the whole packed
  // buffer, still hold their fill pattern.
  auto untouched = [&](uint32_t from, bool packed_too) {
    std::vector<uint8_t> esc(size_t(e + kGuard) * 16), packed(packed_bytes);
    CUDA_CHECK(cudaMemcpy(esc.data(), d_esc, esc.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(packed.data(), d_packed, packed.size(), cudaMemcpyDeviceToHost));
    bool clean = std::all_of(esc.begin() + size_t(from) * 16, esc.end(), [](uint8_t x) { return x == kEscFill; });
    if (packed_too) clean &= std::all_of(packed.begin(), packed.end(), [](uint8_t x) { return x == kPackedFill; });
    return clean;
  };
  bool ok = true;
  EXPECT(e > 0, "case needs escapes");

  reset();
  const cudaError_t short_cap = nf12::detail::encode(d_w, n, k, k, b, d_packed, d_esc, e - 1, d_ws, ws, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT(short_cap == cudaErrorInvalidValue, "capacity e-1 returned %s", cudaGetErrorString(short_cap));
  EXPECT(untouched(0, true), "capacity e-1 wrote output");

  reset();
  const cudaError_t bad_base = nf12::detail::encode(d_w, n, k, k, 121, d_packed, d_esc, e, d_ws, ws, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT(bad_base == cudaErrorInvalidValue, "exp_base 121 returned %s", cudaGetErrorString(bad_base));
  EXPECT(untouched(0, true), "exp_base 121 wrote output");
  const cudaError_t bad_decode = nf12::detail::decode(d_packed, d_esc, 121, n, k, d_w, k, 0);
  EXPECT(bad_decode == cudaErrorInvalidValue, "decode exp_base 121 returned %s", cudaGetErrorString(bad_decode));

  reset();
  CUDA_CHECK(nf12::detail::encode(d_w, n, k, k, b, d_packed, d_esc, e, d_ws, ws, 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT(untouched(e, false), "capacity e wrote past the escape count");
  const Encoded ref = host_encode(w, b);
  std::vector<uint8_t> packed(packed_bytes);
  CUDA_CHECK(cudaMemcpy(packed.data(), d_packed, packed_bytes, cudaMemcpyDeviceToHost));
  EXPECT(ref.escapes.size() == e && ref.packed == packed, "capacity e: packed bytes differ from reference");
  EXPECT(cudaGetLastError() == cudaSuccess, "sticky CUDA error");

  std::printf("  %-4s %-28s %6lldx%-6lld esc=%u (capacity e-1 / exp_base 121 rejected, capacity e ok)\n",
              ok ? "ok" : "FAIL", "argument_contract", (long long)n, (long long)k, e);
  if (!ok) ++g_failures;
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_packed), static_cast<void*>(d_esc), d_ws})
    CUDA_CHECK(cudaFree(p));
}

// The escape-offset scan with injected per-unit counts: totals past 2^32
// (the all-escape case of a ~2^35-element matrix) and arbitrary 64-bit
// values must give the exact exclusive prefix and total.
void run_scan_case(const char* name, int64_t n, const std::function<uint64_t(int64_t)>& count) {
  std::vector<uint64_t> v(n + 1, 0xdeadbeefull), want(n + 1);
  uint64_t run = 0;
  for (int64_t i = 0; i < n; ++i) {
    v[i] = count(i);
    want[i] = run;
    run += v[i];
  }
  want[n] = run;
  uint64_t* d;
  CUDA_CHECK(cudaMalloc(&d, v.size() * 8));
  CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 8, cudaMemcpyHostToDevice));
  CUDA_CHECK(nf12::codec::escape_offsets_scan(d, n, 0));
  CUDA_CHECK(cudaMemcpy(v.data(), d, v.size() * 8, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  bool ok = true;
  EXPECT(v == want, "prefix differs (first at %lld)",
         (long long)(std::mismatch(v.begin(), v.end(), want.begin()).first - v.begin()));
  std::printf("  %-4s %-28s units=%-9lld total=%llu%s\n", ok ? "ok" : "FAIL", name, (long long)n,
              (unsigned long long)want[n], want[n] > UINT32_MAX ? " (> 2^32)" : "");
  if (!ok) ++g_failures;
}

__global__ void fill_kernel(uint4* p, int64_t n, uint4 v) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    p[i] = v;
}

// NF12_HUGE_TESTS=1 requires ~65 GiB free for N = 131072, K = 262144 (2^32 groups).
// All +1.0 with forced exp_base 0 escapes every group. Encode must refuse
// any capacity before writing because the index no longer fits L0. The null
// escape pointer and 16-byte packed buffer would fault on a write.
// Analyze picks B = 56 with no escapes. All +0 escapes at every base and
// must be refused.
void run_huge_case() {
  const int64_t n = 131072, k = 262144;
  const size_t bytes = size_t(n) * k * 2, ws = nf12::detail::encode_workspace_bytes(n, k);
  size_t free_bytes, total_bytes;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < bytes + ws + (size_t(1) << 30)) {
    std::printf("  skip %-28s needs %.1f GiB free, have %.1f\n", "huge_all_escape", (bytes + ws) / 1073741824.0,
                free_bytes / 1073741824.0);
    return;
  }
  __nv_bfloat16* d_w;
  uint8_t* d_packed;
  void* d_ws;
  CUDA_CHECK(cudaMalloc(&d_w, bytes));
  CUDA_CHECK(cudaMalloc(&d_packed, 16));
  CUDA_CHECK(cudaMalloc(&d_ws, ws));
  int sms;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  auto fill = [&](uint32_t pair) {
    fill_kernel<<<sms * 8, 256>>>(reinterpret_cast<uint4*>(d_w), int64_t(bytes / 16), make_uint4(pair, pair, pair, pair));
    CUDA_CHECK(cudaGetLastError());
  };
  bool ok = true;
  uint8_t b = 0xff;
  uint32_t e = 0xffffffffu;

  fill(0x3f803f80u);
  CUDA_CHECK(nf12::detail::encode_analyze(d_w, n, k, k, d_ws, ws, &b, &e, 0));
  EXPECT(b == 56 && e == 0, "all +1.0: B=%d esc=%u, expected B=56 esc=0", b, e);
  for (uint32_t cap : {0u, UINT32_MAX}) {
    const cudaError_t r = nf12::detail::encode(d_w, n, k, k, 0, d_packed, nullptr, cap, d_ws, ws, 0);
    EXPECT(r == cudaErrorInvalidValue, "encode B=0 capacity %u returned %s", cap, cudaGetErrorString(r));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  fill(0u);
  const cudaError_t r = nf12::detail::encode_analyze(d_w, n, k, k, d_ws, ws, &b, &e, 0);
  EXPECT(r == cudaErrorInvalidValue, "analyze all +0 returned %s", cudaGetErrorString(r));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("  %-4s %-28s %6lldx%-6lld 2^32 escaped groups rejected by analyze and encode\n", ok ? "ok" : "FAIL",
              "huge_all_escape", (long long)n, (long long)k);
  if (!ok) ++g_failures;
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_packed), d_ws}) CUDA_CHECK(cudaFree(p));
}

int run_tests() {
  std::printf("correctness\n");
  struct Shape {
    int64_t n, k;
  };
  const Shape shapes[] = {{16, 64}, {32, 128}, {48, 320}, {16, 1024}, {256, 64}, {128, 1088}};
  const struct {
    const char* name;
    Gen gen;
    double min_esc;
  } gens[] = {{"random_bits", gen_random_bits, 0.9},
              {"gaussian", gen_gaussian, 0.0},
              {"gaussian_outliers", gen_gaussian_outliers, 0.0},
              {"specials", gen_specials, 0.5},
              {"wide_exponents", gen_wide, 0.9}};
  uint64_t seed = 1;
  for (const Shape& s : shapes)
    for (const auto& g : gens) {
      run_case(g.name, s.n, s.k, s.k, s.k, seed++, g.gen, true, -1, g.min_esc);
      run_case(g.name, s.n, s.k, s.k + 24, s.k + 8, seed++, g.gen, true, -1, g.min_esc);
    }

  // B = 63 encodes 1.0 with sentinel M == 0. The smallest noncolliding base
  // that holds exponent 127 is 56.
  run_case("all_ones", 64, 256, 256, 264, seed++, gen_ones, true, 56);
  // Zeros collide at B = 0 and escape at every other base.
  run_case("all_zeros", 32, 128, 136, 128, seed++, gen_zeros, true, 0, 1.0);
  run_case("all_neg_zeros", 32, 128, 128, 128, seed++, gen_neg_zeros, true, 0);
  // Forced collision inside an ordinary matrix: one group of positive values
  // with exponent 2B, where B is the base the rest of the matrix selects.
  {
    const Matrix probe = make_matrix(64, 512, 512, 777, gen_gaussian);
    const std::vector<int64_t> esc = host_escape_counts(probe);
    const int b = int(std::min_element(esc.begin(), esc.end()) - esc.begin());
    // Super-atom 3, atom 2, lane 13 with mantissas 0x15 + i in slot i.
    constexpr int64_t kSa = 3;
    constexpr int kAtom = 2, kLane = 13;
    run_case(
        "planted_sentinel", 64, 512, 512, 520, 777, gen_gaussian, true, b, 0.0,
        [b](Matrix& w) {
          const int64_t nb = kSa / (w.k / 64), kb = kSa % (w.k / 64);
          for (int i = 0; i < 8; ++i)
            w.v[(nb * 16 + nf12::slot_row(kLane, i)) * w.ld + kb * 64 + 16 * kAtom + nf12::slot_col(kLane, i)] =
                u16((2 * b) << 7 | (0x15 + i));
        },
        [b](const DeviceRun& d) {
          uint32_t meta, idx;
          std::memcpy(&meta, d.packed.data() + kSa * 1536 + 16 * kLane + 4 * kAtom, 4);
          std::memcpy(&idx, d.packed.data() + kSa * 1536 + 1024 + 16 * kLane + 8 * (kAtom - 2), 4);
          if (meta != 0 || idx >= d.escapes.size()) return false;
          u16 s[8];
          std::memcpy(s, &d.escapes[idx], 16);
          for (int i = 0; i < 8; ++i)
            if (s[i] != u16((2 * b) << 7 | (0x15 + i))) return false;
          return true;
        });
  }
  run_contract_case();
  run_scan_case("scan_max_unit_counts", 5'000'000, [](int64_t) { return uint64_t(8 * 128); });
  run_scan_case("scan_random_wide", 1'000'003, [](int64_t i) { return (uint64_t(i) * 0x9e3779b97f4a7c15ull) >> 24; });
  for (int64_t n : {0, 1, 31, 1024, 1025, 2047, 2048, 2049, 4097})
    run_scan_case("scan_small", n, [](int64_t i) { return uint64_t(i % 7) << 30; });
  run_scan_case("scan_overflow", 4099, [](int64_t i) { return UINT64_MAX - uint64_t(i); });
  if (const char* huge = std::getenv("NF12_HUGE_TESTS"); huge && std::strcmp(huge, "1") == 0) run_huge_case();
  // Large: realistic weights with outliers, checked against the host encoder.
  run_case("gaussian_outliers", 8192, 28672, 28672, 28672, seed++, gen_gaussian_outliers, false);
  run_case("specials", 4096, 4096, 4160, 4104, seed++, gen_specials, false, -1, 0.5);
  run_case("random_bits", 2048, 2048, 2048, 2048, seed++, gen_random_bits, true, -1, 0.9);
  std::printf("%s (%d failing cases)\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}

__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    __stcs(dst + i, __ldcs(src + i));
}

template <typename F>
float time_ms(F&& f, int reps) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) f();
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(a));
    f();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return t[t.size() / 2];
}

void bench_shape(int64_t n, int64_t k, int reps) {
  const Matrix w = make_matrix(n, k, k, 42, gen_gaussian_outliers);
  const size_t bf16_bytes = size_t(n) * k * 2, packed_bytes = size_t(n) * k * 3 / 2;
  __nv_bfloat16 *d_w, *d_out;
  uint8_t *d_packed, *d_bf16t;
  uint4* d_esc;
  void* d_ws;
  const size_t ws = nf12::detail::encode_workspace_bytes(n, k);
  CUDA_CHECK(cudaMalloc(&d_w, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&d_bf16t, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
  CUDA_CHECK(cudaMalloc(&d_ws, ws));
  CUDA_CHECK(cudaMemcpy(d_w, w.v.data(), bf16_bytes, cudaMemcpyHostToDevice));
  uint8_t b;
  uint32_t e;
  CUDA_CHECK(nf12::detail::encode_analyze(d_w, n, k, k, d_ws, ws, &b, &e, 0));
  CUDA_CHECK(cudaMalloc(&d_esc, std::max<size_t>(16, size_t(e) * 16)));
  CUDA_CHECK(nf12::detail::encode(d_w, n, k, k, b, d_packed, d_esc, e, d_ws, ws, 0));

  const double gb = 1e9;
  auto report = [&](const char* what, float ms, double bytes) {
    std::printf("  %-22s %8.3f ms  %8.1f GB/s\n", what, ms, bytes / (ms * 1e-3) / gb);
  };
  std::printf("%lld x %lld  (B=%d, %u escapes = %.3f%% of groups)\n", (long long)n, (long long)k, b, e,
              100.0 * e / (double(n) * k / 8));
  report("memcpy D2D (r+w)", time_ms([&] { CUDA_CHECK(cudaMemcpyAsync(d_out, d_w, bf16_bytes, cudaMemcpyDeviceToDevice)); }, reps),
         2.0 * bf16_bytes);
  const int64_t vecs = int64_t(bf16_bytes / 16);
  int sms;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  report("copy kernel (r+w)", time_ms([&] {
           copy_kernel<<<sms * 8, 256>>>(reinterpret_cast<const uint4*>(d_w), reinterpret_cast<uint4*>(d_out), vecs);
         }, reps),
         2.0 * bf16_bytes);
  report("decode (r+w)", time_ms([&] { CUDA_CHECK(nf12::detail::decode(d_packed, d_esc, b, n, k, d_out, k, 0)); }, reps),
         double(packed_bytes) + 16.0 * e + bf16_bytes);
  report("pack_bf16t (r+w)", time_ms([&] { CUDA_CHECK(nf12::detail::pack_bf16t(d_w, n, k, k, d_bf16t, 0)); }, reps),
         2.0 * bf16_bytes);
  report("encode_analyze (r)", time_ms([&] { CUDA_CHECK(nf12::detail::encode_analyze(d_w, n, k, k, d_ws, ws, &b, &e, 0)); },
                                       reps),
         double(bf16_bytes));
  report("encode (2r+w)", time_ms([&] {
           CUDA_CHECK(nf12::detail::encode(d_w, n, k, k, b, d_packed, d_esc, e, d_ws, ws, 0));
         }, reps),
         2.0 * bf16_bytes + packed_bytes + 16.0 * e);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_out), static_cast<void*>(d_packed),
                  static_cast<void*>(d_bf16t), static_cast<void*>(d_esc), d_ws})
    CUDA_CHECK(cudaFree(p));
}

int run_bench() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  bench_shape(8192, 28672, 50);
  bench_shape(28672, 8192, 50);
  bench_shape(4096, 4096, 200);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return run_bench();
  return run_tests();
}
