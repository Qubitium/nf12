// SPDX-License-Identifier: Apache-2.0
// Device speed-of-light calibration: achievable DRAM streaming-read bandwidth
// (LDG variants, access patterns, occupancies, cp.async.bulk), D2D copy, L2
// capacity/bandwidth curve, cold-L2 read time per buffer size, launch floors
// and the mma.sync m16n8k16 BF16->FP32 peak that NF12 linear kernels issue.
// Standalone (no nf12 dependency). Prints one JSON document on stdout.
//
// Usage: sol_device [--big-mib N] [--cold-bytes a,b,c] [--quick]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "read_kernels.cuh"

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

namespace {

using namespace nf12_bench;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// SM90+ bulk global-to-shared reads with STAGES interleaved chunks in flight
// per CTA. main() skips these configurations on older devices.
template <int STAGES, int CHUNK>
__global__ void bulk_read_kernel(const unsigned char* __restrict__ p, size_t nchunks, uint32_t* sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) uint64_t bar[STAGES];
  const uint32_t smem_base = smem_u32(smem);
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar[s])));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto issue = [&](size_t c, int s) {
    const uint32_t b = smem_u32(&bar[s]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(CHUNK) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     smem_base + s * CHUNK),
                 "l"(p + c * CHUNK), "r"(CHUNK), "r"(b)
                 : "memory");
  };
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      const size_t c = blockIdx.x + (size_t)s * gridDim.x;
      if (c < nchunks) issue(c, s);
    }
  }
  uint32_t acc = 0;
  for (size_t it = 0;; ++it) {
    const size_t c = blockIdx.x + it * gridDim.x;
    if (c >= nchunks) break;
    const int s = static_cast<int>(it % STAGES);
    const uint32_t parity = static_cast<uint32_t>((it / STAGES) & 1);
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra WAIT;\n"
        "}\n" ::"r"(smem_u32(&bar[s])),
        "r"(parity)
        : "memory");
    acc ^= reinterpret_cast<const uint32_t*>(smem + s * CHUNK)[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
      const size_t cn = c + (size_t)STAGES * gridDim.x;
      if (cn < nchunks) issue(cn, s);
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
#endif
}

__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dst[i] = src[i];
}

__global__ void empty_kernel() {}

// Independent accumulator chains hide MMA latency. Thread-dependent register
// operands prevent constant folding.
template <int CHAINS>
__global__ void mma_peak_kernel(int iters, float* sink) {
  const uint32_t a[4] = {0x3c003c00u ^ threadIdx.x, 0x3c003c01u, 0x3c003c02u ^ threadIdx.x, 0x3c003c03u};
  const uint32_t b[2] = {0x3c003c00u ^ blockIdx.x, 0x3c003c01u};
  float acc[CHAINS][4] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
          "{%0,%1,%2,%3};"
          : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
  }
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < CHAINS; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  if (s == 1234.5f) sink[0] = s;
}


struct Dev {
  int id, num_sms, l2_bytes, max_threads_sm, smem_optin, cc_major, cc_minor;
  std::string name;
};

using ReadFn = void (*)(const uint4*, size_t, uint32_t*);
using BulkFn = void (*)(const unsigned char*, size_t, uint32_t*);

struct ReadCfg {
  int ld, unroll, pattern, block, bps;  // bps = CTAs per SM
  ReadFn fn;
};

struct BulkCfg {
  int stages, chunk, block, bps;
  BulkFn fn;
};

template <int LD, int U, int PAT>
void add_read(std::vector<ReadCfg>& v) {
  v.push_back({LD, U, PAT, 0, 0, read_kernel<LD, U, PAT>});
}

template <int LD>
void add_read_ld(std::vector<ReadCfg>& v) {
  add_read<LD, 1, kGridStride>(v);
  add_read<LD, 2, kGridStride>(v);
  add_read<LD, 4, kGridStride>(v);
  add_read<LD, 8, kGridStride>(v);
  add_read<LD, 1, kWarpSlab>(v);
  add_read<LD, 2, kWarpSlab>(v);
  add_read<LD, 4, kWarpSlab>(v);
  add_read<LD, 8, kWarpSlab>(v);
}

std::vector<ReadCfg> read_kernels() {
  std::vector<ReadCfg> v;
  add_read_ld<kLdDefault>(v);
  add_read_ld<kLdNc>(v);
  add_read_ld<kLdNcNoAlloc>(v);
  add_read_ld<kLdNcNoAlloc256>(v);
  add_read_ld<kLdCs>(v);
  return v;
}

std::vector<BulkCfg> bulk_kernels() {
  return {
      {2, 16384, 128, 0, bulk_read_kernel<2, 16384>}, {4, 8192, 128, 0, bulk_read_kernel<4, 8192>},
      {4, 16384, 128, 0, bulk_read_kernel<4, 16384>}, {8, 8192, 128, 0, bulk_read_kernel<8, 8192>},
      {6, 16384, 128, 0, bulk_read_kernel<6, 16384>}, {3, 32768, 128, 0, bulk_read_kernel<3, 32768>},
      {8, 4096, 128, 0, bulk_read_kernel<8, 4096>},
  };
}

struct Timer {
  std::vector<cudaEvent_t> ev;
  explicit Timer(int reps) : ev(2 * reps) {
    for (auto& e : ev) CK(cudaEventCreate(&e));
  }
  ~Timer() {
    for (auto& e : ev) cudaEventDestroy(e);
  }
};

double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Times `reps` launches, each optionally preceded by `pre` (not timed). Returns
// per-launch milliseconds after one untimed warmup.
template <class F>
std::vector<float> time_reps(cudaStream_t s, int reps, F launch) {
  Timer t(reps);
  launch();
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(t.ev[2 * r], s));
    launch();
    CK(cudaEventRecord(t.ev[2 * r + 1], s));
  }
  CK(cudaGetLastError());
  CK(cudaStreamSynchronize(s));
  std::vector<float> ms(reps);
  for (int r = 0; r < reps; ++r) CK(cudaEventElapsedTime(&ms[r], t.ev[2 * r], t.ev[2 * r + 1]));
  return ms;
}

// Captures n launches and returns median per-launch time over reps replays.
// Batching resolves microsecond kernels despite coarse event timestamps
// (2.048 us steps on the SM120 box) and preserves decode-sized launch gaps.
template <class F>
double graph_per_launch_ms(cudaStream_t s, int n, int reps, F launch) {
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < n; ++i) launch(i);
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  const double ms = median(time_reps(s, reps, [&] { CK(cudaGraphLaunch(ge, s)); })) / n;
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  return ms;
}

struct Ctx {
  Dev dev;
  cudaStream_t stream;
  uint32_t* sink;
};

int max_bps(const void* fn, int block, size_t smem) {
  int n = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, block, smem));
  return n;
}

void launch_read(const Ctx& c, const ReadCfg& k, const void* buf, size_t bytes) {
  k.fn<<<c.dev.num_sms * k.bps, k.block, 0, c.stream>>>(reinterpret_cast<const uint4*>(buf), bytes / 16, c.sink);
}

void launch_bulk(const Ctx& c, const BulkCfg& k, const void* buf, size_t bytes) {
  const size_t smem = (size_t)k.stages * k.chunk;
  k.fn<<<c.dev.num_sms * k.bps, k.block, smem, c.stream>>>(reinterpret_cast<const unsigned char*>(buf), bytes / k.chunk, c.sink);
}

std::vector<size_t> parse_sizes(const char* s) {
  std::vector<size_t> v;
  while (*s) {
    char* end;
    v.push_back(strtoull(s, &end, 10));
    s = *end == ',' ? end + 1 : end;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  size_t big_mib = 0;
  std::vector<size_t> cold_sizes;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--big-mib") && i + 1 < argc) big_mib = strtoull(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--cold-bytes") && i + 1 < argc) cold_sizes = parse_sizes(argv[++i]);
    else if (!strcmp(argv[i], "--quick")) quick = true;
    else {
      fprintf(stderr, "usage: %s [--big-mib N] [--cold-bytes a,b,...] [--quick]\n", argv[0]);
      return 2;
    }
  }

  Ctx c{};
  CK(cudaGetDevice(&c.dev.id));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, c.dev.id));
  c.dev.name = prop.name;
  c.dev.num_sms = prop.multiProcessorCount;
  c.dev.l2_bytes = prop.l2CacheSize;
  c.dev.max_threads_sm = prop.maxThreadsPerMultiProcessor;
  c.dev.smem_optin = static_cast<int>(prop.sharedMemPerBlockOptin);
  c.dev.cc_major = prop.major;
  c.dev.cc_minor = prop.minor;
  int bus_width = 0, mem_clock_khz = 0;
  CK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, c.dev.id));
  CK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, c.dev.id));

  const size_t l2 = static_cast<size_t>(c.dev.l2_bytes);
  // The streaming buffer must dwarf L2 so steady state is pure DRAM traffic.
  if (big_mib == 0) big_mib = std::max<size_t>(1024, (16 * l2) >> 20);
  const size_t big = big_mib << 20;
  // One pool: halves serve as copy source/destination, the whole pool holds
  // the rotating copies of the cold sweep.
  unsigned char* pool;
  CK(cudaMalloc(&pool, 2 * big));
  unsigned char* const buf = pool;
  unsigned char* const buf2 = pool + big;
  CK(cudaMalloc(&c.sink, 4096));
  CK(cudaMemset(pool, 0x3c, 2 * big));
  CK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
  CK(cudaDeviceSynchronize());

  const int reps = quick ? 3 : 7;

  printf("{\n");
  printf(
      "\"device\": {\"name\": \"%s\", \"cc\": %d%d, \"num_sms\": %d, \"l2_bytes\": %d, \"smem_optin\": %d, "
      "\"bus_width_bits\": %d, \"mem_clock_khz\": %d, \"theoretical_gbps\": %.1f},\n",
      c.dev.name.c_str(), c.dev.cc_major, c.dev.cc_minor, c.dev.num_sms, c.dev.l2_bytes, c.dev.smem_optin, bus_width,
      mem_clock_khz, 2.0 * mem_clock_khz * 1e3 * (bus_width / 8) / 1e9);
  printf("\"big_buffer_bytes\": %zu,\n", big);

  // (a) streaming read sweep over a buffer >> L2.
  struct Result {
    double gbps;
    std::string desc;
    int kind, idx;  // kind 0 = ldg, 1 = bulk
  };
  std::vector<Result> results;
  std::vector<ReadCfg> rk;
  for (const auto& base : read_kernels()) {
    for (int block : {128, 256, 512, 1024}) {
      const int maxb = max_bps(reinterpret_cast<const void*>(base.fn), block, 0);
      for (int bps : {1, 2, 4, 8, 16}) {
        if (bps > maxb) continue;
        if (quick && (block != 256 || base.pattern != kGridStride)) continue;
        ReadCfg k = base;
        k.block = block;
        k.bps = bps;
        rk.push_back(k);
      }
    }
  }
  printf("\"read_sweep\": [\n");
  bool first = true;
  for (size_t i = 0; i < rk.size(); ++i) {
    const auto& k = rk[i];
    const double ms = median(time_reps(c.stream, reps, [&] { launch_read(c, k, buf, big); }));
    const double gbps = big / ms / 1e6;
    char d[256];
    snprintf(d, sizeof d,
             "{\"kind\": \"ldg\", \"ld\": \"%s\", \"pattern\": \"%s\", \"unroll\": %d, \"block\": %d, "
             "\"ctas_per_sm\": %d, \"warps_per_sm\": %d, \"bytes_in_flight_per_sm\": %d, \"gbps\": %.1f}",
             kLdNames[k.ld], kPatternNames[k.pattern], k.unroll, k.block, k.bps, k.block * k.bps / 32,
             k.block * k.bps * k.unroll * 16, gbps);
    printf("%s  %s", first ? "" : ",\n", d);
    first = false;
    results.push_back({gbps, d, 0, static_cast<int>(i)});
  }
  std::vector<BulkCfg> bk;
  for (const auto& base : bulk_kernels()) {
    const size_t smem = (size_t)base.stages * base.chunk;
    if (smem > (size_t)c.dev.smem_optin || c.dev.cc_major < 9) continue;
    CK(cudaFuncSetAttribute(reinterpret_cast<const void*>(base.fn), cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(smem)));
    const int maxb = max_bps(reinterpret_cast<const void*>(base.fn), base.block, smem);
    for (int bps : {1, 2, 3, 4, 6, 8}) {
      if (bps > maxb) continue;
      BulkCfg k = base;
      k.bps = bps;
      bk.push_back(k);
    }
  }
  for (size_t i = 0; i < bk.size(); ++i) {
    const auto& k = bk[i];
    const size_t bytes = big / k.chunk * k.chunk;
    const double ms = median(time_reps(c.stream, reps, [&] { launch_bulk(c, k, buf, bytes); }));
    const double gbps = bytes / ms / 1e6;
    char d[256];
    snprintf(d, sizeof d,
             "{\"kind\": \"cp.async.bulk\", \"stages\": %d, \"chunk\": %d, \"block\": %d, \"ctas_per_sm\": %d, "
             "\"bytes_in_flight_per_sm\": %d, \"gbps\": %.1f}",
             k.stages, k.chunk, k.block, k.bps, k.stages * k.chunk * k.bps, gbps);
    printf(",\n  %s", d);
    results.push_back({gbps, d, 1, static_cast<int>(i)});
  }
  printf("\n],\n");
  std::vector<Result> ranked = results;
  std::sort(ranked.begin(), ranked.end(), [](const Result& a, const Result& b) { return a.gbps > b.gbps; });
  printf("\"read_best\": %s,\n", ranked[0].desc.c_str());
  // Best per load flavour, so kernels know what each qualifier costs.
  printf("\"read_best_by_ld\": {");
  for (int ld = 0; ld < kNumLd; ++ld) {
    for (const auto& r : ranked) {
      if (r.kind == 0 && rk[r.idx].ld == ld) {
        printf("%s\"%s\": %.1f", ld ? ", " : "", kLdNames[ld], r.gbps);
        break;
      }
    }
  }
  for (const auto& r : ranked) {
    if (r.kind == 1) {
      printf(", \"cp.async.bulk\": %.1f", r.gbps);
      break;
    }
  }
  printf("},\n");

  // Top distinct configurations reused for size sweeps (small buffers favour
  // different grid shapes, so keep several and take the best per size).
  std::vector<Result> top(ranked.begin(), ranked.begin() + std::min<size_t>(6, ranked.size()));
  auto launch_result = [&](const Result& r, const void* b, size_t bytes) {
    if (r.kind == 0) launch_read(c, rk[r.idx], b, bytes);
    else launch_bulk(c, bk[r.idx], b, bk[r.idx].chunk * (bytes / bk[r.idx].chunk));
  };
  // Small buffers: extra grid shapes with one CTA wave or fewer.
  std::vector<Result> small_cfgs = top;
  for (size_t i = 0; i < rk.size(); ++i) {
    const auto& k = rk[i];
    if (k.ld == kLdNcNoAlloc && k.pattern == kGridStride && k.unroll == 4 && k.block == 256)
      small_cfgs.push_back({0, "", 0, static_cast<int>(i)});
  }

  // (b) device-to-device copy.
  {
    const double ms_memcpy = median(
        time_reps(c.stream, reps, [&] { CK(cudaMemcpyAsync(buf2, buf, big, cudaMemcpyDeviceToDevice, c.stream)); }));
    double best_kernel = 1e30;
    int best_bps = 0;
    for (int bps : {1, 2, 4, 8}) {
      const double ms = median(time_reps(c.stream, reps, [&] {
        copy_kernel<<<c.dev.num_sms * bps, 512, 0, c.stream>>>(reinterpret_cast<const uint4*>(buf),
                                                                reinterpret_cast<uint4*>(buf2), big / 16);
      }));
      if (ms < best_kernel) best_kernel = ms, best_bps = bps;
    }
    printf(
        "\"copy\": {\"bytes\": %zu, \"memcpy_ms\": %.4f, \"memcpy_rw_gbps\": %.1f, \"memcpy_bytes_gbps\": %.1f, "
        "\"kernel_ms\": %.4f, \"kernel_rw_gbps\": %.1f, \"kernel_ctas_per_sm\": %d},\n",
        big, ms_memcpy, 2.0 * big / ms_memcpy / 1e6, big / ms_memcpy / 1e6, best_kernel, 2.0 * big / best_kernel / 1e6,
        best_bps);
  }

  // (c) hot read bandwidth vs footprint: the knee locates the usable L2 size.
  {
    std::vector<size_t> sizes;
    for (size_t mib = 1; mib <= std::min<size_t>(big_mib, 8 * (l2 >> 20) + 8); mib = mib < 8 ? mib * 2 : mib + mib / 4)
      sizes.push_back(mib << 20);
    printf("\"l2_hot\": [\n");
    for (size_t si = 0; si < sizes.size(); ++si) {
      const size_t bytes = sizes[si];
      double best = 1e30;
      const int n = static_cast<int>(std::clamp<size_t>((256ull << 20) / bytes, 8, 256));
      for (const auto& r : small_cfgs) {
        // The graph's untimed first replay warms L2.
        best = std::min(best, graph_per_launch_ms(c.stream, n, reps, [&](int) { launch_result(r, buf, bytes); }));
      }
      printf("%s  {\"bytes\": %zu, \"us\": %.3f, \"gbps\": %.1f}", si ? ",\n" : "", bytes, best * 1e3,
             bytes / best / 1e6);
    }
    printf("\n],\n");
  }

  // (d) cold-L2 read time per size: one kernel streaming a weight of that size
  // from DRAM, launch ramp and tail included. Launches rotate over enough
  // copies that each copy has been evicted (>= 3x L2 of other traffic) before
  // it is read again, which is also what consecutive layers of a model see.
  {
    if (cold_sizes.empty()) {
      for (size_t mib = 1; mib <= 1024; mib *= 2) cold_sizes.push_back(mib << 20);
    }
    printf("\"cold\": [\n");
    bool first_cold = true;
    for (size_t si = 0; si < cold_sizes.size(); ++si) {
      const size_t bytes = (cold_sizes[si] + 255) / 256 * 256;
      const size_t copies = std::max<size_t>(2, (3 * l2 + bytes - 1) / bytes + 1);
      if (copies * bytes > 2 * big) {
        fprintf(stderr, "cold size %zu needs %zu bytes of pool, skipped\n", bytes, copies * bytes);
        continue;
      }
      const int n = static_cast<int>(std::max<size_t>(copies, 32) / copies * copies);
      double best = 1e30;
      for (const auto& r : small_cfgs) {
        best = std::min(best, graph_per_launch_ms(c.stream, n, quick ? 3 : 5,
                                                  [&](int i) { launch_result(r, pool + (i % copies) * bytes, bytes); }));
      }
      printf("%s  {\"bytes\": %zu, \"us\": %.3f, \"gbps\": %.1f}", first_cold ? "" : ",\n", bytes, best * 1e3,
             bytes / best / 1e6);
      first_cold = false;
    }
    printf("\n],\n");
  }

  // mma.sync BF16 (FP32 accumulate) throughput, best over warps/SM and chains.
  {
    float* fsink = reinterpret_cast<float*>(c.sink);
    struct MmaCfg {
      int chains;
      void (*fn)(int, float*);
    };
    const MmaCfg cfgs[] = {{2, mma_peak_kernel<2>}, {4, mma_peak_kernel<4>}, {8, mma_peak_kernel<8>}};
    const int iters = quick ? 1024 : 4096;
    double best = 0;
    int best_warps = 0, best_chains = 0;
    printf("\"mma_sync_bf16_f32\": {\"sweep\": [");
    bool first_mma = true;
    for (const auto& k : cfgs) {
      for (int warps : {4, 8, 16, 32}) {
        const int block = std::min(warps, 8) * 32;
        const int grid = c.dev.num_sms * (warps * 32 / block);
        const double ms = median(time_reps(c.stream, reps, [&] { k.fn<<<grid, block, 0, c.stream>>>(iters, fsink); }));
        const double tflops = 4096.0 * k.chains * iters * (double)grid * (block / 32) / ms / 1e9;
        printf("%s{\"chains\": %d, \"warps_per_sm\": %d, \"tflops\": %.1f}", first_mma ? "" : ", ", k.chains, warps,
               tflops);
        first_mma = false;
        if (tflops > best) best = tflops, best_warps = warps, best_chains = k.chains;
      }
    }
    printf("], \"best_tflops\": %.1f, \"best_warps_per_sm\": %d, \"best_chains\": %d},\n", best, best_warps,
           best_chains);
  }

  // Launch floors: event-bracketed single empty kernel, back-to-back stream
  // launches, and kernels inside a CUDA graph (how decoding engines run).
  {
    const double single = median(time_reps(c.stream, 51, [&] { empty_kernel<<<1, 32, 0, c.stream>>>(); }));
    const int n = 1000;
    const double stream = median(time_reps(c.stream, 5, [&] {
                            for (int i = 0; i < n; ++i) empty_kernel<<<1, 32, 0, c.stream>>>();
                          })) /
                          n;
    const double graph =
        graph_per_launch_ms(c.stream, 200, 5, [&](int) { empty_kernel<<<c.dev.num_sms, 128, 0, c.stream>>>(); });
    printf(
        "\"launch\": {\"single_event_us\": %.3f, \"stream_back_to_back_us\": %.3f, \"graph_full_grid_us\": %.3f}\n",
        single * 1e3, stream * 1e3, graph * 1e3);
  }
  printf("}\n");
  CK(cudaStreamDestroy(c.stream));
  CK(cudaFree(pool));
  CK(cudaFree(c.sink));
  return 0;
}
