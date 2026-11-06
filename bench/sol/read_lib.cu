// SPDX-License-Identifier: Apache-2.0
// ctypes streaming reads, timed alongside NF12 with the same graph and rotation.
// The harness selects the fastest configuration per weight.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "read_kernels.cuh"

namespace {

using namespace nf12_bench;

using ReadFn = void (*)(const uint4*, size_t, uint32_t*);

struct Config {
  ReadFn fn;
  int ld, pattern, unroll, block, ctas_per_sm;
};

const Config kConfigs[] = {
    {read_kernel<kLdNcNoAlloc256, 4, kGridStride>, kLdNcNoAlloc256, kGridStride, 4, 1024, 1},
    {read_kernel<kLdNcNoAlloc256, 8, kGridStride>, kLdNcNoAlloc256, kGridStride, 8, 1024, 1},
    {read_kernel<kLdDefault, 4, kGridStride>, kLdDefault, kGridStride, 4, 1024, 1},
    {read_kernel<kLdDefault, 4, kGridStride>, kLdDefault, kGridStride, 4, 256, 8},
    {read_kernel<kLdNcNoAlloc, 4, kGridStride>, kLdNcNoAlloc, kGridStride, 4, 256, 4},
    {read_kernel<kLdNcNoAlloc, 2, kGridStride>, kLdNcNoAlloc, kGridStride, 2, 256, 2},
    {read_kernel<kLdNcNoAlloc, 1, kGridStride>, kLdNcNoAlloc, kGridStride, 1, 512, 2},
    {read_kernel<kLdNc, 2, kGridStride>, kLdNc, kGridStride, 2, 256, 2},
    {read_kernel<kLdDefault, 1, kWarpSlab>, kLdDefault, kWarpSlab, 1, 256, 1},
    {read_kernel<kLdNcNoAlloc, 2, kWarpSlab>, kLdNcNoAlloc, kWarpSlab, 2, 128, 2},
    {read_kernel<kLdNcNoAlloc, 2, kWarpSlab>, kLdNcNoAlloc, kWarpSlab, 2, 256, 2},
    {read_kernel<kLdNcNoAlloc, 4, kWarpSlab>, kLdNcNoAlloc, kWarpSlab, 4, 1024, 1},
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

}  // namespace

extern "C" {

int nf12bench_read_num_configs() { return kNumConfigs; }

// Writes "ld pattern xU blockxCTAs/SM" into buf.
int nf12bench_read_describe(int cfg, char* buf, int len) {
  if (cfg < 0 || cfg >= kNumConfigs) return -1;
  const Config& c = kConfigs[cfg];
  return snprintf(buf, len, "%s %s x%d %dx%d", kLdNames[c.ld], kPatternNames[c.pattern], c.unroll, c.block,
                  c.ctas_per_sm);
}

// Reads `bytes` (a multiple of 512) at `p` on `stream`; `sink` is a 4-byte
// device word the kernel never actually writes. Returns a cudaError_t.
int nf12bench_read(const void* p, size_t bytes, int cfg, uint32_t* sink, cudaStream_t stream) {
  if (cfg < 0 || cfg >= kNumConfigs || bytes % 512) return cudaErrorInvalidValue;
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const Config& c = kConfigs[cfg];
  c.fn<<<sms * c.ctas_per_sm, c.block, 0, stream>>>(static_cast<const uint4*>(p), bytes / 16, sink);
  return cudaGetLastError();
}

}  // extern "C"
