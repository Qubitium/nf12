// SPDX-License-Identifier: Apache-2.0
// Include the implementation so resource queries name the exact kernels
// used by this single-TU harness.
#include "linear/sm89/large_m_narrow.cu"
using namespace nf12::detail;
extern "C" int nf12_narrow_test(int config, int source, const void* x, int64_t m, int64_t ldx,
    const void* weight, const void* escapes, uint32_t num_escapes, uint32_t exp_base, int64_t n, int64_t k,
    const void* bias, void* y, int64_t ldy, void* stream) {
  static DeviceInfo dev = [] {
    DeviceInfo d{};
    cudaGetDevice(&d.device);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, d.device);
    d.sm_arch = p.major * 10 + p.minor;
    d.num_sms = p.multiProcessorCount;
    d.smem_optin_bytes = p.sharedMemPerBlockOptin;
    d.l2_bytes = p.l2CacheSize;
    return d;
  }();
  LinearArgs a{};
  a.x = static_cast<const __nv_bfloat16*>(x); a.m = m; a.ldx = ldx;
  a.source = source ? WeightSource::kBF16T : WeightSource::kNF12;
  a.weight = static_cast<const uint8_t*>(weight); a.escapes = static_cast<const uint4*>(escapes);
  a.num_escapes = num_escapes; a.exp_base = exp_base; a.n = n; a.k = k;
  a.bias = static_cast<const __nv_bfloat16*>(bias); a.y = static_cast<__nv_bfloat16*>(y); a.ldy = ldy;
  a.stream = static_cast<cudaStream_t>(stream); a.dev = &dev;
  return static_cast<int>(linear_large_m_sm8x_narrow(a, config));
}

template <int BM, int SPLITS, int SLOTS>
int narrow_stats(int* out) {
  constexpr auto kernel = narrow_kernel<BM, SPLITS, WeightSource::kNF12, SLOTS>;
  constexpr int bytes = SPLITS * SLOTS * BM * 128 + SPLITS * (BM / 8) * 32 * sizeof(float4);
  cudaFuncAttributes attr{};
  cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (err != cudaSuccess) return err;
  int blocks = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 32 * SPLITS, bytes);
  if (err != cudaSuccess) return err;
  out[0] = attr.numRegs; out[1] = bytes; out[2] = 32 * SPLITS;
  out[3] = blocks; out[4] = BM; out[5] = attr.localSizeBytes;
  return 0;
}

extern "C" int nf12_narrow_stats(int config, int64_t m, int64_t n, int64_t k, int* out) {
  DeviceInfo dev{};
  if (config < 0) config = narrow_sm8x_heuristic(m, n, k, dev);
  out[6] = config;
  switch (config) {
    case 6: return narrow_stats<16, 16, 2>(out);
    case 8: return narrow_stats<8, 16, 2>(out);
    case 9: return narrow_stats<8, 32, 2>(out);
    case 10: return narrow_stats<32, 16, 1>(out);
    default: return cudaErrorInvalidValue;
  }
}
