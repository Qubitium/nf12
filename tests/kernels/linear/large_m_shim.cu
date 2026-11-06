// SPDX-License-Identifier: Apache-2.0
// extern "C" entry points over the large-M kernel families for the Python
// test/bench harness (large_m_harness.py). Not part of the library.
#include <cuda_runtime.h>

#include "linear/large_m.h"
#include "linear/sm80/large_m_mma.h"
#include "linear/sm100/large_m_tc.h"

using namespace nf12::detail;

namespace {

const DeviceInfo& local_device_info() {
  static DeviceInfo info = [] {
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
  return info;
}

}  // namespace

extern "C" {

// Family 0 = mma.sync, 1 = tcgen05, 2 = production linear_large_m() (config ignored).
int nf12_test_large_m_supported(int family) {
  return family == 1 ? large_m_tc_supported(local_device_info()) : 1;
}

int nf12_test_large_m_num_configs(int family) {
  return family == 0 ? large_m_mma_num_configs() : family == 1 ? large_m_tc_num_configs() : 1;
}

const char* nf12_test_large_m_config_name(int family, int config) {
  return family == 0 ? large_m_mma_config_name(config) : family == 1 ? large_m_tc_config_name(config) : "auto";
}

int nf12_test_tile_offset_in_range(int base, int offset, int limit) {
  return tile_offset_in_range(base, offset, limit);
}

int nf12_test_large_m_heuristic(int64_t m, int64_t n, int64_t k) {
  return large_m_mma_heuristic(m, n, k, local_device_info());
}

size_t nf12_test_large_m_ws(int family, int config, int64_t m, int64_t n, int64_t k) {
  const DeviceInfo& dev = local_device_info();
  if (family == 0) return linear_large_m_mma_workspace_bytes(m, n, k, dev, config);
  if (family == 1) return linear_large_m_tc_workspace_bytes(m, n, k, dev, config);
  return linear_large_m_workspace_bytes(m, n, k, dev);
}

int nf12_test_large_m(int family, int config, const void* x, int64_t m, int64_t ldx, int source, const void* weight,
                      const void* escapes, uint32_t num_escapes, uint32_t exp_base, int64_t n, int64_t k,
                      const void* bias, void* y, int64_t ldy, void* workspace, size_t workspace_bytes, void* stream) {
  LinearArgs a{};
  a.x = static_cast<const __nv_bfloat16*>(x);
  a.m = m;
  a.ldx = ldx;
  a.source = source == 0 ? WeightSource::kNF12 : WeightSource::kBF16T;
  a.weight = static_cast<const uint8_t*>(weight);
  a.escapes = static_cast<const uint4*>(escapes);
  a.num_escapes = source == 0 ? num_escapes : 0;
  a.exp_base = exp_base;
  a.n = n;
  a.k = k;
  a.bias = static_cast<const __nv_bfloat16*>(bias);
  a.y = static_cast<__nv_bfloat16*>(y);
  a.ldy = ldy;
  a.workspace = workspace;
  a.workspace_bytes = workspace_bytes;
  a.stream = static_cast<cudaStream_t>(stream);
  a.dev = &local_device_info();
  cudaError_t err = family == 0   ? linear_large_m_mma(a, config)
                    : family == 1 ? linear_large_m_tc(a, config)
                                  : linear_large_m(a);
  return static_cast<int>(err);
}


#ifdef NF12_SHIM_DECODE
// Codec's decode (csrc/codec/decode.cu), for the decode + cuBLAS baseline.
int nf12_test_decode(const void* packed, const void* escapes, uint32_t exp_base, int64_t n, int64_t k, void* out,
                     void* stream) {
  return static_cast<int>(decode(static_cast<const uint8_t*>(packed), static_cast<const uint4*>(escapes), exp_base,
                                 n, k, static_cast<__nv_bfloat16*>(out), k, static_cast<cudaStream_t>(stream)));
}
#endif

}  // extern "C"
