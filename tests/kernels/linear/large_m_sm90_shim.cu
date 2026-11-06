// SPDX-License-Identifier: Apache-2.0
// extern "C" entry points over the SM90 large-M kernel family for the Python
// test/bench harness (large_m_sm90_test.py, large_m_sm90_bench.py). Built
// next to large_m_shim.cu via NF12_EXTRA_SRCS. Not part of the library.
#include <cuda_runtime.h>

#include "linear/sm90/large_m_sm90.h"

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

int nf12_test_sm90_supported() { return large_m_sm90_supported(local_device_info()); }

int nf12_test_sm90_num_configs() { return large_m_sm90_num_configs(); }

int nf12_test_sm90_heuristic(int64_t m, int64_t n, int64_t k) {
  return large_m_sm90_heuristic(m, n, k, local_device_info());
}

const char* nf12_test_sm90_config_name(int config) { return large_m_sm90_config_name(config); }

size_t nf12_test_sm90_ws(int config, int64_t m, int64_t n, int64_t k) {
  return linear_large_m_sm90_workspace_bytes(m, n, k, local_device_info(), config);
}

int nf12_test_sm90(int config, const void* x, int64_t m, int64_t ldx, int source, const void* weight,
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
  return static_cast<int>(linear_large_m_sm90(a, config));
}

}  // extern "C"
