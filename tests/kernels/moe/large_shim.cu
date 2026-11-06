// SPDX-License-Identifier: Apache-2.0
// extern "C" entry points over the grouped (MoE) prefill families for the
// Python harness (large_harness.py): any family and forced config. Not part
// of the library.
#include <cuda_runtime.h>

#include "moe/sm80/large_mma.h"
#include "moe/sm120/large_sm120.h"
#include "moe/sm90/large_sm90.h"
#include "moe/schedule.cuh"

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

constexpr int kFamilyMma = 0, kFamilySm120 = 1, kFamilySm90 = 2;

}  // namespace

extern "C" {

int nf12_test_moe_supported(int family, int64_t num_experts) {
  const DeviceInfo& dev = local_device_info();
  return family == kFamilyMma     ? dev.sm_arch >= 80
         : family == kFamilySm120 ? moe_large_sm120_supported(dev, num_experts)
                                  : moe_large_sm90_supported(dev, num_experts);
}

int nf12_test_moe_num_configs(int family) {
  return family == kFamilyMma     ? moe_large_mma_num_configs()
         : family == kFamilySm120 ? moe_large_sm120_num_configs()
                                  : moe_large_sm90_num_configs();
}

const char* nf12_test_moe_config_name(int family, int config) {
  return family == kFamilyMma     ? moe_large_mma_config_name(config)
         : family == kFamilySm120 ? moe_large_sm120_config_name(config)
                                  : moe_large_sm90_config_name(config);
}

int nf12_test_moe_heuristic(int family, int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k) {
  const DeviceInfo& dev = local_device_info();
  return family == kFamilyMma     ? moe_large_mma_heuristic(max_assignments, num_experts, n, k, dev)
         : family == kFamilySm120 ? moe_large_sm120_heuristic(max_assignments, num_experts, n, k, dev)
                                  : moe_large_sm90_heuristic(max_assignments, num_experts, n, k, dev);
}

int nf12_test_moe_large(int family, int config, const void* x, int64_t ldx, const int32_t* gather, int64_t gather_div,
                        const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments, int source,
                        const void* weight, const void* escapes, const int64_t* escape_offsets,
                        const uint8_t* exp_base, int64_t num_experts, int64_t n, int64_t k, const float* row_scale,
                        void* y, int64_t ldy, void* stream) {
  MoeLinearArgs a{};
  a.x = static_cast<const __nv_bfloat16*>(x);
  a.ldx = ldx;
  a.gather = gather;
  a.gather_div = gather_div;
  a.scatter = scatter;
  a.expert_offsets = expert_offsets;
  a.max_assignments = max_assignments;
  a.source = source == 0 ? WeightSource::kNF12 : WeightSource::kBF16T;
  a.weight = static_cast<const uint8_t*>(weight);
  a.escapes = source == 0 ? static_cast<const uint4*>(escapes) : nullptr;
  a.escape_offsets = source == 0 ? escape_offsets : nullptr;
  a.exp_base = source == 0 ? exp_base : nullptr;
  a.num_experts = num_experts;
  a.n = n;
  a.k = k;
  a.row_scale = row_scale;
  a.y = static_cast<__nv_bfloat16*>(y);
  a.ldy = ldy;
  a.stream = static_cast<cudaStream_t>(stream);
  a.dev = &local_device_info();
  const cudaError_t err = family == kFamilyMma     ? moe_linear_large_mma(a, config)
                          : family == kFamilySm120 ? moe_linear_large_sm120(a, config)
                                                   : moe_linear_large_sm90(a, config);
  return static_cast<int>(err != cudaSuccess ? err : cudaGetLastError());
}

// The m-tile ceiling division every family's schedule uses (moe/schedule.cuh).
int nf12_test_moe_ceil_div(int count, int bm) { return moe_ceil_div(count, bm); }

}  // extern "C"
