// SPDX-License-Identifier: Apache-2.0
// extern "C" entry points over the tcgen05 grouped-GEMM moe_linear family for
// the Python test/bench harness (large_tc_harness.py). Not part of the library.
#include <cuda_runtime.h>

#include "internal.h"
#include "moe/sm100/large_tc.h"
#include "moe/schedule.cuh"

using namespace nf12::detail;

namespace {

const DeviceInfo& current_device() {
  int d = 0;
  cudaGetDevice(&d);
  return device_info(d);
}

}  // namespace

extern "C" {

int nf12_moe_tc_supported() { return moe_large_tc_supported(current_device()) ? 1 : 0; }

int nf12_moe_tc_num_configs() { return moe_large_tc_num_configs(); }

const char* nf12_moe_tc_config_name(int config) { return moe_large_tc_config_name(config); }

size_t nf12_moe_tc_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k, int config) {
  return moe_large_tc_workspace_bytes(max_assignments, num_experts, n, k, current_device(), config);
}

int nf12_moe_tc_linear(int config, const void* x, int64_t ldx, const int32_t* gather, int64_t gather_div,
                       const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments, int source,
                       const void* weight, const void* escapes, const int64_t* escape_offsets, const uint8_t* exp_base,
                       int64_t num_experts, int64_t n, int64_t k, const float* row_scale, void* y, int64_t ldy,
                       void* workspace, size_t workspace_bytes, void* stream) {
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
  a.escapes = static_cast<const uint4*>(escapes);
  a.escape_offsets = escape_offsets;
  a.exp_base = exp_base;
  a.num_experts = num_experts;
  a.n = n;
  a.k = k;
  a.row_scale = row_scale;
  a.y = static_cast<__nv_bfloat16*>(y);
  a.ldy = ldy;
  a.workspace = workspace;
  a.workspace_bytes = workspace_bytes;
  a.stream = static_cast<cudaStream_t>(stream);
  a.dev = &current_device();
  return static_cast<int>(moe_linear_large_tc(a, config));
}

// Raster band of the plan and the m-tile bound it must stay within (host arithmetic).
int nf12_test_moe_tc_band(int64_t budget, int64_t max_assignments, int64_t k, int bn) {
  return moe_large_tc_band(budget, max_assignments, k, bn);
}

int64_t nf12_test_moe_max_mtiles(int64_t max_assignments, int64_t num_experts, int bn) {
  return moe_max_mtiles(max_assignments, num_experts, bn);
}

int nf12_test_moe_ceil_div(int count, int bm) { return moe_ceil_div(count, bm); }

}  // extern "C"
