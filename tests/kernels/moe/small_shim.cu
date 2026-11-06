// SPDX-License-Identifier: Apache-2.0
// extern "C" entry points over the MoE decode kernels and the production
// moe_linear() dispatch for the Python test/bench harness (small_harness.py).
// Not part of the library.
#include <cuda_runtime.h>

#include "internal.h"
#include "moe/sm80/large_mma.h"
#include "moe/sm120/large_sm120.h"
#include "moe/sm100/large_tc.h"
#include "moe/schedule.cuh"
#include "moe/small.h"

using namespace nf12::detail;

namespace {

const DeviceInfo& current_device() {
  int d = 0;
  cudaGetDevice(&d);
  return device_info(d);
}

}  // namespace

extern "C" {

int nf12_moe_small_supported(int mt, int warps, int kc, int slice, int pdl, int kp, int64_t num_experts, int64_t n,
                             int64_t k) {
  return moe_small_config_supported(MoeSmallConfig{mt, warps, kc, slice, pdl, kp}, num_experts, n, k,
                                    current_device())
             ? 1
             : 0;
}

void nf12_moe_small_heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k, int* out) {
  const MoeSmallConfig c = moe_small_heuristic(max_assignments, num_experts, n, k, current_device());
  out[0] = c.mt;
  out[1] = c.warps;
  out[2] = c.kc;
  out[3] = c.slice;
  out[4] = c.pdl;
  out[5] = c.kp;
}

size_t nf12_moe_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k) {
  return moe_linear_workspace_bytes(max_assignments, num_experts, n, k, current_device());
}

// Grouped-GEMM family workspace: family 1 = mma, 2 = sm120, 3 = tc (heuristic configs).
size_t nf12_moe_large_workspace_bytes(int family, int64_t max_assignments, int64_t num_experts, int64_t n,
                                      int64_t k) {
  const DeviceInfo& dev = current_device();
  switch (family) {
    case 1: return moe_linear_large_mma_workspace_bytes(max_assignments, num_experts, n, k, dev);
    case 2: return moe_linear_large_sm120_workspace_bytes(max_assignments, num_experts, n, k, dev);
    case 3: return moe_large_tc_workspace_bytes(max_assignments, num_experts, n, k, dev);
    default: return 0;
  }
}

// mt == 0 runs production nf12::detail::moe_linear(). Negative mt forces
// grouped-GEMM family -mt (as in nf12_moe_large_workspace_bytes).
int nf12_moe_linear(int mt, int warps, int kc, int slice, int pdl, int kp, const void* x, int64_t ldx,
                    const int32_t* gather, int64_t gather_div, const int32_t* scatter, const int64_t* expert_offsets,
                    int64_t max_assignments, int source, const void* weight, const void* escapes,
                    const int64_t* escape_offsets, const uint8_t* exp_base, int64_t num_experts, int64_t n, int64_t k,
                    const float* row_scale, void* y, int64_t ldy, void* workspace, size_t workspace_bytes,
                    void* stream) {
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
  if (mt == 0) return static_cast<int>(moe_linear(a));
  if (mt == -1) return static_cast<int>(moe_linear_large_mma(a));
  if (mt == -2) return static_cast<int>(moe_linear_large_sm120(a));
  if (mt == -3) return static_cast<int>(moe_linear_large_tc(a));
  return static_cast<int>(moe_linear_small(a, MoeSmallConfig{mt, warps, kc, slice, pdl, kp}));
}

int nf12_moe_schedule(const int64_t* expert_offsets, int64_t num_experts, int64_t max_assignments, int bm,
                      void* workspace, void* stream) {
  return static_cast<int>(moe_schedule(expert_offsets, num_experts, max_assignments, bm, workspace,
                                       static_cast<cudaStream_t>(stream)));
}

size_t nf12_moe_schedule_bytes(int64_t max_assignments, int64_t num_experts, int bm) {
  return moe_schedule_bytes(max_assignments, num_experts, bm);
}

size_t nf12_moe_schedule_row_end_offset(int64_t max_assignments, int64_t num_experts, int bm) {
  return moe_schedule_row_end_offset(max_assignments, num_experts, bm);
}

}  // extern "C"
