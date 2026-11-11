// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "internal.h"

namespace nf12::detail {

struct GemvConfig {
  int warps;
  int rows;  // Output rows per CTA: 1 or 4. Zero disables this family.
};

GemvConfig gemv_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Requires validated single-token arguments and a supported configuration.
// Accumulates in FP32 directly from packed weights, without workspace.
cudaError_t linear_gemv(const LinearArgs& args, const GemvConfig& config);

}  // namespace nf12::detail
