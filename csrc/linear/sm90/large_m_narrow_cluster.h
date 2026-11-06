// SPDX-License-Identifier: Apache-2.0
// Cluster-reduced MMA for narrow router weights (dispatch.cu use_narrow()).
#pragma once

#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// Requires SM90, SM100/103 or SM120/121 and an SM90+ body in the loaded image.
bool narrow_sm120_supported(const DeviceInfo& dev);
// config -1 selects the heuristic. Otherwise tf | (wi << 2) | (splits << 4)
// encodes 8 << tf tokens and 1 << wi warps (16-row blocks) per CTA, with
// K split over `splits` CTAs (1-8, or up to 16 on SM90).
// narrow_sm120_heuristic returns the same encoding. No workspace.
int narrow_sm120_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);
cudaError_t linear_narrow_sm120(const LinearArgs& args, int config);

}  // namespace nf12::detail
