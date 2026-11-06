// SPDX-License-Identifier: Apache-2.0
// SM80+ with no split-K or workspace.
// Select heuristically with config < 0, or force a table entry with config >= 0.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

int moe_large_mma_num_configs();
const char* moe_large_mma_config_name(int config);
int moe_large_mma_heuristic(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                            const DeviceInfo& dev);
size_t moe_linear_large_mma_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                            const DeviceInfo& dev, int config = -1);
// Requires moe_linear() argument validation and SM80+.
cudaError_t moe_linear_large_mma(const MoeLinearArgs& args, int config = -1);

}  // namespace nf12::detail
