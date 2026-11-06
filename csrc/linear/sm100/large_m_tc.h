// SPDX-License-Identifier: Apache-2.0
// tcgen05/UMMA (SM100/103/110) family. Negative configs select the heuristic.
// Nonnegative configs force an entry of the config table.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// Requires SM100/103/110 and the architecture-specific tcgen05 kernel body
// in the image loaded for this device.
bool large_m_tc_supported(const DeviceInfo& dev);
int large_m_tc_num_configs();
const char* large_m_tc_config_name(int config);
size_t linear_large_m_tc_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config);
cudaError_t linear_large_m_tc(const LinearArgs& args, int config);

}  // namespace nf12::detail
