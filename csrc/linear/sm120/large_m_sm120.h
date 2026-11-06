// SPDX-License-Identifier: Apache-2.0
// SM120 (RTX Blackwell) large-M kernel family: warp-specialized, TMA-fed,
// persistent mma.sync GEMM that decodes NF12 straight into A fragments.
// `config < 0` selects the heuristic. Nonnegative configs force entry
// `config % 16` with schedule `config / 16 % 16` (0 = heuristic hybrid,
// 1 = all tiles data-parallel, 2 = all tiles stream-K, 3 = every tile split
// k-wise over `config / 256` CTAs).
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// True when `dev` is an SM120/SM121 GPU and the loaded image carries the
// arch-specific (sm_120a/sm_120f/sm_121a) kernel body.
bool large_m_sm120_supported(const DeviceInfo& dev);
int large_m_sm120_num_configs();
int large_m_sm120_heuristic(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);  // encoded as above
const char* large_m_sm120_config_name(int config);
size_t linear_large_m_sm120_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev, int config);
// Requires linear_large_m() preconditions and args.workspace_bytes >=
// linear_large_m_sm120_workspace_bytes(...).
cudaError_t linear_large_m_sm120(const LinearArgs& args, int config);

}  // namespace nf12::detail
