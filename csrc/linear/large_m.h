// SPDX-License-Identifier: Apache-2.0
// Device-family launcher for compute-regime linear. dispatch.cu owns the crossover.
#pragma once

#include <cstddef>
#include <cstdint>

#include "internal.h"

namespace nf12::detail {

// Workspace bytes for (m, n, k) on `dev`, or 0 for an unsplit config.
size_t linear_large_m_workspace_bytes(int64_t m, int64_t n, int64_t k, const DeviceInfo& dev);

// Requires args.m >= 1, n % 16 == 0, k % 64 == 0, 16-byte aligned x/y rows
// (ldx and ldy multiples of 8), and args.workspace_bytes >=
// linear_large_m_workspace_bytes(...).
cudaError_t linear_large_m(const LinearArgs& args);

}  // namespace nf12::detail
