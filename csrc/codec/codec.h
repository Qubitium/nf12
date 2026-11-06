// SPDX-License-Identifier: Apache-2.0
// Exposes codec stages for overflow tests without allocating ~100 GB inputs.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nf12::codec {

// In-place exclusive scan of v[0, n), with the total in v[n].
// Use 64-bit counts because N*K/8 can exceed 2^32 on supported devices.
cudaError_t escape_offsets_scan(uint64_t* v, int64_t n, cudaStream_t stream);

inline constexpr int64_t kMaxGridX = 0x7fffffff;

}
