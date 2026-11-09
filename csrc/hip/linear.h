// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace nf12_hip {
// Internal strided entry points; public contiguous HIP entry points keep their ABI.
hipError_t linear_fused_strided(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                               const void* x, void* out, int64_t n, int64_t k, int64_t m,
                               const void* bias, int64_t ldx, int64_t ldy, bool bf16t,
                               hipStream_t stream);
hipError_t linear_wmma_strided(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                              const void* x, void* out, int64_t n, int64_t k, int64_t m,
                              const void* bias, void* workspace, size_t workspace_bytes,
                              int64_t ldx, int64_t ldy, bool bf16t, hipStream_t stream);
}  // namespace nf12_hip
