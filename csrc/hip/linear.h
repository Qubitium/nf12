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

// CDNA3 matrix-core routes (mfma.hip). Selected by trait, never by arch string.
hipError_t linear_mfma_strided(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                              const void* x, void* out, int64_t n, int64_t k, int64_t m,
                              const void* bias, int64_t ldx, int64_t ldy, bool bf16t,
                              hipStream_t stream);
hipError_t moe_linear_mfma(const uint8_t* packed, const uint4* escapes, const int64_t* escape_offsets,
                           const uint8_t* exp_base, int32_t num_experts, const void* x, int64_t ldx,
                           const int32_t* gather, int64_t gather_div, const int32_t* scatter,
                           const int64_t* expert_offsets, int64_t max_assignments,
                           const float* row_scale, void* out, int64_t ldy, int64_t n, int64_t k,
                           bool bf16t, hipStream_t stream);
int mfma_max_m();
}  // namespace nf12_hip
