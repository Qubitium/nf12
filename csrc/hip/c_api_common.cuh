// SPDX-License-Identifier: Apache-2.0
//
// Shared state and argument validation for the ROCm CUDA-compatibility layer
// (c_api.hip, c_api_moe.hip).
//
// Adapted from GPT-QModel (https://github.com/ModelCloud/GPTQModel) by Qubitium.
//
// The predicates below are transliterations of csrc/api.cu's helpers, with the
// same names and the same clauses. Keeping them identical is the point: the C
// ABI is a contract, and a caller that is valid on CUDA must be valid here and
// rejected here for the same reasons. Before these existed, this layer forwarded
// pointers straight to the kernels, so a null weight segfaulted and a k that was
// not a multiple of 64 silently truncated the GEMM instead of being rejected.
#pragma once

#include "nf12/nf12.h"

#include <cstdint>
#include <limits>

namespace nf12_hip_capi {

// SPEC.md 2 caps exp_base here; csrc/api.cu calls the same constant kMaxExpBase.
inline constexpr uint8_t kMaxExpBase = 120;

// Thread-local so concurrent host threads never see each other's failure, and
// no lock is needed on the free-threaded build.
inline thread_local hipError_t t_last_error = hipSuccess;

// csrc/api.cu maps cudaErrorNotSupported/cudaErrorNoKernelImageForDevice to
// UNSUPPORTED and everything else to CUDA. hipErrorOutOfMemory is deliberately
// NOT mapped to INSUFFICIENT_WORKSPACE: the CUDA build reports a real allocation
// failure as NF12_ERROR_CUDA, and the explicit workspace-size checks below are
// what produce INSUFFICIENT_WORKSPACE on both backends.
inline nf12_status_t status(hipError_t err) {
  if (err == hipSuccess) return NF12_OK;
  // Native HIP entry points also report validation errors this way. Like
  // CUDA's from_codec(), these must not overwrite the last runtime error.
  if (err == hipErrorInvalidValue || err == hipErrorInvalidDevice) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  t_last_error = err;
  if (err == hipErrorNotSupported || err == hipErrorNoBinaryForGpu) {
    return NF12_ERROR_UNSUPPORTED;
  }
  return NF12_ERROR_CUDA;
}

inline bool valid_device(int device) {
  int count = 0;
  return device >= 0 && hipGetDeviceCount(&count) == hipSuccess && device < count;
}

inline bool aligned(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

inline bool aligned16(const void* p) { return aligned(p, 16); }

// Bound intermediate byte counts to int64, as csrc/api.cu does.
inline bool valid_shape(int64_t n, int64_t k) {
  return n > 0 && k > 0 && n % 16 == 0 && k % 64 == 0 &&
         n <= std::numeric_limits<int64_t>::max() / 4 / k;
}

// A row-major BF16 matrix the kernels read or write with 16-byte vectors.
inline bool valid_bf16_matrix(const void* p, int64_t cols, int64_t ld) {
  return p != nullptr && aligned16(p) && cols > 0 && ld >= cols && ld % 8 == 0;
}

inline bool valid_workspace(const void* workspace, size_t bytes) {
  return bytes == 0 || (workspace != nullptr && aligned16(workspace));
}

inline bool valid_weight(const nf12_weight_t* w) {
  return w != nullptr && valid_shape(w->n, w->k) && w->packed != nullptr &&
         aligned16(w->packed) && w->exp_base <= kMaxExpBase && aligned16(w->escapes) &&
         (w->num_escapes == 0 || w->escapes != nullptr);
}

inline bool valid_moe_shape(int64_t num_experts, int64_t n, int64_t k) {
  return num_experts >= 1 && valid_shape(n, k) &&
         num_experts <= std::numeric_limits<int64_t>::max() / 2 / n / k;
}

inline bool valid_moe_weight(const nf12_moe_weight_t* w) {
  return w != nullptr && valid_moe_shape(w->num_experts, w->n, w->k) && w->packed != nullptr &&
         aligned16(w->packed) && aligned16(w->escapes) && w->escape_offsets != nullptr &&
         aligned(w->escape_offsets, 8) && w->exp_base != nullptr;
}

}  // namespace nf12_hip_capi
