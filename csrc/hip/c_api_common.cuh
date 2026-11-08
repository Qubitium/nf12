// SPDX-License-Identifier: Apache-2.0
//
// Shared state for the ROCm CUDA-compatibility layer (c_api.hip, c_api_moe.hip).
//
// Adapted from GPT-QModel (https://github.com/ModelCloud/GPTQModel) by Qubitium.
#pragma once

#include "nf12/nf12.h"

namespace nf12_hip_capi {

// Thread-local so concurrent host threads never see each other's failure, and
// no lock is needed on the free-threaded build.
inline thread_local hipError_t t_last_error = hipSuccess;

inline nf12_status_t status(hipError_t err) {
  t_last_error = err;
  switch (err) {
    case hipSuccess:
      return NF12_OK;
    case hipErrorInvalidValue:
    case hipErrorInvalidDevice:
      return NF12_ERROR_INVALID_ARGUMENT;
    case hipErrorNotSupported:
      return NF12_ERROR_UNSUPPORTED;
    case hipErrorOutOfMemory:
      return NF12_ERROR_INSUFFICIENT_WORKSPACE;
    default:
      return NF12_ERROR_CUDA;
  }
}

}  // namespace nf12_hip_capi
