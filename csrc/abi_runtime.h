// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "backend.h"

namespace nf12::abi {
inline thread_local cudaError_t last_error = cudaSuccess;

inline nf12_status_t from_runtime(cudaError_t err) {
  if (err == cudaSuccess) return NF12_OK;
  last_error = err;
  if (err == cudaErrorNotSupported || err == nf12::backend::kNoKernelImage) return NF12_ERROR_UNSUPPORTED;
  return NF12_ERROR_CUDA;
}

// A capability rejection is not a runtime failure and must not erase a saved error.
inline nf12_status_t from_device(cudaError_t err, bool supported) {
  return err != cudaSuccess ? from_runtime(err) : (supported ? NF12_OK : NF12_ERROR_UNSUPPORTED);
}

// Codec invalid-value errors mean escape capacity/index or grid overflow
// after argument validation. Packed and escape outputs remain untouched.
inline nf12_status_t from_codec(cudaError_t err) {
  return err == nf12::backend::kInvalidValue ? NF12_ERROR_INVALID_ARGUMENT : from_runtime(err);
}

}  // namespace nf12::abi
