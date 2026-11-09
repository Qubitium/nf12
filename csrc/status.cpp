// SPDX-License-Identifier: Apache-2.0
// Shared public status messages for CUDA and HIP.
#include "nf12/nf12.h"

extern "C" const char* nf12_status_string(nf12_status_t status) {
  switch (status) {
    case NF12_OK:
      return "NF12_OK";
    case NF12_ERROR_INVALID_ARGUMENT:
      return "NF12_ERROR_INVALID_ARGUMENT: shape, alignment or pointer violates SPEC.md";
    case NF12_ERROR_UNSUPPORTED:
      return "NF12_ERROR_UNSUPPORTED: device architecture or configuration not supported";
    case NF12_ERROR_INSUFFICIENT_WORKSPACE:
      return "NF12_ERROR_INSUFFICIENT_WORKSPACE";
    case NF12_ERROR_CUDA:
      return "NF12_ERROR_CUDA: see nf12_last_cuda_error()";
  }
  return "NF12: unknown status";
}
