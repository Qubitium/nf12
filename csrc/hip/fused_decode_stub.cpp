// SPDX-License-Identifier: Apache-2.0
//
// The fused single-token MoE fast path (csrc/torch/fused_decode.cu) is a CUDA
// kernel. On ROCm ops.cpp falls back to the composed moe_* path, so this stub
// reports the fast path unavailable instead of duplicating it.
#include "torch/fused_decode.h"

namespace nf12::torch_detail {

bool fused_decode_supported(int, const FusedDecodeArgs&) { return false; }

cudaError_t fused_decode(const FusedDecodeArgs&, cudaStream_t) {
  return cudaErrorNotSupported;
}

}  // namespace nf12::torch_detail
