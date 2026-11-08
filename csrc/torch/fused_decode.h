// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "nf12/nf12.h"
#include <cstdint>

namespace nf12::torch_detail {

struct FusedDecodeWeight {
  const void* packed;
  const void* escapes;
  const int64_t* escape_offsets;
  const uint8_t* exp_base;
  int64_t experts, n, k;
  bool bf16t;
};

struct FusedDecodeArgs {
  FusedDecodeWeight w13, w2;
  const void* x;
  const int32_t* ids;
  const float* scales;
  void* activation;
  void* out;
  int64_t topk, ldx, ldids, ldscales, ldactivation, ldout;
};

// The caller guards device before preflight. Output pointers may be null during
// preflight; their intended strides must already be supplied. No device work,
// allocation or synchronization is performed. Capability preparation is cached.
bool fused_decode_supported(int device, const FusedDecodeArgs& args);

// Both outputs must be nonnull. Preparation completes before either launch;
// launch errors propagate, and callers must not fall back after this call.
cudaError_t fused_decode(const FusedDecodeArgs& args, cudaStream_t stream);

}  // namespace nf12::torch_detail
