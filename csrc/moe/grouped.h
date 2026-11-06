// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>

#include "internal.h"

namespace nf12::detail {

// Write only valid sorted positions, with row stride a.k. Offsets use
// moe_linear's clamped, monotone interpretation.
// Requires a.gather != nullptr and moe_linear() argument validation.
// SM90+ launches use programmatic stream serialization.
cudaError_t moe_gather_rows(const MoeLinearArgs& a, __nv_bfloat16* out);

}  // namespace nf12::detail
