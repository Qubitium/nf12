// SPDX-License-Identifier: Apache-2.0
//
// Mixture-of-experts half of the ROCm CUDA-compatibility layer.
//
// csrc/torch/ops.cpp is backend-agnostic and calls the nf12_* C API from
// include/nf12/nf12.h; these definitions back the MoE ops with the HIP kernels
// in moe.hip and wmma.hip, so the torch op schema is reused unchanged.
//
// Adapted from GPT-QModel (https://github.com/ModelCloud/GPTQModel) by Qubitium.

#include "c_api_common.cuh"

#include "nf12/nf12.h"
#include "nf12/nf12_hip.h"

using nf12_hip_capi::status;

extern "C" {

size_t nf12_moe_align_workspace_bytes(int64_t num_tokens, int64_t topk, int64_t num_experts) {
  if (num_tokens < 0 || topk < 1 || num_experts < 1) return 0;
  if (num_tokens > 0x7fffffffLL / topk) return 0;
  return nf12_hip_moe_align_workspace_bytes(num_tokens * topk, num_experts);
}

nf12_status_t nf12_moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk,
                             int64_t num_experts, int32_t* sorted_out,
                             int64_t* expert_offsets_out, void* workspace,
                             size_t workspace_bytes, cudaStream_t stream) {
  if (expert_offsets_out == nullptr) return NF12_ERROR_INVALID_ARGUMENT;
  if (num_tokens > 0 && (topk_ids == nullptr || sorted_out == nullptr)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip_moe_align(topk_ids, num_tokens, topk, num_experts, sorted_out,
                                   expert_offsets_out, workspace, workspace_bytes, stream));
}

// The native routed kernel serves every shape, so the decode + external grouped
// GEMM route is never selected. That keeps the MoE path free of a dependency on
// aten::_grouped_mm, which ROCm does not implement.
size_t nf12_moe_linear_workspace_bytes(int64_t, int64_t, int64_t, int64_t, int) { return 0; }

int64_t nf12_moe_linear_decode_gemm_rows(int64_t, int64_t, int64_t, int64_t, int) { return 0; }

nf12_status_t nf12_moe_linear(const void* x_bf16, int64_t ldx, const int32_t* gather,
                              int64_t gather_div, const int32_t* scatter,
                              const int64_t* expert_offsets, int64_t max_assignments,
                              const nf12_moe_weight_t* w, const float* row_scale, void* y_bf16,
                              int64_t ldy, void* workspace, size_t workspace_bytes,
                              cudaStream_t stream) {
  (void)workspace;
  (void)workspace_bytes;
  if (w == nullptr || w->packed == nullptr || w->exp_base == nullptr ||
      w->escape_offsets == nullptr || w->num_experts < 1 || w->n <= 0 || w->k <= 0) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (gather_div < 1 || ldx < w->k || ldy < w->n) return NF12_ERROR_INVALID_ARGUMENT;
  if (w->num_experts > 0x7fffffffLL) return NF12_ERROR_INVALID_ARGUMENT;
  return status(nf12_hip_moe_linear(
      static_cast<const uint8_t*>(w->packed), static_cast<const uint4*>(w->escapes),
      w->escape_offsets, w->exp_base, static_cast<int32_t>(w->num_experts), x_bf16, ldx, gather,
      gather_div, scatter, expert_offsets, max_assignments, row_scale, y_bf16, ldy, w->n, w->k,
      stream));
}

nf12_status_t nf12_moe_linear_bf16t(const void* x_bf16, int64_t ldx, const int32_t* gather,
                                    int64_t gather_div, const int32_t* scatter,
                                    const int64_t* expert_offsets, int64_t max_assignments,
                                    const void* w_bf16t, int64_t num_experts, int64_t n, int64_t k,
                                    const float* row_scale, void* y_bf16, int64_t ldy,
                                    void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  (void)workspace;
  (void)workspace_bytes;
  if (w_bf16t == nullptr || num_experts < 1 || num_experts > 0x7fffffffLL || n <= 0 || k <= 0) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (gather_div < 1 || ldx < k || ldy < n) return NF12_ERROR_INVALID_ARGUMENT;
  return status(nf12_hip_moe_linear_bf16t(
      static_cast<const uint8_t*>(w_bf16t), static_cast<int32_t>(num_experts), x_bf16, ldx, gather,
      gather_div, scatter, expert_offsets, max_assignments, row_scale, y_bf16, ldy, n, k, stream));
}

nf12_status_t nf12_moe_combine(const void* y_bf16, int64_t ldy, int64_t num_tokens, int64_t topk,
                               int64_t n, void* out_bf16, int64_t ldo, cudaStream_t stream) {
  return status(nf12_hip_moe_combine(y_bf16, ldy, num_tokens, topk, n, out_bf16, ldo, stream));
}

nf12_status_t nf12_moe_silu_mul(const void* h_bf16, int64_t ldh, int64_t rows, int64_t inter,
                                void* out_bf16, int64_t ldo, cudaStream_t stream) {
  return status(nf12_hip_moe_silu_mul(h_bf16, ldh, rows, inter, out_bf16, ldo, stream));
}

nf12_status_t nf12_moe_decode(const nf12_moe_weight_t* w, const int64_t* expert_offsets,
                              int64_t max_assignments, int64_t row0, int64_t rows, void* out_bf16,
                              int64_t ldo, cudaStream_t stream) {
  if (w == nullptr || w->packed == nullptr || w->exp_base == nullptr || w->num_experts < 1 ||
      w->n <= 0 || w->k <= 0 || w->num_experts > 0xffff) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (w->escapes != nullptr && w->escape_offsets == nullptr) return NF12_ERROR_INVALID_ARGUMENT;
  return status(nf12_hip_moe_decode(static_cast<const uint8_t*>(w->packed),
                                    static_cast<const uint4*>(w->escapes), w->escape_offsets,
                                    w->exp_base, static_cast<int32_t>(w->num_experts), w->n, w->k,
                                    expert_offsets, max_assignments, row0, rows, out_bf16, ldo,
                                    stream));
}

nf12_status_t nf12_moe_grouped_operands(const void* x_bf16, int64_t x_ld, const int32_t* gather,
                                        int64_t gather_div, const int64_t* expert_offsets,
                                        int64_t max_assignments, int64_t num_experts, int64_t k,
                                        void* x_out_bf16, int64_t x_out_ld, int32_t* group_ends,
                                        cudaStream_t stream) {
  if (x_ld < k || x_out_ld < k || gather_div < 1) return NF12_ERROR_INVALID_ARGUMENT;
  return status(nf12_hip_moe_grouped_operands(x_bf16, x_ld, gather, gather_div, expert_offsets,
                                              max_assignments, num_experts, k, x_out_bf16,
                                              x_out_ld, group_ends, stream));
}

}  // extern "C"
