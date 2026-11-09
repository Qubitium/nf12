// SPDX-License-Identifier: Apache-2.0
// CUDA adapter for the shared C ABI frontend. Launch policy stays in detail.
#include "backend.h"
#include "internal.h"

namespace nf12::backend {
namespace {

constexpr int kMinSmArch = 80;

detail::WeightSource native_source(WeightSource source) {
  return source == WeightSource::kNF12 ? detail::WeightSource::kNF12 : detail::WeightSource::kBF16T;
}

}  // namespace

cudaError_t current_device(int* device, bool* supported) {
  *supported = false;
  if (cudaError_t err = cudaGetDevice(device); err != cudaSuccess) return err;
  *supported = detail::device_info(*device).sm_arch >= kMinSmArch;
  return cudaSuccess;
}

bool device_supported(int device) {
  if (device < 0) return false;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return device < count && detail::device_info(device).sm_arch >= kMinSmArch;
}

size_t linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int device) {
  return detail::linear_workspace_bytes(m, n, k, detail::device_info(device));
}

cudaError_t linear(const LinearArgs& a) {
  const detail::LinearArgs native{
      static_cast<const __nv_bfloat16*>(a.x), a.m, a.ldx, native_source(a.source), a.weight,
      static_cast<const uint4*>(a.escapes), a.num_escapes, a.exp_base, a.n, a.k,
      static_cast<const __nv_bfloat16*>(a.bias), static_cast<__nv_bfloat16*>(a.y), a.ldy,
      a.workspace, a.workspace_bytes, a.stream, &detail::device_info(a.device)};
  return detail::linear(native);
}

int64_t linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, int device) {
  return detail::linear_decode_gemm_rows(m, n, k, detail::device_info(device));
}

size_t encode_workspace_bytes(int64_t n, int64_t k) {
  return detail::encode_workspace_bytes(n, k);
}

cudaError_t encode_analyze(const void* w, int64_t n, int64_t k, int64_t ldw, void* workspace,
                           size_t workspace_bytes, uint8_t* exp_base_out, uint32_t* num_escapes_out,
                           cudaStream_t stream) {
  return detail::encode_analyze(static_cast<const __nv_bfloat16*>(w), n, k, ldw, workspace,
                                workspace_bytes, exp_base_out, num_escapes_out, stream);
}

cudaError_t encode(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base, uint8_t* packed,
                   void* escapes, uint32_t escapes_capacity, void* workspace, size_t workspace_bytes,
                   cudaStream_t stream) {
  return detail::encode(static_cast<const __nv_bfloat16*>(w), n, k, ldw, exp_base, packed,
                        static_cast<uint4*>(escapes), escapes_capacity, workspace, workspace_bytes, stream);
}

cudaError_t decode(const uint8_t* packed, const void* escapes, uint32_t exp_base, int64_t n, int64_t k,
                   void* out, int64_t ldo, cudaStream_t stream) {
  return detail::decode(packed, static_cast<const uint4*>(escapes), exp_base, n, k,
                        static_cast<__nv_bfloat16*>(out), ldo, stream);
}

cudaError_t pack_bf16t(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t* out,
                       cudaStream_t stream) {
  return detail::pack_bf16t(static_cast<const __nv_bfloat16*>(w), n, k, ldw, out, stream);
}

cudaError_t moe_decode(const uint8_t* packed, const void* escapes, const int64_t* escape_offsets,
                       const uint8_t* exp_base, int64_t num_experts, int64_t n, int64_t k,
                       const int64_t* expert_offsets, int64_t max_assignments, int64_t row0, int64_t rows,
                       void* out, int64_t ldo, cudaStream_t stream) {
  return detail::moe_decode(packed, static_cast<const uint4*>(escapes), escape_offsets, exp_base,
                            num_experts, n, k, expert_offsets, max_assignments, row0, rows,
                            static_cast<__nv_bfloat16*>(out), ldo, stream);
}

size_t moe_align_workspace_bytes(int64_t num_tokens, int64_t topk, int64_t num_experts) {
  return detail::moe_align_workspace_bytes(num_tokens, topk, num_experts);
}

cudaError_t moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk, int64_t num_experts,
                      int32_t* sorted, int64_t* expert_offsets, void* workspace, size_t workspace_bytes,
                      cudaStream_t stream) {
  return detail::moe_align(topk_ids, num_tokens, topk, num_experts, sorted, expert_offsets,
                           workspace, workspace_bytes, stream);
}

size_t moe_linear_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                  int device) {
  return detail::moe_linear_workspace_bytes(max_assignments, num_experts, n, k, detail::device_info(device));
}

cudaError_t moe_linear(const MoeLinearArgs& a) {
  const detail::MoeLinearArgs native{
      static_cast<const __nv_bfloat16*>(a.x), a.ldx, a.gather, a.gather_div, a.scatter, a.expert_offsets,
      a.max_assignments, native_source(a.source), a.weight, static_cast<const uint4*>(a.escapes),
      a.escape_offsets, a.exp_base, a.num_experts, a.n, a.k, a.row_scale,
      static_cast<__nv_bfloat16*>(a.y), a.ldy, a.workspace, a.workspace_bytes, a.stream,
      &detail::device_info(a.device)};
  return detail::moe_linear(native);
}

cudaError_t moe_combine(const void* y, int64_t ldy, int64_t num_tokens, int64_t topk, int64_t n,
                        void* out, int64_t ldo, cudaStream_t stream) {
  return detail::moe_combine(static_cast<const __nv_bfloat16*>(y), ldy, num_tokens, topk, n,
                             static_cast<__nv_bfloat16*>(out), ldo, stream);
}

cudaError_t moe_silu_mul(const void* h, int64_t ldh, int64_t rows, int64_t inter, void* out,
                         int64_t ldo, cudaStream_t stream) {
  return detail::moe_silu_mul(static_cast<const __nv_bfloat16*>(h), ldh, rows, inter,
                              static_cast<__nv_bfloat16*>(out), ldo, stream);
}

int64_t moe_linear_decode_gemm_rows(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k,
                                    int device) {
  return detail::moe_linear_decode_gemm_rows(max_assignments, num_experts, n, k, detail::device_info(device));
}

cudaError_t moe_grouped_operands(const void* x, int64_t ldx, const int32_t* gather, int64_t gather_div,
                                 const int64_t* expert_offsets, int64_t max_assignments, int64_t num_experts,
                                 int64_t k, void* x_out, int64_t ldo, int32_t* group_ends,
                                 cudaStream_t stream) {
  return detail::moe_grouped_operands(static_cast<const __nv_bfloat16*>(x), ldx, gather, gather_div,
                                      expert_offsets, max_assignments, num_experts, k,
                                      static_cast<__nv_bfloat16*>(x_out), ldo, group_ends, stream);
}

cudaError_t embedding(const uint8_t* packed, const void* escapes, uint32_t num_escapes, uint32_t exp_base,
                      int64_t n, int64_t k, const int32_t* ids32, const int64_t* ids64, int64_t num_ids,
                      void* out, int64_t ldo, cudaStream_t stream) {
  return detail::embedding(packed, static_cast<const uint4*>(escapes), num_escapes, exp_base, n, k,
                           ids32, ids64, num_ids, static_cast<__nv_bfloat16*>(out), ldo, stream);
}

}  // namespace nf12::backend
