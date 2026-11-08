// SPDX-License-Identifier: Apache-2.0
//
// CUDA-API compatibility layer for the ROCm torch extension.
//
// The torch extension (csrc/torch/ops.cpp) is backend-agnostic: it calls the
// nf12_* C API from include/nf12/nf12.h. On ROCm those symbols are provided
// here on top of the HIP kernels, so ops.cpp is reused unchanged instead of
// being duplicated. Every op is backed by a real kernel: nothing is emulated and
// nothing silently falls back to a reference implementation.
//
// Adapted from GPT-QModel (https://github.com/ModelCloud/GPTQModel) by Qubitium.

#include "c_api_common.cuh"

#include "nf12/nf12.h"
#include "nf12/nf12_hip.h"

namespace nf12_hip {
hipError_t pack_bf16t(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t* out,
                      hipStream_t stream);
hipError_t embedding(const void* packed, const uint4* escapes, uint32_t num_escapes,
                     uint32_t exp_base, int64_t n, int64_t k, const int32_t* ids32,
                     const int64_t* ids64, int64_t num_ids, void* out, int64_t ldo,
                     hipStream_t stream);
}  // namespace nf12_hip

using nf12_hip_capi::status;
using nf12_hip_capi::aligned16;
using nf12_hip_capi::valid_bf16_matrix;
using nf12_hip_capi::valid_shape;
using nf12_hip_capi::valid_weight;
using nf12_hip_capi::valid_workspace;

extern "C" {

const char* nf12_status_string(nf12_status_t s) {
  switch (s) {
    case NF12_OK:
      return "ok";
    case NF12_ERROR_INVALID_ARGUMENT:
      return "invalid argument";
    case NF12_ERROR_UNSUPPORTED:
      return "unsupported";
    case NF12_ERROR_INSUFFICIENT_WORKSPACE:
      return "insufficient workspace";
    case NF12_ERROR_CUDA:
      return "cuda error";
  }
  return "unknown";
}

cudaError_t nf12_last_cuda_error(void) { return nf12_hip_capi::t_last_error; }

size_t nf12_packed_bytes(int64_t n, int64_t k) {
  // csrc/api.cu returns 0 for an invalid shape rather than a wrapped size_t.
  // Without this a caller computing a buffer size from n < 0 asks for ~2^63
  // bytes instead of failing.
  return valid_shape(n, k) ? static_cast<size_t>(n) * static_cast<size_t>(k) * 3 / 2 : 0;
}

size_t nf12_bf16t_bytes(int64_t n, int64_t k) {
  return valid_shape(n, k) ? static_cast<size_t>(n) * static_cast<size_t>(k) * 2 : 0;
}

size_t nf12_encode_workspace_bytes(int64_t n, int64_t k) {
  return valid_shape(n, k) ? nf12_hip_encode_workspace_bytes(n, k) : 0;
}

nf12_status_t nf12_encode_analyze(const void* w, int64_t n, int64_t k, int64_t ldw, void* workspace,
                                  size_t workspace_bytes, uint8_t* exp_base_out, uint32_t* num_escapes_out,
                                  cudaStream_t stream) {
  if (!valid_shape(n, k) || !valid_bf16_matrix(w, k, ldw) || exp_base_out == nullptr ||
      num_escapes_out == nullptr || !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (workspace_bytes < nf12_hip_encode_workspace_bytes(n, k)) {
    return NF12_ERROR_INSUFFICIENT_WORKSPACE;
  }
  return status(nf12_hip_encode_analyze(w, n, k, ldw, workspace, workspace_bytes, exp_base_out,
                                        num_escapes_out, stream));
}

nf12_status_t nf12_encode(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base,
                          void* packed_out, void* escapes_out, uint32_t escapes_capacity,
                          void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (!valid_shape(n, k) || !valid_bf16_matrix(w, k, ldw) || exp_base > nf12_hip_capi::kMaxExpBase ||
      packed_out == nullptr || !aligned16(packed_out) || !aligned16(escapes_out) ||
      (escapes_capacity > 0 && escapes_out == nullptr) ||
      !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (workspace_bytes < nf12_hip_encode_workspace_bytes(n, k)) {
    return NF12_ERROR_INSUFFICIENT_WORKSPACE;
  }
  return status(nf12_hip_encode(w, n, k, ldw, exp_base, static_cast<uint8_t*>(packed_out),
                                static_cast<uint4*>(escapes_out), escapes_capacity, workspace,
                                workspace_bytes, stream));
}

nf12_status_t nf12_decode(const nf12_weight_t* w, void* out_bf16, int64_t ldo, cudaStream_t stream) {
  if (!valid_weight(w) || !valid_bf16_matrix(out_bf16, w->k, ldo)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip_decode(static_cast<const uint8_t*>(w->packed),
                                static_cast<const uint4*>(w->escapes), w->exp_base, w->n, w->k,
                                out_bf16, ldo, stream));
}

size_t nf12_linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int) {
  if (m <= 0 || !valid_shape(n, k) || m > nf12_hip_wmma_max_m()) return 0;
  return nf12_hip_linear_wmma_workspace_bytes(n, k, m);
}

nf12_status_t nf12_linear(const void* x_bf16, int64_t m, int64_t ldx, const nf12_weight_t* w,
                          const void* bias_bf16, void* y_bf16, int64_t ldy, void* workspace,
                          size_t workspace_bytes, cudaStream_t stream) {
  // Mirrors run_linear() in csrc/api.cu clause for clause, in the same order, so
  // a caller gets the same status for the same mistake.
  if (!valid_weight(w)) return NF12_ERROR_INVALID_ARGUMENT;
  if (m < 0 || (bias_bf16 != nullptr && !aligned16(bias_bf16)) ||
      !valid_workspace(workspace, workspace_bytes)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (m == 0) return NF12_OK;
  if (!valid_bf16_matrix(x_bf16, w->k, ldx) || !valid_bf16_matrix(y_bf16, w->n, ldy)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  // HIP kernels address x and y as contiguous rows of k and n, so unlike CUDA
  // they cannot honour a wider leading dimension. Reject rather than compute the
  // wrong rows.
  if (ldx != w->k || ldy != w->n) return NF12_ERROR_INVALID_ARGUMENT;
  if (workspace_bytes < nf12_hip_linear_wmma_workspace_bytes(w->n, w->k, m)) {
    return NF12_ERROR_INSUFFICIENT_WORKSPACE;
  }
  if (m < 16) {
    return status(nf12_hip_linear_fused(static_cast<const uint8_t*>(w->packed),
                                        static_cast<const uint4*>(w->escapes),
                                        w->exp_base, x_bf16, y_bf16, w->n, w->k, m, bias_bf16,
                                        stream));
  }
  if (m <= nf12_hip_wmma_max_m()) {
    return status(nf12_hip_linear_wmma(static_cast<const uint8_t*>(w->packed),
                                       static_cast<const uint4*>(w->escapes),
                                       w->exp_base, x_bf16, y_bf16, w->n, w->k, m, bias_bf16,
                                       workspace, workspace_bytes, stream));
  }
  return NF12_ERROR_UNSUPPORTED;
}

// Above the matrix-core crossover ops.cpp decodes slices and calls torch's
// GEMM, so no HIP GEMM is needed here.
int64_t nf12_linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, int) {
  if (m <= 0 || m > std::numeric_limits<int32_t>::max() || !valid_shape(n, k)) return 0;
  return m > nf12_hip_wmma_max_m() ? 64 : 0;
}

nf12_status_t nf12_pack_bf16t(const void* w, int64_t n, int64_t k, int64_t ldw, void* out,
                              cudaStream_t stream) {
  if (!valid_shape(n, k) || !valid_bf16_matrix(w, k, ldw) || out == nullptr || !aligned16(out)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip::pack_bf16t(w, n, k, ldw, static_cast<uint8_t*>(out), stream));
}

nf12_status_t nf12_linear_bf16t(const void* x_bf16, int64_t m, int64_t ldx, const void* w_bf16t,
                                int64_t n, int64_t k, const void* bias_bf16, void* y_bf16,
                                int64_t ldy, void* workspace, size_t workspace_bytes,
                                cudaStream_t stream) {
  if (!valid_shape(n, k) || w_bf16t == nullptr || !aligned16(w_bf16t) || m < 0) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  // No workspace check here: the HIP twin always runs one K pass and needs no
  // scratch, so requiring the nf12 route's buffer would reject a valid call.
  if (m == 0) return NF12_OK;
  if (!valid_bf16_matrix(x_bf16, k, ldx) || !valid_bf16_matrix(y_bf16, n, ldy)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (ldx != k || ldy != n) return NF12_ERROR_INVALID_ARGUMENT;
  const uint8_t* const packed = static_cast<const uint8_t*>(w_bf16t);
  if (m < 16) {
    return status(
        nf12_hip_linear_fused_bf16t(packed, x_bf16, y_bf16, n, k, m, bias_bf16, stream));
  }
  return status(nf12_hip_linear_wmma_bf16t(packed, x_bf16, y_bf16, n, k, m, bias_bf16,
                                           workspace, workspace_bytes, stream));
}

// The MoE C API lives in c_api_moe.hip: it is the same contract, but keeping it
// in its own translation unit stops this file from becoming a second dispatcher.

nf12_status_t nf12_embedding(const nf12_weight_t* w, const int32_t* ids, int64_t num_ids,
                             void* out_bf16, int64_t ldo, cudaStream_t stream) {
  if (!valid_weight(w) || num_ids < 0 || !nf12_hip_capi::aligned(ids, sizeof(int32_t))) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (num_ids == 0) return NF12_OK;
  if (ids == nullptr || !valid_bf16_matrix(out_bf16, w->k, ldo) ||
      num_ids > std::numeric_limits<int64_t>::max() / 2 / ldo) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip::embedding(w->packed, static_cast<const uint4*>(w->escapes),
                                    w->num_escapes, w->exp_base, w->n, w->k, ids, nullptr,
                                    num_ids, out_bf16, ldo, stream));
}

nf12_status_t nf12_embedding_i64(const nf12_weight_t* w, const int64_t* ids, int64_t num_ids,
                                 void* out_bf16, int64_t ldo, cudaStream_t stream) {
  if (!valid_weight(w) || num_ids < 0 || !nf12_hip_capi::aligned(ids, sizeof(int64_t))) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (num_ids == 0) return NF12_OK;
  if (ids == nullptr || !valid_bf16_matrix(out_bf16, w->k, ldo) ||
      num_ids > std::numeric_limits<int64_t>::max() / 2 / ldo) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip::embedding(w->packed, static_cast<const uint4*>(w->escapes),
                                    w->num_escapes, w->exp_base, w->n, w->k, nullptr, ids,
                                    num_ids, out_bf16, ldo, stream));
}

}  // extern "C"
