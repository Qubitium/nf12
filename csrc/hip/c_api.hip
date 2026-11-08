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
  return static_cast<size_t>(n) * static_cast<size_t>(k) * 3 / 2;
}

size_t nf12_bf16t_bytes(int64_t n, int64_t k) {
  return static_cast<size_t>(n) * static_cast<size_t>(k) * 2;
}

size_t nf12_encode_workspace_bytes(int64_t n, int64_t k) {
  return nf12_hip_encode_workspace_bytes(n, k);
}

nf12_status_t nf12_encode_analyze(const void* w, int64_t n, int64_t k, int64_t ldw, void* workspace,
                                  size_t workspace_bytes, uint8_t* exp_base_out, uint32_t* num_escapes_out,
                                  cudaStream_t stream) {
  return status(nf12_hip_encode_analyze(w, n, k, ldw, workspace, workspace_bytes, exp_base_out,
                                        num_escapes_out, stream));
}

nf12_status_t nf12_encode(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base,
                          void* packed_out, void* escapes_out, uint32_t escapes_capacity,
                          void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  return status(nf12_hip_encode(w, n, k, ldw, exp_base, static_cast<uint8_t*>(packed_out),
                                static_cast<uint4*>(escapes_out), escapes_capacity, workspace,
                                workspace_bytes, stream));
}

nf12_status_t nf12_decode(const nf12_weight_t* w, void* out_bf16, int64_t ldo, cudaStream_t stream) {
  return status(nf12_hip_decode(static_cast<const uint8_t*>(w->packed),
                                static_cast<const uint4*>(w->escapes), w->exp_base, w->n, w->k,
                                out_bf16, ldo, stream));
}

size_t nf12_linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int) {
  if (m < 16 || m > nf12_hip_wmma_max_m()) return 0;
  return nf12_hip_linear_wmma_workspace_bytes(n, k, m);
}

nf12_status_t nf12_linear(const void* x_bf16, int64_t m, int64_t ldx, const nf12_weight_t* w,
                          const void* bias_bf16, void* y_bf16, int64_t ldy, void* workspace,
                          size_t workspace_bytes, cudaStream_t stream) {
  // The CUDA build gates this through valid_weight() before touching w; without
  // the null check the shape comparison below would dereference a null pointer.
  if (w == nullptr || w->packed == nullptr || w->n <= 0 || w->k <= 0 || w->n % 16 != 0 ||
      w->k % 64 != 0) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  if (ldx != w->k || ldy != w->n) return NF12_ERROR_INVALID_ARGUMENT;
  // Same workspace contract as csrc/api.cu (run_linear): a non-zero buffer must
  // be 16-byte aligned, and anything smaller than the shape needs is
  // INSUFFICIENT_WORKSPACE rather than INVALID_ARGUMENT. Aligning the code here
  // matters because a caller ports from CUDA by matching status values.
  if (workspace_bytes != 0 &&
      (workspace == nullptr || (reinterpret_cast<uintptr_t>(workspace) & 15u) != 0)) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
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
int64_t nf12_linear_decode_gemm_rows(int64_t m, int64_t, int64_t, int) {
  return m > nf12_hip_wmma_max_m() ? 64 : 0;
}

nf12_status_t nf12_pack_bf16t(const void* w, int64_t n, int64_t k, int64_t ldw, void* out,
                              cudaStream_t stream) {
  if (n <= 0 || k <= 0 || n % 16 != 0 || k % 64 != 0 || w == nullptr || out == nullptr) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip::pack_bf16t(w, n, k, ldw, static_cast<uint8_t*>(out), stream));
}

nf12_status_t nf12_linear_bf16t(const void* x_bf16, int64_t m, int64_t ldx, const void* w_bf16t,
                                int64_t n, int64_t k, const void* bias_bf16, void* y_bf16,
                                int64_t ldy, void* workspace, size_t workspace_bytes,
                                cudaStream_t stream) {
  if (n <= 0 || k <= 0 || n % 16 != 0 || k % 64 != 0) return NF12_ERROR_INVALID_ARGUMENT;
  if (w_bf16t == nullptr || x_bf16 == nullptr || y_bf16 == nullptr) {
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
  if (w == nullptr || ids == nullptr || out_bf16 == nullptr || w->n <= 0 || w->k <= 0) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip::embedding(w->packed, static_cast<const uint4*>(w->escapes),
                                    w->num_escapes, w->exp_base, w->n, w->k, ids, nullptr,
                                    num_ids, out_bf16, ldo, stream));
}

nf12_status_t nf12_embedding_i64(const nf12_weight_t* w, const int64_t* ids, int64_t num_ids,
                                 void* out_bf16, int64_t ldo, cudaStream_t stream) {
  if (w == nullptr || ids == nullptr || out_bf16 == nullptr || w->n <= 0 || w->k <= 0) {
    return NF12_ERROR_INVALID_ARGUMENT;
  }
  return status(nf12_hip::embedding(w->packed, static_cast<const uint4*>(w->escapes),
                                    w->num_escapes, w->exp_base, w->n, w->k, nullptr, ids,
                                    num_ids, out_bf16, ldo, stream));
}

}  // extern "C"
