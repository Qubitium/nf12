// SPDX-License-Identifier: Apache-2.0
// HIP implementation of the private backend boundary; public validation is in api.cpp.
#include "../backend.h"
#include "linear.h"
#include "nf12/nf12_hip.h"
#include <algorithm>
#include <limits>

namespace nf12_hip {
hipError_t pack_bf16t(const void*, int64_t, int64_t, int64_t, uint8_t*, hipStream_t);
hipError_t embedding(const void*, const uint4*, uint32_t, uint32_t, int64_t, int64_t,
                     const int32_t*, const int64_t*, int64_t, void*, int64_t, hipStream_t);
}

namespace nf12::backend {
cudaError_t current_device(int* device, bool* supported) {
  *supported = true;  // Individual HIP launchers check architecture-specific capabilities.
  return hipGetDevice(device);
}
bool device_supported(int device) {
  int count = 0;
  if (device < 0) return false;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return device < count;
}

size_t encode_workspace_bytes(int64_t n, int64_t k) {
  return nf12_hip_encode_workspace_bytes(n, k);
}
cudaError_t encode_analyze(const void* w, int64_t n, int64_t k, int64_t ldw,
                           void* ws, size_t bytes, uint8_t* base, uint32_t* count, cudaStream_t stream) {
  return nf12_hip_encode_analyze(w, n, k, ldw, ws, bytes, base, count, stream);
}
cudaError_t encode(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t base,
                   uint8_t* packed, void* escapes, uint32_t capacity,
                   void* ws, size_t bytes, cudaStream_t stream) {
  return nf12_hip_encode(w, n, k, ldw, base, packed, static_cast<uint4*>(escapes),
                         capacity, ws, bytes, stream);
}
cudaError_t decode(const uint8_t* packed, const void* escapes, uint32_t base,
                   int64_t n, int64_t k, void* out, int64_t ldo, cudaStream_t stream) {
  return nf12_hip_decode(packed, static_cast<const uint4*>(escapes), base, n, k, out, ldo, stream);
}
cudaError_t pack_bf16t(const void* w, int64_t n, int64_t k, int64_t ldw,
                       uint8_t* out, cudaStream_t stream) {
  return nf12_hip::pack_bf16t(w, n, k, ldw, out, stream);
}

size_t linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int) {
  // A tail can require a larger split plane than the full M chunk.
  const int64_t cap = nf12_hip_wmma_max_m();
  return std::max(nf12_hip_linear_wmma_workspace_bytes(n, k, std::min(m, cap)),
                  nf12_hip_linear_wmma_workspace_bytes(n, k, m % cap));
}
cudaError_t linear(const LinearArgs& a) {
  const bool twin = a.source == WeightSource::kBF16T;
  const int64_t chunk = nf12_hip_wmma_max_m();
  for (int64_t m0 = 0; m0 < a.m; ) {
    const int64_t rows = std::min(chunk, a.m - m0);
    const auto* x = static_cast<const uint8_t*>(a.x) + m0 * a.ldx * 2;
    auto* y = static_cast<uint8_t*>(a.y) + m0 * a.ldy * 2;
    const auto* escapes = static_cast<const uint4*>(a.escapes);
    const hipError_t err = rows < 16
        ? nf12_hip::linear_fused_strided(a.weight, escapes, a.exp_base, x, y, a.n, a.k,
                                         rows, a.bias, a.ldx, a.ldy, twin, a.stream)
        : nf12_hip::linear_wmma_strided(a.weight, escapes, a.exp_base, x, y, a.n, a.k,
                                        rows, a.bias, a.workspace, a.workspace_bytes,
                                        a.ldx, a.ldy, twin, a.stream);
    if (err != hipSuccess) return err;
    m0 += rows;
  }
  return hipSuccess;
}
int64_t linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, int device) {
  if (m < 16 || (m <= nf12_hip_wmma_max_m() && nf12_hip_wmma_supported_device(m, device))) return 0;
  constexpr int64_t kSliceBytes = int64_t(256) << 20;
  const int64_t cap = std::max<int64_t>(16, kSliceBytes / (2 * k) / 16 * 16);
  const int64_t slices = (n + cap - 1) / cap;
  return (n / 16 + slices - 1) / slices * 16;
}

size_t moe_align_workspace_bytes(int64_t tokens, int64_t topk, int64_t experts) {
  return nf12_hip_moe_align_workspace_bytes(tokens * topk, experts);
}
cudaError_t moe_align(const int32_t* ids, int64_t tokens, int64_t topk, int64_t experts,
                      int32_t* sorted, int64_t* offsets, void* ws, size_t bytes, cudaStream_t stream) {
  return nf12_hip_moe_align(ids, tokens, topk, experts, sorted, offsets, ws, bytes, stream);
}
size_t moe_linear_workspace_bytes(int64_t, int64_t, int64_t, int64_t, int) { return 0; }
// Native routed GEMM handles all M; no aten::_grouped_mm dependency on ROCm.
int64_t moe_linear_decode_gemm_rows(int64_t, int64_t, int64_t, int64_t, int) { return 0; }
cudaError_t moe_linear(const MoeLinearArgs& a) {
  if (a.num_experts > std::numeric_limits<int32_t>::max()) return hipErrorNotSupported;
  if (a.source == WeightSource::kBF16T) {
    return nf12_hip_moe_linear_bf16t(a.weight, static_cast<int32_t>(a.num_experts), a.x, a.ldx,
        a.gather, a.gather_div, a.scatter, a.expert_offsets, a.max_assignments, a.row_scale,
        a.y, a.ldy, a.n, a.k, a.stream);
  }
  return nf12_hip_moe_linear(a.weight, static_cast<const uint4*>(a.escapes), a.escape_offsets,
      a.exp_base, static_cast<int32_t>(a.num_experts), a.x, a.ldx, a.gather, a.gather_div, a.scatter,
      a.expert_offsets, a.max_assignments, a.row_scale, a.y, a.ldy, a.n, a.k, a.stream);
}
cudaError_t moe_combine(const void* y, int64_t ldy, int64_t tokens, int64_t topk,
                        int64_t n, void* out, int64_t ldo, cudaStream_t stream) {
  return nf12_hip_moe_combine(y, ldy, tokens, topk, n, out, ldo, stream);
}
cudaError_t moe_silu_mul(const void* h, int64_t ldh, int64_t rows, int64_t inter,
                         void* out, int64_t ldo, cudaStream_t stream) {
  return nf12_hip_moe_silu_mul(h, ldh, rows, inter, out, ldo, stream);
}
cudaError_t moe_decode(const uint8_t* packed, const void* escapes, const int64_t* escape_offsets,
                       const uint8_t* base, int64_t experts, int64_t n, int64_t k,
                       const int64_t* offsets, int64_t assignments, int64_t row0, int64_t rows,
                       void* out, int64_t ldo, cudaStream_t stream) {
  if (experts > 0xffff) return hipErrorNotSupported;
  return nf12_hip_moe_decode(packed, static_cast<const uint4*>(escapes), escape_offsets, base,
      static_cast<int32_t>(experts), n, k, offsets, assignments, row0, rows, out, ldo, stream);
}
cudaError_t moe_grouped_operands(const void* x, int64_t ldx, const int32_t* gather, int64_t div,
                                 const int64_t* offsets, int64_t assignments, int64_t experts,
                                 int64_t k, void* out, int64_t ldo, int32_t* ends, cudaStream_t stream) {
  return nf12_hip_moe_grouped_operands(x, ldx, gather, div, offsets, assignments, experts,
                                      k, out, ldo, ends, stream);
}
cudaError_t embedding(const uint8_t* packed, const void* escapes, uint32_t count, uint32_t base,
                      int64_t n, int64_t k, const int32_t* ids32, const int64_t* ids64,
                      int64_t num_ids, void* out, int64_t ldo, cudaStream_t stream) {
  return nf12_hip::embedding(packed, static_cast<const uint4*>(escapes), count, base,
                             n, k, ids32, ids64, num_ids, out, ldo, stream);
}
}  // namespace nf12::backend
