// SPDX-License-Identifier: Apache-2.0
// HIP implementation of the private backend boundary; public validation is in api.cpp.
#include "../backend.h"
#include "capability.h"
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

// M band the fused CDNA3 GEMM covers. Above it the M blocks re-read the packed
// weight, and decode-once + rocBLAS is faster again.
constexpr int64_t kMfmaFusedM = 128;

size_t linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int device) {
  // A tail can require a larger split plane than the full M chunk.
  const int64_t cap = nf12_hip_wmma_max_m();
  const size_t wmma = std::max(nf12_hip_linear_wmma_workspace_bytes(n, k, std::min(m, cap)),
                               nf12_hip_linear_wmma_workspace_bytes(n, k, m % cap));
  if (nf12_hip::device_traits(device).matrix == nf12_hip::MatrixCore::kMfma) {
    // The fused route stages no decoded weight; it only needs the FP32 partial
    // plane when K is split across CTAs. Above kMfmaFusedM the torch op takes
    // the decode route, so the wider of the two must be available.
    const int splits = nf12_hip::mfma_dense_splits(m, n, k);
    const size_t part = splits > 1 ? size_t(m) * size_t(n) * size_t(splits) * sizeof(float) : 0;
    return m <= kMfmaFusedM ? part : std::max(wmma, part);
  }
  return wmma;
}

// Which linear route this device takes. Trait-driven: RDNA4 waves take the
// v_wmma path, CDNA3 waves take v_mfma, anything else (or a device the build
// did not target) takes decode-then-GEMM through the fused small-M kernel and
// the caller's GEMM. No arch string is consulted here.
namespace {
static nf12_hip::MatrixCore linear_matrix_core(const LinearArgs& a) {
  return nf12_hip::device_traits(a.device).matrix;
}
}  // namespace

cudaError_t linear(const LinearArgs& a) {
  const bool twin = a.source == WeightSource::kBF16T;
  const nf12_hip::MatrixCore mc = linear_matrix_core(a);
  if (mc == nf12_hip::MatrixCore::kMfma) {
    // The MFMA tile is 16 rows wide, so m < 16 would have to read a 16-row A
    // window from a shorter tensor. The fused GEMV is exact for that band on
    // every device (the same split the RDNA4 route uses).
    if (a.m < 16) {
      return nf12_hip::linear_fused_strided(a.weight, static_cast<const uint4*>(a.escapes), a.exp_base,
                                            a.x, a.y, a.n, a.k, a.m, a.bias, a.ldx, a.ldy, twin,
                                            a.stream);
    }
    return nf12_hip::linear_mfma_strided(a.weight, static_cast<const uint4*>(a.escapes), a.exp_base,
                                         a.x, a.y, a.n, a.k, a.m, a.bias, a.ldx, a.ldy,
                                         a.workspace, a.workspace_bytes, twin, a.stream);
  }
  // The RDNA4 matrix-core route keeps its own M chunking. It reports
  // hipErrorNotSupported when the device lacks wmma-128b-insts, and the fused
  // kernel below covers those shapes exactly.
  if (mc == nf12_hip::MatrixCore::kWmma) {
    const int64_t chunk = nf12_hip_wmma_max_m();
    for (int64_t m0 = 0; m0 < a.m;) {
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
  // No matrix core: the fused decode+GEMM kernel handles small M exactly; above
  // it the caller's decode-then-GEMM route has already been chosen by
  // linear_decode_gemm_rows below, so this path is only the tail.
  if (a.m >= 16) return hipErrorNotSupported;
  return nf12_hip::linear_fused_strided(a.weight, static_cast<const uint4*>(a.escapes), a.exp_base,
                                        a.x, a.y, a.n, a.k, a.m, a.bias, a.ldx, a.ldy, twin,
                                        a.stream);
}
int64_t linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, int device) {
  const nf12_hip::MatrixCore mc = nf12_hip::device_traits(device).matrix;
  // Both matrix cores cover the small-M band themselves, so small M never
  // decodes the weight first (that is the shared API contract, tests/api).
  //
  // Above the band they diverge, and the split is a TRAIT, not an ISA:
  //  - RDNA4 (kWmma) waves are narrow and few, so the fused WMMA kernel keeps
  //    the packed weight in registers and never materialises a decoded tile.
  //    It covers all M up to its own cap.
  //  - CDNA3 (kMfma) splits K across the CTA's waves (mfma_split_kernel), so
  //    the fused route keeps the packed weight resident and never materialises
  //    a decoded copy. That halves the bytes read against a BF16 GEMM, which is
  //    the only way to stay ahead of rocBLAS on a bandwidth-bound part. Above
  //    kMfmaFusedM the M blocks re-read the packed weight, so the decode-once +
  //    rocBLAS route wins again.
  constexpr int64_t kSmallM = 64;
  if (mc == nf12_hip::MatrixCore::kNone) {
    // No matrix core: only the fused m < 16 kernel exists, so everything above
    // it must decode first (this is the pre-existing fallback route).
    if (m < 16) return 0;
  } else if (mc == nf12_hip::MatrixCore::kMfma) {
    if (m <= kMfmaFusedM) return 0;
  } else if (m <= kSmallM) {
    return 0;
  }
  if (mc == nf12_hip::MatrixCore::kWmma && m <= nf12_hip_wmma_max_m()) return 0;
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
// Routed GEMM policy. The fused MFMA/packed kernel is the ONLY routed route on
// HIP, and it is what both the C API and the torch op use (rows == 0).
//
// The CUDA build can hand a gathered launch to decode + aten::_grouped_mm. That
// is not usable here: on this ROCm build _grouped_mm stages its `offs`
// argument through an unpinned host tensor, so it raises during CUDA graph
// capture (tests/api/test_decode_route.py::test_moe_route_graph_capture). The
// fused routed kernel keeps the packed weight resident, needs no workspace and
// is capturable, so it stays the single route. Selection is still trait-based:
// a device without a matrix core reports kNone and the fused kernel handles it
// exactly.
int64_t moe_linear_decode_gemm_rows(int64_t, int64_t, int64_t, int64_t, int) { return 0; }

cudaError_t moe_linear(const MoeLinearArgs& a) {
  if (a.num_experts > std::numeric_limits<int32_t>::max()) return hipErrorNotSupported;
  const bool twin = a.source == WeightSource::kBF16T;
  const nf12_hip::MatrixCore mc = nf12_hip::device_traits(a.device).matrix;
  if (mc == nf12_hip::MatrixCore::kMfma) {
    return nf12_hip::moe_linear_mfma(
        a.weight, static_cast<const uint4*>(a.escapes), a.escape_offsets, a.exp_base,
        static_cast<int32_t>(a.num_experts), a.x, a.ldx, a.gather, a.gather_div, a.scatter,
        a.expert_offsets, a.max_assignments, a.row_scale, a.y, a.ldy, a.n, a.k, twin, a.stream);
  }
  if (mc == nf12_hip::MatrixCore::kWmma) {
    if (a.source == WeightSource::kBF16T) {
      return nf12_hip_moe_linear_bf16t(a.weight, static_cast<int32_t>(a.num_experts), a.x, a.ldx,
          a.gather, a.gather_div, a.scatter, a.expert_offsets, a.max_assignments, a.row_scale,
          a.y, a.ldy, a.n, a.k, a.stream);
    }
    return nf12_hip_moe_linear(a.weight, static_cast<const uint4*>(a.escapes), a.escape_offsets,
        a.exp_base, static_cast<int32_t>(a.num_experts), a.x, a.ldx, a.gather, a.gather_div, a.scatter,
        a.expert_offsets, a.max_assignments, a.row_scale, a.y, a.ldy, a.n, a.k, a.stream);
  }
  return hipErrorNotSupported;
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
  if (experts > 0xffff) return hipErrorInvalidValue;
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

// Reuse the shared validation and chunk dispatcher, including small-M tails
// and workspace rejection before any output writes. The dispatcher calls the
// strided kernel entry points, so this native wrapper does not recurse.
extern "C" hipError_t nf12_hip_linear_wmma_bf16t(
    const uint8_t* packed, const void* x, void* out, int64_t n, int64_t k, int64_t m,
    const void* bias, void* workspace, size_t workspace_bytes, hipStream_t stream) {
  if (m < 16) return hipErrorInvalidValue;
  const auto status = nf12_linear_bf16t(x, m, k, packed, n, k, bias, out, n,
                                       workspace, workspace_bytes, stream);
  if (status == NF12_OK) return hipSuccess;
  if (status == NF12_ERROR_UNSUPPORTED) return hipErrorNotSupported;
  if (status == NF12_ERROR_CUDA) return nf12_last_cuda_error();
  return hipErrorInvalidValue;
}
