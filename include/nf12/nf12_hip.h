/* SPDX-License-Identifier: Apache-2.0
 * NF12 RDNA4 (HIP) public C API.
 *
 * The HIP port implements the codec and the fused linear routes of nf12.h, but
 * takes hipStream_t and returns hipError_t. There is no CUDA-compatibility
 * layer, so include this header rather than nf12.h.
 *
 * Pointers refer to device memory. Weight descriptors and analysis outputs are
 * host-resident. No function allocates device memory and the library keeps no
 * mutable device state, so calls on different streams (or from different host
 * threads) never share scratch and are independent. nf12_hip_encode_analyze
 * and nf12_hip_encode synchronize the supplied stream.
 */
#ifndef NF12_NF12_HIP_H_
#define NF12_NF12_HIP_H_

#include <stddef.h>
#include <stdint.h>

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Workspace required by analyze + encode. Both synchronize the stream and
 * cannot be graph-captured. */
size_t nf12_hip_encode_workspace_bytes(int64_t n, int64_t k);

/* Analyze selects exp_base and counts escapes before allocation. */
hipError_t nf12_hip_encode_analyze(const void* w_bf16, int64_t n, int64_t k, int64_t ldw,
                                   void* workspace, size_t workspace_bytes,
                                   uint8_t* exp_base_out, uint32_t* num_escapes_out,
                                   hipStream_t stream);

/* Packing is deterministic for a fixed exp_base. Capacity is checked before
 * any output is written. */
hipError_t nf12_hip_encode(const void* w_bf16, int64_t n, int64_t k, int64_t ldw,
                           uint8_t exp_base, uint8_t* packed_out, uint4* escapes_out,
                           uint32_t escapes_capacity, void* workspace,
                           size_t workspace_bytes, hipStream_t stream);

/* Exact BF16 reconstruction. out_bf16 is row-major [n, k] with row stride ldo. */
hipError_t nf12_hip_decode(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                           int64_t n, int64_t k, void* out_bf16, int64_t ldo,
                           hipStream_t stream);

/* Uncompressed bf16t twin (SPEC.md 4.4): inverse of nf12.pack_bf16t. */
hipError_t nf12_hip_unpack_bf16t(const uint8_t* packed, int64_t n, int64_t k, void* out_bf16,
                                 int64_t ldo, hipStream_t stream);

/* Fused decode+GEMM for m <= 16. Never materialises the dense weight. */
int nf12_hip_linear_fused_small_m_supported(int64_t m);
hipError_t nf12_hip_linear_fused(const uint8_t* packed, const uint4* escapes,
                                 uint32_t exp_base, const void* x, void* out,
                                 int64_t n, int64_t k, int64_t m, const void* bias,
                                 hipStream_t stream);

/* bf16t twin of the small-M route: same kernel, uncompressed source. */
hipError_t nf12_hip_linear_fused_bf16t(const uint8_t* packed, const void* x, void* out,
                                       int64_t n, int64_t k, int64_t m, const void* bias,
                                       hipStream_t stream);

/* Matrix-core fused path for 17 <= m <= nf12_hip_wmma_max_m(). Returns 0 from
 * nf12_hip_wmma_supported on devices without wmma-128b-insts (gfx1250), where
 * the caller should use decode-then-rocBLAS instead. */
int nf12_hip_wmma_max_m(void);
int nf12_hip_wmma_supported(int64_t m);

/* Device-parameterised form: the same query for an explicit device ordinal.
 * The op layer uses it to route a non-WMMA device to decode+GEMM without
 * switching the caller's current device. */
int nf12_hip_wmma_supported_device(int64_t m, int device);

/* Device scratch for the split-K reduction, owned by the caller. Returns 0 when
 * the launch chosen for (n, k, m) needs no scratch.
 *
 * This mirrors the CUDA build (csrc/api.cu, run_linear) exactly: the library
 * owns no scratch, allocates nothing, and keeps no per-device or per-stream
 * cache. Size the buffer with this function using the SAME (n, k, m) passed to
 * the launch, and allocate it from the caller's own caching allocator -- on
 * ROCm that is PyTorch's, which is per device and stream ordered.
 *
 * A null or undersized buffer is rejected rather than aliased: nf12_linear
 * returns NF12_ERROR_INSUFFICIENT_WORKSPACE, the same status the CUDA build
 * returns, so a caller ported from CUDA sees identical behaviour. Allocating
 * here instead would mean a single per-device buffer shared by every stream,
 * which corrupts concurrent launches. */
size_t nf12_hip_linear_wmma_workspace_bytes(int64_t n, int64_t k, int64_t m);
hipError_t nf12_hip_linear_wmma(const uint8_t* packed, const uint4* escapes,
                                uint32_t exp_base, const void* x, void* out,
                                int64_t n, int64_t k, int64_t m, const void* bias,
                                void* workspace, size_t workspace_bytes,
                                hipStream_t stream);

/* bf16t twin of the matrix-core route. Runs the identical (tiles, splits) plan
 * as nf12_hip_linear_wmma so the two are bitwise comparable, which means it
 * needs the same split-K partial plane: size it with
 * nf12_hip_linear_wmma_workspace_bytes(n, k, min(m, nf12_hip_wmma_max_m())).
 * Any m >= 16 is accepted; above nf12_hip_wmma_max_m() the M axis is chunked. */
hipError_t nf12_hip_linear_wmma_bf16t(const uint8_t* packed, const void* x, void* out,
                                      int64_t n, int64_t k, int64_t m, const void* bias,
                                      void* workspace, size_t workspace_bytes,
                                      hipStream_t stream);

/* Mixture-of-experts routes (SPEC.md section 9). All are host-synchronisation
 * free: per-expert counts stay device-side and the host only supplies bounds,
 * so they capture into a graph. */
size_t nf12_hip_moe_align_workspace_bytes(int64_t num_assignments, int64_t num_experts);
hipError_t nf12_hip_moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk,
                              int64_t num_experts, int32_t* sorted_out,
                              int64_t* expert_offsets_out, void* workspace,
                              size_t workspace_bytes, hipStream_t stream);
hipError_t nf12_hip_moe_linear(const uint8_t* packed, const uint4* escapes,
                               const int64_t* escape_offsets, const uint8_t* exp_base,
                               int32_t num_experts, const void* x, int64_t ldx,
                               const int32_t* gather, int64_t gather_div,
                               const int32_t* scatter, const int64_t* expert_offsets,
                               int64_t max_assignments, const float* row_scale, void* out,
                               int64_t ldy, int64_t n, int64_t k, hipStream_t stream);
hipError_t nf12_hip_moe_linear_bf16t(const uint8_t* packed, int32_t num_experts, const void* x,
                                     int64_t ldx, const int32_t* gather, int64_t gather_div,
                                     const int32_t* scatter, const int64_t* expert_offsets,
                                     int64_t max_assignments, const float* row_scale, void* out,
                                     int64_t ldy, int64_t n, int64_t k, hipStream_t stream);
hipError_t nf12_hip_moe_combine(const void* y_bf16, int64_t ldy, int64_t num_tokens, int64_t topk,
                                int64_t n, void* out_bf16, int64_t ldo, hipStream_t stream);
hipError_t nf12_hip_moe_silu_mul(const void* h_bf16, int64_t ldh, int64_t rows, int64_t inter,
                                 void* out_bf16, int64_t ldo, hipStream_t stream);
hipError_t nf12_hip_moe_decode(const uint8_t* packed, const uint4* escapes,
                               const int64_t* escape_offsets, const uint8_t* exp_base,
                               int32_t num_experts, int64_t n, int64_t k,
                               const int64_t* expert_offsets, int64_t max_assignments,
                               int64_t row0, int64_t rows, void* out, int64_t ldo,
                               hipStream_t stream);
hipError_t nf12_hip_moe_grouped_operands(const void* x_bf16, int64_t ldx,
                                         const int32_t* gather, int64_t gather_div,
                                         const int64_t* expert_offsets, int64_t max_assignments,
                                         int64_t num_experts, int64_t k, void* x_out_bf16,
                                         int64_t ldo, int32_t* group_ends, hipStream_t stream);

/* Launches a kernel that stamps 0x4E463132. Fails loudly if the build produced
 * no device code for this target. */
hipError_t nf12_hip_selftest(uint32_t* out, hipStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NF12_NF12_HIP_H_ */
