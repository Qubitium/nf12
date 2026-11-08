/* SPDX-License-Identifier: Apache-2.0
 * NF12 RDNA4 (HIP) public C API.
 *
 * The HIP port implements the codec and the fused linear routes of nf12.h, but
 * takes hipStream_t and returns hipError_t. There is no CUDA-compatibility
 * layer, so include this header rather than nf12.h.
 *
 * Pointers refer to device memory. Weight descriptors and analysis outputs are
 * host-resident. No function allocates device memory. nf12_hip_encode_analyze
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

/* Fused decode+GEMM for m <= 16. Never materialises the dense weight. */
int nf12_hip_linear_fused_small_m_supported(int64_t m);
hipError_t nf12_hip_linear_fused(const uint8_t* packed, const uint4* escapes,
                                 uint32_t exp_base, const void* x, void* out,
                                 int64_t n, int64_t k, int64_t m, const void* bias,
                                 hipStream_t stream);

/* Matrix-core fused path for 17 <= m <= nf12_hip_wmma_max_m(). Returns 0 from
 * nf12_hip_wmma_supported on devices without wmma-128b-insts (gfx1250), where
 * the caller should use decode-then-rocBLAS instead. */
int nf12_hip_wmma_max_m(void);
int nf12_hip_wmma_supported(int64_t m);
hipError_t nf12_hip_linear_wmma(const uint8_t* packed, const uint4* escapes,
                                uint32_t exp_base, const void* x, void* out,
                                int64_t n, int64_t k, int64_t m, const void* bias,
                                hipStream_t stream);

/* Launches a kernel that stamps 0x4E463132. Fails loudly if the build produced
 * no device code for this target. */
hipError_t nf12_hip_selftest(uint32_t* out, hipStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NF12_NF12_HIP_H_ */
