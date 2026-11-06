// SPDX-License-Identifier: Apache-2.0
// Streaming reads shared by the device probe and in-run calibration.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nf12_bench {

enum Ld { kLdDefault, kLdNc, kLdNcNoAlloc, kLdNcNoAlloc256, kLdCs, kNumLd };
const char* const kLdNames[kNumLd] = {"ld.global", "ld.global.nc", "ld.global.nc.L1::no_allocate",
                                      "ld.global.nc.L1::no_allocate.L2::256B", "ld.global.cs"};

// Grid-stride reads sweep the buffer. Warp-slab reads assign contiguous 512 B rows per warp.
enum Pattern { kGridStride, kWarpSlab };
const char* const kPatternNames[] = {"grid_stride", "warp_slab"};

template <int LD>
__device__ __forceinline__ uint4 ld16(const uint4* p) {
  uint4 v;
  if constexpr (LD == kLdDefault) {
    asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  } else if constexpr (LD == kLdNc) {
    asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  } else if constexpr (LD == kLdNcNoAlloc) {
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
  } else if constexpr (LD == kLdNcNoAlloc256) {
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
  } else {
    asm("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  }
  return v;
}

__device__ __forceinline__ uint32_t fold(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

// The conditional sink store prevents the compiler from removing loads.
template <int LD, int U, int PAT>
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, uint32_t* sink) {
  uint32_t acc = 0;
  if constexpr (PAT == kGridStride) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n16; i += U * stride) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) v[u] = ld16<LD>(p + i + u * stride);
#pragma unroll
      for (int u = 0; u < U; ++u) acc ^= fold(v[u]);
    }
    for (; i < n16; i += stride) acc ^= fold(ld16<LD>(p + i));
  } else {
    const size_t lane = threadIdx.x & 31;
    const size_t warp = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const size_t warps = ((size_t)gridDim.x * blockDim.x) >> 5;
    const size_t rows = n16 / 32;  // 512 B "rows"
    const size_t per = (rows + warps - 1) / warps;
    const size_t r0 = warp * per;
    const size_t r1 = min(r0 + per, rows);
    size_t r = r0;
    for (; r + U <= r1; r += U) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) v[u] = ld16<LD>(p + (r + u) * 32 + lane);
#pragma unroll
      for (int u = 0; u < U; ++u) acc ^= fold(v[u]);
    }
    for (; r < r1; ++r) acc ^= fold(ld16<LD>(p + r * 32 + lane));
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

}  // namespace nf12_bench
