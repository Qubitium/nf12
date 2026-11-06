// SPDX-License-Identifier: Apache-2.0
// extern "C" entry points over the small-M kernels, the production dispatch
// and a streaming-read calibration kernel for the Python test/bench harness
// (small_m_harness.py). Not part of the library.
#include <cuda_runtime.h>

#include "internal.h"
#include "linear/common.cuh"
#include "linear/small_m.h"
#include "linear/small_m_core.cuh"

using namespace nf12::detail;

namespace {

const DeviceInfo& current_device() {
  int d = 0;
  cudaGetDevice(&d);
  return device_info(d);
}

LinearArgs make_args(const void* x, int64_t m, int64_t ldx, int source, const void* weight, const void* escapes,
                     uint32_t num_escapes, uint32_t exp_base, int64_t n, int64_t k, const void* bias, void* y,
                     int64_t ldy, void* stream) {
  LinearArgs a{};
  a.x = static_cast<const __nv_bfloat16*>(x);
  a.m = m;
  a.ldx = ldx;
  a.source = source == 0 ? WeightSource::kNF12 : WeightSource::kBF16T;
  a.weight = static_cast<const uint8_t*>(weight);
  a.escapes = static_cast<const uint4*>(escapes);
  a.num_escapes = num_escapes;
  a.exp_base = exp_base;
  a.n = n;
  a.k = k;
  a.bias = static_cast<const __nv_bfloat16*>(bias);
  a.y = static_cast<__nv_bfloat16*>(y);
  a.ldy = ldy;
  a.stream = static_cast<cudaStream_t>(stream);
  a.dev = &current_device();
  return a;
}

// Match GEMV's load instruction and per-warp depth on contiguous 512-byte
// rows to calibrate streaming-read bandwidth for this access pattern.
template <int DEPTH>
__global__ void __launch_bounds__(256) stream_read_kernel(const uint8_t* __restrict__ p, size_t lines,
                                                          uint32_t* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const size_t warp = (blockIdx.x * size_t{blockDim.x} + threadIdx.x) >> 5;
  const size_t nwarps = (gridDim.x * size_t{blockDim.x}) >> 5;
  uint32_t acc = 0;
  for (size_t l = warp * DEPTH; l < lines; l += nwarps * DEPTH) {
    uint4 v[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d)
      if (l + d < lines) v[d] = ldg_stream(p + (l + d) * 512 + lane * 16);
#pragma unroll
    for (int d = 0; d < DEPTH; ++d)
      if (l + d < lines) acc ^= v[d].x ^ v[d].y ^ v[d].z ^ v[d].w;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;  // keeps the loads alive
}

}  // namespace

extern "C" {

int nf12_sm_supported(int mt, int warps, int kc, int slice, int kp, int64_t m, int64_t n, int64_t k) {
  return small_m_config_supported(SmallMConfig{mt, warps, kc, slice, kp}, m, n, k, current_device()) ? 1 : 0;
}

// Weight ring (small_m_stages) and granularity of SmallMConfig::slice per K
// part for mt-tile configs on an n x k weight on this device.
int nf12_sm_stages(int mt, int64_t n, int64_t k) {
  return small_m_stages(SmallMConfig{mt, 0, 0, 0}, n, k, current_device());
}
int nf12_sm_slice_step(int mt, int64_t n, int64_t k) { return small_m_step(nf12_sm_stages(mt, n, k)); }

int nf12_sm_resident_ctas(int mt, int warps, int kc, int slice, int kp, int64_t m, int64_t n, int64_t k) {
  return small_m_resident_ctas(SmallMConfig{mt, warps, kc, slice, kp}, m, n, k, current_device());
}

void nf12_sm_heuristic(int64_t m, int64_t n, int64_t k, int* out) {
  const SmallMConfig c = small_m_heuristic(m, n, k, current_device());
  out[0] = c.mt;
  out[1] = c.warps;
  out[2] = c.kc;
  out[3] = c.slice;
  out[4] = c.kp;
}

// mt <= 0 runs the production entry point nf12::detail::linear().
int nf12_sm_linear(int mt, int warps, int kc, int slice, int kp, const void* x, int64_t m, int64_t ldx, int source,
                   const void* weight, const void* escapes, uint32_t num_escapes, uint32_t exp_base, int64_t n,
                   int64_t k, const void* bias, void* y, int64_t ldy, void* workspace, size_t workspace_bytes,
                   void* stream) {
  LinearArgs a = make_args(x, m, ldx, source, weight, escapes, num_escapes, exp_base, n, k, bias, y, ldy, stream);
  a.workspace = workspace;
  a.workspace_bytes = workspace_bytes;
  if (mt == 0) return static_cast<int>(linear(a));
  return static_cast<int>(linear_small_m(a, SmallMConfig{mt, warps, kc, slice, kp}));
}

size_t nf12_sm_workspace_bytes(int64_t m, int64_t n, int64_t k) {
  return linear_workspace_bytes(m, n, k, current_device());
}

size_t nf12_sm_config_workspace_bytes(int mt, int warps, int kc, int slice, int kp, int64_t m, int64_t n) {
  return small_m_workspace_bytes(SmallMConfig{mt, warps, kc, slice, kp}, m, n, current_device());
}

int nf12_sm_heuristic_slice(int mt, int warps, int kc, int kp, int64_t m, int64_t n, int64_t k) {
  return small_m_heuristic_slice(SmallMConfig{mt, warps, kc, 0, kp}, m, n, k, current_device());
}

int nf12_sm_stream_read(const void* p, size_t bytes, int blocks, int depth, void* out, void* stream) {
  const size_t lines = bytes / 512;
  auto s = static_cast<cudaStream_t>(stream);
  const auto* src = static_cast<const uint8_t*>(p);
  auto* o = static_cast<uint32_t*>(out);
  switch (depth) {
    case 2: stream_read_kernel<2><<<blocks, 256, 0, s>>>(src, lines, o); break;
    case 4: stream_read_kernel<4><<<blocks, 256, 0, s>>>(src, lines, o); break;
    case 8: stream_read_kernel<8><<<blocks, 256, 0, s>>>(src, lines, o); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
