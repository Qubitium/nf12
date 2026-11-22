// SPDX-License-Identifier: Apache-2.0
// Dense linear using Metal 4 tensor operations. The runtime
// compiler prepends nf12_metal.h and builds this as a separate library.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace nf12;

namespace {

constant constexpr uint kTensorSimdgroups = 4;
constant constexpr uint kTensorN = kSuperAtomRows;
constant constexpr uint kTensorK = kSuperAtomCols;
using Extents = metal::dextents<int, 2>;
using Strides = metal::array<int, 2>;
using Activation = metal::tensor<device bfloat, Extents, metal::tensor_inline>;
using WeightTile = metal::tensor<threadgroup bfloat, Extents, metal::tensor_inline>;

template <uint BM, uint KS>
constant constexpr uint kScratchWords = kTensorN * (KS > 1 && BM > kTensorK / 2 ? BM : kTensorK / 2);

template <uint BM, uint KS, typename Src>
inline void linear_tensor(Src src, device const bfloat* x, device const ushort* bias, device ushort* y,
                          constant LinearParams& p, uint2 tg, uint sg, uint lane,
                          threadgroup uint (&scratch)[kTensorSimdgroups][kScratchWords<BM, KS>]) {
  const uint n0 = (tg.x * (kTensorSimdgroups / KS) + sg / KS) * kTensorN;
  // Move the last partial tile inside the bounds so its tensor loads stay
  // full sized. This threadgroup only stores the rows assigned to it.
  const uint first_row = tg.y * BM;
  const uint m0 = min(first_row, p.m - BM);
  const bool valid = n0 < p.n;
  if constexpr (KS == 1) {
    if (!valid) return;
  }
  const uint k = p.k_blocks * kTensorK;
  Activation a(const_cast<device bfloat*>(x) + ulong(m0) * p.ldx, Extents(k, BM),
               Strides{1, int(p.ldx)});
  WeightTile w(reinterpret_cast<threadgroup bfloat*>(scratch[sg]), Extents(kTensorK, kTensorN),
               Strides{1, kTensorK});
  auto w_tile = w.template slice<kTensorK, kTensorN>(0, 0);
  using ActivationTile = decltype(a.template slice<kTensorK, BM>(0, 0));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM, kTensorN, kTensorK, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto acc = op.template get_destination_cooperative_tensor<ActivationTile, decltype(w_tile), float>();
  #pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  const uint kb0 = (sg % KS) * p.kb_per_split;
  const uint kb1 = min(kb0 + p.kb_per_split, p.k_blocks);
  if (valid) {
    const ulong first_atom = ulong(n0 / kTensorN) * p.k_blocks;
    for (uint kb = kb0; kb < kb1; ++kb) {
      uint regs[4][4];
      src.load(first_atom + kb, lane, regs);
      stage_bf16_tile<kTensorK>(scratch[sg], lane, regs);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      auto a_tile = a.template slice<kTensorK, BM>(kb * kTensorK, 0);
      op.run(a_tile, w_tile, acc);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Reuse the decoded tile for split-K partials and add them in split order.
  if constexpr (KS > 1) {
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i)
      scratch[sg][lane * acc.get_capacity() + i] = as_type<uint>(acc[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid || sg % KS != 0) return;
  }
  #pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    const auto coord = acc.get_multidimensional_index(i);
    const uint m = m0 + coord[1], n = n0 + coord[0];
    if (m < first_row) continue;
    float sum = acc[i];
    #pragma unroll
    for (uint split = 1; split < KS; ++split)
      sum += as_type<float>(scratch[sg + split][lane * acc.get_capacity() + i]);
    store_linear(y, bias, p, m, n, sum);
  }
}

}  // namespace

#define NF12_TENSOR_KERNEL(BM, KS)                                                                                  \
  NF12_THREADS(32 * kTensorSimdgroups) kernel void nf12_linear_tensor_nf12_m##BM##_s##KS(                           \
      device const bfloat* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],                               \
      device const uint4* escapes [[buffer(2)]], device const ushort* bias [[buffer(3)]],                           \
      device ushort* y [[buffer(4)]], constant LinearParams& p [[buffer(5)]],                                       \
      device const uchar* exp_base [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                        \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                        \
    threadgroup uint scratch[kTensorSimdgroups][kScratchWords<BM, KS>];                                             \
    linear_tensor<BM, KS>(NF12Source{packed, escapes, exp_base_x4(exp_base[0])}, x, bias, y, p, tg, sg, lane,         \
                          scratch);                                                                                \
  }                                                                                                                \
  NF12_THREADS(32 * kTensorSimdgroups) kernel void nf12_linear_tensor_bf16t_m##BM##_s##KS(                          \
      device const bfloat* x [[buffer(0)]], device const uint4* packed [[buffer(1)]],                               \
      device const ushort* bias [[buffer(3)]], device ushort* y [[buffer(4)]], constant LinearParams& p [[buffer(5)]], \
      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                        \
      uint lane [[thread_index_in_simdgroup]]) {                                                                    \
    threadgroup uint scratch[kTensorSimdgroups][kScratchWords<BM, KS>];                                             \
    linear_tensor<BM, KS>(Bf16tSource{packed}, x, bias, y, p, tg, sg, lane, scratch);                                \
  }

NF12_TENSOR_KERNEL(16, 1)
NF12_TENSOR_KERNEL(16, 4)
NF12_TENSOR_KERNEL(32, 1)
NF12_TENSOR_KERNEL(32, 4)
NF12_TENSOR_KERNEL(64, 1)
NF12_TENSOR_KERNEL(64, 4)
