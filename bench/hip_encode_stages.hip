// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// Benchmark-only access to the actual kernels; no copies of codec logic.
#include "encode.hip"
#include "pack_bf16t.hip"

extern "C" hipError_t nf12_bench_encode_stages(
    int stage, const void* input, int64_t n, int64_t k, uint8_t base,
    void* packed, void* escapes, void* workspace, hipStream_t stream) {
  using namespace nf12_hip;
  const auto* w = static_cast<const __hip_bfloat16*>(input);
  const int64_t sa = num_super_atoms(n, k), units = num_units(sa);
  const unsigned blocks = (units + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto* counts = static_cast<uint64_t*>(workspace);
  // stage 0: analyze histogram; 1: count/scan/write; 2: both; 3: BF16t.
  // Base/capacity are validated before timing. No host selection or copies.
  if (stage == 0 || stage == 2) {
    auto err = hipMemsetAsync(workspace, 0, kAnalyzeWorkspaceBytes, stream);
    if (err != hipSuccess) return err;
    analyze_kernel<<<grid_blocks<kAnalyzeSuperAtomsPerWarp>(sa), kThreadsPerBlock, 0, stream>>>(
        w, k, sa, k / kSuperAtomCols, static_cast<unsigned long long*>(workspace));
  }
  if (stage == 1 || stage == 2) {
    count_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        w, k, sa, k / kSuperAtomCols, exp_base_x4(base), counts, units);
    scan_kernel<<<1, kScanThreads, 0, stream>>>(counts, units);
    write_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        w, k, sa, k / kSuperAtomCols, exp_base_x4(base), counts, units,
        static_cast<uint4*>(packed), static_cast<uint4*>(escapes));
  }
  if (stage == 3) return pack_bf16t(input, n, k, k, static_cast<uint8_t*>(packed), stream);
  return hipGetLastError();
}
