// SPDX-License-Identifier: Apache-2.0
//
// Device capabilities as traits, not an arch-string table.
//
// The RDNA4 port picked every code path from `gcnArchName` (and, worse, from
// the device-pass-only `__gfx1201__`). That is the wrong axis. gfx942 (CDNA3)
// and gfx1201 (RDNA4) differ in wave width, matrix-core family, packed-dot
// support, CU count, L2 size and XCC (chiplet) count, and those are what the
// kernels actually branch on. Two devices can share an arch string and still
// differ in CU count, and one process can see a mix, so every decision below
// is a runtime query of the device the work runs on, cached per ordinal.
//
// The traits are a host-side POD that launchers read; no declaration changes
// between the host and device passes, so a device compile and a host compile
// can never disagree about a symbol.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace nf12_hip {

// Lanes that own one NF12 super-atom. This is a property of the FORMAT (16x64,
// four atoms, one lane per (g,t) with g = lane>>2, t = lane&3), not of the
// hardware wave. On a wave32 device it equals the wave; on a wave64 device two
// of these groups share one hardware wave. Every codec kernel indexes by this,
// which is why the same code is correct on both.
inline constexpr int kAtomLanes = 32;

// 8 lane-groups per CTA on both backends: 256 threads at wave32, and 256
// threads made of two 32-lane groups per wave at wave64. Keeping the CTA size
// fixed keeps the RDNA4 occupancy tuning valid and sidesteps the case where
// one half of a wave early-returns while the other hits a wave barrier.
inline constexpr int kGroupsPerBlock = 8;

// Which bf16 matrix-core family this device's wave has. `kNone` selects the
// decode-then-GEMM route, which every device supports.
enum class MatrixCore {
  kNone = 0,  // no bf16 matrix core we can drive
  kWmma,      // RDNA wave32 v_wmma_*_gfx12 (gfx1200 / gfx1201)
  kMfma,      // CDNA wave64 v_mfma_* (gfx90a / gfx942 / gfx950)
};

// A snapshot of one device. Cheap to copy; the launchers take it by value.
struct DeviceTraits {
  int device = -1;
  int wave_size = 32;      // hardware wave: 32 (RDNA) or 64 (CDNA)
  int compute_units = 0;   // hipDeviceAttributeMultiprocessorCount
  int xccs = 1;            // chiplets (XCDs); a grid hint, not a launch bound
  int l2_bytes = 0;        // LLC size, sets the streaming/occupancy budget
  int smem_per_cu = 0;     // shared memory per CU
  int max_threads_per_cu = 0;
  int clock_khz = 0;
  int memory_bus_bits = 0;
  int memory_clock_khz = 0;
  MatrixCore matrix = MatrixCore::kNone;
  bool packed_dot = false;  // v_dot2_f32_bf16 (dot12-insts)
  bool perm = false;        // v_perm_b32

  // Bytes an L2-resident working set can count on before streaming starts.
  int64_t llc_budget() const { return l2_bytes > 0 ? l2_bytes : 0; }
};

// Traits for an explicit ordinal (marked speed-cached). Returns a default
// (kNone) record for an ordinal that does not exist rather than reading past
// the per-device table.
DeviceTraits device_traits(int device);

// Wave width for an ordinal; falls back to 32 for an unknown device so callers
// that only need a launch shape never crash.
int wave_size(int device);

// Whole waves a grid of `ctas` CTAs covers on this device's CU count. The
// split-K and tile policy reason in waves; a 304-CU CDNA3 part is ~9.5x the
// width of the 32-CU RDNA4 part the original constants were measured on, so
// the wave size has to come from the device, never from a literal.
inline int64_t waves_for(int64_t ctas, const DeviceTraits& t) {
  const int64_t per = t.compute_units > 0 ? t.compute_units : 1;
  return (ctas + per - 1) / per;
}

}  // namespace nf12_hip
