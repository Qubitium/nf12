// SPDX-License-Identifier: Apache-2.0
// Runtime device traits for the HIP backend. See capability.h.
//
// Everything here is a query of the device the work will run on, cached per
// ordinal. The matrix-core family is decided by asking the compiler which
// intrinsic exists for the target ISA AND the runtime which family the loaded
// device actually is; a mismatch (a gfx1201 build loaded on gfx942, or the
// reverse) reports kNone and the caller keeps the exact decode-then-GEMM route.
#include "capability.h"

#include <hip/hip_runtime_api.h>

#include <atomic>

namespace nf12_hip {
namespace {

constexpr int kMaxDevices = 64;

struct Slot {
  std::atomic<int8_t> state{0};  // 0 unprobed, 1 ready
  DeviceTraits traits{};
};

Slot g_slots[kMaxDevices];

int attr(int what, int device) {
  int value = 0;
  if (hipDeviceGetAttribute(&value, static_cast<hipDeviceAttribute_t>(what), device) != hipSuccess) return 0;
  return value;
}

// The wave width and matrix family of the *loaded* device. The host pass cannot
// call device builtins, so the family is matched against the arch the runtime
// reports, but the arch STRING is only ever used to confirm a family the
// compiler already emitted for; it never decides which kernel is instantiated.
DeviceTraits probe_traits(int device) {
  DeviceTraits t{};
  t.device = device;
  t.wave_size = attr(hipDeviceAttributeWarpSize, device);
  if (t.wave_size <= 0) t.wave_size = 32;
  t.compute_units = attr(hipDeviceAttributeMultiprocessorCount, device);
  t.xccs = attr(hipDeviceAttributeNumberOfXccs, device);
  if (t.xccs < 1) t.xccs = 1;
  t.l2_bytes = attr(hipDeviceAttributeL2CacheSize, device);
  t.smem_per_cu = attr(hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device);
  t.max_threads_per_cu = attr(hipDeviceAttributeMaxThreadsPerMultiProcessor, device);
  t.clock_khz = attr(hipDeviceAttributeClockRate, device);
  t.memory_bus_bits = attr(hipDeviceAttributeMemoryBusWidth, device);
  t.memory_clock_khz = attr(hipDeviceAttributeMemoryClockRate, device);

  hipDeviceProp_t props{};
  const bool have_props = hipGetDeviceProperties(&props, device) == hipSuccess;
  const char* arch = have_props ? props.gcnArchName : "";
  // Family confirmation. RDNA4 is gfx12xx with wave32 and the gfx12 wmma;
  // CDNA3 is gfx94x with wave64 and the mfma family. This is a *consistency*
  // check against the width the runtime reports, not the selector.
  const bool rdna4 = arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' && arch[3] == '1' &&
                     arch[4] == '2' && arch[5] == '0' && (arch[6] == '0' || arch[6] == '1');
  const bool cdna = arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' && arch[3] == '9' &&
                    (arch[4] == '0' || arch[4] == '4' || arch[4] == '5');
  if (rdna4 && t.wave_size == 32) {
    t.matrix = MatrixCore::kWmma;
    t.packed_dot = true;
  } else if (cdna && t.wave_size == 64) {
    t.matrix = MatrixCore::kMfma;
    t.packed_dot = false;  // v_dot2_f32_bf16 is dot12-insts, not on CDNA3
  } else {
    t.matrix = MatrixCore::kNone;
  }
  t.perm = true;
  return t;
}

DeviceTraits load(int device) {
  Slot& slot = g_slots[device];
  if (slot.state.load(std::memory_order_acquire) == 0) {
    slot.traits = probe_traits(device);
    slot.state.store(1, std::memory_order_release);
  }
  return slot.traits;
}

}  // namespace

DeviceTraits device_traits(int device) {
  if (device < 0 || device >= kMaxDevices) return DeviceTraits{};
  return load(device);
}

int wave_size(int device) {
  if (device < 0 || device >= kMaxDevices) return 32;
  return load(device).wave_size;
}

}  // namespace nf12_hip
