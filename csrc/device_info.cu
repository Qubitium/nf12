// SPDX-License-Identifier: Apache-2.0
#include <memory>
#include <mutex>

#include "internal.h"

namespace nf12::detail {

namespace {

struct DeviceCache {
  int count = 0;
  std::unique_ptr<std::once_flag[]> once;
  std::unique_ptr<DeviceInfo[]> info;

  DeviceCache() {
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
      cudaGetLastError();  // leave the caller's CUDA error state clear
      count = 0;
    }
    once.reset(new std::once_flag[count]);
    info.reset(new DeviceInfo[count]);
  }
};

DeviceCache& cache() {
  static DeviceCache c;
  return c;
}

int attribute(cudaDeviceAttr attr, int device) {
  int value = 0;
  cudaDeviceGetAttribute(&value, attr, device);
  return value;
}

}

// The C API validates device against the device count before calling.
const DeviceInfo& device_info(int device) {
  DeviceCache& c = cache();
  std::call_once(c.once[device], [&] {
    DeviceInfo& d = c.info[device];
    d.device = device;
    d.sm_arch = attribute(cudaDevAttrComputeCapabilityMajor, device) * 10 +
                attribute(cudaDevAttrComputeCapabilityMinor, device);
    d.num_sms = attribute(cudaDevAttrMultiProcessorCount, device);
    d.smem_optin_bytes = static_cast<size_t>(attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    d.l2_bytes = static_cast<size_t>(attribute(cudaDevAttrL2CacheSize, device));
  });
  return c.info[device];
}

}
