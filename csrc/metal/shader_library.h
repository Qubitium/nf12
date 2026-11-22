// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

namespace nf12::metal {

// Both Metal host APIs refer to the same Objective-C objects. The caller
// owns the returned library, as for Metal's newLibrary methods.
void* new_shader_library(void* device, bool tensor = false);
bool supports_tensor_linear(void* device);

inline bool is_tensor_kernel(std::string_view name) {
  return name.starts_with("nf12_linear_tensor_");
}

}  // namespace nf12::metal
