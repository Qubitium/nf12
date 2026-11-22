// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace nf12::metal {

// Both Metal host APIs refer to the same Objective-C objects. The caller
// owns the returned library, as for Metal's newLibrary methods.
void* new_shader_library(void* device);

}  // namespace nf12::metal
