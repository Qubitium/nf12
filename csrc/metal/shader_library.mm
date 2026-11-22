// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "nf12_shader_source.h"
#include "shader_library.h"

namespace nf12::metal {

bool supports_tensor_linear(void* device) {
  if (@available(macOS 27, *)) {
    // Apple10 introduces matrix acceleration for TensorOps. Use the enum
    // value so the host extension can also build against older SDKs.
    constexpr auto tensor_accelerator_family = static_cast<MTLGPUFamily>(1010);  // MTLGPUFamilyApple10
    id<MTLDevice> gpu = static_cast<id<MTLDevice>>(device);
    return [gpu supportsFamily:tensor_accelerator_family];
  }
  return false;
}

void* new_shader_library(void* device, bool tensor) {
  @autoreleasepool {
    id<MTLDevice> gpu = static_cast<id<MTLDevice>>(device);
    NSError* error = nil;
    id<MTLLibrary> library;
    const char* path = std::getenv("NF12_METALLIB");
    const bool precompiled = !tensor && path != nullptr && *path != '\0';
    if (precompiled) {
      library = [gpu newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
    } else {
      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
      // Use the numeric value to allow SDKs predating Metal 4.1.
      options.languageVersion = tensor ? static_cast<MTLLanguageVersion>((4 << 16) + 1) : MTLLanguageVersion3_1;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
      if (tensor) {
        if (@available(macOS 15, *)) {
          options.mathMode = MTLMathModeSafe;
          options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        }
      }
#endif
      const char* source = tensor ? kTensorShaderSource : kShaderSource;
      library = [gpu newLibraryWithSource:@(source) options:options error:&error];
      [options release];
    }
    if (library == nil) {
      const std::string source = precompiled ? path : "embedded Metal source";
      throw std::runtime_error("nf12: failed to create Metal library from " + source + ": " +
                               (error ? error.localizedDescription.UTF8String : "unknown error"));
    }
    library.label = tensor ? @"nf12_tensor" : @"nf12";
    return library;
  }
}

}  // namespace nf12::metal
