// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "nf12_shader_source.h"
#include "shader_library.h"

namespace nf12::metal {

void* new_shader_library(void* device) {
  @autoreleasepool {
    id<MTLDevice> gpu = static_cast<id<MTLDevice>>(device);
    NSError* error = nil;
    id<MTLLibrary> library;
    const char* path = std::getenv("NF12_METALLIB");
    const bool precompiled = path != nullptr && *path != '\0';
    if (precompiled) {
      library = [gpu newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
    } else {
      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
      options.languageVersion = MTLLanguageVersion3_1;
      // Metal 3.1 uses fastMathEnabled. Keep the flag for the macOS 14
      // deployment target and preserve the offline build's precise math.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
      library = [gpu newLibraryWithSource:@(kShaderSource) options:options error:&error];
      [options release];
    }
    if (library == nil) {
      const std::string source = precompiled ? path : "embedded Metal source";
      throw std::runtime_error("nf12: failed to create Metal library from " + source + ": " +
                               (error ? error.localizedDescription.UTF8String : "unknown error"));
    }
    library.label = @"nf12";
    return library;
  }
}

}  // namespace nf12::metal
