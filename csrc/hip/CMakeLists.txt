# SPDX-License-Identifier: Apache-2.0
#
# Standalone build for the RDNA4 HIP kernels:
#
#   cmake -S csrc/hip -B build-hip -DNF12_HIP_ARCHS="gfx1201"
#   cmake --build build-hip -j
#
# Also included from the top-level build when -DNF12_ENABLE_HIP=ON.
cmake_minimum_required(VERSION 3.26)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(nf12_hip LANGUAGES CXX HIP)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# gfx1200/gfx1201 are validated. gfx1250 has no wmma-128b-insts, so wmma.hip
# compiles to an empty translation unit there and the caller keeps the
# decode-then-GEMM route.
set(NF12_HIP_ARCHS "gfx1200;gfx1201" CACHE STRING "AMD GPU targets to build for")

add_library(nf12_hip SHARED
  decode.hip
  encode.hip
  wmma.hip
  embedding.hip
  pack_bf16t.hip
  c_api.hip)
target_include_directories(nf12_hip
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# csrc/torch/ops.cpp includes nf12/nf12.h; NF12_HIP selects the HIP aliases
# there so the same op layer builds for both backends.
target_compile_definitions(nf12_hip PUBLIC NF12_HIP)
# C++ consumers of nf12.h (the torch op layer) need the HIP headers and the
# platform macro; the HIP compiler supplies them for the .hip sources.
get_filename_component(NF12_ROCM_ROOT "${CMAKE_HIP_COMPILER}" DIRECTORY)
get_filename_component(NF12_ROCM_ROOT "${NF12_ROCM_ROOT}" DIRECTORY)
target_include_directories(nf12_hip INTERFACE "${NF12_ROCM_ROOT}/include")
target_compile_definitions(nf12_hip INTERFACE __HIP_PLATFORM_AMD__)
# Land next to libnf12.so so a single directory holds either backend.
set_target_properties(nf12_hip PROPERTIES
  HIP_ARCHITECTURES "${NF12_HIP_ARCHS}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Install next to libnf12.so (the top-level CUDA target also installs to
# ``nf12``), so a native HIP build can be packaged the same way as a CUDA one.
install(TARGETS nf12_hip LIBRARY DESTINATION nf12)

# --use_fast_math would violate the numerical contract in SPEC.md section 6.
target_compile_options(nf12_hip PRIVATE -O3 -Wall -Wextra)

# The torch extension is the same csrc/torch/ops.cpp as the CUDA build. It
# reaches the HIP kernels through the nf12_* compatibility layer in c_api.cpp,
# so the op schema, dispatch and graph handling are not duplicated.
if(NF12_BUILD_TORCH)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE NF12_TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  unset(NF12_TORCH_CPU_LIB CACHE)
  unset(NF12_TORCH_HIP_LIB CACHE)
  find_library(NF12_TORCH_CPU_LIB torch_cpu PATHS "${NF12_TORCH_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(NF12_TORCH_HIP_LIB torch_hip PATHS "${NF12_TORCH_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(nf12_torch SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/../torch/ops.cpp
    fused_decode_stub.cpp)
  target_include_directories(nf12_torch SYSTEM PRIVATE "${NF12_TORCH_DIR}/include")
  target_include_directories(nf12_torch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
  # USE_ROCM exposes the CUDA-named stream shim in torch's stable ABI header.
  target_compile_definitions(nf12_torch PRIVATE USE_ROCM
    TORCH_TARGET_VERSION=0x020A000000000000ULL)
  target_link_libraries(nf12_torch PRIVATE nf12_hip "${NF12_TORCH_CPU_LIB}" "${NF12_TORCH_HIP_LIB}")
  set_target_properties(nf12_torch PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "$ORIGIN;${NF12_TORCH_DIR}/lib"
    INSTALL_RPATH "$ORIGIN")
  install(TARGETS nf12_torch LIBRARY DESTINATION nf12)
endif()
