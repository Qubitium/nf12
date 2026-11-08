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
  wmma.hip)
target_include_directories(nf12_hip
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Land next to libnf12.so so a single directory holds either backend.
set_target_properties(nf12_hip PROPERTIES
  HIP_ARCHITECTURES "${NF12_HIP_ARCHS}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# --use_fast_math would violate the numerical contract in SPEC.md section 6.
target_compile_options(nf12_hip PRIVATE -O3 -Wall -Wextra)
