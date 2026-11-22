# SPDX-License-Identifier: Apache-2.0
# The MPS and MLX extensions embed the same Metal source and compile it at
# first GPU use. Apple Command Line Tools can build both host extensions.

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)  # torch >= 2.13 headers require C++20
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Metal's in-process compiler has no include path for project headers.
set(_nf12_metal_dir "${CMAKE_CURRENT_SOURCE_DIR}/csrc/metal")
function(nf12_embed_metal output)
  set(source "")
  foreach(file IN LISTS ARGN)
    set(path "${_nf12_metal_dir}/${file}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
    file(READ "${path}" contents)
    string(REGEX REPLACE [=[#pragma once|#include "(nf12_metal|linear_core)[.]h"]=] "" contents "${contents}")
    string(APPEND source "#line 1 \"${file}\"\n${contents}\n")
  endforeach()
  set(${output} "${source}" PARENT_SCOPE)
endfunction()
nf12_embed_metal(_nf12_shader_source nf12_metal.h linear_core.h codec.metal linear.metal moe.metal)
nf12_embed_metal(_nf12_tensor_source nf12_metal.h linear_tensor.metal)
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/nf12_shader_source.h" CONTENT [=[
#pragma once
namespace nf12::metal {
inline constexpr char kShaderSource[] = R"nf12metal(@_nf12_shader_source@)nf12metal";
inline constexpr char kTensorShaderSource[] = R"nf12metal(@_nf12_tensor_source@)nf12metal";
}
]=] @ONLY)

find_package(Python COMPONENTS Interpreter REQUIRED)

# --- nf12._mlx_ext (MLX) --------------------------------------------------------
# AUTO builds it when the build interpreter can import mlx and nanobind.
set(NF12_BUILD_MLX AUTO CACHE STRING "Build the MLX extension (ON, OFF or AUTO)")
set_property(CACHE NF12_BUILD_MLX PROPERTY STRINGS ON OFF AUTO)
if(NOT NF12_BUILD_MLX STREQUAL "OFF")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import os, mlx.core, nanobind; print(os.path.dirname(mlx.core.__file__)); print(nanobind.cmake_dir()); print(mlx.core.__version__)"
    OUTPUT_VARIABLE _nf12_mlx_probe
    RESULT_VARIABLE _nf12_mlx_rc
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_nf12_mlx_rc EQUAL 0)
    string(REPLACE "\n" ";" _nf12_mlx_probe "${_nf12_mlx_probe}")
    list(GET _nf12_mlx_probe 0 NF12_MLX_DIR)
    list(GET _nf12_mlx_probe 1 _nf12_nanobind_dir)
    list(GET _nf12_mlx_probe 2 NF12_MLX_VERSION)
    message(STATUS "nf12: MLX ${NF12_MLX_VERSION} at ${NF12_MLX_DIR}")
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED PATHS "${_nf12_nanobind_dir}" NO_DEFAULT_PATH)
    unset(NF12_MLX_LIB CACHE)
    find_library(NF12_MLX_LIB mlx PATHS "${NF12_MLX_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
    # MLX's nanobind bindings use the full (not stable) ABI in the "mlx" domain;
    # sharing both lets nf12 accept mx.array directly.
    nanobind_add_module(_mlx_ext NB_STATIC NB_DOMAIN mlx NOMINSIZE csrc/mlx/nf12_mlx.cpp csrc/metal/shader_library.mm)
    target_include_directories(_mlx_ext SYSTEM PRIVATE "${NF12_MLX_DIR}/include" "${NF12_MLX_DIR}/include/metal_cpp")
    target_include_directories(_mlx_ext PRIVATE "${_nf12_metal_dir}" "${CMAKE_CURRENT_BINARY_DIR}")
    # MLX has no stable C++ ABI; nf12.mlx rebuilds when the runtime version differs.
    target_compile_definitions(_mlx_ext PRIVATE _METAL_ ACCELERATE_NEW_LAPACK NF12_MLX_VERSION="${NF12_MLX_VERSION}")
    target_compile_options(_mlx_ext PRIVATE -Wall -Wextra)
    # MLX's exported CMake target hard-codes the frameworks of the SDK it was
    # built with, so link the library and system frameworks directly.
    target_link_libraries(_mlx_ext PRIVATE "${NF12_MLX_LIB}" "-framework Metal" "-framework Foundation")
    set_target_properties(_mlx_ext PROPERTIES
      CXX_STANDARD 20
      BUILD_RPATH "${NF12_MLX_DIR}/lib"
      INSTALL_RPATH "@loader_path/../mlx/lib;${NF12_MLX_DIR}/lib")
    install(TARGETS _mlx_ext LIBRARY DESTINATION nf12)
  elseif(NF12_BUILD_MLX STREQUAL "ON")
    message(FATAL_ERROR "nf12: NF12_BUILD_MLX=ON but ${Python_EXECUTABLE} cannot import mlx and nanobind")
  else()
    message(STATUS "nf12: mlx or nanobind not importable; skipping the MLX extension")
  endif()
endif()

# --- libnf12_mps.dylib --------------------------------------------------------
option(NF12_BUILD_TORCH "Build the torch extension (libnf12_mps.dylib)" ON)
if(NOT NF12_BUILD_TORCH)
  return()
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.dirname(torch.__file__)); print(torch.__version__)"
  OUTPUT_VARIABLE _nf12_torch_probe
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" _nf12_torch_probe "${_nf12_torch_probe}")
list(GET _nf12_torch_probe 0 NF12_TORCH_DIR)
list(GET _nf12_torch_probe 1 NF12_TORCH_VERSION)
message(STATUS "nf12: torch ${NF12_TORCH_VERSION} at ${NF12_TORCH_DIR}")
# Isolated builds can locate torch in a different temporary environment each time.
foreach(_lib torch torch_cpu c10)
  unset(NF12_TORCH_${_lib}_LIB CACHE)
  find_library(NF12_TORCH_${_lib}_LIB ${_lib} PATHS "${NF12_TORCH_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
endforeach()

# MPS stream access (ATen/mps/MPSStream.h) is not part of torch's stable ABI, so
# this library is built against the installed torch (see python/nf12/_jit.py).
add_library(nf12_mps SHARED csrc/metal/ops_mps.mm csrc/metal/shader_library.mm)
target_include_directories(nf12_mps PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(nf12_mps SYSTEM PRIVATE
  "${NF12_TORCH_DIR}/include" "${NF12_TORCH_DIR}/include/torch/csrc/api/include")
target_compile_options(nf12_mps PRIVATE -Wall -Wextra -fno-objc-arc)
target_link_libraries(nf12_mps PRIVATE
  "${NF12_TORCH_torch_LIB}" "${NF12_TORCH_torch_cpu_LIB}" "${NF12_TORCH_c10_LIB}"
  "-framework Metal" "-framework Foundation")
set_target_properties(nf12_mps PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  OBJCXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  BUILD_RPATH "${NF12_TORCH_DIR}/lib"
  INSTALL_RPATH "@loader_path/../torch/lib;${NF12_TORCH_DIR}/lib")
install(TARGETS nf12_mps LIBRARY DESTINATION nf12)
# The torch this library was built against: python/nf12/_library.py loads it only
# under the same torch version and otherwise rebuilds (python/nf12/_jit.py).
set(NF12_MPS_TORCH_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/libnf12_mps.torch_version")
file(WRITE "${NF12_MPS_TORCH_VERSION_FILE}" "${NF12_TORCH_VERSION}\n")
install(FILES "${NF12_MPS_TORCH_VERSION_FILE}" DESTINATION nf12)
