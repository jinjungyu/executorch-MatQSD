# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for selective build demo.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.24) # 3.24 is required for WHOLE_ARCHIVE
project(llama_runner)

# Duplicating options as root CMakeLists.txt
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build the optimized kernels" OFF)

include(CMakeDependentOption)
#
# pthreadpool: build pthreadpool library. Disable on unsupported platforms
#
cmake_dependent_option(
  EXECUTORCH_BUILD_PTHREADPOOL "Build pthreadpool library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)
#
# cpuinfo: build cpuinfo library. Disable on unsupported platforms
#
cmake_dependent_option(
  EXECUTORCH_BUILD_CPUINFO "Build cpuinfo library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)

option(EXECUTORCH_BUILD_KERNELS_TORCHAO_MPS "Build the torchao MPS kernels" OFF)

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  # Can't set to 11 due to executor_runner.cpp make_unique
endif()

if(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  set(CMAKE_TOOLCHAIN_IOS ON)
else()
  set(CMAKE_TOOLCHAIN_IOS OFF)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# For some reason android build is not able to find where gflags is and hence
# cannot find corresponding .cmake file
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

#
# llama_main: test binary to run llama, with tokenizer and sampler integrated
#

# find `executorch` libraries. CMAKE_PREFIX_PATH would work for host
# compilation, but CMAKE_FIND_ROOT_PATH appears to be necessary for
# cross-compiling (e.g., to Android) to work as well.
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

# llama_runner library
add_subdirectory(runner)

set(link_libraries executorch gflags)
set(_srcs main.cpp)

if(TARGET optimized_native_cpu_ops_lib)
  list(
    APPEND
    link_libraries
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas
  )
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()

# quantized_ops_lib: Register quantized op kernels into the runtime
executorch_target_link_options_shared_lib(quantized_ops_lib)
list(APPEND link_libraries quantized_kernels quantized_ops_lib)

if(TARGET custom_ops)
  executorch_target_link_options_shared_lib(custom_ops)
  list(APPEND link_libraries custom_ops)
endif()

if(TARGET torchao_ops_executorch)
  executorch_target_link_options_shared_lib(torchao_ops_executorch)
  list(APPEND link_libraries torchao_ops_executorch)
else()
  # When building llama as a standalone sub-project, the torchao cmake target
  # is not available. Link the pre-built static libraries directly.
  set(_torchao_lib "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake-out/third-party/ao/torchao/csrc/cpu/libtorchao_ops_executorch.a")
  set(_torchao_kernels_lib "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake-out/third-party/ao/torchao/csrc/cpu/torch_free_kernels/aarch64/libtorchao_kernels_aarch64.a")
  if(EXISTS ${_torchao_lib})
    message(STATUS "Linking torchao_ops_executorch from pre-built: ${_torchao_lib}")
    # Use -force_load to prevent linker from stripping self-registering ops
    list(APPEND link_libraries "-Wl,-force_load,${_torchao_lib}")
    if(EXISTS ${_torchao_kernels_lib})
      list(APPEND link_libraries ${_torchao_kernels_lib})
    endif()
  endif()
endif()

if(EXECUTORCH_BUILD_KERNELS_TORCHAO_MPS)
  # Currently only enable this on Arm-based Macs
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR STREQUAL
                                             "arm64"
  )
    if(EXECUTORCH_BUILD_MPS)
      add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/ao/torchao/experimental/ops/mps
        ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/ao/torchao/experimental/ops/mps
      )
      executorch_target_link_options_shared_lib(torchao_ops_mps_executorch)
      list(APPEND link_libraries torchao_ops_mps_executorch)
    endif()
  endif()
endif()

set(XNNPACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../backends/xnnpack)
# Extra compile option and include dir for pthreadpool
if(EXECUTORCH_BUILD_PTHREADPOOL)
  list(APPEND link_libraries extension_threadpool pthreadpool)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/pthreadpool/include
  )
endif()

# Extra sources for cpuinfo
if(EXECUTORCH_BUILD_CPUINFO)
  list(APPEND link_libraries extension_threadpool cpuinfo)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/cpuinfo/include
  )
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_backend_libs})
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()

# Vulkan backend
if(TARGET vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
  executorch_target_link_options_shared_lib(vulkan_backend)
endif()

# Qnn backend
if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
  executorch_target_link_options_shared_lib(qnn_executorch_backend)
endif()

# MPS backend
if(TARGET mpsdelegate)
  list(
    APPEND
    link_libraries
    mpsdelegate
    "-framework Foundation"
    "-weak_framework MetalPerformanceShaders"
    "-weak_framework MetalPerformanceShadersGraph"
    "-weak_framework Metal"
  )
  executorch_target_link_options_shared_lib(mpsdelegate)
endif()

if(TARGET coremldelegate)
  find_library(SQLITE_LIBRARY sqlite3)
  list(
    APPEND
    link_libraries
    coremldelegate
    sqlite3
    "-framework Foundation"
    "-framework CoreML"
    "-framework Accelerate"
  )
  executorch_target_link_options_shared_lib(coremldelegate)
endif()

# This one is needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

add_executable(llama_main ${_srcs})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(llama_main)
  if(NOT APPLE)
    target_link_options(llama_main PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(llama_main PUBLIC ${_common_include_directories})
target_link_libraries(llama_main PUBLIC llama_runner ${link_libraries})
target_compile_options(llama_main PUBLIC ${_common_compile_options})
if(APPLE)
  target_link_options(llama_main PRIVATE -Wl,-rpath,@loader_path)
elseif(UNIX)
  set_target_properties(llama_main PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
endif()
# Windows doesn't need rpath - DLLs are found via standard Windows search order

# MatQSD test binary
add_executable(matqsd_test ${CMAKE_CURRENT_LIST_DIR}/matqsd_test.cpp)
target_include_directories(matqsd_test PUBLIC ${_common_include_directories})
target_link_libraries(matqsd_test PUBLIC extension_module_static extension_tensor ${link_libraries})
target_compile_options(matqsd_test PUBLIC ${_common_compile_options})

# PPL evaluator
add_executable(ppl_eval ${CMAKE_CURRENT_LIST_DIR}/ppl_eval.cpp)
target_include_directories(ppl_eval PUBLIC ${_common_include_directories})
target_link_libraries(ppl_eval PUBLIC llama_runner ${link_libraries})
target_compile_options(ppl_eval PUBLIC ${_common_compile_options})

# KV test
add_executable(kv_test ${CMAKE_CURRENT_LIST_DIR}/kv_test.cpp)
target_include_directories(kv_test PUBLIC ${_common_include_directories})
target_link_libraries(kv_test PUBLIC extension_module_static extension_tensor ${link_libraries})
target_compile_options(kv_test PUBLIC ${_common_compile_options})

# SD KV bench (real SD with KV cache)
add_executable(sd_kv_bench ${CMAKE_CURRENT_LIST_DIR}/sd_kv_bench.cpp)
target_include_directories(sd_kv_bench PUBLIC ${_common_include_directories})
target_link_libraries(sd_kv_bench PUBLIC extension_module_static extension_tensor ${link_libraries})
target_compile_options(sd_kv_bench PUBLIC ${_common_compile_options})

# SD bench (MatQSD draft-verify cycle timing)
add_executable(sd_bench ${CMAKE_CURRENT_LIST_DIR}/sd_bench.cpp)
target_include_directories(sd_bench PUBLIC ${_common_include_directories})
target_link_libraries(sd_bench PUBLIC extension_module_static extension_tensor ${link_libraries})
target_compile_options(sd_bench PUBLIC ${_common_compile_options})

# SD runner
add_executable(sd_runner ${CMAKE_CURRENT_LIST_DIR}/sd_runner.cpp)
target_include_directories(sd_runner PUBLIC ${_common_include_directories})
target_link_libraries(sd_runner PUBLIC llama_runner ${link_libraries})
target_compile_options(sd_runner PUBLIC ${_common_compile_options})

add_executable(kv_multi_test ${CMAKE_CURRENT_LIST_DIR}/kv_multi_test.cpp)
target_include_directories(kv_multi_test PUBLIC ${_common_include_directories})
target_link_libraries(kv_multi_test PUBLIC extension_module_static extension_tensor ${link_libraries})
target_compile_options(kv_multi_test PUBLIC ${_common_compile_options})

add_executable(test_s3 ${CMAKE_CURRENT_LIST_DIR}/test_s3.cpp)
target_include_directories(test_s3 PUBLIC ${_common_include_directories})
target_link_libraries(test_s3 PUBLIC extension_module_static extension_tensor ${link_libraries})
target_compile_options(test_s3 PUBLIC ${_common_compile_options})

add_executable(test_mqint8 ${CMAKE_CURRENT_BINARY_DIR}/test_mqint8.cpp)
target_include_directories(test_mqint8 PUBLIC ${_common_include_directories})
target_link_libraries(test_mqint8 PUBLIC extension_module_static extension_tensor ${link_libraries})
target_compile_options(test_mqint8 PUBLIC ${_common_compile_options})

# MatQSD demo (ncurses TUI: AR-8bit vs MatQSD side-by-side)
find_library(NCURSES_LIB ncurses)
if(NCURSES_LIB)
  add_executable(demo ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)
  target_include_directories(demo PUBLIC ${_common_include_directories})
  target_link_libraries(demo PUBLIC llama_runner ${link_libraries} ${NCURSES_LIB} "-framework IOKit" "-framework CoreFoundation")
  target_compile_options(demo PUBLIC ${_common_compile_options})

  add_executable(demo_gemma4 ${CMAKE_CURRENT_LIST_DIR}/demo_gemma4.cpp)
  target_include_directories(demo_gemma4 PUBLIC ${_common_include_directories})
  target_link_libraries(demo_gemma4 PUBLIC llama_runner ${link_libraries} ${NCURSES_LIB} "-framework IOKit" "-framework CoreFoundation")
  target_compile_options(demo_gemma4 PUBLIC ${_common_compile_options})
endif()
