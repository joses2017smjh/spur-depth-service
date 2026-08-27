cmake_minimum_required(VERSION 3.16)
project(spur_trt_harness LANGUAGES CXX)

# C2: load a serialized TensorRT engine and run one dummy pass.
# Headers/libs come from a pip wheel (SPUR_TRT_PATH) or TENSORRT_ROOT.
# This target is skipped when NvInfer.h is missing so CPU CI stays green.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_TRT_HINTS
  $ENV{TENSORRT_ROOT}
  $ENV{SPUR_TRT_PATH}
  /nfs/hpc/share/sanchej7/opt/py-tensorrt
)
find_path(NVINFER_INC NvInfer.h
  HINTS ${_TRT_HINTS}
  PATH_SUFFIXES include tensorrt/include nvidia/tensorrt/include)
find_library(NVINFER_LIB nvinfer
  HINTS ${_TRT_HINTS}
  PATH_SUFFIXES lib lib64 tensorrt_libs nvidia/tensorrt_libs)

if(NOT NVINFER_INC OR NOT NVINFER_LIB)
  message(STATUS "TensorRT not found — skipping C2 harness (set SPUR_TRT_PATH)")
  add_custom_target(trt_harness_skipped ALL
    COMMAND ${CMAKE_COMMAND} -E echo "C2 skipped: NvInfer.h not on this machine")
  return()
endif()

add_executable(trt_harness src/harness.cpp)
target_include_directories(trt_harness PRIVATE ${NVINFER_INC})
target_link_libraries(trt_harness PRIVATE ${NVINFER_LIB})
install(TARGETS trt_harness RUNTIME DESTINATION bin)
