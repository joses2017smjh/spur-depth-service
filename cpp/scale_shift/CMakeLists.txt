cmake_minimum_required(VERSION 3.16)
project(scale_shift LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SCALE_SHIFT_OPENMP "Build with OpenMP" ON)
option(SCALE_SHIFT_BUILD_TESTS "Build GoogleTest suite" ON)
option(SCALE_SHIFT_ENABLE_ASAN "AddressSanitizer" OFF)
option(SCALE_SHIFT_ENABLE_UBSAN "UndefinedBehaviorSanitizer" OFF)

add_library(scale_shift_lib
  src/npy.cpp
  src/fit_scale_shift.cpp
)
target_include_directories(scale_shift_lib PUBLIC include)

if(SCALE_SHIFT_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(scale_shift_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(scale_shift_lib PUBLIC _OPENMP)
  endif()
endif()

function(spur_sanitizers tgt)
  if(SCALE_SHIFT_ENABLE_ASAN)
    target_compile_options(${tgt} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(${tgt} PRIVATE -fsanitize=address)
  endif()
  if(SCALE_SHIFT_ENABLE_UBSAN)
    target_compile_options(${tgt} PRIVATE -fsanitize=undefined)
    target_link_options(${tgt} PRIVATE -fsanitize=undefined)
  endif()
endfunction()

spur_sanitizers(scale_shift_lib)

add_executable(scale_shift_fit src/main.cpp)
target_link_libraries(scale_shift_fit PRIVATE scale_shift_lib)
spur_sanitizers(scale_shift_fit)

if(SCALE_SHIFT_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(scale_shift_tests tests/test_fit.cpp)
  target_link_libraries(scale_shift_tests PRIVATE scale_shift_lib GTest::gtest)
  spur_sanitizers(scale_shift_tests)
  add_test(NAME scale_shift_tests COMMAND scale_shift_tests)
endif()
