cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(fastJEPA
  VERSION 0.1.0
  LANGUAGES C
  DESCRIPTION "Fast JEPA in C and CUDA"
)

option(USE_CUDA "Enable CUDA backend" OFF)

if(USE_CUDA)
  if(NOT CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit install prefix")
  endif()
  if(NOT CMAKE_CUDA_COMPILER AND DEFINED CUDAToolkit_ROOT)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;89" CACHE STRING "CUDA architectures")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_definitions(FJ_WITH_CUDA)
endif()

if(NOT WIN32)
  add_compile_definitions(_POSIX_C_SOURCE=200809L)
endif()

add_compile_options(-Wall -Wextra)

include_directories(csrc tests)

# Placeholder: add sources as implemented.
# add_executable(fastjepa csrc/main.c)
# target_link_libraries(fastjepa m)
"