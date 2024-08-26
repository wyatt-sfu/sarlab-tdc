cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cudatypes VERSION 1.0.0 LANGUAGES CXX CUDA)

# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package(CUDAToolkit REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_library(cudatypes INTERFACE)

target_include_directories(cudatypes INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(cudatypes INTERFACE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
