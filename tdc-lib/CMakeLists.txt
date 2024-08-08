cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(tdc-lib VERSION 1.0.0 LANGUAGES CXX CUDA)

# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package(CUDAToolkit REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_library(tdc-lib STATIC
    src/tdckernels.cu
    src/tdcprocessor.cpp
)

target_include_directories(tdc-lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(tdc-lib PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# This breaks pybind11! No idea why.
# Unfortunately this means we can't use separable compilation
# set_target_properties(tdc-lib PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(tdc-lib PUBLIC
    fmt::fmt
    spdlog::spdlog
    CUDA::cudart_static
    CUDA::nppc
)
