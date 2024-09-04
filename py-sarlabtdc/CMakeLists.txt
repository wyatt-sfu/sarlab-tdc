cmake_minimum_required(VERSION 3.22)

file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" VCPKG_ROOT)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(_sarlabtdc VERSION 1.0.0 LANGUAGES CXX CUDA)

find_package(pybind11 CONFIG REQUIRED)

# Configure the C++ library as a subdirectory
set(TDCLIB_DIR ${CMAKE_SOURCE_DIR}/../tdc-lib)
message("TDCLIB_DIR: ${TDCLIB_DIR}")
add_subdirectory(${TDCLIB_DIR} ${CMAKE_SOURCE_DIR}/build/tdc-lib)

set(CUDATYPES_DIR ${CMAKE_SOURCE_DIR}/../cudatypes)
message("CUDATYPES_DIR: ${CUDATYPES_DIR}")
add_subdirectory(${CUDATYPES_DIR} ${CMAKE_SOURCE_DIR}/build/cudatypes)

# Create the pybind module
pybind11_add_module(_sarlabtdc
    src/interface.cpp
    src/tdcprocessor_wrapper.cpp
)

target_include_directories(_sarlabtdc PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(_sarlabtdc PRIVATE
    tdc-lib
    cudatypes
)
