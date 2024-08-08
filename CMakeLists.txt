cmake_minimum_required(VERSION 3.22)

file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" VCPKG_ROOT)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

project(sarlab-tdc VERSION 1.0.0 LANGUAGES CXX CUDA)

# Enables testing
include(CTest)
enable_testing()

# ##############################################################################
# SUBPROJECTS
# ##############################################################################
add_subdirectory(tdc-lib)
add_subdirectory(tdctest)
