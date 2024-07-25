cmake_minimum_required(VERSION 3.22)

file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" VCPKG_ROOT)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

project(sarlab-tdc VERSION 1.0.0 LANGUAGES CXX)

# Enables testing
include(CTest)
enable_testing()

# ##############################################################################
# PLATFORM DEPENDENT CONFIGURATION
# ##############################################################################
IF(WIN32)
    # Windows made the ridiculous choice to define min/max macros so you need
    # to manually disable this to use std::min/std::max
    add_compile_definitions(NOMINMAX)

    # This is required for parallel builds (why this is not on by default is
    # anyone's guess...)
    add_compile_options(/MP)
ENDIF()

# ##############################################################################
# SUBPROJECTS
# ##############################################################################
add_subdirectory(tdc-lib)
