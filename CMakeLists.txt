cmake_minimum_required(VERSION 3.22)

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
add_subdirectory(sarlab-tdc)
add_subdirectory(tdc-lib)

