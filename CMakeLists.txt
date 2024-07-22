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
# BUILD OPTIONS
# ##############################################################################

# Uncomment this line if you want to print out a profiling report
# add_compile_definitions(PROFILE_BUILD)

# Uncomment this line if you want lots of debug logs
# add_compile_definitions(DEBUG_LOGGING)

# Uncomment this line if you want to run address sanitizer
# See https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170
# add_compile_options(/fsanitize=address)
# add_link_options(/fsanitize=address)

# Uncomment this (on Linux, requires GCC) to generate code coverage
# Note: this is untested!
# add_compile_options(--coverage)
# add_link_options(--coverage)

# ##############################################################################
# SUBPROJECTS
# ##############################################################################
add_subdirectory(sarlab-tdc)
add_subdirectory(tdc-lib)

