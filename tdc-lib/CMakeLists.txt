cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(tdc-lib VERSION 1.0.0 LANGUAGES CXX)

# Library target
add_library(tdc-lib STATIC
    src/tdc_processor.cpp
)

# Include directories
target_include_directories(tdc-lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
