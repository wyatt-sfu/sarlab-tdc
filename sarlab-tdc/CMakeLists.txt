cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(sarlab-tdc VERSION 1.0.0 LANGUAGES CXX)

add_executable(sarlab-tdc
    src/main.cpp
    src/argparser.cpp
)

target_include_directories(sarlab-tdc PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
