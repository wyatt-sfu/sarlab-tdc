cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(sarlabtdc VERSION 1.0.0 LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

pybind11_add_module(sarlabtdc test.cpp)
