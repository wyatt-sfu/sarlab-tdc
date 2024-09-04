cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(tdctest VERSION 1.0.0 LANGUAGES CXX)

# Libraries
find_package(GTest REQUIRED)

# Executable target
add_executable(tdctest
    tdctest.cpp
    mathtest.cpp
)

# Linking
target_link_libraries(tdctest PRIVATE
    GTest::gtest_main
    GTest::gtest
    tdc-lib
)

include(GoogleTest)
gtest_discover_tests(tdctest
    TEST_PREFIX tdctest::)
