# sarlab-tdc
Open source time-domain correlation SAR processor for high bandwidth mmWave radars

## Computer Requirements
This library uses CUDA, which requires a compatible NVidia GPU.

## Building C++ Library
To use vcpkg to manage dependencies, add the following argument when calling cmake:
 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake
Where <VCPKG_ROOT> is the install location of vcpkg.

In VSCode this can be done by adding to the "Cmake: Configure Args" setting.

## Python Library Setup
1) Install pybind in your Python environment (i.e. pip install pybind11).
2) cd <py-tdc location>
3) Run pip install .
