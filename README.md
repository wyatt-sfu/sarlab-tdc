# sarlab-tdc
Open source time-domain correlation SAR processor for high bandwidth FMCW mmWave radars

## Computer Requirements
This library uses CUDA, which requires a compatible Nvidia GPU.

## Building C++ Library
This library uses vcpkg to manage depencies. The vcpkg toolchain file is found
by using the VCPKG_ROOT environment variable (check that this variable exists on
your machine).

## Python Library Setup
1) Install pybind in your Python environment (i.e. pip install pybind11).
2) cd `GIT CLONE LOCATION`/py-sarlabtdc
3) pip install .
