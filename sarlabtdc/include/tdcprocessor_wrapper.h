#ifndef TDCPROCESSOR_WRAPPER_H
#define TDCPROCESSOR_WRAPPER_H

/* Standard library headers */
#include <complex>
#include <memory>

/* Pybind headers */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

/* tdc-lib headers */
#include "tdcprocessor.h"

namespace py = pybind11;

/**
 * This class wraps the TdcProcessor class. See tdcprocessor.h for documentation
 * details.
 */
class TdcProcessorWrapper
{
public:
    TdcProcessorWrapper(int gpuNum);
    void start();
    void setRawData(
        py::array_t<std::complex<float>, py::array::c_style> rawData,
        py::array_t<float, py::array::c_style> priTimes,
        py::array_t<float, py::array::c_style> position,
        py::array_t<float, py::array::c_style> attitude, float modRate,
        float sampleRate);

private:
    std::unique_ptr<TdcProcessor> tdcProc;
};

#endif // TDCPROCESSOR_WRAPPER_H
