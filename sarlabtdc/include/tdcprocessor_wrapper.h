#ifndef TDCPROCESSOR_WRAPPER_H
#define TDCPROCESSOR_WRAPPER_H

/* Standard library headers */
#include <memory>

/* Pybind headers */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

/* tdc-lib headers */
#include "tdcprocessor.h"

namespace py = pybind11;

/**
 * This class wraps the TdcProcessor class. See that class for documentation
 * details.
 */
class TdcProcessorWrapper
{
public:
    TdcProcessorWrapper(int gpuNum);
    void start();
    void arrayTest(py::array_t<float, py::array::c_style> data);

private:
    std::unique_ptr<TdcProcessor> tdcProc;
};

#endif // TDCPROCESSOR_WRAPPER_H
