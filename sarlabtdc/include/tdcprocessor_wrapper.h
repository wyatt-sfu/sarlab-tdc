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
    void start(float dopplerBandwidth, bool applyRangeWindow);
    void setRawData(
        py::array_t<std::complex<float>, py::array::c_style> rawData,
        py::array_t<float, py::array::c_style> priTimes,
        py::array_t<float, py::array::c_style> sampleTimes,
        py::array_t<float, py::array::c_style> position,
        py::array_t<float, py::array::c_style> velocity,
        py::array_t<float, py::array::c_style> attitude, float modRate,
        float startFreq);
    void setFocusGrid(py::array_t<float, py::array::c_style> focusGrid);
    py::array_t<std::complex<float>, py::array::c_style> getFocusedImage();

private:
    /* Methods */
    void setupLogging();

    /* Underlying C++ object */
    std::unique_ptr<TdcProcessor> tdcProc;

    /* Numpy arrays (needed to stop Python from garbage collecting) */
    py::array_t<std::complex<float>, py::array::c_style> rawData;
    py::array_t<float, py::array::c_style> priTimes;
    py::array_t<float, py::array::c_style> sampleTimes;
    py::array_t<float, py::array::c_style> position;
    py::array_t<float, py::array::c_style> velocity;
    py::array_t<float, py::array::c_style> attitude;
    py::array_t<float, py::array::c_style> focusGrid;

    /* Focus grid shape */
    int gridNumRows;
    int gridNumCols;
};

#endif // TDCPROCESSOR_WRAPPER_H
