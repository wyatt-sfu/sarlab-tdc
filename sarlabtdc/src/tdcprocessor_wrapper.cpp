#include "tdcprocessor_wrapper.h"
#include <stdexcept>

TdcProcessorWrapper::TdcProcessorWrapper(int gpuNum)
{
    tdcProc = std::make_unique<TdcProcessor>(gpuNum);
}

void TdcProcessorWrapper::start()
{
    tdcProc->start();
}

void TdcProcessorWrapper::setRawData(
    py::array_t<std::complex<float>, py::array::c_style> rawData,
    py::array_t<float, py::array::c_style> priTimes,
    py::array_t<float, py::array::c_style> position,
    py::array_t<float, py::array::c_style> attitude, float modRate,
    float sampleRate)
{
    // Get array info structures
    auto dataInfo = rawData.request();
    auto timeInfo = priTimes.request();
    auto posInfo = position.request();
    auto attInfo = attitude.request();

    // Check array dimensions
    if (dataInfo.ndim != 2) {
        throw std::runtime_error("rawData must be 2D");
    }

    if (timeInfo.ndim != 1) {
        throw std::runtime_error("priTimes must be 1D");
    }

    if (posInfo.ndim != 3) {
        throw std::runtime_error("position must be 3D");
    }

    if (attInfo.ndim != 3) {
        throw std::runtime_error("attitude must be 3D");
    }

    // Get number of PRIs and number of samples from the data shape
    int nPri = dataInfo.shape[0];
    int nSamples = dataInfo.shape[1];

    // Check the shapes of the other arrays
    if (timeInfo.shape[0] != nPri) {
        throw std::runtime_error("timeInfo shape is incorrect");
    }

    if (posInfo.shape[0] != nPri || posInfo.shape[1] != nSamples
        || posInfo.shape[2] != 3) {
        throw std::runtime_error("position shape is incorrect");
    }

    if (attInfo.shape[0] != nPri || attInfo.shape[1] != nSamples
        || attInfo.shape[2] != 3) {
        throw std::runtime_error("attitude shape is incorrect");
    }

    // Get the pointers to the underlying data in the arrays
    auto *dataPtr = reinterpret_cast<std::complex<float> const *>(dataInfo.ptr);
    auto *timePtr = reinterpret_cast<float const *>(timeInfo.ptr);
    auto *posPtr = reinterpret_cast<float const *>(posInfo.ptr);
    auto *attPtr = reinterpret_cast<float const *>(attInfo.ptr);

    // Call the underlying C++ function
    tdcProc->setRawData(dataPtr, timePtr, posPtr, attPtr, nPri, nSamples,
                        modRate, sampleRate);
}

void TdcProcessorWrapper::setFocusGrid(py::array_t<float, py::array::c_style> focusGrid)
{
    auto gridInfo = focusGrid.request();
    if (gridInfo.ndim != 3) {
        throw std::runtime_error("focusGrid must be 3D");
    }

    // Get array shape
    int numRows = gridInfo.shape[0];
    int numCols = gridInfo.shape[1];
    
    // Check the array shape
    if (gridInfo.shape[2] != 3) {
        throw std::runtime_error("focusGrid shape is incorrect");
    }

    // Get the pointer to the underlying data in the array
    auto *gridPtr = reinterpret_cast<float const *>(gridInfo.ptr);

    // Call the underlying C++ function
    tdcProc->setFocusGrid(gridPtr, numRows, numCols);
}
