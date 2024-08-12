/* Standard library headers */
#include <memory>
#include <stdexcept>
#include <vector>

/* 3rd party library headers */
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

/* Class header */
#include "tdcprocessor_wrapper.h"

TdcProcessorWrapper::TdcProcessorWrapper(int gpuNum)
{
    tdcProc = std::make_unique<TdcProcessor>(gpuNum);
    setupLogging();
}

void TdcProcessorWrapper::start()
{
    tdcProc->start();
}

void TdcProcessorWrapper::setRawData(
    py::array_t<std::complex<float>, py::array::c_style> rawData,
    py::array_t<float, py::array::c_style> priTimes,
    py::array_t<float, py::array::c_style> sampleTimes,
    py::array_t<float, py::array::c_style> position,
    py::array_t<float, py::array::c_style> velocity,
    py::array_t<float, py::array::c_style> attitude, float modRate,
    float startFreq)
{
    // Assign to member variables to prevent Python garbage collection
    this->rawData = rawData;
    this->priTimes = priTimes;
    this->sampleTimes = sampleTimes;
    this->position = position;
    this->velocity = velocity;
    this->attitude = attitude;

    // Get array info structures
    auto dataInfo = rawData.request();
    auto priTimeInfo = priTimes.request();
    auto sampleTimeInfo = sampleTimes.request();
    auto posInfo = position.request();
    auto velInfo = velocity.request();
    auto attInfo = attitude.request();

    // Check array dimensions
    if (dataInfo.ndim != 2) {
        throw std::runtime_error("rawData must be 2D");
    }

    if (priTimeInfo.ndim != 1) {
        throw std::runtime_error("priTimes must be 1D");
    }

    if (sampleTimeInfo.ndim != 1) {
        throw std::runtime_error("sampleTimes must be 1D");
    }

    if (posInfo.ndim != 3) {
        throw std::runtime_error("position must be 3D");
    }

    if (velInfo.ndim != 3) {
        throw std::runtime_error("velocity must be 3D");
    }

    if (attInfo.ndim != 3) {
        throw std::runtime_error("attitude must be 3D");
    }

    // Get number of PRIs and number of samples from the data shape
    int nPri = dataInfo.shape[0];
    int nSamples = dataInfo.shape[1];

    // Check the shapes of the arrays
    if (priTimeInfo.shape[0] != nPri) {
        throw std::runtime_error("priTimes shape is incorrect");
    }

    if (sampleTimeInfo.shape[0] != nSamples) {
        throw std::runtime_error("sampleTimes shape is incorrect");
    }

    if (posInfo.shape[0] != nPri || posInfo.shape[1] != nSamples
        || posInfo.shape[2] != 4) {
        throw std::runtime_error("position shape is incorrect");
    }

    if (velInfo.shape[0] != nPri || velInfo.shape[1] != nSamples
        || velInfo.shape[2] != 4) {
        throw std::runtime_error("velocity shape is incorrect");
    }

    if (attInfo.shape[0] != nPri || attInfo.shape[1] != nSamples
        || attInfo.shape[2] != 4) {
        throw std::runtime_error("attitude shape is incorrect");
    }

    // Get the pointers to the underlying data in the arrays
    auto *dataPtr = reinterpret_cast<std::complex<float> const *>(dataInfo.ptr);
    auto *priTimePtr = reinterpret_cast<float const *>(priTimeInfo.ptr);
    auto *sampleTimePtr = reinterpret_cast<float const *>(sampleTimeInfo.ptr);
    auto *posPtr = reinterpret_cast<float const *>(posInfo.ptr);
    auto *velPtr = reinterpret_cast<float const *>(velInfo.ptr);
    auto *attPtr = reinterpret_cast<float const *>(attInfo.ptr);

    // Call the underlying C++ function
    tdcProc->setRawData(dataPtr, priTimePtr, sampleTimePtr, posPtr, velPtr,
                        attPtr, nPri, nSamples, modRate, startFreq);
}

void TdcProcessorWrapper::setFocusGrid(
    py::array_t<float, py::array::c_style> focusGrid)
{
    auto gridInfo = focusGrid.request();
    if (gridInfo.ndim != 3) {
        throw std::runtime_error("focusGrid must be 3D");
    }

    // Get array shape
    int numRows = gridInfo.shape[0];
    int numCols = gridInfo.shape[1];

    // Check the array shape
    if (gridInfo.shape[2] != 4) {
        throw std::runtime_error("focusGrid shape is incorrect");
    }

    // Get the pointer to the underlying data in the array
    auto *gridPtr = reinterpret_cast<float const *>(gridInfo.ptr);

    // Call the underlying C++ function
    tdcProc->setFocusGrid(gridPtr, numRows, numCols);
}

void TdcProcessorWrapper::setupLogging()
{
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinkList = {consoleSink};
    tdcProc->setLoggerSinks(sinkList);
}
