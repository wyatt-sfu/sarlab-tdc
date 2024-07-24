#include "tdcprocessor_wrapper.h"

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
    py::array_t<float, py::array::c_style> priTimes, float modRate,
    float sampleRate)
{
    // Check array sizes
    auto dataInfo = rawData.request();
    if (dataInfo.ndim != 2) {
        throw std::runtime_error("rawData must be 2D");
    }

    auto timeInfo = priTimes.request();
    if (timeInfo.ndim != 1) {
        throw std::runtime_error("priTimes must be 1D");
    }

    if (dataInfo.shape[0] != timeInfo.shape[0]) {
        throw std::runtime_error("Shapes do not match");
    }

    int nPri = dataInfo.shape[0];
    int nSamples = dataInfo.shape[1];
    std::complex<float> const *dataPtr =
        reinterpret_cast<std::complex<float> const *>(dataInfo.ptr);
    float const *timePtr = reinterpret_cast<float const *>(timeInfo.ptr);

    tdcProc->setRawData(dataPtr, timePtr, nPri, nSamples, modRate, sampleRate);
}
