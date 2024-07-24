#include "tdcprocessor_wrapper.h"

TdcProcessorWrapper::TdcProcessorWrapper(int gpuNum)
{
    tdcProc = std::make_unique<TdcProcessor>(gpuNum);
}

void TdcProcessorWrapper::start()
{
    tdcProc->start();
}

void TdcProcessorWrapper::arrayTest(py::array_t<float, py::array::c_style> data)
{
    auto dataInfo = data.request();
    if (dataInfo.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be one");
    }

    int size = dataInfo.shape[0];
    tdcProc->arrayTest(reinterpret_cast<float *>(dataInfo.ptr), size);
}