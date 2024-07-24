/* Standard library headers */
#include <complex>
#include <iostream>
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>

/* Class header */
#include "tdcprocessor.h"

TdcProcessor::TdcProcessor(int gpuNum)
{
    cudaError_t err = cudaSetDevice(gpuNum);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void TdcProcessor::start()
{
    std::cout << "TdcProcessor::start() called\n";
}

void TdcProcessor::arrayTest(float *data, int arrLen) {
    std::cout << "TdcProcessor::arrayTest() called\n";
    for (int i = 0; i < arrLen; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";
}

void TdcProcessor::setRawData(const std::complex<float> *rawData,
                              const float *priTimes, int nPri, int nSamples,
                              float modRate, float sampleRate)
{
}
