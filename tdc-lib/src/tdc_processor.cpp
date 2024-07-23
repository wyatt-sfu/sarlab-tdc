/* Standard library headers */
#include <complex
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>

/* Class header */
#include "tdc_processor.h"

TdcProcessor::TdcProcessor(int gpuNum)
{
    cudaError_t err = cudaSetDevice(gpuNum);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void TdcProcessor::start() {}

void TdcProcessor::setRawData(const std::complex<float> *rawData,
                              const float *priTimes, int nPri, int nSamples,
                              float modRate, float sampleRate)
{
}
