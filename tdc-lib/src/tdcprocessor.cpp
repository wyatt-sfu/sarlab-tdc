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

void TdcProcessor::setRawData(std::complex<float> const *rawData,
                              float const *priTimes, float const *position,
                              float const *attitude, int nPri, int nSamples,
                              float modRate, float sampleRate)
{
    this->rawData = rawData;
    this->priTimes = priTimes;
    this->position = position;
    this->attitude = attitude;
    this->nPri = nPri;
    this->nSamples = nSamples;
    this->modRate = modRate;
    this->sampleRate = sampleRate;
}

void TdcProcessor::setFocusGrid(float const *focusGrid, int nRows, int nCols)
{
    this->focusGrid = focusGrid;
    gridNumRows = nRows;
    gridNumCols = nCols;
}
