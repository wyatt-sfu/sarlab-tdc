/* CUDA headers */
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <vector_types.h>

/* Project headers */
#include "tuning.h"

/* Class header */
#include "tdckernels.h"

/** Global variable used for storing the maximum of the window array */
__device__ float windowMaxValue[NUM_STREAMS];

void *getWindowMaxValuePtr()
{
    void *devPtr;
    cudaGetSymbolAddress(&devPtr, windowMaxValue);
    return devPtr;
}

__global__ void createWindowKernel(float *window, int chunkIdx, int nPri,
                                   int nSamp)
{
    unsigned int const priWindowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const priGlobalIdx = chunkIdx * PRI_CHUNKSIZE + priWindowIdx;
    float winVal = 0;

    if (priWindowIdx < PRI_CHUNKSIZE && sampleIdx < nSamp) {
        if (priGlobalIdx < nPri) {
            // TODO: Compute window based on Doppler
            winVal = 1.0;
        } else {
            winVal = 0.0;
        }
    }

    window[priWindowIdx * nSamp + sampleIdx] = winVal;
}

void createWindow(float *window, int chunkIdx, int nPri, int nSamples,
                  cudaStream_t stream)
{
    dim3 const blockSize(WindowKernel::BlockSizeX, WindowKernel::BlockSizeY, 0);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                        (nPri + blockSize.y - 1) / blockSize.y, 0);
    createWindowKernel<<<gridSize, blockSize, 0, stream>>>(window, chunkIdx,
                                                           nPri, nSamples);
}

__global__ void focusToGridPointKernel() {}

void focusToGridPoint(float2 const *rawData, float2 const *reference,
                      float *window, float4 const *position,
                      float4 const *velocity, float4 const *attitude,
                      float const *priTimes, float const *sampleTimes,
                      float4 const *focusGrid, float2 const *image,
                      float modRate, float startFreq, int chunkIdx, int nPri,
                      int nSamples, int streamIdx, cudaStream_t stream)
{
    focusToGridPointKernel<<<1, 1, 0, stream>>>();
}
