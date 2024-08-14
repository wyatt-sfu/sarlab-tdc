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
__device__ float WindowMaxValue[NUM_STREAMS];

void *getWindowMaxValuePtr()
{
    void *devPtr;
    cudaGetSymbolAddress(&devPtr, WindowMaxValue);
    return devPtr;
}

/**
 * Cuda kernel for computing the window function for the specified chunk of
 * raw data.
 */
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

/**
 * Wrapper around the cuda kernel createWindowKernel
 */
void createWindow(float *window, int chunkIdx, int nPri, int nSamples,
                  cudaStream_t stream)
{
    dim3 const blockSize(WindowKernel::BlockSizeX, WindowKernel::BlockSizeY, 0);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                        (PRI_CHUNKSIZE + blockSize.y - 1) / blockSize.y, 0);
    createWindowKernel<<<gridSize, blockSize, 0, stream>>>(window, chunkIdx,
                                                           nPri, nSamples);
}

/**
 * Cuda kernel which computes the reference response from a target at the
 * specified position.
 */
__global__ void reference_response_tdc(float2 *reference,
                                       float const *__restrict__ window,
                                       float4 const *__restrict__ position,
                                       int nPri, int nSamples, float target_x,
                                       float target_y, float target_z,
                                       float const *__restrict__ sample_time,
                                       float modRate, float startFreq)
{
}

/**
 * Cuda kernel for focusing the data to the specified grid point.
 * This kernel only does work if the window function is non-zero.
 */
__global__ void focusToGridPointKernel(
    float2 const *rawData, float2 const *reference, float *window,
    float4 const *position, float4 const *velocity, float4 const *attitude,
    float const *priTimes, float const *sampleTimes, float2 const *image,
    float3 target, float modRate, float startFreq, int chunkIdx, int nPri,
    int nSamples, int streamIdx)
{
    float winMax = WindowMaxValue[streamIdx];

    // Only process the chunk if the window is non-zero
    if (winMax > WINDOW_LOWER_BOUND) {
        // Create the reference response
        dim3 const blockSize(ReferenceResponseKernel::BlockSizeX,
                             ReferenceResponseKernel::BlockSizeY, 0);
        dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                            (PRI_CHUNKSIZE + blockSize.y - 1) / blockSize.y, 0);
        // reference_response_tdc<<<gridSize, blockSize>>>(reference, window,
        // position, nPri, nSamples);
    }
}

/**
 * Wrapper around the cuda kernel focusToGridPointKernel
 */
void focusToGridPoint(float2 const *rawData, float2 const *reference,
                      float *window, float4 const *position,
                      float4 const *velocity, float4 const *attitude,
                      float const *priTimes, float const *sampleTimes,
                      float2 const *image, float3 target, float modRate,
                      float startFreq, int chunkIdx, int nPri, int nSamples,
                      int streamIdx, cudaStream_t stream)
{
    focusToGridPointKernel<<<1, 1, 0, stream>>>(
        rawData, reference, window, position, velocity, attitude, priTimes,
        sampleTimes, image, target, modRate, startFreq, chunkIdx,
        nPri, nSamples, streamIdx);
}
