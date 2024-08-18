/* CUDA headers */
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <math_constants.h>
#include <vector_types.h>

/* Project headers */
#include "tuning.h"

/* Class header */
#include "tdckernels.h"

#define SPEED_OF_LIGHT_F 299792458.0F

/* Global variable used for storing the maximum of the window array */
__device__ float WindowMaxValue[NUM_STREAMS];

/**
 * Return a device pointer to the WindowMaxValue array.
 */
void *getWindowMaxValuePtr()
{
    void *devPtr;
    cudaGetSymbolAddress(&devPtr, WindowMaxValue);
    return devPtr;
}

/**
 * Cuda kernel for initializing the range window array.
 *
 * rgWin: 1D array that will be filled with the range window weights
 * nSamples: Number of samples
 */
__global__ void initRangeWindowKernel(float *rgWin, int nSamples)
{
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIdx >= nSamples) {
        return;
    }

    rgWin[sampleIdx] = RANGE_WINDOW_A_PARAMETER
                       - ((1.0 - RANGE_WINDOW_A_PARAMETER)
                          * cosf(2.0F * CUDART_PI_F * sampleIdx / nSamples));
}

/**
 * Initialize the range window array on the GPU
 *
 * rgWin: 1D array that will be filled with the range window weights
 * nSamples: Number of samples
 */
void initRangeWindow(float *rgWin, int nSamples)
{
    dim3 const blockSize(32, 0, 0);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x, 0, 0);
    initRangeWindowKernel<<<gridSize, blockSize>>>(rgWin, nSamples);
}

__global__ void dopplerCentroid(float4 const *velocity, float4 const *attitude,
                                float lambda, int chunkIdx, int nPri, int nSamp)
{
    float lambdaFac = 2.0f / lambda;
}

/**
 * Cuda kernel for computing the window function for the specified chunk of
 * raw data.
 */
__global__ void createWindowKernel(float *window, float const *rangeWindow,
                                   float4 const *velocity,
                                   float4 const *attitude, float lambda,
                                   int chunkIdx, int nPri, int nSamp)
{
    unsigned int const priChunkIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const priGlobalIdx = chunkIdx * PRI_CHUNKSIZE + priChunkIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE
        && sampleIdx < nSamp) {
        float lambdaFac = 2.0f / lambda;
        // Compute the Doppler centroid for this pulse

        // TODO: Compute window based on Doppler
        window[priChunkIdx * nSamp + sampleIdx] = 1.0;
    }
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
    // createWindowKernel<<<gridSize, blockSize, 0, stream>>>();
}

/**
 * Cuda kernel which computes the reference response from a target at the
 * specified position.
 */
__global__ void reference_response_tdc(float2 *reference,
                                       float const *__restrict__ window,
                                       float4 const *__restrict__ position,
                                       float const *__restrict__ sampleTime,
                                       float3 target, float modRate,
                                       float startFreq, int chunkIdx, int nPri,
                                       int nSamples)
{
    unsigned int const priChunkIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const priGlobalIdx = chunkIdx * PRI_CHUNKSIZE + priChunkIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE
        && sampleIdx < nSamples) {
        const float4 phase_centre =
            position[priChunkIdx * nSamples + sampleIdx];
        const float winVal = window[priChunkIdx * nSamples + sampleIdx];
        float dist_to_target =
            norm3df(phase_centre.x - target.x, phase_centre.y - target.y,
                    phase_centre.z - target.z);

        const float freq = fmaf(sampleTime[sampleIdx], modRate, startFreq);
        const float phi =
            -4.0F * CUDART_PI_F * dist_to_target * (freq / SPEED_OF_LIGHT_F);
        float sinval = 0.0;
        float cosval = 0.0;
        sincosf(phi, &sinval, &cosval);
        reference[priChunkIdx * nSamples + sampleIdx] = {cosval * winVal,
                                                         sinval * winVal};
    }
}

/**
 * Cuda kernel for focusing the data to the specified grid point.
 * This kernel only does work if the window function is non-zero.
 */
__global__ void focusToGridPointKernel(
    float2 const *rawData, float2 *reference, float const *window,
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
        reference_response_tdc<<<gridSize, blockSize>>>(
            reference, window, position, sampleTimes, target, modRate,
            startFreq, chunkIdx, nPri, nSamples);
    }
}

/**
 * Wrapper around the cuda kernel focusToGridPointKernel
 */
void focusToGridPoint(float2 const *rawData, float2 *reference,
                      float const *window, float4 const *position,
                      float4 const *velocity, float4 const *attitude,
                      float const *priTimes, float const *sampleTimes,
                      float2 const *image, float3 target, float modRate,
                      float startFreq, int chunkIdx, int nPri, int nSamples,
                      int streamIdx, cudaStream_t stream)
{
    focusToGridPointKernel<<<1, 1, 0, stream>>>(
        rawData, reference, window, position, velocity, attitude, priTimes,
        sampleTimes, image, target, modRate, startFreq, chunkIdx, nPri,
        nSamples, streamIdx);
}
