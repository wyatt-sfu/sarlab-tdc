/* Standard library headers */
#include <cstddef>

/* CUDA headers */
#include <cstdlib>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
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
__global__ void createWindowKernel(
    // Window arrays
    float *window, // 2D full window to apply to raw data chunk
    float const *rangeWindow, // 1D range window

    // Positioning data
    float3 const *velocity, // [m] 2D, x,y,z velocity at each PRI/sample
    float4 const *attitude, // 2D quaternion at each PRI/sample

    // Radar parameters
    float lambda, // [m] Radar carrier wavelength
    float dopplerBw, // [Hz] Bandwidth for windowing

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples // Number of samples per PRI
)
{
    unsigned int const priChunkIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const priGlobalIdx = chunkIdx * PRI_CHUNKSIZE + priChunkIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE
        && sampleIdx < nSamples) {
        float lambdaFac = 2.0f / lambda;
        // Compute the Doppler centroid for this pulse

        // TODO: Compute window based on Doppler
        window[priChunkIdx * nSamples + sampleIdx] = rangeWindow[sampleIdx];
    }
}

/**
 * Wrapper around the cuda kernel createWindowKernel
 */
void createWindow(
    // Window arrays
    float *window, // 2D full window to apply to raw data chunk
    float const *rangeWindow, // 1D range window

    // Positioning data
    float3 const *velocity, // [m] 2D, x,y,z velocity at each PRI/sample
    float4 const *attitude, // 2D quaternion at each PRI/sample

    // Radar parameters
    float lambda, // [m] Radar carrier wavelength
    float dopplerBw, // [Hz] Bandwidth for windowing

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples, // Number of samples per PRI
    cudaStream_t stream // Stream to run the kernel in
)
{
    dim3 const blockSize(WindowKernel::BlockSizeX, WindowKernel::BlockSizeY, 0);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                        (PRI_CHUNKSIZE + blockSize.y - 1) / blockSize.y, 0);
    createWindowKernel<<<gridSize, blockSize, 0, stream>>>(
        window, rangeWindow, velocity, attitude, lambda, dopplerBw, chunkIdx,
        nPri, nSamples);
}

/**
 * Cuda kernel which computes the reference response from a target at the
 * specified position.
 */
__global__ void referenceResponseKernel(
    // Data array parameters
    float2 *reference, // 2D, IQ data to contain reference data
    float const *window, // 2D full window to apply to raw data chunk
    float3 const *position, // [m] 2D, x,y,z position at each PRI/sample
    float const *sampleTimes, // [s] 1D, Time of each sample in a PRI
    float3 target, // [m] Location on the focus grid

    // Radar operating parameters
    float startFreq, // [Hz] PRI start frequency
    float modRate, // [Hz/s] Modulation rate

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples // Number of samples per PRI
)
{
    unsigned int const priChunkIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const priGlobalIdx = chunkIdx * PRI_CHUNKSIZE + priChunkIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE
        && sampleIdx < nSamples) {
        const float3 phase_centre =
            position[priChunkIdx * nSamples + sampleIdx];
        const float winVal = window[priChunkIdx * nSamples + sampleIdx];
        float dist_to_target =
            norm3df(phase_centre.x - target.x, phase_centre.y - target.y,
                    phase_centre.z - target.z);

        const float freq = fmaf(sampleTimes[sampleIdx], modRate, startFreq);
        const float phi =
            -4.0F * CUDART_PI_F * dist_to_target * (freq / SPEED_OF_LIGHT_F);
        float sinval = 0.0;
        float cosval = 0.0;
        sincosf(phi, &sinval, &cosval);
        reference[priChunkIdx * nSamples + sampleIdx] = {cosval * winVal,
                                                         sinval * winVal};
    }
}

void referenceResponse(
    // Data array parameters
    float2 *reference, // 2D, IQ data to contain reference data
    float const *window, // 2D full window to apply to raw data chunk
    float3 const *position, // [m] 2D, x,y,z position at each PRI/sample
    float const *sampleTimes, // [s] 1D, Time of each sample in a PRI
    float3 target, // [m] Location on the focus grid

    // Radar operating parameters
    float startFreq, // [Hz] PRI start frequency
    float modRate, // [Hz/s] Modulation rate

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples, // Number of samples per PRI
    cudaStream_t stream // Stream to run the kernel in
)
{
    dim3 const refBlockSize(ReferenceResponseKernel::BlockSizeX,
                            ReferenceResponseKernel::BlockSizeY, 0);
    dim3 const refGridSize(
        (nSamples + refBlockSize.x - 1) / refBlockSize.x,
        (PRI_CHUNKSIZE + refBlockSize.y - 1) / refBlockSize.y, 0);

    referenceResponseKernel<<<refGridSize, refBlockSize, 0, stream>>>(
        reference, window, position, sampleTimes, target, startFreq, modRate,
        chunkIdx, nPri, nSamples);
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
