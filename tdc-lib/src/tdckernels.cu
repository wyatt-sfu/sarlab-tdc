/* Standard library headers */
#include <algorithm>
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
#include "gpumath.h"
#include "tuning.h"

/* Class header */
#include "tdckernels.h"

#define SPEED_OF_LIGHT_F 299792458.0F

/* Global variable used for storing the maximum of the window array */
__device__ float2 SumResults[NUM_STREAMS];

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
 * Correlate the raw data with the reference response.
 */
__global__ void correlateWithReference(
    // Argument list
    const float2 *__restrict__ raw, // Raw radar data
    float2 *reference, // Reference response. Correlation is written back into
                       // this array.
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs
    int nSamples // Number of samples in a PRI
)
{
    unsigned int const priChunkIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const priGlobalIdx = chunkIdx * PRI_CHUNKSIZE + priChunkIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE
        && sampleIdx < nSamples) {
        const float2 v1 = raw[priChunkIdx * nSamples + sampleIdx];
        float2 v2 = reference[priChunkIdx * nSamples + sampleIdx];

        v2.y *= -1.0; // conjugate
        reference[priChunkIdx * nSamples + sampleIdx].x =
            (v1.x * v2.x) - (v1.y * v2.y);
        reference[priChunkIdx * nSamples + sampleIdx].y =
            (v1.x * v2.y) + (v1.y * v2.x);
    }
}

__global__ void addToImage(float2 *pixel, float2 *sumVal)
{
    *pixel = (*pixel) + (*sumVal);
}

/**
 * Correlate the raw data with the reference array and put the result in the
 * focused image
 */
void correlateAndSum(
    // Data array parameters
    float2 const *raw, // 2D, IQ data chunk
    float2 *reference, // 2D, Reference response to correlate with
    void *scratch, // Scratch space for sum reduction
    size_t scratchSize, // Size of sum scratch space

    // Focus image
    float2 *pixel, // Pointer to the current pixel

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples, // Number of samples per PRI
    int streamIdx, // Stream index
    cudaStream_t stream // Stream to run the kernel in
)
{
    // First correlate the reference and raw data
    dim3 const blockSize(WindowKernel::BlockSizeX, WindowKernel::BlockSizeY, 0);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                        (PRI_CHUNKSIZE + blockSize.y - 1) / blockSize.y, 0);
    correlateWithReference<<<gridSize, blockSize, 0, stream>>>(
        raw, reference, chunkIdx, nPri, nSamples);

    // Then sum the result
    void *devPtr;
    cudaGetSymbolAddress(&devPtr, SumResults);
    float2 *sumResult = reinterpret_cast<float2 *>(devPtr) + streamIdx;
    size_t priIndex = chunkIdx * PRI_CHUNKSIZE;
    size_t prisToSum = std::min(PRI_CHUNKSIZE, nPri - priIndex);
    cub::DeviceReduce::Sum(scratch, scratchSize, reference, sumResult,
                           prisToSum * nSamples, stream);
    addToImage<<<1, 1, 0, stream>>>(pixel, sumResult);
}

/**
 * Returns the scratch size needed in bytes for the correlateAndSum function
 */
size_t sumScratchSize(int nSamples)
{
    void *scratch = nullptr;
    size_t scratchSize = 0;
    float2 *dataIn = nullptr;
    float2 dataOut = {0, 0};
    cub::DeviceReduce::Sum(scratch, scratchSize, dataIn, &dataOut,
                           PRI_CHUNKSIZE * nSamples);
    return scratchSize;
}
