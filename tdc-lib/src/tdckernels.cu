/* Standard library headers */
#include <algorithm>
#include <cstddef>
#include <stdexcept>

/* CUDA headers */
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <fmt/core.h>
#include <math_constants.h>
#include <nppdefs.h>
#include <npps.h>
#include <npps_statistics_functions.h>
#include <vector_types.h>

/* 3rd party headers */
#include <fmt/format.h>

/* Project headers */
#include "gpumath.h"
#include "tuning.h"
#include "windowing.h"

/* Class header */
#include "tdckernels.h"

#define SPEED_OF_LIGHT_F 299792458.0F

/**
 * Cuda kernel for initializing the range window array.
 */
__global__ void initRangeWindowKernel(float *__restrict__ rgWin, // 1D range window
                                      int nSamples, // Number of samples
                                      bool applyRangeWindow // Flag to control behaviour
)
{
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIdx >= nSamples) {
        return;
    }

    if (applyRangeWindow) {
        rgWin[sampleIdx] =
            RANGE_WINDOW_A_PARAMETER
            - ((1.0 - RANGE_WINDOW_A_PARAMETER) * cospif(2.0F * sampleIdx / nSamples));
    } else {
        rgWin[sampleIdx] = 1.0F;
    }
}

/**
 * Initialize the range window array with a Hamming window if applyRangeWindow is true.
 * Else this initializes the range window to all ones.
 */
void initRangeWindow(float *rgWin, // 1D range window
                     int nSamples, // Number of samples
                     bool applyRangeWindow // Flag to control behaviour
)
{
    dim3 const blockSize(32, 1, 1);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x, 1, 1);
    initRangeWindowKernel<<<gridSize, blockSize>>>(rgWin, nSamples, applyRangeWindow);
}

/**
 * Cuda kernel for computing the window function for the specified chunk of
 * raw data.
 */
__global__ void createWindowKernel(
    // Window arrays
    float *__restrict__ window, // 2D full window to apply to raw data chunk
    float const *__restrict__ rangeWindow, // 1D range window

    // Positioning data
    float3 const *position, // [m] 2D, x,y,z position at each PRI/sample
    float3 const *__restrict__ velocity, // [m] 2D, x,y,z velocity at each PRI/sample
    float4 const *__restrict__ attitude, // 2D quaternion at each PRI/sample
    float3 target, // [m] Location on the focus grid
    float3 bodyBoresight, // (x, y, z) Boresight vector in body coordinate system

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
    const ptrdiff_t elementIdx =
        static_cast<ptrdiff_t>(priChunkIdx) * nSamples + sampleIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE && sampleIdx < nSamples) {
        // Read out the radar position/attitude data
        float3 radarPos = position[elementIdx];
        float3 vel = velocity[elementIdx];
        float4 att = attitude[elementIdx];

        // First we need to compute the pointing angle of the radar in local coordinates
        // using the attitude quaternion. This quaternion rotates the boresight vector
        // from body coordinates to the local coordinate system we are focusing the
        // image in.
        float3 antPointing = q_rot(att, bodyBoresight); // This is already a unit vector

        // Then get the Doppler centroid
        float fDopCentroid = dopplerCentroid(vel, antPointing, lambda);

        // Now compute the Doppler freq to the target being focused
        float fDop = dopplerFreq(radarPos, vel, target, lambda);

        // Window based on the difference to the Doppler centroid
        float azWin = dopplerWindow(fDop, fDopCentroid, dopplerBw);

        window[elementIdx] = rangeWindow[sampleIdx] * azWin;
    }
}

/**
 * Wrapper around the cuda kernel createWindowKernel
 */
void createWindow(
    // Window arrays
    float *__restrict__ window, // 2D full window to apply to raw data chunk
    float const *__restrict__ rangeWindow, // 1D range window

    // Positioning data
    float3 const *position, // [m] 2D, x,y,z position at each PRI/sample
    float3 const *__restrict__ velocity, // [m] 2D, x,y,z velocity at each PRI/sample
    float4 const *__restrict__ attitude, // 2D quaternion at each PRI/sample
    float3 target, // [m] Location on the focus grid
    float3 bodyBoresight, // (x, y, z) Boresight vector in body coordinate system

    // Radar parameters
    float lambda, // [m] Radar carrier wavelength
    float dopplerBw, // [Hz] Bandwidth for windowing

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples // Number of samples per PRI
)
{
    dim3 const blockSize(WindowKernel::BlockSizeX, WindowKernel::BlockSizeY, 1);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                        (PRI_CHUNKSIZE + blockSize.y - 1) / blockSize.y, 1);
    createWindowKernel<<<gridSize, blockSize>>>(window, rangeWindow, position, velocity,
                                                attitude, target, bodyBoresight, lambda,
                                                dopplerBw, chunkIdx, nPri, nSamples);
}

/**
 * Cuda kernel which computes the reference response from a target at the
 * specified position.
 */
__global__ void referenceResponseKernel(
    // Data array parameters
    float2 *__restrict__ reference, // 2D, IQ data to contain reference data
    float const *__restrict__ window, // 2D full window to apply to raw data chunk
    float3 const *__restrict__ position, // [m] 2D, x,y,z position at each PRI/sample
    float const *__restrict__ sampleTimes, // [s] 1D, Time of each sample in a PRI
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
    const ptrdiff_t elementIdx =
        static_cast<ptrdiff_t>(priChunkIdx) * nSamples + sampleIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE && sampleIdx < nSamples) {

        const float3 phase_centre = position[elementIdx];
        const float winVal = window[elementIdx];
        float dist_to_target =
            norm3df(phase_centre.x - target.x, phase_centre.y - target.y,
                    phase_centre.z - target.z);

        const float freq = fmaf(sampleTimes[sampleIdx], modRate, startFreq);
        const float phi = -4.0F * dist_to_target * (freq / SPEED_OF_LIGHT_F);
        float sinval = 0.0;
        float cosval = 0.0;
        sincospif(phi, &sinval, &cosval);
        reference[elementIdx] = {cosval * winVal, sinval * winVal};
    }
}

void referenceResponse(
    // Data array parameters
    float2 *__restrict__ reference, // 2D, IQ data to contain reference data
    float const *__restrict__ window, // 2D full window to apply to raw data chunk
    float3 const *__restrict__ position, // [m] 2D, x,y,z position at each PRI/sample
    float const *__restrict__ sampleTimes, // [s] 1D, Time of each sample in a PRI
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
    dim3 const refBlockSize(ReferenceResponseKernel::BlockSizeX,
                            ReferenceResponseKernel::BlockSizeY, 1);
    dim3 const refGridSize((nSamples + refBlockSize.x - 1) / refBlockSize.x,
                           (PRI_CHUNKSIZE + refBlockSize.y - 1) / refBlockSize.y, 1);

    referenceResponseKernel<<<refGridSize, refBlockSize>>>(
        reference, window, position, sampleTimes, target, startFreq, modRate, chunkIdx,
        nPri, nSamples);
}

/**
 * Correlate the raw data with the reference response.
 */
__global__ void correlateWithReference(
    // Argument list
    const float2 *__restrict__ raw, // Raw radar data
    float2 *__restrict__ reference, // Reference response. Correlation is
                                    // written back into this array.
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs
    int nSamples // Number of samples in a PRI
)
{
    unsigned int const priChunkIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int const sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const priGlobalIdx = chunkIdx * PRI_CHUNKSIZE + priChunkIdx;
    const ptrdiff_t elementIdx =
        static_cast<ptrdiff_t>(priChunkIdx) * nSamples + sampleIdx;

    if (priGlobalIdx < nPri && priChunkIdx < PRI_CHUNKSIZE && sampleIdx < nSamples) {
        const float2 v1 = raw[elementIdx];
        float2 v2 = reference[elementIdx];

        v2.y *= -1.0; // conjugate
        reference[elementIdx].x = (v1.x * v2.x) - (v1.y * v2.y);
        reference[elementIdx].y = (v1.x * v2.y) + (v1.y * v2.x);
    }
}

__global__ void addToImage(float2 *__restrict__ pixel, float2 *__restrict__ sumVal)
{
    *pixel = (*pixel) + (*sumVal);
}

/**
 * Correlate the raw data with the reference array and put the result in the
 * focused image
 */
void correlateAndSum(
    // Data array parameters
    float2 const *__restrict__ raw, // 2D, IQ data chunk
    float2 *__restrict__ reference, // 2D, Reference response to correlate with
    void *__restrict__ scratch, // Scratch space for sum reduction
    size_t scratchSize, // Size of sum scratch space
    float2 *sumVal, // The sum result will be placed here

    // Focus image
    float2 *__restrict__ pixel, // Pointer to the current pixel

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples // Number of samples per PRI
)
{
    // First correlate the reference and raw data
    dim3 const blockSize(CorrelateKernel::BlockSizeX, CorrelateKernel::BlockSizeY, 1);
    dim3 const gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                        (PRI_CHUNKSIZE + blockSize.y - 1) / blockSize.y, 1);
    correlateWithReference<<<gridSize, blockSize>>>(raw, reference, chunkIdx, nPri,
                                                    nSamples);

    // Then sum the result
    size_t priIndex = chunkIdx * PRI_CHUNKSIZE;
    size_t prisToSum = std::min(PRI_CHUNKSIZE, nPri - priIndex);
    nppsSum_32fc((const Npp32fc *) reference, prisToSum * nSamples, (Npp32fc *) sumVal,
                 (Npp8u *) scratch);
    addToImage<<<1, 1>>>(pixel, sumVal);
}

/**
 * Returns the scratch size needed in bytes for the correlateAndSum function
 */
size_t sumScratchSize(int nSamples)
{
    size_t scratchSize = 0;
    NppStatus status =
        nppsSumGetBufferSize_32fc(PRI_CHUNKSIZE * nSamples, &scratchSize);
    if (status != 0) {
        throw std::runtime_error(fmt::format(
            "Error while computing scratch space size: {}", static_cast<int>(status)));
    }
    return scratchSize;
}
