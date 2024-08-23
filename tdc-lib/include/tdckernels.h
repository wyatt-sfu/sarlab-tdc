#ifndef TDCKERNELS_H
#define TDCKERNELS_H

/* Standard library headers */
#include <cstdlib>

/* CUDA headers */
#include <driver_types.h>
#include <vector_types.h>

/**
 * Header for calling the kernel wrappers from pure C++ files.
 * See tdckernels.cu for detailed documentation.
 */

/**
 * Initialize the range window array with a Hamming window.
 */
void initRangeWindow(float *rgWin, int nSamples);

/**
 * Create the window array for the specified chunk of raw data.
 */
void createWindow(
    // Window arrays
    float *window, // 2D full window to apply to raw data chunk
    float const *rangeWindow, // 1D range window

    // Position related arguments
    float3 const *velocity, // [m] 2D, x,y,z velocity at each PRI/sample
    float4 const *attitude, // 2D quaternion at each PRI/sample

    // Radar parameters
    float lambda, // [m] Radar carrier wavelength
    float dopplerBw, // [Hz] Bandwidth for windowing

    // Data shape arguments
    int chunkIdx, // Current chunk index
    int nPri, // Number of PRIs in the full acquisition
    int nSamples // Number of samples per PRI
);

/**
 * Compute the response of a single target at the specified position.
 */
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
    int nSamples // Number of samples per PRI
);

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
    int nSamples // Number of samples per PRI
);

/**
 * Returns the scratch size needed in bytes for the correlateAndSum function
 */
size_t sumScratchSize(int nSamples);

#endif // TDCKERNELS_H
