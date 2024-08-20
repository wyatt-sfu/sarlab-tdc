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
    int nSamples, // Number of samples per PRI
    cudaStream_t stream // Stream to run the kernel in
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
    int nSamples, // Number of samples per PRI
    cudaStream_t stream // Stream to run the kernel in
);

/**
 * Process the next chunk of data and focus it to the specified grid location.
 */
void focusToGridPoint(float2 const *rawData, float2 *reference,
                      float const *window, float4 const *position,
                      float4 const *velocity, float4 const *attitude,
                      float const *priTimes, float const *sampleTimes,
                      float2 const *image, float3 target, float modRate,
                      float startFreq, int chunkIdx, int nPri, int nSamples,
                      int streamIdx, cudaStream_t stream);

#endif // TDCKERNELS_H
