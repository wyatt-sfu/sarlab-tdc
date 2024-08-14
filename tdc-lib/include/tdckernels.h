#ifndef TDCKERNELS_H
#define TDCKERNELS_H

/* CUDA headers */
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>

/**
 * Return a device pointer to the WindowMaxValue array.
 * See tdckernels.cu for details.
 */
void *getWindowMaxValuePtr();

/**
 * Create the window array for the specified chunk of raw data.
 */
void createWindow(float *window, int chunkIdx, int nPri, int nSamples,
                  cudaStream_t stream);

/**
 * Process the next chunk of data and focus it to the specified grid location.
 */
void focusToGridPoint(float2 const *rawData, float2 const *reference,
                      float *window, float4 const *position,
                      float4 const *velocity, float4 const *attitude,
                      float const *priTimes, float const *sampleTimes,
                      float4 const *focusGrid, float2 const *image,
                      float modRate, float startFreq, int chunkIdx, int nPri,
                      int nSamples, int streamIdx, cudaStream_t stream);

#endif // TDCKERNELS_H
