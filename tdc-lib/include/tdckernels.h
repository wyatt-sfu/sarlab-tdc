#ifndef TDCKERNELS_H
#define TDCKERNELS_H

/* CUDA headers */
#include <cuda_runtime.h>
#include <driver_types.h>

/**
 * Return a device pointer to the windowMaxValue array
 */
void *getWindowMaxValuePtr();

/**
 * Create the window array
 */
void createWindow(float *window, int chunkIdx, int nPri, int nSamples,
                  cudaStream_t stream);

/**
 * Process the next chunk of data and focus it to the specified grid location.
 */
void focusToGridPoint(float *window, int chunkIdx, int nPri, int nSamples,
                      cudaStream_t stream);

#endif // TDCKERNELS_H
