#ifndef TDCKERNELS_H
#define TDCKERNELS_H

/* CUDA headers */
#include <cuda_runtime.h>

/**
 * Return a device pointer to the windowMaxValue array
 */
void *getWindowMaxValuePtr();

/**
 * Create the window array
 */
void createWindow(float *window, int chunkIdx, int nPri, int nSamples,
                  cudaStream_t stream);

#endif // TDCKERNELS_H
