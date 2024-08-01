#ifndef TDC_CUH
#define TDC_CUH

/* Standard library headers */
#include <cstddef>

/* CUDA headers */
#include <cuda_runtime.h>
#include <device_types.h>
#include <driver_types.h>
#include <vector_types.h>

/* Constants */
constexpr size_t PRI_CHUNKSIZE = 128;
constexpr size_t NUM_STREAMS = 2;

/**
 * Create the window array
 */
void createWindow(float *window, int chunkIdx, int nPri, int nSamples,
                  cudaStream_t stream);
__global__ void createWindowKernel(float *window, int chunkIdx, int nPri,
                                   int nSamples);

#endif // TDC_CUH
