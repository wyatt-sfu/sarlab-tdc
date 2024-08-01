/* CUDA headers */
#include <cuda_runtime.h>
#include <device_types.h>
#include <driver_types.h>
#include <vector_types.h>

/* Class header */
#include "tdckernels.cuh"

void createWindow(float *window, int chunkIdx, int nPri, int nSamples,
                  cudaStream_t stream)
{
    dim3 blockSize(8, PRI_CHUNKSIZE, 0);
    dim3 gridSize((nSamples + blockSize.x - 1) / blockSize.x,
                  (nPri + blockSize.y - 1) / blockSize.y, 0);
    createWindowKernel<<<gridSize, blockSize, 0, stream>>>(window, chunkIdx,
                                                           nPri, nSamples);
}

/**
 * Correlate the raw data with the reference response.
 */
__global__ void createWindowKernel(float *window, int chunkIdx, int nPri,
                                   int nSamp)
{
    int priWindowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int priGlobalIdx =
        static_cast<int>(chunkIdx * PRI_CHUNKSIZE) + priWindowIdx;
    float winVal = 0;

    if (priWindowIdx < PRI_CHUNKSIZE && sampleIdx < nSamp) {
        if (priGlobalIdx < nPri) {
            // TODO: Compute window based on Doppler
            winVal = 1.0;
        } else {
            winVal = 0.0;
        }
    }

    window[priWindowIdx * nSamp + sampleIdx] = winVal;
}
