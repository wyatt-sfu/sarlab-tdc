/* Standard library headers */
#include <cstdlib>
#include <stdexcept>

/* Cuda headers */
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

/* 3rd party headers */
#include <fmt/core.h>

/* Class header */
#include "cubscratch.h"
#include "gpumath.h"

namespace CubHelpers {

size_t floatMaxScratchSize(size_t numItems)
{
    size_t scratchSize = 0;
    float *tmpArr = nullptr;
    float *maxVal = nullptr;
    cudaError_t err =
        cub::DeviceReduce::Max(tmpArr, scratchSize, tmpArr, maxVal, numItems);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("Failed to compute scratch space size: {}",
                        cudaGetErrorString(err)));
    }

    return scratchSize;
}

size_t float2SumScratchSize(size_t numItems)
{
    size_t scratchSize = 0;
    float2 *tmpArr = nullptr;
    float2 *sumVal = nullptr;
    cudaError_t err =
        cub::DeviceReduce::Sum(tmpArr, scratchSize, tmpArr, sumVal, numItems);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("Failed to compute scratch space size: {}",
                        cudaGetErrorString(err)));
    }

    return scratchSize;
}

}
