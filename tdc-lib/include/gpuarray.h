#ifndef GPUARRAY_H
#define GPUARRAY_H

/* Standard library headers */
#include <cstddef>
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>

/* 3rd party headers */
#include <fmt/core.h>

/**
 * RAII class for managing memory allocated on the GPU
 *
 */
template <typename T>
class GpuArray
{
public:
    /**
     * Construct a new GpuArray object and allocate the required space on
     * the GPU.
     */
    GpuArray(size_t arraySize)
    {
        this->arraySize = arraySize;
        void *gpuMem = nullptr;
        cudaError_t err = cudaMalloc(&gpuMem, arraySize);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                fmt::format("Failed to allocate array on the GPU: {}",
                            cudaGetErrorString(err)));
        }

        array = reinterpret_cast<T *>(gpuMem);
    }

    /**
     * Free the memory on the GPU.
     */
    ~GpuArray() { cudaFree(array); }

    /**
     * Delete the copy constructor
     */
    GpuArray(const GpuArray& other) = delete;

    /**
     * Delete the copy-assignment constructor
     */
    GpuArray& operator=(const GpuArray& other) = delete;

    /**
     * Use the default move constructor
     */
    GpuArray(GpuArray&& other) = default;

    /**
     * Use the default move-assignment constructor
     */
    GpuArray& operator=(GpuArray&& other) = default;

    /**
     * Copy data in hostArray to the device. Size of hostArray must be large
     * enough (no checks are performed).
     */
    void hostToDevice(T const *hostArray)
    {
        cudaError_t err = cudaMemcpy(array, hostArray, arraySize * sizeof(T),
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                fmt::format("Failed to copy memory to device: {}",
                            cudaGetErrorString(err)));
        }
    }

    /**
     * Copy data from the array on the GPU to the block of memory pointed to by
     * hostArray. Size of hostArray must be large enough (no checks are
     * performed).
     */
    void deviceToHost(T *hostArray)
    {
        cudaError_t err = cudaMemcpy(hostArray, array, arraySize * sizeof(T),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format(
                "Failed to copy memory to host: {}", cudaGetErrorString(err)));
        }
    }

private:
    T *array;
    size_t arraySize;
};

#endif // GPUARRAY_H
