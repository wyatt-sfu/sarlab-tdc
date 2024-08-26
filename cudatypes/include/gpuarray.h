#ifndef GPUARRAY_H
#define GPUARRAY_H

/* Standard library headers */
#include <cstddef>
#include <memory>
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>

/* 3rd party headers */
#include <fmt/core.h>

/**
 * RAII class for managing memory allocated on the GPU
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
        cudaError_t err = cudaMalloc(&gpuMem, arraySize * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format(
                "Failed to allocate array on the GPU: {}", cudaGetErrorString(err)));
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
    GpuArray(const GpuArray &other) = delete;

    /**
     * Delete the copy-assignment constructor
     */
    GpuArray &operator=(const GpuArray &other) = delete;

    /**
     * Use the default move constructor
     */
    GpuArray(GpuArray &&other) = default;

    /**
     * Use the default move-assignment constructor
     */
    GpuArray &operator=(GpuArray &&other) = default;

    /**
     * Returns the underlying device pointer
     */
    T *ptr() const { return array; }

    /**
     * Returns the size of the array.
     */
    size_t size() const { return arraySize; }

    /**
     * Copy data in hostArray to the device. Size of hostArray must be large
     * enough (no checks are performed).
     */
    void hostToDevice(T const *hostArray)
    {
        cudaError_t err =
            cudaMemcpy(array, hostArray, arraySize * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to copy memory to device: {}",
                                                 cudaGetErrorString(err)));
        }
    }

    /**
     * Asynchronously copies data in hostArray to the device. The size of
     * hostArray must be large enough (no checks are performed), AND
     * hostArray must be allocated from page locked memory using cudaMallocHost.
     * This function will return before the transfer is complete.
     */
    void hostToDeviceAsync(T const *hostArray, cudaStream_t stream)
    {
        cudaError_t err = cudaMemcpyAsync(array, hostArray, arraySize * sizeof(T),
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format(
                "Failed to async copy memory to device: {}", cudaGetErrorString(err)));
        }
    }

    /**
     * Copy data from the array on the GPU to the block of memory pointed to by
     * hostArray. Size of hostArray must be large enough (no checks are
     * performed).
     */
    void deviceToHost(T *hostArray) const
    {
        cudaError_t err =
            cudaMemcpy(hostArray, array, arraySize * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to copy memory to host: {}",
                                                 cudaGetErrorString(err)));
        }
    }

    /**
     * Asynchronously copies data from the array on the GPU to the block of
     * memory pointed to by hostArray. The size of hostArray must be large
     * enough (no checks are performed), AND hostArray must be allocated from
     * page locked memory using cudaMallocHost. This function will return
     * before the transfer is complete.
     */
    void deviceToHostAsync(T *hostArray, cudaStream_t stream) const
    {
        cudaError_t err = cudaMemcpyAsync(hostArray, array, arraySize * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to copy memory to host: {}",
                                                 cudaGetErrorString(err)));
        }
    }

private:
    T *array;
    size_t arraySize;
};

/**
 * Type aliases for more compact naming
 */
template <typename T>
using GpuArrayPtr = std::unique_ptr<GpuArray<T>>;

#endif // GPUARRAY_H
