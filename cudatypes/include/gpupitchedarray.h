#ifndef GPUPITCHEDARRAY_H
#define GPUPITCHEDARRAY_H

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
 * RAII class for managing 2D arrays allocated on the GPU
 */
template <typename T>
class GpuPitchedArray
{
public:
    /**
     * Construct a new GpuPitchedArray object and allocate the required space on
     * the GPU.
     */
    GpuPitchedArray(size_t numRows, size_t numCols)
    {
        this->numRows = numRows;
        this->numCols = numCols;

        void *gpuMem = nullptr;
        cudaError_t err =
            cudaMallocPitch(&gpuMem, &arrayPitch, numCols * sizeof(T), numRows);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format(
                "Failed to allocate array on the GPU: {}", cudaGetErrorString(err)));
        }

        array = reinterpret_cast<T *>(gpuMem);
    }

    /**
     * Free the memory on the GPU.
     */
    ~GpuPitchedArray() { cudaFree(array); }

    /**
     * Delete the copy constructor
     */
    GpuPitchedArray(const GpuPitchedArray &other) = delete;

    /**
     * Delete the copy-assignment constructor
     */
    GpuPitchedArray &operator=(const GpuPitchedArray &other) = delete;

    /**
     * Use the default move constructor
     */
    GpuPitchedArray(GpuPitchedArray &&other) = default;

    /**
     * Use the default move-assignment constructor
     */
    GpuPitchedArray &operator=(GpuPitchedArray &&other) = default;

    /**
     * Returns the pitch of the array on the GPU
     */
    size_t pitch() const { return arrayPitch; }

    /**
     * Returns the underlying device pointer
     */
    T *ptr() const { return array; }

    /**
     * Copy data in hostArray to the device. Size of hostArray must be large
     * enough (no checks are performed). The hostPitch is specified in bytes.
     */
    void hostToDevice(T const *hostArray, size_t hostPitch)
    {
        cudaError_t err =
            cudaMemcpy2D(array, arrayPitch, hostArray, hostPitch, numCols * sizeof(T),
                         numRows, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to copy memory to device: {}",
                                                 cudaGetErrorString(err)));
        }
    }

    /**
     * Asynchronously copies data in hostArray to the device. The size of
     * hostArray must be large enough (no checks are performed), AND
     * hostArray must be allocated from page locked memory using cudaMallocHost.
     * The hostPitch is specified in bytes. This function will return before
     * the transfer is complete.
     */
    void hostToDeviceAsync(T const *hostArray, size_t hostPitch, cudaStream_t stream)
    {
        cudaError_t err = cudaMemcpy2DAsync(array, arrayPitch, hostArray, hostPitch,
                                            numCols * sizeof(T), numRows,
                                            cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to copy memory to device: {}",
                                                 cudaGetErrorString(err)));
        }
    }

    /**
     * Copy data from the array on the GPU to the block of memory pointed to by
     * hostArray. Size of hostArray must be large enough (no checks are
     * performed). The hostPitch is specified in bytes.
     */
    void deviceToHost(T *hostArray, size_t hostPitch) const
    {
        cudaError_t err =
            cudaMemcpy2D(hostArray, hostPitch, array, arrayPitch, numCols * sizeof(T),
                         numRows, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to copy memory to host: {}",
                                                 cudaGetErrorString(err)));
        }
    }

    /**
     * Asynchronously copies data from the array on the GPU to the block of
     * memory pointed to by hostArray. The size of hostArray must be large
     * enough (no checks are performed), AND hostArray must be allocated from
     * page locked memory using cudaMallocHost. The hostPitch is specified in
     * bytes. This function will return before the transfer is complete.
     */
    void deviceToHostAsync(T *hostArray, size_t hostPitch, cudaStream_t stream) const
    {
        cudaError_t err = cudaMemcpy2DAsync(hostArray, hostPitch, array, arrayPitch,
                                            numCols * sizeof(T), numRows,
                                            cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to copy memory to host: {}",
                                                 cudaGetErrorString(err)));
        }
    }

private:
    T *array;
    size_t numRows;
    size_t numCols;
    size_t arrayPitch;
};

/**
 * Type aliases for more compact naming
 */
template <typename T>
using GpuPitchedArrayPtr = std::unique_ptr<GpuPitchedArray<T>>;

#endif // GPUPITCHEDARRAY_H
