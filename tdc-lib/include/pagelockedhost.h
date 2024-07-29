#ifndef PAGELOCKEDHOST_H
#define PAGELOCKEDHOST_H

/* Standard library headers */
#include <cstddef>
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>

/* 3rd party headers */
#include <fmt/core.h>

/**
 * RAII class for managing page locked host memory
 */
class PageLockedHost
{
public:
    /**
     * Allocate size bytes in page locked host memory
     */
    PageLockedHost(size_t size)
    {
        cudaError_t err = cudaMallocHost(&hostMemory, size);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                fmt::format("Failed to allocate page locked memory on host: {}",
                            cudaGetErrorString(err)));
        }
    }

    /**
     * Free memory
     */
    ~PageLockedHost() { cudaFreeHost(hostMemory); }

    /**
     * Delete the copy constructor
     */
    PageLockedHost(const PageLockedHost &other) = delete;

    /**
     * Delete the copy-assignment constructor
     */
    PageLockedHost &operator=(const PageLockedHost &other) = delete;

    /**
     * Use the default move constructor
     */
    PageLockedHost(PageLockedHost &&other) = default;

    /**
     * Use the default move-assignment constructor
     */
    PageLockedHost &operator=(PageLockedHost &&other) = default;

    /**
     * Returns a pointer to the host memory
     */
    void *ptr() { return hostMemory; }

private:
    void *hostMemory;
};

#endif // PAGELOCKEDHOST_H
