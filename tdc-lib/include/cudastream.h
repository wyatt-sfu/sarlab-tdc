#ifndef CUDASTREAM_H
#define CUDASTREAM_H

/* Standard library headers */
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>

/* 3rd party headers */
#include <fmt/core.h>

/**
 * RAII class for managing CUDA streams
 */
class CudaStream
{
public:
    /**
     * Create a cudaStream_t
     */
    CudaStream()
    {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("Failed to create stream: {}",
                                                 cudaGetErrorString(err)));
        }
    }

    /**
     * Destroy the cudaStream_t
     */
    ~CudaStream() { cudaStreamDestroy(stream); }

    /**
     * Delete the copy constructor
     */
    CudaStream(const CudaStream &other) = delete;

    /**
     * Delete the copy-assignment constructor
     */
    CudaStream &operator=(const CudaStream &other) = delete;

    /**
     * Use the default move constructor
     */
    CudaStream(CudaStream &&other) = default;

    /**
     * Use the default move-assignment constructor
     */
    CudaStream &operator=(CudaStream &&other) = default;

    /**
     * Returns a pointer to the cudaStream_t
     */
    cudaStream_t *ptr() { return &stream; }

private:
    cudaStream_t stream;
};

#endif // CUDASTREAM_h
