/* Cuda headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nppdefs.h>

/* File header */
#include "npp_utils.h"

NppStreamContext createNppStreamContext(cudaStream_t s)
{
    NppStreamContext ctx{};
    ctx.hStream = s;
    cudaGetDevice(&ctx.nCudaDeviceId);

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, ctx.nCudaDeviceId);

    ctx.nMultiProcessorCount = p.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = p.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = p.minor;
    cudaStreamGetFlags(s, &ctx.nStreamFlags);

    return ctx;
}
