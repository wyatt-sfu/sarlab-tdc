#ifndef GPUMATH_H
#define GPUMATH_H

/* Cuda headers */
#include <cuda_runtime.h>
#include <device_types.h>
#include <vector_types.h>

/**
 * Dot product of 3d vectors
 */
__host__ __device__ inline float v3_dot(float3 const *v1, float3 const *v2)
{
    return (v1->x * v2->x) + (v1->y * v2->y) + (v1->z * v2->z);
}

#endif // GPUMATH_H
