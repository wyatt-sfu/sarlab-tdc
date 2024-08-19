#ifndef GPUMATH_H
#define GPUMATH_H

/* Cuda headers */
#include <cuda_runtime.h>
#include <device_types.h>
#include <vector_types.h>

/**
 * Sum operator for float2 arrays (which we use to represent complex numbers)
 */
__host__ __device__ inline float2 operator+(const float2 &lhs,
                                            const float2 &rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}

/**
 * Dot product of 3d vectors
 */
__host__ __device__ inline float v3_dot(float3 const *v1, float3 const *v2)
{
    return (v1->x * v2->x) + (v1->y * v2->y) + (v1->z * v2->z);
}

#endif // GPUMATH_H
