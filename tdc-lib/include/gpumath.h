#ifndef GPUMATH_H
#define GPUMATH_H

/* Cuda headers */
#include <cuda_runtime.h>
#include <device_types.h>
#include <vector_types.h>

/**
 * Sum operator for float2 arrays (which we use to represent complex numbers)
 */
__host__ __device__ inline float2 operator+(const float2 &lhs, const float2 &rhs)
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

/**
 * Quaternion conjugate
 */
__host__ __device__ inline float4 q_conj(float4 q)
{
    float *vals = reinterpret_cast<float *>(&q);
    vals[1] *= -1;
    vals[2] *= -1;
    vals[3] *= -1;
    return q;
}

/**
 * Product of two quaternions
 */
__host__ __device__ inline float4 q_prod(float4 q1, float4 q2)
{
    return {(q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z) - (q1.w * q2.w),
            (q1.x * q2.y) + (q1.y * q2.x) + (q1.z * q2.w) - (q1.w * q2.z),
            (q1.x * q2.z) - (q1.y * q2.w) + (q1.z * q2.x) + (q1.w * q2.y),
            (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y) + (q1.w * q2.x)};
}

/**
 * Rotate the vector v by the quaternion q. Assumes q is a unit quaternion.
 */
__host__ __device__ inline float3 q_rot(float4 q, float3 v)
{
    float4 vq;
    float *_vqF32 = reinterpret_cast<float *>(&vq);
    _vqF32[0] = 0.0F;
    _vqF32[1] = v.x;
    _vqF32[2] = v.y;
    _vqF32[3] = v.z;
    vq = q_prod(q_prod(q, vq), q_conj(q));
    return {vq.y, vq.z, vq.w};
}

#endif // GPUMATH_H
