/* CUDA headers*/
#include <vector_types.h>

/* Project headers */
#include "gpumath.h"

/**
 * Returns the Doppler centroid using the platform velocity vector and the
 * radar antenna pointing vector (both in the local coordinate system).
 */
__device__ inline float dopplerCentroid(const float3 &vel,
                                                 const float3 &antPointing,
                                                 float lambda)
{
    return 2.0F / lambda * v3_dot(vel, antPointing);
}

/**
 * Returns the Doppler frequency to the specified target position.
 */
__device__ inline float dopplerFreq(const float3 &pos, const float3 &vel,
                                             const float3 &target, float lambda)
{
    float3 radarToTarget = target - pos;
    float fDop = 2.0F / lambda * v3_dot(vel, radarToTarget)
                 / norm3df(radarToTarget.x, radarToTarget.y, radarToTarget.z);

    return fDop;
}