#ifndef WINDOWING_H
#define WINDOWING_H

/* Standard library headers */
#include <cmath>

/* CUDA headers*/
#include <math_constants.h>
#include <vector_types.h>

/* Project headers */
#include "gpumath.h"
#include "tuning.h"

/**
 * Returns the Doppler centroid using the platform velocity vector and the
 * radar antenna pointing vector (both in the local coordinate system).
 */
__host__ __device__ inline float dopplerCentroid(const float3 &vel,
                                                 const float3 &antPointing,
                                                 float lambda)
{
    return 2.0F / lambda * v3_dot(vel, antPointing);
}

/**
 * Returns the Doppler frequency to the specified target position.
 */
__host__ __device__ inline float dopplerFreq(const float3 &pos, const float3 &vel,
                                             const float3 &target, float lambda)
{
    float3 radarToTarget = target - pos;
    float fDop = 2.0F / lambda * v3_dot(vel, radarToTarget)
                 / hdNorm3df(radarToTarget.x, radarToTarget.y, radarToTarget.z);

    return fDop;
}

/**
 * Compute a window based on the difference between the Doppler frequency and
 * the Doppler centroid.
 */
__host__ __device__ inline float dopplerWindow(float fDop, float fDopCenter,
                                               float dopplerBw, bool taper = true)
{
    float deltaFDop = fDop - fDopCenter;
    float azWin = 0.0;
    if (fabs(deltaFDop) <= dopplerBw / 2.0) {
        if (taper) {
            azWin = AZIMUTH_WINDOW_A_PARAMETER
                    - ((1.0F - AZIMUTH_WINDOW_A_PARAMETER)
                       * cosf((2.0F * PI_F * deltaFDop / dopplerBw) - PI_F));
        } else {
            azWin = 1.0;
        }
    }
    return azWin;
}

#endif // WINDOWING_H
