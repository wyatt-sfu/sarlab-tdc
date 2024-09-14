/* Standard library headers */
#include <cassert>
#include <cmath>

/* 3rd party libraries */
#include <gtest/gtest.h>
#include <vector_types.h>

/* Project headers */
#include "gpumath.h"

TEST(GpuMathTest, norm3df)
{
    float a = 2.0f;
    float b = 4.0f;
    float c = 6.0f;
    float norm = sqrtf((a * a) + (b * b) + (c * c));
    ASSERT_FLOAT_EQ(norm, hdNorm3df(a, b, c));
}

TEST(GpuMathTest, float2Addition)
{
    float2 a = {1.0, 2.0};
    float2 b = {3.0, 5.0};
    float2 val = a + b;

    ASSERT_FLOAT_EQ(val.x, 4.0);
    ASSERT_FLOAT_EQ(val.y, 7.0);

    b = {-3.0, -5.0};
    val = a + b;
    ASSERT_FLOAT_EQ(val.x, -2.0);
    ASSERT_FLOAT_EQ(val.y, -3.0);

    a = {0.0, 0.0};
    b = {1.0, 1.0};
    val = a + b;
    ASSERT_FLOAT_EQ(val.x, 1.0);
    ASSERT_FLOAT_EQ(val.y, 1.0);

    a = {1.0, 1.0};
    b = {0.0, 0.0};
    val = a + b;
    ASSERT_FLOAT_EQ(val.x, 1.0);
    ASSERT_FLOAT_EQ(val.y, 1.0);
}

TEST(GpuMathTest, float3Subtraction)
{
    float3 a = {1.0, 2.0, 3.0};
    float3 b = {4.0, 8.0, 12.0};
    float3 val = a - b;

    ASSERT_FLOAT_EQ(val.x, -3.0);
    ASSERT_FLOAT_EQ(val.y, -6.0);
    ASSERT_FLOAT_EQ(val.z, -9.0);

    b = {-4.0, -8.0, -12.0};
    val = a - b;
    ASSERT_FLOAT_EQ(val.x, 5.0);
    ASSERT_FLOAT_EQ(val.y, 10.0);
    ASSERT_FLOAT_EQ(val.z, 15.0);

    a = {1.0, 1.0, 1.0};
    b = {0.0, 0.0, 0.0};
    val = a - b;
    ASSERT_FLOAT_EQ(val.x, 1.0);
    ASSERT_FLOAT_EQ(val.y, 1.0);
    ASSERT_FLOAT_EQ(val.z, 1.0);

    a = {0.0, 0.0, 0.0};
    b = {1.0, 1.0, 1.0};
    val = a - b;
    ASSERT_FLOAT_EQ(val.x, -1.0);
    ASSERT_FLOAT_EQ(val.y, -1.0);
    ASSERT_FLOAT_EQ(val.z, -1.0);
}

TEST(GpuMathTest, v3Dot)
{
    float3 a = {1.0, 0.0, 0.0};
    float3 b = {4.0, 8.0, 12.0};
    float val = v3_dot(a, b);
    ASSERT_FLOAT_EQ(a.x * b.x + a.y * b.y + a.z * b.z, 4.0);

    a = {0.0, 1.0, 0.0};
    b = {4.0, 8.0, 12.0};
    val = v3_dot(a, b);
    ASSERT_FLOAT_EQ(a.x * b.x + a.y * b.y + a.z * b.z, 8.0);

    a = {0.0, 0.0, 1.0};
    b = {4.0, 8.0, 12.0};
    val = v3_dot(a, b);
    ASSERT_FLOAT_EQ(a.x * b.x + a.y * b.y + a.z * b.z, 12.0);

    a = {1.0, 1.0, 1.0};
    b = {4.0, 8.0, 12.0};
    val = v3_dot(a, b);
    ASSERT_FLOAT_EQ(a.x * b.x + a.y * b.y + a.z * b.z, 24.0);
}

TEST(GpuMathTest, qConj)
{
    float4 q = {1.0, 2.0, 3.0, 4.0};
    float4 qc = q_conj(q);
    ASSERT_FLOAT_EQ(qc.x, q.x);
    ASSERT_FLOAT_EQ(qc.y, -q.y);
    ASSERT_FLOAT_EQ(qc.z, -q.z);
    ASSERT_FLOAT_EQ(qc.w, -q.w);
}

TEST(GpuMathTest, qProd)
{
    float4 q1 = {1.0, 2.0, 3.0, 4.0};
    float4 q2 = {8.0, 7.0, 6.0, 5.0};
    float4 prod = q_prod(q1, q2);
    ASSERT_FLOAT_EQ(prod.x, -44);
    ASSERT_FLOAT_EQ(prod.y, 14);
    ASSERT_FLOAT_EQ(prod.z, 48);
    ASSERT_FLOAT_EQ(prod.w, 28);
}

TEST(GpuMathTest, qRot)
{
    float3 vec = {1.0, 0.0, 0.0};
    float4 q = {sqrtf(2) / 2.0f, 0, 0, sqrtf(2) / 2.0f};
    float3 vecOut = q_rot(q, vec);
    ASSERT_FLOAT_EQ(vecOut.x, 0);
    ASSERT_FLOAT_EQ(vecOut.y, 1);
    ASSERT_FLOAT_EQ(vecOut.z, 0);

    q = {sqrtf(2) / 2.0f, 0, 0, -sqrtf(2) / 2.0f};
    vecOut = q_rot(q, vec);
    ASSERT_FLOAT_EQ(vecOut.x, 0);
    ASSERT_FLOAT_EQ(vecOut.y, -1);
    ASSERT_FLOAT_EQ(vecOut.z, 0);
}
