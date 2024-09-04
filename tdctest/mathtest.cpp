/* Standard library headers */
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
