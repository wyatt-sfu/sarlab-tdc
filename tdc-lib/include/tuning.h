/**
 * This file is used to contain CUDA parameters that can be changed while
 * performance tuning the software.
 */

/* Standard library headers */
#include <cstddef>

/* CUDA headers */
#include <vector_types.h>

/**
 * Controls how many PRI's are processed in each chunk.
 */
constexpr size_t PRI_CHUNKSIZE = 128;

/**
 * Controls how many streams are used at once. This should probably be set to
 * either 1 or 2.
 */
constexpr size_t NUM_STREAMS = 2;

/**
 * If the maximum value in the window array for a specific chunk is below this
 * value then it will not be processed.
 */
constexpr float WINDOW_LOWER_BOUND = 0.01;

/**
 * The "a" parameter for the range window function.
 * Setting this value to 25/46 creates a Hamming window, and setting it to
 * 0.5 creates a Hann window.
 */
constexpr float RANGE_WINDOW_A_PARAMETER = 25.0 / 46.0;

/**
 * Tuning parameters for the window generation kernel
 */
namespace WindowKernel {
constexpr unsigned int BlockSizeX = 16;
constexpr unsigned int BlockSizeY = 16;
}

namespace ReferenceResponseKernel {
constexpr unsigned int BlockSizeX = 16;
constexpr unsigned int BlockSizeY = 16;
}

namespace CorrelateKernel {
constexpr unsigned int BlockSizeX = 16;
constexpr unsigned int BlockSizeY = 16;
}
