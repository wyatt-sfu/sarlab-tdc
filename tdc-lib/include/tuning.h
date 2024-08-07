/**
 * This file is used to contain CUDA parameters that can be changed while
 * performance tuning the software.
 */

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
 * Tuning parameters for the window generation kernel
 */
namespace WindowKernel {
constexpr unsigned int BlockSizeX = 8;
constexpr unsigned int BlockSizeY = PRI_CHUNKSIZE;
}
