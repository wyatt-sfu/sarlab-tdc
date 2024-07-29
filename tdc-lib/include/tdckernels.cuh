#ifndef TDC_CUH
#define TDC_CUH

/* Standard library headers */
#include <cstddef>

/* CUDA headers */
#include <cuda_runtime.h>

/* Constants */
constexpr size_t PRI_CHUNKSIZE = 128;
constexpr size_t NUM_STREAMS = 2;

#endif // TDC_CUH
