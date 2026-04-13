#ifndef NPP_UTILS_H
#define NPP_UTILS_H

/* Cuda headers */
#include <driver_types.h>
#include <nppdefs.h>

/**
 * Create a Npp stream context object given a cuda stream
 */
NppStreamContext createNppStreamContext(cudaStream_t);

#endif // NPP_UTILS_H