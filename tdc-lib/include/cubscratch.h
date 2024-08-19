/* Standard library headers */
#include <cstdlib>

namespace CubHelpers {

/**
 * Returns the scratch size required for computing a float max.
 */
size_t floatMaxScratchSize(size_t numItems);

/**
 * Returns the scratch size required for computing a float2 sum.
 */
size_t float2SumScratchSize(size_t numItems);

}
