#ifndef TDCPROCESSOR_H
#define TDCPROCESSOR_H

/* Standard library headers */
#include <array>
#include <complex>
#include <memory>
#include <vector>

/* CUDA headers */
#include <vector_types.h>

/* 3rd party headers */
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

/* Project headers */
#include "cudastream.h"
#include "gpuarray.h"
#include "gpupitchedarray.h"
#include "pagelockedhost.h"
#include "tdckernels.cuh"

/**
 * Time-domain correlation SAR processor.
 *
 * Usage:
 * 1) Create an instance of the class
 * 2) Initialize with setRawData() and setFocusGrid()
 * 3) Call start()
 *
 * Coordinate system definition:
 * Body Coordinate System:
 * -----------------------
 * Axes:
 *     +Z: Upwards
 *     +Y: Boresight
 *     +X: To the right, from the perspective of the radar (left when
 *         looking directly at radar)
 *
 * Local Coordinate System:
 * ------------------------
 *     +Z: Upwards
 *     +X: Along direction of motion
 *     +Y: Defined by right-hand rule
 */
class TdcProcessor
{
public:
    TdcProcessor(int gpuNum);

    /**
     * Start the SAR processor. Returns when the processing is complete.
     */
    void start();

    /**
     * Configures the raw radar data input to the processor. This needs to be
     * called before start().
     *
     * Arguments:
     * -----------
     * rawData: Pointer to a 2D array of raw IQ data
     *          - Shape is nPri x nSamples x float[2].
     *          - Stored row-wise.
     * priTimes: Pointer to a 1D array containing the start time of each PRI.
                 - Shape is nPri
     * sampleTimes: Pointer to a 1D array containing the time of each sample
     *              relative to the start of the PRI.
     *              - Shape is nSamples
     * position: Pointer to a 3D array array of radar phase center positions
     *           - Shape is nPri x nSamples x 4
     *           - Last dimension is ordered (x, y, z, 0)
     *           - Stored row-wise
     *           - Last element of each value is 0 for GPU performance
     * attitude: Pointer to a 3D array of radar attitude values
     *           - Orientation represented as a quaternion rotation from body
     *             coordinate system to the local coordinate system
     *           - Shape is nPri x nSamples x 4
     *           - Last dimension is ordered (a, b, c, d)
     *           - Stored row-wise
     * nPri: Number of PRIs in raw data.
     * nSamples: Number of samples per PRI.
     * modRate: The modulation rate in Hz/s.
     * startFreq: Start frequency of the linear FMCW chirp in Hz.
     */
    void setRawData(std::complex<float> const *rawData, float const *priTimes,
                    float const *sampleTimes, float const *position,
                    float const *attitude, int nPri, int nSamples,
                    float modRate, float startFreq);

    /**
     * Configures the focus grid. This needs to be called before start().
     *
     * Arguments:
     * ------------
     * focusGrid: Pointer to a 3D array containing focus grid locations
     *            - Shape is nRows x nCols x 4
     *            - Last dimension is ordered (x, y, z, 0)
     *            - Stored row-wise
     *            - Last element of each value is 0 for GPU performance
     * nRows: Number of rows in the focus grid
     * nCols: Number of columns in the focus grid
     */
    void setFocusGrid(float const *focusGrid, int nRows, int nCols);

private:
    /* Methods */
    void initLogging();
    void allocateGpuMemory();
    void allocateHostMemory();
    void initGpuData();
    void stageNextChunk(int chunkIdx);

    /* Raw data fields */
    std::complex<float> const *rawData = nullptr;
    float const *priTimes = nullptr;
    float const *sampleTimes = nullptr;
    int nPri = 0;
    int nSamples = 0;
    int nChunks = 0;
    float modRate = 0.0;
    float startFreq = 0.0;

    /* Radar position fields */
    float const *position = nullptr;
    float const *attitude = nullptr;

    /* Focus grid fields */
    float const *focusGrid = nullptr;
    int gridNumRows = 0;
    int gridNumCols = 0;

    /* Staging buffers for transferring data to the GPU */
    std::unique_ptr<PageLockedHost> rawStaging;
    std::unique_ptr<PageLockedHost> posStaging;
    std::unique_ptr<PageLockedHost> attitudeStaging;

    /* GPU data structures */
    std::array<std::unique_ptr<CudaStream>, NUM_STREAMS> streams;
    std::array<GpuPitchedArrayPtr<float2>, NUM_STREAMS> rawDataGpu;
    std::array<GpuPitchedArrayPtr<float2>, NUM_STREAMS> windowGpu;
    std::array<GpuPitchedArrayPtr<float4>, NUM_STREAMS> positionGpu;
    std::array<GpuPitchedArrayPtr<float4>, NUM_STREAMS> attitudeGpu;
    GpuArrayPtr<float> priTimesGpu;
    GpuArrayPtr<float> sampleTimesGpu;
    GpuPitchedArrayPtr<float4> focusGridGpu;
    GpuPitchedArrayPtr<float2> imageGpu;

    /* Logging */
    std::vector<spdlog::sink_ptr> sinkList;
    std::shared_ptr<spdlog::logger> log;
};

#endif // TDCPROCESSOR_H
