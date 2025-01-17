#ifndef TDCPROCESSOR_H
#define TDCPROCESSOR_H

/* Standard library headers */
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/* CUDA headers */
#include <vector_types.h>

/* 3rd party headers */
#include <spdlog/common.h>
#include <spdlog/logger.h>

/* Project headers */
#include "gpuarray.h"
#include "pagelockedhost.h"

/**
 * Time-domain correlation SAR processor.
 *
 * Usage:
 * 1) Create an instance of the class
 * 2) Initialize with setRawData() and setFocusGrid()
 * 3) Call start()
 * 4) Call imageBuffer() to get a pointer to the focused image data
 *
 * Coordinate system:
 * This SAR processor is independent of the coordinate system used. Two coordinate
 * systems are implied: (i) body and (ii) local.
 *
 * (i) Body: The processor assumes that the radar antenna pointing direction
 *           remains fixed in the body coordinate system.
 * (ii) Local: The local coordinate system is the coordinates in which the focus
 *             grid is defined. The attitude quaternion defines the rotation from
 *             the body to the local coordinate system.
 */
class TdcProcessor
{
public:
    TdcProcessor(int gpuNum);
    ~TdcProcessor();

    /**
     * Start the SAR processor. Returns when the processing is complete.
     *
     * This function should only be called after setRawData and setFocusGrid.
     *
     * Arguments:
     * -------------
     * dopplerWinCenter: [Hz] If dopCentroidWin is false, then this parameter is used
     *                   to set the center frequency of the Doppler band used for
     *                   processing.
     * dopplerBandwidth: [Hz] Controls the width of the Doppler band used for
     *                   processing.
     * dopCentroidWin: If this is true, the Doppler centroid is used for windowing
     *                 and the dopplerWinCenter argument is ignored.
     * applyRangeWin: If true, the raw data will be range windowed. Set to
     *                False if you have already range windowed the raw data.
     */
    void start(float dopplerWinCenter, float dopplerBandwidth, bool dopCentroidWin,
               bool applyRangeWin);

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
     *           - Shape is nPri x nSamples x 3
     *           - Last dimension is ordered (x, y, z)
     *           - Stored row-wise
     * velocity: Pointer to a 3D array array of radar phase center velocities
     *           - Shape is nPri x nSamples x 3
     *           - Last dimension is ordered (x, y, z)
     *           - Stored row-wise
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
     * bodyBoresight: (x, y, z) of the boresight vector in the body coordinate system
     */
    void setRawData(std::complex<float> const *rawData, float const *priTimes,
                    float const *sampleTimes, float const *position,
                    float const *velocity, float const *attitude, int nPri,
                    int nSamples, float modRate, float startFreq, float3 bodyBoresight);

    /**
     * Configures the focus grid. This needs to be called before start().
     *
     * Arguments:
     * ------------
     * focusGrid: Pointer to a 3D array containing focus grid locations
     *            - Shape is nRows x nCols x 3
     *            - Last dimension is ordered (x, y, z)
     *            - Stored row-wise
     * nRows: Number of rows in the focus grid
     * nCols: Number of columns in the focus grid
     */
    void setFocusGrid(float const *focusGrid, int nRows, int nCols);

    /**
     * Configures the class to log to the specified sinks.
     *
     * Logging is only enabled after this function is called.
     */
    void setLoggerSinks(const std::vector<spdlog::sink_ptr> &sinks);

    /**
     * Returns a pointer to the focused image buffer. This should be called
     * after start() as finished. The TdcProcessor class retains ownership
     * of the buffer.
     */
    float2 const *imageBuffer() const;

private:
    /* Methods */
    void allocateGpuMemory();
    void allocateHostMemory();
    void initGpuData();
    void transferNextChunk(int chunkIdx);
    void stageNextChunk(int chunkIdx);
    bool windowIsNonZero(float3 target, float3 bodyBoresight, float dopplerBw,
                         float dopplerWinCenter, bool dopCentroidWin);

    /* Raw data fields */
    std::complex<float> const *rawData = nullptr;
    float const *priTimes = nullptr;
    float const *sampleTimes = nullptr;
    int nPri = 0;
    int nSamples = 0;
    int nChunks = 0;
    float modRate = 0.0;
    float startFreq = 0.0;
    float3 bodyBoresight = {0, 0, 0};
    float wavelengthCenter = 0.0;

    /* Radar position fields */
    float const *position = nullptr;
    float const *velocity = nullptr;
    float const *attitude = nullptr;

    /* Focus grid fields */
    float3 const *focusGrid = nullptr;
    int gridNumRows = 0;
    int gridNumCols = 0;
    std::unique_ptr<float2[]> focusedImage;

    /* Processor options */
    bool applyRangeWindow = true;

    /* Staging buffers for transferring data to the GPU */
    std::unique_ptr<PageLockedHost> rawStaging;
    std::unique_ptr<PageLockedHost> positionStaging;
    std::unique_ptr<PageLockedHost> velocityStaging;
    std::unique_ptr<PageLockedHost> attitudeStaging;

    /* GPU data structures */
    GpuArrayPtr<float2> rawDataGpu;
    GpuArrayPtr<float2> referenceGpu;
    GpuArrayPtr<float> windowGpu;
    GpuArrayPtr<uint8_t> sumScratchGpu;
    GpuArrayPtr<float3> positionGpu;
    GpuArrayPtr<float3> velocityGpu;
    GpuArrayPtr<float4> attitudeGpu;
    GpuArrayPtr<float> priTimesGpu;
    GpuArrayPtr<float> sampleTimesGpu;
    GpuArrayPtr<float> rangeWindowGpu;
    GpuArrayPtr<float2> imageGpu;
    GpuArrayPtr<float2> sumValueGpu;

    /* Logging */
    std::shared_ptr<spdlog::logger> log;

    /* Windowing speedup */
    static constexpr int PointsPerRgLine = 3;
    static constexpr int PointsPerChunk = 3;
    std::array<float, static_cast<size_t>(PointsPerChunk *PointsPerRgLine)> dopFreqMag;
};

#endif // TDCPROCESSOR_H
