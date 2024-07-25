#ifndef TDCPROCESSOR_H
#define TDCPROCESSOR_H

/* Standard library headers */
#include <complex>

/**
 * Time-domain correlation SAR processor.
 *
 * Usage:
 * 1) Create an instance of the class
 * 2) Call setRawData()
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
     * Configures the raw radar data input to the processor.
     *
     * Arguments:
     * -----------
     * rawData: Pointer to a 2D array of raw IQ data
     *          - Shape is nPri x nSamples.
     *          - Stored row-wise.
     * priTimes: Pointer to a 1D array containing the start time of each PRI.
                 - Shape is nPri.
     * position: Pointer to a 3D array array of radar phase center positions
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
     * sampleRate: The data sample rate in Hz.
     */
    void setRawData(std::complex<float> const *rawData, float const *priTimes,
                    float const *position, float const *attitude, int nPri,
                    int nSamples, float modRate, float sampleRate);

private:
    /* Raw data fields */
    std::complex<float> const *rawData = nullptr;
    float const *priTimes = nullptr;
    float const *position = nullptr;
    float const *attitude = nullptr;
    int nPri = 0;
    int nSamples = 0;
    float modRate = 0.0;
    float sampleRate = 0.0;
};

#endif // TDCPROCESSOR_H
