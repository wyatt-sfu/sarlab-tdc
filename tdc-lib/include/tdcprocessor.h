#ifndef TDCPROCESSOR_H
#define TDCPROCESSOR_H

/* Standard library headers */
#include <complex>

class TdcProcessor
{
public:
    TdcProcessor(int gpuNum);

    /**
     * Start the SAR processor. Returns when the processing is complete.
     */
    void start();

    /* Test for pybind, delete */
    void arrayTest(float *data, int arrLen);

    /**
     * Configures the raw radar data input to the processor.
     *
     * Arguments:
     * -----------
     * rawData: Pointer to a 2D array of raw IQ data
     *          - Size nPri x nSamples.
     *          - Stored row-wise.
     * priTimes: Pointer to a 1D array containing the start time of each PRI.
                 - The size is nPri.
     * nPri: Number of PRIs in raw data.
     * nSamples: Number of samples per PRI.
     * modRate: The modulation rate in Hz/s.
     * sampleRate: The data sample rate in Hz.
     */
    void setRawData(const std::complex<float> *rawData, const float *priTimes,
                    int nPri, int nSamples, float modRate, float sampleRate);

private:
};

#endif // TDCPROCESSOR_H
