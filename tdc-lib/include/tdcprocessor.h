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
    /* Raw data fields */
    std::complex<float> const *rawData = nullptr;
    float const *priTimes = nullptr;
    int nPri = 0;
    int nSamples = 0;
    float modRate = 0.0;
    float sampleRate = 0.0;
};

#endif // TDCPROCESSOR_H
