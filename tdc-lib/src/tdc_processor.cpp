/* Standard library headers */
#include <complex>

/* Class header */
#include "tdc_processor.h"

TdcProcessor::TdcProcessor(int gpuNum)
{
    this->gpuNum = gpuNum;
}

void TdcProcessor::start() {}

void TdcProcessor::setRawData(const std::complex<float> *dataBuffer,
                              const float *priTimes, int nPri, int nSamples,
                              float modRate, float sampleRate)
{
}
