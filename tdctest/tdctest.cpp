/* Standard library headers */
#include <complex>
#include <memory>
#include <vector>

/* 3rd party libraries */
#include <gtest/gtest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/common.h>
#include <vector_types.h>

/* Project headers */
#include "tdcprocessor.h"

/**
 * The purpose of this test is to just run the SAR processor and make sure
 * it does not crash.
 */
TEST(TdcProcessorTest, SmokeTest)
{
    TdcProcessor tdc(0);

    // Setup the focus grid
    int nRows = 20;
    int nCols = 20;
    int gridNumElem = nRows * nCols * 3;
    std::vector<float> grid(gridNumElem);
    tdc.setFocusGrid(grid.data(), nRows, nCols);

    // Setup the raw data
    int nPri = 750;
    int nSamples = 1500;
    int rawNumElem = nPri * nSamples;
    std::vector<std::complex<float>> raw(rawNumElem);
    std::vector<float> priTimes(rawNumElem);
    std::vector<float> sampleTimes(nSamples);
    std::vector<float> position(rawNumElem * 3);
    std::vector<float> velocity(rawNumElem * 3);
    std::vector<float> attitude(rawNumElem * 4);
    float3 bodyBoresight = {0, 1, 0};
    tdc.setRawData(raw.data(), priTimes.data(), sampleTimes.data(), position.data(),
                   velocity.data(), attitude.data(), nPri, nSamples, 1.0, 1.0,
                   bodyBoresight);

    // Setup logging
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinkList = {consoleSink};
    tdc.setLoggerSinks(sinkList);

    float dopplerBandwidth = 5.0;
    bool applyRangeWin = true;

    tdc.start(dopplerBandwidth, applyRangeWin);
    float2 const *img = tdc.imageBuffer();
    ASSERT_NE(img, nullptr);
}
