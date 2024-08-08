/* Standard library headers */
#include <vector>

/* 3rd party libraries */
#include <gtest/gtest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

/* Project headers */
#include "tdcprocessor.h"

TEST(TdcProcessorTest, BasicTest)
{
    TdcProcessor tdc(0);

    // Setup the focus grid
    int nRows = 1000;
    int nCols = 1000;
    int gridNumElem = nRows * nCols * 4;
    std::vector<float> grid(gridNumElem);
    tdc.setFocusGrid(grid.data(), nRows, nCols);

    // Setup the raw data
    int nPri = 2000;
    int nSamples = 4000;
    int rawNumElem = nPri * nSamples;
    std::vector<std::complex<float>> raw(rawNumElem);
    std::vector<float> priTimes(rawNumElem);
    std::vector<float> sampleTimes(nSamples);
    std::vector<float> position(rawNumElem * 4);
    std::vector<float> attitude(rawNumElem * 4);
    tdc.setRawData(raw.data(), priTimes.data(), sampleTimes.data(),
                   position.data(), attitude.data(), nPri, nSamples, 0.0, 0.0);

    // Setup logging
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinkList = {consoleSink};
    tdc.setLoggerSinks(sinkList);

    tdc.start();
}
