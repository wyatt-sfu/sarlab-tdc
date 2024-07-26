/* Standard library headers */
#include <chrono>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>

/* 3rd party headers */
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

/* Project headers */
#include "gpuarray.h"

/* Class header */
#include "tdcprocessor.h"

TdcProcessor::TdcProcessor(int gpuNum)
{
    initLogging();
    log = spdlog::get("TDCPROC");

    // Initialize the GPU
    cudaError_t err = cudaSetDevice(gpuNum);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void TdcProcessor::start()
{
    log->info("Starting the TDC processor");

    // Allocate memory on the GPU
    focusGridGpu = std::make_unique<GpuArray<float>>(gridNumRows * gridNumCols);
}

void TdcProcessor::setRawData(std::complex<float> const *rawData,
                              float const *priTimes, float const *position,
                              float const *attitude, int nPri, int nSamples,
                              float modRate, float sampleRate)
{
    this->rawData = rawData;
    this->priTimes = priTimes;
    this->position = position;
    this->attitude = attitude;
    this->nPri = nPri;
    this->nSamples = nSamples;
    this->modRate = modRate;
    this->sampleRate = sampleRate;
}

void TdcProcessor::setFocusGrid(float const *focusGrid, int nRows, int nCols)
{
    log->info("Configure the grid with size {} x {}", nRows, nCols);
    this->focusGrid = focusGrid;
    gridNumRows = nRows;
    gridNumCols = nCols;
}

void TdcProcessor::initLogging()
{
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinkList = {consoleSink};

    // Create loggers and register them with spdlog
    // Registering allows the logger objects to be retrieved with spdlog::get
    auto defaultLogger = std::make_shared<spdlog::logger>(
        "DEFAULT", sinkList.begin(), sinkList.end());
    spdlog::register_logger(defaultLogger);

    auto tdcLogger = std::make_shared<spdlog::logger>(
        "TDCPROC", sinkList.begin(), sinkList.end());
    spdlog::register_logger(tdcLogger);

    spdlog::set_default_logger(defaultLogger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");
    spdlog::flush_every(std::chrono::seconds(2));

    spdlog::set_level(spdlog::level::debug);
    tdcLogger->info("Completed logging setup");
}
