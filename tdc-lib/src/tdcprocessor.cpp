/* Standard library headers */
#include <chrono>
#include <complex>
#include <memory>
#include <stdexcept>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

/* 3rd party headers */
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

/* Project headers */
#include "gpuarray.h"
#include "gpupitchedarray.h"

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
    allocateGpuMemory();
    initGpuData();
}

void TdcProcessor::setRawData(std::complex<float> const *rawData,
                              float const *priTimes, float const *sampleTimes,
                              float const *position, float const *attitude,
                              int nPri, int nSamples, float modRate,
                              float sampleRate)
{
    log->info("Setting raw data with {} PRIs and {} samples/PRI", nPri,
              nSamples);
    this->rawData = rawData;
    this->priTimes = priTimes;
    this->sampleTimes = sampleTimes;
    this->position = position;
    this->attitude = attitude;
    this->nPri = nPri;
    this->nSamples = nSamples;
    this->modRate = modRate;
    this->sampleRate = sampleRate;
}

void TdcProcessor::setFocusGrid(float const *focusGrid, int nRows, int nCols)
{
    log->info("Configure the focus grid with shape {} x {}", nRows, nCols);
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

void TdcProcessor::allocateGpuMemory()
{
    log->info("Allocating GPU memory for raw data ...");
    rawDataGpu = std::make_unique<GpuPitchedArray<float2>>(nPri, nSamples);
    priTimesGpu = std::make_unique<GpuArray<float>>(nPri);
    sampleTimesGpu = std::make_unique<GpuArray<float>>(nSamples);
    log->info("... Done allocating GPU memory for raw data");

    log->info("Allocating GPU memory for position data ...");
    positionGpu = std::make_unique<GpuPitchedArray<float4>>(nPri, nSamples);
    attitudeGpu = std::make_unique<GpuPitchedArray<float4>>(nPri, nSamples);
    log->info("... Done allocating GPU memory for position data");

    log->info("Allocating GPU memory for focus grid ...");
    focusGridGpu =
        std::make_unique<GpuPitchedArray<float4>>(gridNumRows, gridNumCols);
    log->info("... Done allocating GPU memory for focus grid");
}

void TdcProcessor::initGpuData()
{
    log->info("Transferring raw data to the GPU ...");
    rawDataGpu->hostToDevice(reinterpret_cast<const float2 *>(rawData),
                             nSamples * sizeof(float2));
    priTimesGpu->hostToDevice(priTimes);
    sampleTimesGpu->hostToDevice(sampleTimes);
    log->info("... Done transferring raw data to the GPU");

    log->info("Transferring position data to the GPU");
    positionGpu->hostToDevice(reinterpret_cast<const float4 *>(position),
                              nSamples * sizeof(float4));
    attitudeGpu->hostToDevice(reinterpret_cast<const float4 *>(attitude),
                              nSamples * sizeof(float4));
    log->info("... Done transferring position data to the GPU");

    log->info("Transferring focus grid to the GPU");
    focusGridGpu->hostToDevice(reinterpret_cast<const float4 *>(focusGrid),
                               gridNumCols * sizeof(float4));
    log->info("... Done transferring focus grid to the GPU");
}
