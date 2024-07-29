/* Standard library headers */
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
#include "cudastream.h"
#include "gpuarray.h"
#include "gpupitchedarray.h"
#include "tdckernels.cuh"

/* Class header */
#include "tdcprocessor.h"

TdcProcessor::TdcProcessor(int gpuNum)
{
    initLogging();
    log = spdlog::get("TDCPROC");

    // Initialize the GPU
    log->info("Initializing GPU {}", gpuNum);
    cudaError_t err = cudaSetDevice(gpuNum);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Create the stream objects for concurrency
    log->info("Creating CUDA streams");
    for (int i = 0; i < NUM_STREAMS; ++i) {
        streams[i] = std::make_unique<CudaStream>();
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
                              float startFreq)
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
    this->startFreq = startFreq;
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
    spdlog::set_level(spdlog::level::debug);
    tdcLogger->flush_on(spdlog::level::info);
    tdcLogger->info("Completed logging setup");
}

void TdcProcessor::allocateGpuMemory()
{
    log->info("Allocating GPU memory for raw data ...");
    rawDataGpu =
        std::make_unique<GpuPitchedArray<float2>>(PRI_CHUNKSIZE, nSamples);
    priTimesGpu = std::make_unique<GpuArray<float>>(PRI_CHUNKSIZE);
    sampleTimesGpu = std::make_unique<GpuArray<float>>(nSamples);
    log->info("... Done allocating GPU memory for raw data");

    log->info("Allocating GPU memory for position data ...");
    positionGpu =
        std::make_unique<GpuPitchedArray<float4>>(PRI_CHUNKSIZE, nSamples);
    attitudeGpu =
        std::make_unique<GpuPitchedArray<float4>>(PRI_CHUNKSIZE, nSamples);
    log->info("... Done allocating GPU memory for position data");

    log->info("Allocating GPU memory for focus grid ...");
    focusGridGpu =
        std::make_unique<GpuPitchedArray<float4>>(gridNumRows, gridNumCols);
    log->info("... Done allocating GPU memory for focus grid");

    log->info("Allocating GPU memory for focused scene ...");
    imageGpu =
        std::make_unique<GpuPitchedArray<float2>>(gridNumRows, gridNumCols);
    log->info("... Done allocating GPU memory for focused scene");
}

void TdcProcessor::initGpuData()
{
    log->info("Transferring timing data to the GPU ...");
    priTimesGpu->hostToDevice(priTimes);
    sampleTimesGpu->hostToDevice(sampleTimes);
    log->info("... Done transferring raw data to the GPU");

    log->info("Transferring focus grid to the GPU");
    focusGridGpu->hostToDevice(reinterpret_cast<const float4 *>(focusGrid),
                               gridNumCols * sizeof(float4));
    log->info("... Done transferring focus grid to the GPU");

    log->info("Initializing focused image to zeros ...");
    cudaError_t err = cudaMemset2D(imageGpu->ptr(), imageGpu->pitch(), 0,
                                   gridNumCols, gridNumRows);
    log->info("... Done initializing focused image");
}
