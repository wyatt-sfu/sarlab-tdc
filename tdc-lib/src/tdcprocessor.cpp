/* Standard library headers */
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

/* CUDA headers */
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

/* 3rd party headers */
#include <fmt/core.h>
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

/* Project headers */
#include "cubscratch.h"
#include "cudastream.h"
#include "gpuarray.h"
#include "pagelockedhost.h"
#include "tdckernels.h"

/* Class header */
#include "tdcprocessor.h"

TdcProcessor::TdcProcessor(int gpuNum)
{
    // Initialize with logging to null
    log = spdlog::null_logger_mt("NULL");

    // Initialize the GPU
    log->info("Initializing GPU {}", gpuNum);
    cudaError_t const err = cudaSetDevice(gpuNum);
    if (err != cudaSuccess) {
        throw std::runtime_error(fmt::format("Failed to set cuda device: {}",
                                             cudaGetErrorString(err)));
    }

    // Create the stream objects for concurrency
    log->info("Creating CUDA streams");
    for (int i = 0; i < NUM_STREAMS; ++i) {
        streams[i] = std::make_unique<CudaStream>();
    }
}

TdcProcessor::~TdcProcessor()
{
    cudaDeviceReset();
}

void TdcProcessor::start()
{
    log->info("Starting the TDC processor");
    allocateGpuMemory();
    allocateHostMemory();
    initGpuData();

    nChunks = static_cast<int>(std::max(nPri / PRI_CHUNKSIZE, 1ULL));
    size_t streamIdx = 0;
    size_t nextStreamIdx = 1;

    // Initialize the range window
    initRangeWindow(rangeWindowGpu->ptr(), nSamples);

    // Before starting the loop we need to transfer the data for the first chunk
    transferNextChunk(0, streamIdx);
    cudaDeviceSynchronize();

    // Start looping through each chunk of data
    for (int i = 0; i < nChunks; ++i) {
        streamIdx = i % NUM_STREAMS;
        nextStreamIdx = (i + 1) % NUM_STREAMS;

        // Transfer the next chunk of data while we process the current chunk
        if (i + 1 < nChunks) {
            transferNextChunk(i + 1, nextStreamIdx);
        }

        for (int j = 0; j < gridNumRows; ++j) {
            for (int k = 0; k < gridNumCols; ++k) {
                // Create the window array
                createWindow(
                    // Window arrays
                    windowGpu[streamIdx]->ptr(), //
                    rangeWindowGpu->ptr(), //

                    // Position related arguments
                    velocityGpu[streamIdx]->ptr(), //
                    attitudeGpu[streamIdx]->ptr(), //

                    // Radar parameters
                    1.0F, 1.0F,

                    // Data shape arguments
                    i, nPri, nSamples, streams[streamIdx]->ptr());

                float3 target = focusGrid[(j * gridNumCols) + i];

                // Create the reference response for this grid location
                referenceResponse(
                    // Data array parameters
                    referenceGpu[streamIdx]->ptr(), //
                    windowGpu[streamIdx]->ptr(), //
                    positionGpu[streamIdx]->ptr(), //
                    sampleTimesGpu->ptr(), //
                    target, //

                    // Radar operating parameters
                    startFreq, modRate,

                    // Data shape arguments
                    i, nPri, nSamples, streams[streamIdx]->ptr());

                // Focus the chunk of data to the specified grid point
                focusToGridPoint(
                    rawDataGpu[streamIdx]->ptr(),
                    referenceGpu[streamIdx]->ptr(), windowGpu[streamIdx]->ptr(),
                    positionGpu[streamIdx]->ptr(),
                    velocityGpu[streamIdx]->ptr(),
                    attitudeGpu[streamIdx]->ptr(), priTimesGpu->ptr(),
                    sampleTimesGpu->ptr(), imageGpu->ptr(), target, modRate,
                    startFreq, i, nPri, nSamples, static_cast<int>(streamIdx),
                    streams[streamIdx]->ptr());
            }
        }
        cudaDeviceSynchronize();
    }

    log->info("Completed SAR processing");
}

void TdcProcessor::setRawData(std::complex<float> const *rawData,
                              float const *priTimes, float const *sampleTimes,
                              float const *position, float const *velocity,
                              float const *attitude, int nPri, int nSamples,
                              float modRate, float startFreq)
{
    log->info("Setting raw data with {} PRIs and {} samples/PRI", nPri,
              nSamples);
    this->rawData = rawData;
    this->priTimes = priTimes;
    this->sampleTimes = sampleTimes;
    this->position = position;
    this->velocity = velocity;
    this->attitude = attitude;
    this->nPri = nPri;
    this->nSamples = nSamples;
    this->modRate = modRate;
    this->startFreq = startFreq;

    // Error checking
    if (nPri <= 0) {
        throw std::invalid_argument("nPri must be greater than 0");
    }

    if (nSamples <= 0) {
        throw std::invalid_argument("nSamples must be greater than 0");
    }
}

void TdcProcessor::setFocusGrid(float const *focusGrid, int nRows, int nCols)
{
    log->info("Configure the focus grid with shape {} x {}", nRows, nCols);
    this->focusGrid = reinterpret_cast<float3 const *>(focusGrid);
    gridNumRows = nRows;
    gridNumCols = nCols;
}

void TdcProcessor::setLoggerSinks(const std::vector<spdlog::sink_ptr> &sinks)
{
    auto tdcLogger =
        std::make_shared<spdlog::logger>("TDCPROC", sinks.begin(), sinks.end());
    spdlog::register_logger(tdcLogger);

    spdlog::set_default_logger(tdcLogger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");
    spdlog::set_level(spdlog::level::debug);
    tdcLogger->flush_on(spdlog::level::info);
    tdcLogger->info("Completed logging setup");
    log = spdlog::get("TDCPROC");
}

void TdcProcessor::allocateGpuMemory()
{
    log->info("Allocating GPU memory for raw data ...");
    for (int i = 0; i < NUM_STREAMS; ++i) {
        rawDataGpu[i] =
            std::make_unique<GpuArray<float2>>(PRI_CHUNKSIZE * nSamples);
        referenceGpu[i] =
            std::make_unique<GpuArray<float2>>(PRI_CHUNKSIZE * nSamples);
        windowGpu[i] =
            std::make_unique<GpuArray<float>>(PRI_CHUNKSIZE * nSamples);
        positionGpu[i] =
            std::make_unique<GpuArray<float3>>(PRI_CHUNKSIZE * nSamples);
        velocityGpu[i] =
            std::make_unique<GpuArray<float3>>(PRI_CHUNKSIZE * nSamples);
        attitudeGpu[i] =
            std::make_unique<GpuArray<float4>>(PRI_CHUNKSIZE * nSamples);
    }

    priTimesGpu = std::make_unique<GpuArray<float>>(nPri);
    sampleTimesGpu = std::make_unique<GpuArray<float>>(nSamples);
    log->info("... Done allocating GPU memory for raw data");

    log->info("Allocating GPU memory for focused scene ...");
    imageGpu = std::make_unique<GpuArray<float2>>(gridNumRows * gridNumCols);
    log->info("... Done allocating GPU memory for focused scene");

    log->info("Allocating GPU scratch space ...");
    size_t maxScratchSize =
        CubHelpers::floatMaxScratchSize(PRI_CHUNKSIZE * nSamples);
    size_t sumScratchSize =
        CubHelpers::float2SumScratchSize(PRI_CHUNKSIZE * nSamples);

    log->info("Maximum scratch size: {}", maxScratchSize);
    log->info("Sum scratch size: {}", sumScratchSize);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        maxScratchGpu[i] = std::make_unique<GpuArray<uint8_t>>(maxScratchSize);
        sumScratchGpu[i] = std::make_unique<GpuArray<uint8_t>>(sumScratchSize);
    }

    rangeWindowGpu = std::make_unique<GpuArray<float>>(nSamples);
    log->info("... Done allocating GPU scratch space");
}

void TdcProcessor::allocateHostMemory()
{
    size_t stagingSizeData = PRI_CHUNKSIZE * nSamples * sizeof(float2);
    size_t stagingSizePos = PRI_CHUNKSIZE * nSamples * sizeof(float3);
    size_t stagingSizeVel = PRI_CHUNKSIZE * nSamples * sizeof(float3);
    size_t stagingSizeAtt = PRI_CHUNKSIZE * nSamples * sizeof(float4);

    log->info("Allocating page locked host memory ...");
    rawStaging = std::make_unique<PageLockedHost>(stagingSizeData);
    positionStaging = std::make_unique<PageLockedHost>(stagingSizePos);
    velocityStaging = std::make_unique<PageLockedHost>(stagingSizeVel);
    attitudeStaging = std::make_unique<PageLockedHost>(stagingSizeAtt);
    log->info("... Done allocating page locked host memory");
}

void TdcProcessor::initGpuData()
{
    log->info("Transferring timing data to the GPU ...");
    priTimesGpu->hostToDevice(priTimes);
    sampleTimesGpu->hostToDevice(sampleTimes);
    log->info("... Done transferring timing data to the GPU");

    log->info("Initializing focused image to zeros ...");
    cudaError_t err = cudaMemset(imageGpu->ptr(), 0,
                                 static_cast<size_t>(gridNumCols * gridNumRows)
                                     * sizeof(float2));
    log->info("... Done initializing focused image");

    log->info("Initializing range window ...");
    initRangeWindow(rangeWindowGpu->ptr(), nSamples);
    log->info("... Done initializing range window");
}

void TdcProcessor::transferNextChunk(int chunkIdx, size_t streamIdx)
{
    stageNextChunk(chunkIdx);
    rawDataGpu[streamIdx]->hostToDeviceAsync(
        reinterpret_cast<const float2 *>(rawStaging->ptr()),
        streams[streamIdx]->ptr());
    positionGpu[streamIdx]->hostToDeviceAsync(
        reinterpret_cast<const float3 *>(positionStaging->ptr()),
        streams[streamIdx]->ptr());
    velocityGpu[streamIdx]->hostToDeviceAsync(
        reinterpret_cast<const float3 *>(velocityStaging->ptr()),
        streams[streamIdx]->ptr());
    attitudeGpu[streamIdx]->hostToDeviceAsync(
        reinterpret_cast<const float4 *>(attitudeStaging->ptr()),
        streams[streamIdx]->ptr());
}

void TdcProcessor::stageNextChunk(int chunkIdx)
{
    // Transfer the next chunk of data into the staging area in page locked
    // memory. This is so that we can concurrently transfer data to the GPU
    // while processing is occuring.
    size_t priIndex = chunkIdx * PRI_CHUNKSIZE;
    size_t prisToStage = std::min(PRI_CHUNKSIZE, nPri - priIndex);
    size_t stagingSizeData = prisToStage * nSamples * sizeof(float2);
    size_t stagingSizePos = prisToStage * nSamples * sizeof(float3);
    size_t stagingSizeVel = prisToStage * nSamples * sizeof(float3);
    size_t stagingSizeAtt = prisToStage * nSamples * sizeof(float4);

    const auto *rawPtr = reinterpret_cast<const uint8_t *>(rawData);
    const auto *posPtr = reinterpret_cast<const uint8_t *>(position);
    const auto *velPtr = reinterpret_cast<const uint8_t *>(velocity);
    const auto *attPtr = reinterpret_cast<const uint8_t *>(attitude);

    log->info("Staging chunk {} of {}", chunkIdx + 1, nChunks);

    std::memcpy(rawStaging->ptr(),
                rawPtr + (priIndex * nSamples * sizeof(float2)),
                stagingSizeData);
    std::memcpy(positionStaging->ptr(),
                posPtr + (priIndex * nSamples * sizeof(float3)),
                stagingSizePos);
    std::memcpy(velocityStaging->ptr(),
                velPtr + (priIndex * nSamples * sizeof(float3)),
                stagingSizeVel);
    std::memcpy(attitudeStaging->ptr(),
                attPtr + (priIndex * nSamples * sizeof(float4)),
                stagingSizeAtt);
}
