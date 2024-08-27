/* Standard library headers */
#include <algorithm>
#include <chrono>
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
#include "gpuarray.h"
#include "pagelockedhost.h"
#include "tdckernels.h"
#include "tuning.h"

/* Class header */
#include "tdcprocessor.h"

#define SPEED_OF_LIGHT 299792458.0

TdcProcessor::TdcProcessor(int gpuNum)
{
    // Initialize with logging to null
    log = spdlog::null_logger_mt("NULL");

    // Initialize the GPU
    log->info("Initializing GPU {}", gpuNum);
    cudaError_t const err = cudaSetDevice(gpuNum);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("Failed to set cuda device: {}", cudaGetErrorString(err)));
    }
}

TdcProcessor::~TdcProcessor()
{
    cudaDeviceReset();
}

void TdcProcessor::start(float dopplerBandwidth, bool applyRangeWindow)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    log->info("Starting the TDC processor");
    this->applyRangeWindow = applyRangeWindow;
    allocateGpuMemory();
    allocateHostMemory();
    initGpuData();

    // The data is chunked so that large data collections can still be processed
    // without running out of RAM on the GPU. Useful for laptops.
    nChunks = static_cast<int>(std::max(nPri / PRI_CHUNKSIZE, 1ULL));
    cudaDeviceSynchronize();

    // Start looping through each chunk of data
    for (int i = 0; i < nChunks; ++i) {
        // Transfer the next chunk of data while we process the current chunk
        transferNextChunk(i);
        log->info("Processing chunk {} of {}", i + 1, nChunks);

        for (int j = 0; j < gridNumRows; ++j) {
            for (int k = 0; k < gridNumCols; ++k) {
                float3 target = focusGrid[(j * gridNumCols) + k];

                // Create the window array
                createWindow(
                    // Window arrays
                    windowGpu->ptr(), rangeWindowGpu->ptr(), //

                    // Position related arguments
                    velocityGpu->ptr(), attitudeGpu->ptr(), target, //

                    // Radar parameters
                    wavelengthCenter, dopplerBandwidth,

                    // Data shape arguments
                    i, nPri, nSamples);

                // Create the reference response for this grid location
                referenceResponse(
                    // Data array parameters
                    referenceGpu->ptr(), //
                    windowGpu->ptr(), //
                    positionGpu->ptr(), //
                    sampleTimesGpu->ptr(), //
                    target, //

                    // Radar operating parameters
                    startFreq, modRate,

                    // Data shape arguments
                    i, nPri, nSamples);

                // Correlate the raw and reference arrays and then add the
                // result to the focused image
                float2 *pixelPtr =
                    imageGpu->ptr() + (static_cast<ptrdiff_t>(j) * gridNumCols) + k;
                correlateAndSum(
                    // Data array parameters
                    rawDataGpu->ptr(), //
                    referenceGpu->ptr(), //
                    sumScratchGpu->ptr(), //
                    sumScratchGpu->size(), //

                    // Focus image pixel
                    pixelPtr,

                    // Data shape
                    i, nPri, nSamples);
            }
        }

        // Sanity check for any cuda runtime errors
        // At this point there really shouldn't be any errors so I just check
        // after each chunk is processed.
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                fmt::format("Cuda runtime error occured during processing: {}",
                            cudaGetErrorString(err)));
        }
        cudaDeviceSynchronize();
    }

    log->info("Transferring focused image off the GPU ...");
    imageGpu->deviceToHost(focusedImage.get());
    log->info("... Done transferring the focused image off the GPU");

    log->info("Completed SAR processing");
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    log->info("Elapsed time: {}s",
              std::chrono::duration_cast<std::chrono::seconds>(end - begin).count());
}

void TdcProcessor::setRawData(std::complex<float> const *rawData, float const *priTimes,
                              float const *sampleTimes, float const *position,
                              float const *velocity, float const *attitude, int nPri,
                              int nSamples, float modRate, float startFreq)
{
    log->info("Setting raw data with {} PRIs and {} samples/PRI", nPri, nSamples);
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

    // Compute the wavelength of the center of the chirp
    double chirpTime = sampleTimes[nSamples - 1] - sampleTimes[0];
    double endFreq = startFreq + (static_cast<double>(modRate) * chirpTime);
    double centerFreq = (startFreq + endFreq) / 2.0;
    wavelengthCenter = static_cast<float>(SPEED_OF_LIGHT / centerFreq);
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

float2 const *TdcProcessor::imageBuffer() const
{
    return focusedImage.get();
}

void TdcProcessor::allocateGpuMemory()
{
    log->info("Allocating GPU memory for raw data ...");
    rawDataGpu = std::make_unique<GpuArray<float2>>(PRI_CHUNKSIZE * nSamples);
    referenceGpu = std::make_unique<GpuArray<float2>>(PRI_CHUNKSIZE * nSamples);
    windowGpu = std::make_unique<GpuArray<float>>(PRI_CHUNKSIZE * nSamples);
    positionGpu = std::make_unique<GpuArray<float3>>(PRI_CHUNKSIZE * nSamples);
    velocityGpu = std::make_unique<GpuArray<float3>>(PRI_CHUNKSIZE * nSamples);
    attitudeGpu = std::make_unique<GpuArray<float4>>(PRI_CHUNKSIZE * nSamples);

    priTimesGpu = std::make_unique<GpuArray<float>>(nPri);
    sampleTimesGpu = std::make_unique<GpuArray<float>>(nSamples);
    log->info("... Done allocating GPU memory for raw data");

    log->info("Allocating GPU memory for focused scene ...");
    imageGpu = std::make_unique<GpuArray<float2>>(gridNumRows * gridNumCols);
    log->info("... Done allocating GPU memory for focused scene");

    log->info("Allocating GPU scratch space ...");
    size_t scratchSize = sumScratchSize(nSamples);
    log->info("Sum scratch size: {}", scratchSize);
    sumScratchGpu = std::make_unique<GpuArray<uint8_t>>(scratchSize);
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

    log->info("Allocating host memory for focused image ...");
    focusedImage =
        std::make_unique<float2[]>(static_cast<size_t>(gridNumRows) * gridNumCols);
    log->info("... Done allocating host memory for focused image.");
}

void TdcProcessor::initGpuData()
{
    log->info("Transferring timing data to the GPU ...");
    priTimesGpu->hostToDevice(priTimes);
    sampleTimesGpu->hostToDevice(sampleTimes);
    log->info("... Done transferring timing data to the GPU");

    log->info("Initializing focused image to zeros ...");
    cudaError_t err =
        cudaMemset(imageGpu->ptr(), 0,
                   static_cast<size_t>(gridNumCols * gridNumRows) * sizeof(float2));
    log->info("... Done initializing focused image");

    log->info("Initializing range window ...");
    initRangeWindow(rangeWindowGpu->ptr(), nSamples, applyRangeWindow);
    log->info("... Done initializing range window");
}

void TdcProcessor::transferNextChunk(int chunkIdx)
{
    log->info("Transferring chunk {} of {}", chunkIdx + 1, nChunks);
    stageNextChunk(chunkIdx);
    rawDataGpu->hostToDevice(reinterpret_cast<const float2 *>(rawStaging->ptr()));
    positionGpu->hostToDevice(reinterpret_cast<const float3 *>(positionStaging->ptr()));
    velocityGpu->hostToDevice(reinterpret_cast<const float3 *>(velocityStaging->ptr()));
    attitudeGpu->hostToDevice(reinterpret_cast<const float4 *>(attitudeStaging->ptr()));
}

void TdcProcessor::stageNextChunk(int chunkIdx)
{
    // Transfer the next chunk of data into the staging area in page locked
    // memory. This simplifies the memory transfers and allows for
    // experimentation with cuda streams and asynchronous memory transfers.
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

    std::memcpy(rawStaging->ptr(), rawPtr + (priIndex * nSamples * sizeof(float2)),
                stagingSizeData);
    std::memcpy(positionStaging->ptr(), posPtr + (priIndex * nSamples * sizeof(float3)),
                stagingSizePos);
    std::memcpy(velocityStaging->ptr(), velPtr + (priIndex * nSamples * sizeof(float3)),
                stagingSizeVel);
    std::memcpy(attitudeStaging->ptr(), attPtr + (priIndex * nSamples * sizeof(float4)),
                stagingSizeAtt);
}
