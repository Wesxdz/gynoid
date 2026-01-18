#pragma once

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <chrono>

/**
 * DINO Embedder - Async DINOv2 embedding computation using dinov2.cpp/ggml
 *
 * Runs inference in a background thread to avoid blocking the main loop.
 * Frames are submitted and results retrieved asynchronously.
 */

// Forward declarations
struct dino_model;
struct dino_params;
struct ggml_gallocr;
typedef struct ggml_gallocr* ggml_gallocr_t;

// Callback for embedding results
using DinoEmbedCallback = std::function<void(float cosDiff, int frameCount)>;

class DinoEmbedder {
public:
    DinoEmbedder(int maxFrames = 10);
    ~DinoEmbedder();

    // Load the GGUF model file and start worker thread
    bool loadModel(const std::string& modelPath, int numThreads = 4);

    // Stop the worker thread
    void stop();

    // Check if model is loaded
    bool isLoaded() const { return modelLoaded; }

    // Submit frame for async processing (non-blocking)
    // Returns true if queued, false if busy (frame dropped)
    bool submitFrame(const uint8_t* bgraPixels, int width, int height, int pitch);

    // Check if busy processing
    bool isBusy() const { return processing.load(); }

    // Get latest result (returns true if new result available)
    bool getResult(float& cosDiff, int& frameCount);

    // Get statistics
    uint64_t getFramesProcessed() const { return framesProcessed; }
    uint64_t getFramesDropped() const { return framesDropped; }

    // Set callback for results (called from worker thread)
    void setCallback(DinoEmbedCallback cb) { callback = cb; }

private:
    // Model state
    dino_model* model = nullptr;
    dino_params* params = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::atomic<bool> modelLoaded{false};

    // Worker thread
    std::thread worker;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> processing{false};

    // Frame queue (single slot - only latest frame kept)
    struct PendingFrame {
        std::vector<uint8_t> pixels;
        int width = 0;
        int height = 0;
        int pitch = 0;
        bool valid = false;
    };
    PendingFrame pendingFrame;
    std::mutex frameMutex;
    std::condition_variable frameCV;

    // Results
    std::atomic<float> lastCosDiff{0.0f};
    std::atomic<int> lastFrameCount{0};
    std::atomic<bool> hasNewResult{false};

    // Rolling average state
    int maxFrames;
    std::deque<std::vector<float>> embeddings;
    std::vector<float> rollingAverage;
    std::mutex stateMutex;

    // Statistics
    std::atomic<uint64_t> framesProcessed{0};
    std::atomic<uint64_t> framesDropped{0};

    // Inference size
    int inferenceWidth = 112;
    int inferenceHeight = 112;

    // Callback
    DinoEmbedCallback callback;

    // Worker thread function
    void workerLoop();

    // Internal helpers
    std::vector<float> extractEmbedding(const uint8_t* bgraPixels, int width, int height, int pitch);
    float computeCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    void updateRollingAverage(const std::vector<float>& embedding);
};
