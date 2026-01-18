#include "dino_embedder.h"

#include "dinov2.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>

DinoEmbedder::DinoEmbedder(int maxFrames)
    : maxFrames(maxFrames) {
    model = new dino_model();
    params = new dino_params();
}

DinoEmbedder::~DinoEmbedder() {
    stop();

    if (modelLoaded) {
        if (model->ctx) ggml_free(model->ctx);
        if (allocator) ggml_gallocr_free(allocator);
        if (model->buffer) ggml_backend_buffer_free(model->buffer);
        if (model->backend) ggml_backend_free(model->backend);
    }
    delete model;
    delete params;
}

void DinoEmbedder::stop() {
    if (worker.joinable()) {
        shouldStop = true;
        frameCV.notify_all();
        worker.join();
    }
}

bool DinoEmbedder::loadModel(const std::string& modelPath, int numThreads) {
    if (modelLoaded) {
        std::cerr << "[DINO] Model already loaded" << std::endl;
        return true;
    }

    // Check if model file exists
    std::ifstream f(modelPath);
    if (!f.good()) {
        std::cerr << "[DINO] Model file not found: " << modelPath << std::endl;
        return false;
    }
    f.close();

    std::cout << "[DINO] Initializing ggml..." << std::endl;
    std::cout.flush();
    ggml_time_init();

    params->model = modelPath;
    params->n_threads = numThreads;
    params->classify = false;

    // Use small inference size for speed (position embedding will be interpolated)
    inferenceWidth = 112;
    inferenceHeight = 112;
    cv::Size inferenceSize(inferenceWidth, inferenceHeight);

    std::cout << "[DINO] Loading model from " << modelPath << std::endl;
    std::cout << "[DINO] Inference size: " << inferenceWidth << "x" << inferenceHeight << std::endl;
    std::cout.flush();

    if (!dino_model_load(inferenceSize, params->model, *model, *params)) {
        std::cerr << "[DINO] Failed to load model from " << modelPath << std::endl;
        return false;
    }

    std::cout << "[DINO] Model file loaded, creating allocator..." << std::endl;
    std::cout.flush();

    allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));

    // Reserve the allocator with a test graph to pre-allocate memory
    // This prevents reallocation issues during inference
    {
        // Calculate the actual inference size (padded to patch_size multiples, like dino_preprocess does)
        const int patchSize = model->hparams.patch_size;
        const int paddedWidth = (inferenceWidth / patchSize + 1) * patchSize;
        const int paddedHeight = (inferenceHeight / patchSize + 1) * patchSize;
        cv::Size actualSize(paddedWidth, paddedHeight);

        struct ggml_init_params test_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context* ctx_test = ggml_init(test_params);
        struct ggml_cgraph* test_graph = build_graph(actualSize, ctx_test, *model, *params);

        if (!ggml_gallocr_reserve(allocator, test_graph)) {
            std::cerr << "[DINO] Failed to reserve allocator" << std::endl;
            ggml_free(ctx_test);
            return false;
        }
        std::cout << "[DINO] Allocator reserved for " << paddedWidth << "x" << paddedHeight << " input" << std::endl;
        ggml_free(ctx_test);
    }

    modelLoaded = true;
    std::cout << "[DINO] Model loaded successfully (async mode)" << std::endl;
    std::cout << "[DINO] Hidden size: " << model->hparams.hidden_size << std::endl;
    std::cout << "[DINO] Patch size: " << model->hparams.patch_size << std::endl;
    std::cout.flush();

    // Start worker thread
    shouldStop = false;
    worker = std::thread(&DinoEmbedder::workerLoop, this);

    return true;
}

bool DinoEmbedder::submitFrame(const uint8_t* bgraPixels, int width, int height, int pitch) {
    if (!modelLoaded) return false;

    // If busy, drop this frame
    if (processing.load()) {
        framesDropped++;
        return false;
    }

    // Copy frame to pending slot
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        size_t dataSize = (size_t)height * pitch;
        pendingFrame.pixels.resize(dataSize);
        memcpy(pendingFrame.pixels.data(), bgraPixels, dataSize);
        pendingFrame.width = width;
        pendingFrame.height = height;
        pendingFrame.pitch = pitch;
        pendingFrame.valid = true;
    }

    frameCV.notify_one();
    return true;
}

bool DinoEmbedder::getResult(float& cosDiff, int& frameCount) {
    if (!hasNewResult.load()) {
        return false;
    }

    cosDiff = lastCosDiff.load();
    frameCount = lastFrameCount.load();
    hasNewResult.store(false);
    return true;
}

void DinoEmbedder::workerLoop() {
    std::cout << "[DINO] Worker thread started" << std::endl;

    while (!shouldStop) {
        PendingFrame frame;

        // Wait for a frame
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCV.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                return pendingFrame.valid || shouldStop;
            });

            if (shouldStop) break;
            if (!pendingFrame.valid) continue;

            frame = std::move(pendingFrame);
            pendingFrame.valid = false;
        }

        processing.store(true);

        // Extract embedding
        std::vector<float> embedding = extractEmbedding(
            frame.pixels.data(), frame.width, frame.height, frame.pitch);

        if (!embedding.empty()) {
            // Print the embedding
            std::cout << "[DINO] Embedding (" << embedding.size() << " dims): [";
            size_t printCount = std::min(embedding.size(), (size_t)10);
            for (size_t i = 0; i < printCount; i++) {
                if (i > 0) std::cout << ", ";
                std::cout << embedding[i];
            }
            if (embedding.size() > 10) {
                std::cout << ", ... (+" << (embedding.size() - 10) << " more)";
            }
            std::cout << "]" << std::endl;

            // Compute cosine diff from rolling average
            float cosDiff = 0.0f;
            int frameCount = 0;
            {
                std::lock_guard<std::mutex> lock(stateMutex);
                if (!rollingAverage.empty()) {
                    float cosSim = computeCosineSimilarity(embedding, rollingAverage);
                    cosDiff = 1.0f - cosSim;
                }
                frameCount = (int)embeddings.size();
            }

            // Update rolling average
            updateRollingAverage(embedding);
            framesProcessed++;

            // Store result
            lastCosDiff.store(cosDiff);
            lastFrameCount.store(frameCount + 1);
            hasNewResult.store(true);

            // Callback if set
            if (callback) {
                callback(cosDiff, frameCount + 1);
            }

            std::cout << "[DINO] cos_diff=" << cosDiff << ", frames=" << (frameCount + 1)
                      << " (processed=" << framesProcessed << ", dropped=" << framesDropped << ")" << std::endl;
        }

        processing.store(false);
    }

    std::cout << "[DINO] Worker thread stopped" << std::endl;
}

std::vector<float> DinoEmbedder::extractEmbedding(const uint8_t* bgraPixels, int width, int height, int pitch) {
    std::vector<float> result;

    if (!modelLoaded) {
        return result;
    }

    // Convert BGRA to BGR cv::Mat
    cv::Mat bgraMat(height, width, CV_8UC4);
    for (int y = 0; y < height; y++) {
        memcpy(bgraMat.ptr(y), bgraPixels + y * pitch, width * 4);
    }

    cv::Mat bgrMat;
    cv::cvtColor(bgraMat, bgrMat, cv::COLOR_BGRA2BGR);

    // Resize to inference size
    cv::Mat resized;
    cv::resize(bgrMat, resized, cv::Size(inferenceWidth, inferenceHeight), 0, 0, cv::INTER_LINEAR);

    // Preprocess for DINOv2
    cv::Mat input = dino_preprocess(resized, resized.size(), model->hparams);

    // Run inference
    ggml_backend_synchronize(model->backend);
    std::unique_ptr<dino_output> output = dino_predict(*model, input, *params, allocator);
    ggml_backend_synchronize(model->backend);

    if (!output || !output->patch_tokens.has_value()) {
        std::cerr << "[DINO] Inference failed" << std::endl;
        return result;
    }

    // Extract mean embedding from patch tokens
    const cv::Mat& patchTokens = output->patch_tokens.value();
    cv::Mat meanEmbed;
    cv::reduce(patchTokens, meanEmbed, 0, cv::REDUCE_AVG, CV_32F);

    result.resize(meanEmbed.cols);
    for (int i = 0; i < meanEmbed.cols; i++) {
        result[i] = meanEmbed.at<float>(0, i);
    }

    return result;
}

float DinoEmbedder::computeCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    normA = std::sqrt(normA);
    normB = std::sqrt(normB);

    if (normA < 1e-8f || normB < 1e-8f) return 0.0f;
    return dot / (normA * normB);
}

void DinoEmbedder::updateRollingAverage(const std::vector<float>& embedding) {
    std::lock_guard<std::mutex> lock(stateMutex);

    embeddings.push_back(embedding);
    while ((int)embeddings.size() > maxFrames) {
        embeddings.pop_front();
    }

    if (embeddings.empty()) {
        rollingAverage.clear();
        return;
    }

    size_t dim = embeddings[0].size();
    rollingAverage.assign(dim, 0.0f);

    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < dim; i++) {
            rollingAverage[i] += emb[i];
        }
    }

    float scale = 1.0f / embeddings.size();
    for (size_t i = 0; i < dim; i++) {
        rollingAverage[i] *= scale;
    }
}
