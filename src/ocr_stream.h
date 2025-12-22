#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <flecs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <GLFW/glfw3.h>
#include "debug_log.h"

// OCR word with bounding box (same as doctr_client.h but self-contained)
struct OCRWord {
    std::string text;
    int xmin, ymin, xmax, ymax;
    float confidence;  // Optional: for future use
    double creationTime;  // Timestamp when word was first detected (from glfwGetTime())

    OCRWord() : xmin(0), ymin(0), xmax(0), ymax(0), confidence(1.0f), creationTime(0.0) {}
};

struct OCRResult {
    std::string timestamp;
    std::vector<OCRWord> words;
    int word_count;
    bool valid;

    OCRResult() : word_count(0), valid(false) {}
};

struct VNC_OCR_Data {
    OCRResult data;
    mutable std::mutex mutex;
    bool has_data = false;
    bool initial_ocr_done = false;
    double last_update_time = 0.0;

    void update(const OCRResult& new_data, bool is_full_screen = false) {
        std::lock_guard<std::mutex> lock(mutex);
        data = new_data;
        has_data = true;
        double currentTime = glfwGetTime();
        last_update_time = currentTime;

        // Set creation time for all words
        for (auto& word : data.words) {
            word.creationTime = currentTime;
        }

        LOG_DEBUG(LogCategory::OCR_STREAM, "VNC_OCR_Data::update() - words: {}, valid: {}, full_screen: {}",
                  new_data.words.size(), new_data.valid, is_full_screen);
    }

    OCRResult get() const {
        std::lock_guard<std::mutex> lock(mutex);
        LOG_DEBUG(LogCategory::OCR_STREAM, "VNC_OCR_Data::get() - quadrant: {}, words: {}, valid: {}",
                  data.quadrant, data.words.size(), data.valid);
        return data;
    }

    bool is_stale(double max_age_seconds = 5.0) const {
        std::lock_guard<std::mutex> lock(mutex);
        return (glfwGetTime() - last_update_time) > max_age_seconds;
    }
};

// OCR streaming configuration
struct OCRStreamConfig {
    bool enabled = true;
    bool auto_request = true;       // Automatically request OCR periodically
    float request_interval = 0.5f;  // Seconds between OCR requests (reduced for faster response)
    bool render_bounds = true;      // Render bounding boxes
    bool render_text = true;        // Render detected text
    uint32_t color_palette[8] = {   // Colors for rendering (RGBA)
        0x3498dbff,  // Blue
        0xe74c3cff,  // Red
        0x2ecc71ff,  // Green
        0xf39c12ff,  // Orange
        0x9b59b6ff,  // Purple
        0x1abc9cff,  // Turquoise
        0xe67e22ff,  // Carrot
        0xecf0f1ff   // Silver
    };
};

// Forward declarations
struct SDL_Surface;
struct VNCClient;  // For accessing VNC client data

// OCR request for async processing
struct OCRRequest {
    std::string image_path;
    SDL_Surface* surface;
    flecs::entity vnc_entity;

    OCRRequest() : surface(nullptr) {}

    // Legacy constructor with file path (deprecated)
    OCRRequest(int q, const std::string& path, flecs::entity entity)
        : image_path(path), surface(nullptr), vnc_entity(entity) {}

    // New constructor with surface (preferred)
    OCRRequest(int q, SDL_Surface* surf, flecs::entity entity)
        : surface(surf), vnc_entity(entity) {}
};

// TCP client for DocTR server
struct OCRClient {
    std::string host;
    int port;
    bool connected;
    std::atomic<bool> running;
    std::vector<std::thread> worker_threads;  // Multiple worker threads
    int num_workers = 2;  // Number of concurrent OCR workers

    // Async request queue
    std::queue<OCRRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    OCRClient() : host("127.0.0.1"), port(9996),
                  connected(false), running(false) {}

    ~OCRClient() {
        stop_worker();
        connected = false;
    }

    void start_worker();
    void stop_worker();
    void queue_request(const OCRRequest& request);
    void worker_loop();

    // Test connection (used by auto-connect system)
    bool connect() {
        int test_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (test_fd < 0) {
            return false;
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
            close(test_fd);
            return false;
        }

        if (::connect(test_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(test_fd);
            return false;
        }

        close(test_fd);
        connected = true;
        LOG_INFO(LogCategory::OCR_STREAM, "DocTR server available at {}:{}", host, port);
        return true;
    }

    // Each worker creates its own connection
    bool send_request_with_new_connection(const std::string& json_request, std::string& response) {
        // Create fresh connection for this request
        int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_fd < 0) {
            return false;
        }

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
            close(sock_fd);
            return false;
        }

        if (::connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sock_fd);
            return false;
        }

        // Send request with newline delimiter
        std::string request_with_newline = json_request + "\n";
        ssize_t sent = send(sock_fd, request_with_newline.c_str(),
                           request_with_newline.length(), 0);

        if (sent < 0) {
            close(sock_fd);
            return false;
        }

        // Receive response (newline-delimited)
        char buffer[65536];  // Large buffer for OCR results
        ssize_t received = recv(sock_fd, buffer, sizeof(buffer) - 1, 0);

        close(sock_fd);

        if (received <= 0) {
            return false;
        }

        buffer[received] = '\0';
        response = std::string(buffer);

        // Remove trailing newline if present
        if (!response.empty() && response.back() == '\n') {
            response.pop_back();
        }

        return true;
    }
};

bool parse_ocr_response(const std::string& json, OCRResult& result);

bool request_ocr_for_quadrant(OCRClient& client, const OCRRequest& req, OCRResult& result);

void OCRStreamModule(flecs::world& ecs);
