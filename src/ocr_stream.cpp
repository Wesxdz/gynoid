#include "ocr_stream.h"
#include "debug_log.h"
#include "vision_processor.h"
#include <sstream>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

// Base64 encoding table
static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

// Encode binary data to base64 string
static std::string base64_encode(const unsigned char* data, size_t len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (len--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; i < 4; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }

    return ret;
}

// Encode SDL_Surface to PNG in memory and return as base64
static std::string encode_surface_to_base64_png(SDL_Surface* surface) {
    if (!surface) {
        LOG_ERROR(LogCategory::OCR_STREAM, "Null surface provided to encode_surface_to_base64_png");
        return "";
    }

    // Create a memory buffer using SDL_RWops
    SDL_RWops* rw = SDL_RWFromMem(nullptr, 0);
    if (!rw) {
        // If SDL_RWFromMem with nullptr fails, allocate our own buffer
        rw = SDL_AllocRW();
        if (!rw) {
            LOG_ERROR(LogCategory::OCR_STREAM, "Failed to create SDL_RWops");
            return "";
        }
    }

    // Actually, we need to use a dynamic memory approach
    // Let's use a vector to hold the PNG data
    std::vector<unsigned char> png_data;

    // Create a custom RWops that writes to our vector
    auto write_func = [](SDL_RWops* context, const void* ptr, size_t size, size_t num) -> size_t {
        auto* vec = static_cast<std::vector<unsigned char>*>(context->hidden.unknown.data1);
        const unsigned char* bytes = static_cast<const unsigned char*>(ptr);
        vec->insert(vec->end(), bytes, bytes + (size * num));
        return num;
    };

    SDL_RWops* mem_rw = SDL_AllocRW();
    if (!mem_rw) {
        LOG_ERROR(LogCategory::OCR_STREAM, "Failed to allocate SDL_RWops");
        return "";
    }

    mem_rw->write = write_func;
    mem_rw->close = [](SDL_RWops* context) -> int { SDL_FreeRW(context); return 0; };
    mem_rw->hidden.unknown.data1 = &png_data;

    // Save PNG to memory
    if (IMG_SavePNG_RW(surface, mem_rw, 0) != 0) {
        LOG_ERROR(LogCategory::OCR_STREAM, "Failed to encode surface to PNG: {}", SDL_GetError());
        SDL_FreeRW(mem_rw);
        return "";
    }

    SDL_FreeRW(mem_rw);

    // Encode to base64
    std::string base64_png = base64_encode(png_data.data(), png_data.size());

    LOG_DEBUG(LogCategory::OCR_STREAM, "Encoded surface to PNG ({}bytes) -> base64 ({}bytes)",
              png_data.size(), base64_png.size());

    return base64_png;
}

// Simple JSON parser for OCR response
// Parses: {"status":"success","word_count":N,"words":[{"text":"...","bbox":[x,y,x,y]},...]}
bool parse_ocr_response(const std::string& json, OCRResult& result) {
    // Log first 500 chars of response
    std::string json_preview = json.length() > 500 ? json.substr(0, 500) + "..." : json;
    LOG_DEBUG(LogCategory::OCR_STREAM, "Received JSON response: {}", json_preview);
    // Helper to extract string value
    auto extract_string = [&json](const std::string& key, size_t start_pos = 0) -> std::string {
        size_t pos = json.find("\"" + key + "\"", start_pos);
        if (pos == std::string::npos) return "";
        pos = json.find(":", pos);
        if (pos == std::string::npos) return "";
        pos = json.find("\"", pos);
        if (pos == std::string::npos) return "";
        size_t end = json.find("\"", pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    };

    // Helper to extract integer value
    auto extract_int = [&json](const std::string& key, size_t start_pos = 0) -> int {
        size_t pos = json.find("\"" + key + "\"", start_pos);
        if (pos == std::string::npos) return 0;
        pos = json.find(":", pos);
        if (pos == std::string::npos) return 0;
        pos = json.find_first_of("0123456789-", pos);
        if (pos == std::string::npos) return 0;
        return std::stoi(json.substr(pos));
    };

    // Check status
    std::string status = extract_string("status");
    LOG_DEBUG(LogCategory::OCR_STREAM, "Parsing OCR response, status: {}", status);

    if (status != "success") {
        std::string message = extract_string("message");
        LOG_ERROR(LogCategory::OCR_STREAM, "OCR request failed: {}", message);
        result.valid = false;
        return false;
    }

    result.word_count = extract_int("word_count");
    result.valid = true;

    LOG_DEBUG(LogCategory::OCR_STREAM, "OCR response parsed successfully, word_count: {}", result.word_count);

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    result.timestamp = ss.str();

    // Parse words array
    size_t words_pos = json.find("\"words\"");
    if (words_pos == std::string::npos) {
        LOG_WARN(LogCategory::OCR_STREAM, "No 'words' key found in JSON");
        result.words.clear();
        return true;  // No words found, but request was successful
    }

    size_t array_start = json.find("[", words_pos);
    if (array_start == std::string::npos) {
        LOG_WARN(LogCategory::OCR_STREAM, "No '[' found after 'words' key");
        return true;
    }

    // Find matching closing bracket by counting depth
    size_t array_end = std::string::npos;
    int bracket_depth = 0;
    for (size_t i = array_start; i < json.length(); i++) {
        if (json[i] == '[') bracket_depth++;
        else if (json[i] == ']') {
            bracket_depth--;
            if (bracket_depth == 0) {
                array_end = i;
                break;
            }
        }
    }

    if (array_end == std::string::npos) {
        LOG_ERROR(LogCategory::OCR_STREAM, "No matching ']' found for words array");
        return false;
    }

    LOG_DEBUG(LogCategory::OCR_STREAM, "Words array: start={}, end={}, length={}",
              array_start, array_end, array_end - array_start);

    // Log a snippet of the words array
    if (array_end - array_start > 0 && array_end - array_start < 500) {
        std::string words_snippet = json.substr(array_start, std::min(size_t(200), array_end - array_start + 1));
        LOG_DEBUG(LogCategory::OCR_STREAM, "Words array content: {}", words_snippet);
    }

    // Parse each word object in the array
    size_t pos = array_start + 1;
    int loop_count = 0;
    while (pos < array_end) {
        loop_count++;
        if (loop_count > 1000) {
            LOG_ERROR(LogCategory::OCR_STREAM, "Infinite loop detected in word parsing!");
            break;
        }
        size_t obj_start = json.find("{", pos);
        if (obj_start == std::string::npos || obj_start >= array_end) break;

        size_t obj_end = json.find("}", obj_start);
        if (obj_end == std::string::npos || obj_end > array_end) break;

        std::string word_obj = json.substr(obj_start, obj_end - obj_start + 1);

        OCRWord word;

        // Extract text
        size_t text_pos = word_obj.find("\"text\"");
        if (text_pos != std::string::npos) {
            size_t text_start = word_obj.find("\"", text_pos + 6);
            if (text_start != std::string::npos) {
                size_t text_end = word_obj.find("\"", text_start + 1);
                if (text_end != std::string::npos) {
                    word.text = word_obj.substr(text_start + 1, text_end - text_start - 1);
                }
            }
        }

        // Extract bbox array [xmin, ymin, xmax, ymax]
        size_t bbox_pos = word_obj.find("\"bbox\"");
        if (bbox_pos != std::string::npos) {
            size_t bbox_start = word_obj.find("[", bbox_pos);
            size_t bbox_end = word_obj.find("]", bbox_start);
            if (bbox_start != std::string::npos && bbox_end != std::string::npos) {
                std::string bbox_str = word_obj.substr(bbox_start + 1, bbox_end - bbox_start - 1);

                // Parse comma-separated integers
                std::istringstream bbox_stream(bbox_str);
                std::string val;
                int idx = 0;
                while (std::getline(bbox_stream, val, ',') && idx < 4) {
                    // Trim whitespace
                    val.erase(0, val.find_first_not_of(" \t"));
                    val.erase(val.find_last_not_of(" \t") + 1);

                    int num = std::stoi(val);
                    switch (idx) {
                        case 0: word.xmin = num; break;
                        case 1: word.ymin = num; break;
                        case 2: word.xmax = num; break;
                        case 3: word.ymax = num; break;
                    }
                    idx++;
                }
            }
        }

        LOG_DEBUG(LogCategory::OCR_STREAM, "Extracted word: text='{}' (len={}), bbox=[{},{},{},{}]",
                 word.text, word.text.length(), word.xmin, word.ymin, word.xmax, word.ymax);

        if (!word.text.empty()) {
            result.words.push_back(word);
            LOG_DEBUG(LogCategory::OCR_STREAM, "Added word #{}: '{}' bbox=[{},{},{},{}]",
                     result.words.size(), word.text, word.xmin, word.ymin, word.xmax, word.ymax);
        } else {
            LOG_WARN(LogCategory::OCR_STREAM, "Skipping word with empty text! word_obj: '{}'", word_obj);
        }

        pos = obj_end + 1;
    }

    LOG_DEBUG(LogCategory::OCR_STREAM, "Finished parsing, total words: {}", result.words.size());
    return true;
}

// Request OCR using OCRRequest object (handles both surface and file path modes)
bool request_ocr_for_quadrant(OCRClient& client, const OCRRequest& req, OCRResult& result) {
    if (!client.connected) {
        LOG_WARN(LogCategory::OCR_STREAM, "Cannot request OCR for quadrant {}: client not connected", req.quadrant);
        return false;
    }

    // Build JSON request
    std::ostringstream request;
    request << "{\"command\":\"ocr\",";

    // Use surface data if provided, otherwise fall back to file path
    if (req.surface) {
        LOG_DEBUG(LogCategory::OCR_STREAM, "Requesting OCR for quadrant {} using direct surface data ({}x{})",
                 req.quadrant, req.surface->w, req.surface->h);

        // Encode surface to base64 PNG
        std::string base64_png = encode_surface_to_base64_png(req.surface);
        if (base64_png.empty()) {
            LOG_ERROR(LogCategory::OCR_STREAM, "Failed to encode surface to base64 PNG");
            return false;
        }

        request << "\"image_data\":\"" << base64_png << "\",\"image_format\":\"png\"";
    } else if (!req.image_path.empty()) {
        LOG_DEBUG(LogCategory::OCR_STREAM, "Requesting OCR for quadrant {} using file path: {}",
                 req.quadrant, req.image_path);
        request << "\"image_path\":\"" << req.image_path << "\"";
    } else {
        LOG_ERROR(LogCategory::OCR_STREAM, "OCR request has neither surface nor image_path!");
        return false;
    }

    request << ",\"return_image\":false"
            << ",\"container_id\":" << (req.quadrant + 1);  // Convert 0-indexed quadrant to 1-indexed container

    request << "}";

    std::string response;
    if (!client.send_request_with_new_connection(request.str(), response)) {
        LOG_ERROR(LogCategory::OCR_STREAM, "Failed to send OCR request for quadrant {}", req.quadrant);
        return false;
    }

    LOG_TRACE(LogCategory::OCR_STREAM, "Received response for quadrant {}: {}", req.quadrant, response);

    // Parse response
    if (!parse_ocr_response(response, result)) {
        LOG_ERROR(LogCategory::OCR_STREAM, "Failed to parse OCR response for quadrant {}", req.quadrant);
        return false;
    }

    result.quadrant = req.quadrant;
    LOG_INFO(LogCategory::OCR_STREAM, "OCR request successful for quadrant {}: {} words found", req.quadrant, result.word_count);
    return true;
}

// Worker thread methods for async OCR processing
void OCRClient::start_worker() {
    if (running.load()) {
        LOG_WARN(LogCategory::OCR_STREAM, "Worker threads already running");
        return;
    }

    running.store(true);
    for (int i = 0; i < num_workers; i++) {
        worker_threads.emplace_back(&OCRClient::worker_loop, this);
    }
    LOG_INFO(LogCategory::OCR_STREAM, "Started {} OCR worker threads", num_workers);
}

void OCRClient::stop_worker() {
    if (!running.load()) {
        return;
    }

    running.store(false);
    queue_cv.notify_all();

    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads.clear();

    LOG_INFO(LogCategory::OCR_STREAM, "Worker threads stopped");
}

void OCRClient::queue_request(const OCRRequest& request) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);

        // Drop old requests for the same quadrant to keep only the latest
        std::queue<OCRRequest> filtered_queue;
        int dropped = 0;
        while (!request_queue.empty()) {
            OCRRequest& existing = request_queue.front();
            if (existing.quadrant != request.quadrant) {
                filtered_queue.push(std::move(existing));
            } else {
                dropped++;
            }
            request_queue.pop();
        }
        request_queue = std::move(filtered_queue);

        request_queue.push(request);

        if (dropped > 0) {
            LOG_DEBUG(LogCategory::OCR_STREAM, "Dropped {} old OCR requests for quadrant {}", dropped, request.quadrant);
        }
    }
    queue_cv.notify_one();
    LOG_TRACE(LogCategory::OCR_STREAM, "Queued OCR request for quadrant {} (queue size: ~{})", request.quadrant, request_queue.size());
}

void OCRClient::worker_loop() {
    LOG_INFO(LogCategory::OCR_STREAM, "Worker loop started");

    while (running.load()) {
        OCRRequest request;

        // Wait for a request
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this] { return !request_queue.empty() || !running.load(); });

            if (!running.load() && request_queue.empty()) {
                break;
            }

            if (request_queue.empty()) {
                continue;
            }

            request = request_queue.front();
            request_queue.pop();
        }

        // Process the request (blocking recv is OK here in worker thread)
        OCRResult result;
        // bool is_full_screen = request.region.isFullScreen;
        bool is_full_screen = true;

        if (request_ocr_for_quadrant(*this, request, result)) {
            // Update the quadrant entity with results
            if (request.vnc_entity.is_alive()) {
                if (request.vnc_entity.has<VNC_OCR_Data>()) {
                    auto& quad_data = request.vnc_entity.ensure<VNC_OCR_Data>();

                    quad_data.update(result, true);
                } else {
                    LOG_ERROR(LogCategory::OCR_STREAM, "VNC entity has no VNC_OCR_Data component!");
                }
            } else {
                LOG_ERROR(LogCategory::OCR_STREAM, "VNC entity is not alive!");
            }
        }
    }

    LOG_INFO(LogCategory::OCR_STREAM, "Worker loop exited");
}

// Flecs module initialization
void OCRStreamModule(flecs::world& ecs) {
    // Register components
    ecs.component<VNC_OCR_Data>();
    ecs.component<OCRStreamConfig>();
    ecs.component<OCRClient>();

    // Create global config entity
    auto config_entity = ecs.entity("OCRStreamConfig")
        .add<OCRStreamConfig>();

    // Create OCR client entity with initialized component
    auto client_entity = ecs.entity("OCRClient");
    auto& client = client_entity.ensure<OCRClient>();

    // Log initialization
    LOG_INFO(LogCategory::OCR_STREAM, "OCRClient entity created: host={}, port={}, connected={}",
             client.host, client.port, client.connected);

    // Create 4 quadrant entities for OCR data storage
    for (int i = 0; i < 4; i++) {
        std::string name = "OCRQuadrant" + std::to_string(i);
        auto quad_entity = ecs.entity(name.c_str());
        quad_entity.ensure<VNC_OCR_Data>();
        LOG_DEBUG(LogCategory::OCR_STREAM, "Created quadrant entity: {}", name);
    }

    LOG_INFO(LogCategory::OCR_STREAM, "Module initialized (4 quadrants created)");

    // System: Auto-connect to DocTR server on startup
    auto ocr_connect = ecs.system<OCRClient>("OCRAutoConnect")
        .kind(flecs::OnUpdate)
        .interval(0.5f)  // Try to connect every 0.5 seconds if disconnected
        .each([](flecs::entity e, OCRClient& client) {
            static bool first_run = true;
            if (first_run) {
                LOG_INFO(LogCategory::OCR_STREAM, "OCRAutoConnect system started for entity: {}", e.name().c_str());
                LOG_INFO(LogCategory::OCR_STREAM, "Initial state: host={}, port={}, connected={}",
                         client.host, client.port, client.connected);
                first_run = false;
            }

            if (!client.connected && !client.running.load()) {
                LOG_INFO(LogCategory::OCR_STREAM, "Attempting to connect to DocTR server at {}:{}...",
                         client.host, client.port);
                bool success = client.connect();
                if (!success) {
                    LOG_WARN(LogCategory::OCR_STREAM, "Connection attempt failed");
                } else {
                    LOG_INFO(LogCategory::OCR_STREAM, "Successfully connected!");
                    // Start worker thread for async OCR processing
                    client.start_worker();
                }
            }
        });
    ocr_connect.run();

    // System: Periodic OCR request for all quadrants
    ecs.system<OCRStreamConfig>("OCRPeriodicRequest")
        .kind(flecs::OnUpdate)
        .each([&ecs](flecs::entity e, OCRStreamConfig& config) {
            if (!config.enabled || !config.auto_request) {
                return;
            }

            static double last_request_time = -999.0;  // Start with large negative to trigger immediately
            double current_time = glfwGetTime();

            if (current_time - last_request_time < config.request_interval) {
                return;
            }

            last_request_time = current_time;

            // Get OCR client
            auto client_entity = ecs.lookup("OCRClient");
            if (!client_entity.is_alive()) return;

            auto& client = client_entity.ensure<OCRClient>();
            if (!client.connected) {
                static int warn_count = 0;
                if (warn_count++ % 20 == 0) {  // Warn every 20 attempts
                    LOG_WARN(LogCategory::OCR_STREAM, "Client not connected, skipping OCR request");
                }
                return;
            }

            // Check VNC view state to determine which quadrants to process
            const VNCViewState* viewState = ecs.try_get<VNCViewState>();
            bool fullscreenMode = viewState && !viewState->quadrantView;
            int startQuadrant = fullscreenMode ? viewState->activeQuadrant : 0;
            int endQuadrant = fullscreenMode ? viewState->activeQuadrant + 1 : 4;

            if (fullscreenMode) {
                LOG_DEBUG(LogCategory::OCR_STREAM, "Fullscreen mode - queueing OCR request for quadrant {}", viewState->activeQuadrant);
            } else {
                LOG_DEBUG(LogCategory::OCR_STREAM, "Quadrant mode - queueing OCR requests for all quadrants");
            }

            // NOTE: This periodic OCR system is now largely replaced by the tile cluster system
            // in main.cpp which triggers OCR immediately when changes are detected and uses
            // direct VNC surface data. This system is kept as a fallback but disabled by default.
            //
            // To use this fallback system, you would need to either:
            // 1. Still save PNG files periodically for it to read, OR
            // 2. Pass surface data through a different mechanism
            //
            // For now, just log that we're skipping since the tile cluster system handles it.
            static int skip_log_counter = 0;
            if (skip_log_counter++ == 0) {
                LOG_INFO(LogCategory::OCR_STREAM,
                    "Periodic OCR system is disabled - OCR is triggered by tile cluster system with direct VNC textures");
            }

            LOG_DEBUG(LogCategory::OCR_STREAM, "Completed queueing OCR requests");
        });

    LOG_INFO(LogCategory::OCR_STREAM, "Systems registered (auto-connect, periodic-request)");
}
