#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <stack>
#include <algorithm>
#include <variant>
#include <string>

#define GLAD_GL_IMPLEMENTATION
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <flecs.h>

#include <nanovg.h>
#define NANOVG_GL3_IMPLEMENTATION
#include <nanovg_gl.h>

#include <float.h>
#include <unordered_map>
#include <map>
#include <sstream>
#include <raymath.h>

#include <ctime>
#include <chrono>
#include <sys/socket.h>
#include <sys/un.h>

#include <filesystem>
#include <system_error>
namespace fs = std::filesystem;

// LibVNC
#include <rfb/rfbclient.h>

// SDL for texture creation from VNC framebuffer
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

// fpng for fast PNG encoding
#include <fpng.h>

// For multithreaded screenshot saving
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

#include "mel_spec_render.h"
#include "vision_processor.h"
#include "debug_log.h"
#include "x11_outline.h"

#include "spatial_index.h"
#include "query_server.h"
#include "dino_embedder.h"
#include "frame_diff.h"

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <stack>

#include <libssh2.h>
#include <nlohmann/json.hpp>

#include <enet/enet.h>

using json = nlohmann::json;

#include "components.h"
#include "thorn_vfx.h"
#include "panel3d.h"

#include "tradewinds.h"

using namespace tradewinds;

std::vector<std::string> splitLinesPreserve(std::string s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t nl = s.find('\n', start);
        if (nl == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, nl - start)); // can be empty
        start = nl + 1;
        if (start == s.size()) { // trailing newline -> final empty line
            out.push_back(std::string{});
            break;
        }
    }
    return out;
}

TextSize measureText(NVGcontext* vg,
                            std::string text,
                            float wrapWidth)
{
    float asc = 0, desc = 0, lineh = 0;
    nvgTextMetrics(vg, &asc, &desc, &lineh);

    float w = 0.0f;
    float h = 0.0f;

    auto lines = splitLinesPreserve(text);

    if (wrapWidth > 0.0f) {
        NVGtextRow rows[16];

        for (auto line : lines) {
            // Even an empty explicit line consumes vertical space.
            if (line.empty()) { h += lineh; continue; }

            const char* start = line.data();
            const char* end   = start + line.size();

            while (start < end) {
                int n = nvgTextBreakLines(vg, start, end, wrapWidth, rows, 16);
                if (n <= 0) break;

                for (int i = 0; i < n; ++i) {
                    w = std::max(w, rows[i].width);
                    h += lineh;
                }
                start = rows[n - 1].next;
            }
        }

        // Optional: if you want “reported width” to be the box width:
        // w = std::min(w, wrapWidth);

    } else {
        float bounds[4];

        for (auto line : lines) {
            h += lineh;                  // explicit line always counts
            if (line.empty()) continue;  // blank line => width 0

            nvgTextBounds(vg, 0, 0, line.data(), line.data() + line.size(), bounds);
            w = std::max(w, bounds[2] - bounds[0]);
        }
    }

    return { w, h };
}

static const char* VNC_SURFACE_TAG = "vnc_surface";
static const char* VNC_CLIENT_TAG = "vnc_client";
static struct timeval g_last_screenshot_time;

// Vision processing job queue
struct VisionProcessingJob {
    SDL_Surface* surface;
    int quadrant; // TODO: Remove this since it's an artifact of Shephleetess having 4 VNC Streams...
    std::string paletteFile;
    std::string outputPath;
    int width;
    int height;
    int pitch;
    std::vector<uint8_t> pixelData;  // Copy of pixel data to avoid race conditions
};

class VisionJobQueue {
private:
    std::deque<VisionProcessingJob> jobs;  // Changed from queue to deque for frame dropping
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> isStopped{false};  // Track if stop() has been called
    std::vector<std::thread> workers;
    std::atomic<size_t> droppedFrames{0};

public:
    void start(int numThreads = 2) {
        shouldStop = false;
        isStopped = false;
        for (int i = 0; i < numThreads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    VisionProcessingJob job;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [this]() { return !jobs.empty() || shouldStop; });

                        if (shouldStop && jobs.empty()) {
                            return;
                        }

                        if (jobs.empty()) continue;

                        job = std::move(jobs.front());
                        jobs.pop_front();
                    }

                    // Process the job outside the lock
                    std::cout << "[VISION WORKER] Processing quadrant " << job.quadrant << std::endl;
                    process_vnc_vision(job.pixelData.data(), job.width, job.height,
                                      job.pitch, job.quadrant,
                                      job.paletteFile.c_str(), job.outputPath.c_str());
                }
            });
        }
        std::cout << "[VISION QUEUE] Started " << numThreads << " worker threads" << std::endl;
    }

    void submit(const VisionProcessingJob& job) {
        size_t dropped = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);

            // Remove any existing jobs for this quadrant to keep only the latest frame
            auto it = jobs.begin();
            while (it != jobs.end()) {
                if (it->quadrant == job.quadrant) {
                    it = jobs.erase(it);
                    dropped++;
                } else {
                    ++it;
                }
            }

            jobs.push_back(job);
        }

        if (dropped > 0) {
            droppedFrames += dropped;
            std::cout << "[VISION QUEUE] Dropped " << dropped << " old frame(s) for quadrant "
                      << job.quadrant << " (total dropped: " << droppedFrames << ")" << std::endl;
        }

        cv.notify_one();
    }

    void stop() {
        // Prevent double-stop (e.g., explicit call + destructor call)
        bool expected = false;
        if (!isStopped.compare_exchange_strong(expected, true)) {
            // Already stopped, nothing to do
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            shouldStop = true;
        }
        cv.notify_all();

        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();
        std::cout << "[VISION QUEUE] All worker threads stopped" << std::endl;
    }

    size_t pendingJobs() {
        std::lock_guard<std::mutex> lock(mutex);
        return jobs.size();
    }

    ~VisionJobQueue() {
        stop();
    }
};

// Global vision processing job queue
static VisionJobQueue g_visionQueue;

// Global DINO embedder (runs async in background thread)
static DinoEmbedder g_dinoEmbedder(10);  // Rolling average of 10 frames

// PNG save job for background thread
struct PNGSaveJob {
    std::vector<uint8_t> pixelData;
    int width;
    int height;
    std::string filename;
};

class PNGSaveQueue {
private:
    std::deque<PNGSaveJob> jobs;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> isStopped{false};
    std::thread worker;

public:
    void start() {
        shouldStop = false;
        isStopped = false;
        worker = std::thread([this]() {
            while (true) {
                PNGSaveJob job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [this]() { return !jobs.empty() || shouldStop; });

                    if (shouldStop && jobs.empty()) {
                        return;
                    }

                    if (jobs.empty()) continue;

                    job = std::move(jobs.front());
                    jobs.pop_front();
                }

                // Save PNG outside the lock
                if (fpng::fpng_encode_image_to_file(job.filename.c_str(), job.pixelData.data(),
                                                     job.width, job.height, 4, 0)) {
                    std::cout << "[PNG SAVE] Saved " << job.filename << " (background)" << std::endl;
                } else {
                    std::cerr << "[PNG SAVE ERROR] Failed to save " << job.filename << std::endl;
                }
            }
        });
        std::cout << "[PNG SAVE QUEUE] Started background worker thread" << std::endl;
    }

    void submit(PNGSaveJob&& job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }

    void stop() {
        bool expected = false;
        if (!isStopped.compare_exchange_strong(expected, true)) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            shouldStop = true;
        }
        cv.notify_all();

        if (worker.joinable()) {
            worker.join();
        }
        std::cout << "[PNG SAVE QUEUE] Worker thread stopped" << std::endl;
    }

    ~PNGSaveQueue() {
        stop();
    }
};

// Global PNG save queue
static PNGSaveQueue g_pngSaveQueue;

#include "vnc_struct.h"


float get_time_of_day_normalized() {
    using namespace std::chrono;

    // 1. Get current time point
    auto now = system_clock::now();

    // 2. Convert to time_t to break down into local time (hours, min, sec)
    std::time_t t = system_clock::to_time_t(now);
    std::tm local_tm = *std::localtime(&t); // Note: Not thread-safe (see below)

    // 3. Calculate total seconds elapsed today (Hours + Minutes + Seconds)
    long seconds_since_midnight = 
        (local_tm.tm_hour * 3600) + 
        (local_tm.tm_min * 60) + 
        local_tm.tm_sec;

    // 4. Retrieve milliseconds for higher precision
    //    Get duration since epoch, extract seconds, and find the remainder
    auto duration = now.time_since_epoch();
    auto sec_duration = duration_cast<std::chrono::seconds>(duration);
    auto ms_duration = duration_cast<std::chrono::milliseconds>(duration) - 
                       duration_cast<std::chrono::milliseconds>(sec_duration);
    
    float ms_fraction = ms_duration.count() / 1000.0f;

    // 5. Combine and normalize
    //    Total seconds in a day = 86400 (24 * 60 * 60)
    float total_seconds_today = static_cast<float>(seconds_since_midnight) + ms_fraction;
    
    return total_seconds_today / 86400.0f;
    // spin counterclockwise
    // return local_tm.tm_sec/60.0f;f
}

QuadraticBezierRenderable get_hour_segment(size_t i, float start_angle = 0.0f)
{
    float radius = 1080/2;

    float centerX = radius+16;
    float centerY = radius+16;
    int segments = 24;
    float thickness = 16.0f;

    uint32_t dayA   = 0x5f9c00FF; // Green
    uint32_t dayB   = 0xfc9800FF; // Orange
    uint32_t nightA = 0x6868fbFF; // Light Blue
    uint32_t nightB = 0x002df4FF; // Dark Blue

    float angleStep = (2.0f * M_PI) / segments;

    // Start at PI (Left/West) and go around
    float thetaStart = start_angle + (i * angleStep) + M_PI;
    float thetaEnd   = start_angle + ((i + 1) * angleStep) + M_PI;

    // Standard Bezier Arc Math
    float x1 = centerX + radius * cos(thetaStart);
    float y1 = centerY + radius * sin(thetaStart);
    float x2 = centerX + radius * cos(thetaEnd);
    float y2 = centerY + radius * sin(thetaEnd);

    float midTheta = (thetaStart + thetaEnd) / 2.0f;
    // Calculate mid-point slightly further out for the control point
    float ctrlDist = radius / cos(angleStep / 2.0f);
    float cx = centerX + ctrlDist * cos(midTheta);
    float cy = centerY + ctrlDist * sin(midTheta);

    uint32_t segColor;
    
    // TODO: Get location dependent
    bool isDaytime = i > 7 && i < 17; 
    // bool isDaytime = i < 12; 

    if (isDaytime) {
        // DAY: Switch between Green and Orange
        // Use modulus on the index 'i' to alternate colors
        segColor = (i % 2 == 0) ? dayA : dayB;
    } else {
        // NIGHT: Switch between Light Blue and Dark Blue
        segColor = (i % 2 == 0) ? nightA : nightB;
    }
    return {x1, y1, cx, cy, x2, y2, thickness, segColor};
}

struct ChatPanel {
    flecs::entity messages_panel;
    flecs::entity input_panel;
    flecs::entity input_text;
    flecs::entity message_list;
};

// Async interpretation of chat messages
struct PendingInterpretation {
    std::string draft;
    std::string result;
    std::atomic<bool> completed{false};
    flecs::entity message_list;  // Parent for badges
};

std::mutex pending_interpretations_mutex;
std::vector<std::shared_ptr<PendingInterpretation>> pending_interpretations;

// TODO: There are some 'layers of context'
// std::unordered_map<std::string, EntityBinding> entityBindingContext;

// Known entities from previous interpretations (for context/binding)
struct KnownEntity {
    std::string id;
    std::string label;
    std::string color;  // Hex color string
    std::string display_symbol;  // The symbol for this entity (digit or letter)
};

// Token types for annotated sentence
enum class TokenType {
    PlainText,       // Regular word
    Entity,          // {{text, n}} - entity binding
    Relationship,    // {{text, R:src:tgt}} - relationship with source/target digits
    Concept,         // [[Qn - opens a comprehension frame; members follow unchanged
    ConceptEnd       // ]] - closes the frame
};

// Binding type for relationship slots
enum class SlotBindingType {
    Standard,  // Single bound entity
    Set,       // Multiple bound entities (could also work for standard entity badges?)
    Wildcard   // Unbound/intensional slot
};

// Token in annotated sentence
struct SentenceToken {
    std::string text;
    std::string binding_symbol;
    std::string source_symbol;
    std::string target_symbol;
    std::string reified_symbol;
    TokenType type;

    bool is_binding() const { return type != TokenType::PlainText; }
    // For selection: relationships have 3 selectable parts (source, rel, target)
    // Reified indicator is inside badge but not separately selectable
    int selection_width() const {
        return type == TokenType::Relationship ? 3 : 1;
    }
};

// A relationship "joins" an adjacent entity whose symbol matches one of its
// slots: the entity's badge renders inside the block, and -- because badge and
// slot are then the same thing -- they collapse to a single selection stop.
// One definition, consumed by rendering and by every key handler, because the
// selection index space is derived from these widths; two opinions about what
// joined means would desynchronise the arrow keys from the screen.
static bool relationship_src_joined(const std::vector<SentenceToken>& tokens, size_t rel) {
    return rel > 0
        && tokens[rel].type == TokenType::Relationship
        && tokens[rel - 1].type == TokenType::Entity
        && !tokens[rel - 1].binding_symbol.empty()
        && tokens[rel].source_symbol == tokens[rel - 1].binding_symbol;
}

static bool relationship_tgt_joined(const std::vector<SentenceToken>& tokens, size_t rel) {
    return rel + 1 < tokens.size()
        && tokens[rel].type == TokenType::Relationship
        && tokens[rel + 1].type == TokenType::Entity
        && !tokens[rel + 1].binding_symbol.empty()
        && tokens[rel].target_symbol == tokens[rel + 1].binding_symbol;
}

// Selection stops this token contributes: a relationship loses one stop per
// joined side, since the joined badge is counted by its own entity token.
// Every symbol in use across the sentence: entity bindings, relationship
// slots, reified triples, and concept frames. One pool, one collector --
// entities, reified particulars and concepts share the digit space so any of
// them can be bound into a relationship slot or referenced by symbol, and
// split scans would hand two of them the same digit.
static std::set<std::string> collect_used_symbols(const std::vector<SentenceToken>& tokens) {
    std::set<std::string> used;
    for (const auto& tok : tokens) {
        if (!tok.binding_symbol.empty()) used.insert(tok.binding_symbol);
        if (!tok.source_symbol.empty()) used.insert(tok.source_symbol);
        if (!tok.target_symbol.empty()) used.insert(tok.target_symbol);
        if (!tok.reified_symbol.empty()) used.insert(tok.reified_symbol);
    }
    return used;
}

static int token_selection_width(const std::vector<SentenceToken>& tokens, size_t i) {
    if (tokens[i].type == TokenType::ConceptEnd) return 0;
    int width = tokens[i].selection_width();
    if (tokens[i].type == TokenType::Relationship) {
        if (relationship_src_joined(tokens, i)) width--;
        if (relationship_tgt_joined(tokens, i)) width--;
    }
    return width;
}

// Which roles (0=source, 1=whole triple, 2=target) a relationship's remaining
// stops map to, in order. Handlers translate a sub-part index through this
// rather than assuming the un-joined 0/1/2 layout.
static std::vector<int> relationship_slot_roles(const std::vector<SentenceToken>& tokens, size_t rel) {
    std::vector<int> roles;
    if (!relationship_src_joined(tokens, rel)) roles.push_back(0);
    roles.push_back(1);
    if (!relationship_tgt_joined(tokens, rel)) roles.push_back(2);
    return roles;
}

// TODO: Automatically register from create badge?
// Cache for entity colors (binding_symbol -> color)
std::unordered_map<std::string, uint32_t> entity_color_cache = {
    {"w", 0x6df0ffFF},  // Wesley - cyan
    {"h", 0xff75baFF},   // Heonae - pink
    {"g", 0xe66c25ff}
};

// TODO: Refactor this to Tradewinds...
// Get color via Unix socket to persistent Python server (fast after first call)
uint32_t get_entity_color(std::string binding_symbol, const std::string& entity_text) {
    // Check cache first
    auto it = entity_color_cache.find(binding_symbol);
    if (it != entity_color_cache.end()) {
        return it->second;
    }

    // Build list of taken colors
    std::string taken_json = "[";
    bool first = true;
    for (const auto& [id, color] : entity_color_cache) {
        if (!first) taken_json += ",";
        first = false;
        char hex[7];
        snprintf(hex, sizeof(hex), "%06x", (color >> 8) & 0xFFFFFF);
        taken_json += "\"" + std::string(hex) + "\"";
    }
    taken_json += "]";

    // Try to connect to server
    const char* socket_path = "/tmp/entity_color.sock";
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        entity_color_cache[binding_symbol] = 0x4d9be6FF;
        return 0x4d9be6FF;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        // Server not running, start it
        close(sock);
        system("python3 ../scripts/entity_color_server.py &");
        usleep(2000000); // Wait 2s for server to start and load model

        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock < 0 || connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            if (sock >= 0) close(sock);
            entity_color_cache[binding_symbol] = 0x4d9be6FF;
            return 0x4d9be6FF;
        }
    }

    // Send request
    std::string request = "{\"text\":\"" + entity_text + "\",\"taken\":" + taken_json + "}";
    send(sock, request.c_str(), request.length(), 0);

    // Receive response
    char buffer[1024];
    int n = recv(sock, buffer, sizeof(buffer) - 1, 0);
    close(sock);

    if (n <= 0) {
        entity_color_cache[binding_symbol] = 0x4d9be6FF;
        return 0x4d9be6FF;
    }
    buffer[n] = '\0';
    std::string result(buffer);

    // Parse JSON response
    uint32_t color = 0x4d9be6FF;
    size_t color_pos = result.find("\"color\"");
    if (color_pos != std::string::npos) {
        size_t start = result.find("\"", color_pos + 7);
        if (start != std::string::npos) {
            size_t end = result.find("\"", start + 1);
            if (end != std::string::npos) {
                std::string hex = result.substr(start + 1, end - start - 1);
                if (hex.length() == 6) {
                    unsigned int rgb = std::stoul(hex, nullptr, 16);
                    color = (rgb << 8) | 0xFF;
                }
            }
        }
    }

    entity_color_cache[binding_symbol] = color;
    return color;
}

// TODO: This should be probably be prechewed into a more imperically digestable form with Python first...
// Parse sentence template string into tokens
// Formats:
//   "Hello {{world, 5}} text" - entity binding
//   "{{*, S3}} {{loves, R5}} {{*, T7}}" - relationship with wildcards
std::vector<SentenceToken> parse_sentence_template(const std::string& sentence) {
    std::vector<SentenceToken> tokens;
    size_t i = 0;
    while (i < sentence.size()) {
        // Skip whitespace
        while (i < sentence.size() && std::isspace(sentence[i])) i++;
        if (i >= sentence.size()) break;

        // Comprehension frame markers travel as bare words in the template:
        // "[[Q0" opens a frame, "]]" closes it. Members between them are
        // ordinary tokens, bindings intact -- the frame contains, never absorbs.
        if (sentence.compare(i, 3, "[[Q") == 0) {
            size_t end = i + 3;
            while (end < sentence.size() && !std::isspace(sentence[end])) end++;
            std::string spec = sentence.substr(i + 3, end - i - 3);
            // "Qi:e" -- i is the intension's id, e the extension set's. Both are
            // entities; the extension id lives in reified_symbol, which is
            // semantically exact: the output set is the reified query result.
            std::string intension = spec, extension = "";
            size_t colon = spec.find(':');
            if (colon != std::string::npos) {
                intension = spec.substr(0, colon);
                extension = spec.substr(colon + 1);
            }
            tokens.push_back({"", intension, "", "", extension, TokenType::Concept});
            i = end;
            continue;
        }
        if (sentence.compare(i, 2, "]]") == 0) {
            tokens.push_back({"", "", "", "", "", TokenType::ConceptEnd});
            i += 2;
            continue;
        }

        // Check for binding {{text, spec}}
        if (i + 1 < sentence.size() && sentence[i] == '{' && sentence[i+1] == '{') {
            size_t start = i + 2;
            size_t end = sentence.find("}}", start);
            if (end != std::string::npos) {
                std::string binding = sentence.substr(start, end - start);
                size_t comma = binding.rfind(',');
                if (comma != std::string::npos) {
                    std::string text = binding.substr(0, comma);
                    // Trim whitespace from text
                    while (!text.empty() && std::isspace(text.back())) text.pop_back();
                    while (!text.empty() && std::isspace(text.front())) text.erase(0, 1);
                    std::string spec = binding.substr(comma + 1);
                    // Trim whitespace from spec
                    while (!spec.empty() && std::isspace(spec.front())) spec.erase(0, 1);
                    while (!spec.empty() && std::isspace(spec.back())) spec.pop_back();

                    // Parse spec: digit for entity, R (src:tgt)reified or R src:tgt for relationship
                    TokenType type = TokenType::Entity;
                    std::string bind_str = "";
                    std::string src_str = "";
                    std::string tgt_str = "";
                    std::string reified_str = "";

                    if (!spec.empty()) {
                        char prefix = spec[0];
                        if (prefix == 'Q' && spec.size() >= 2) {
                            type = TokenType::Concept;
                            bind_str = spec.substr(1);
                        }
                        else if (prefix == 'R' || prefix == 'r') {
                            type = TokenType::Relationship;
                            // Check for reified format: R (src:tgt)reified
                            size_t paren_open = spec.find('(');
                            size_t paren_close = spec.find(')');
                            if (paren_open != std::string::npos && paren_close != std::string::npos && paren_close > paren_open) {
                                // Reified format: R (src:tgt)reified
                                std::string inner = spec.substr(paren_open + 1, paren_close - paren_open - 1);
                                reified_str = spec.substr(paren_close + 1);
                                size_t colon = inner.find(':');
                                if (colon != std::string::npos) {
                                    src_str = inner.substr(0, colon);
                                    tgt_str = inner.substr(colon + 1);
                                    // Convert * to empty string for wildcards
                                    if (src_str == "*") src_str = "";
                                    if (tgt_str == "*") tgt_str = "";
                                }
                            } else {
                                // Non-reified format: R src:tgt
                                size_t space = spec.find(' ');
                                std::string rest = (space != std::string::npos) ? spec.substr(space + 1) : spec.substr(1);
                                // Trim whitespace
                                while (!rest.empty() && std::isspace(rest.front())) rest.erase(0, 1);
                                size_t colon = rest.find(':');
                                if (colon != std::string::npos) {
                                    src_str = rest.substr(0, colon);
                                    tgt_str = rest.substr(colon + 1);
                                    // Convert * to empty string for wildcards
                                    if (src_str == "*") src_str = "";
                                    if (tgt_str == "*") tgt_str = "";
                                }
                            }
                        } else {
                            bind_str = spec;
                            // Optional lemma annotation after ':' -- a
                            // kind-promoted member's canonical form.
                            size_t colon = bind_str.find(':');
                            if (colon != std::string::npos) {
                                reified_str = bind_str.substr(colon + 1);
                                bind_str = bind_str.substr(0, colon);
                            }
                            // Convert * to empty string for wildcards
                            if (bind_str == "*") bind_str = "";
                        }
                    }
                    tokens.push_back({text, bind_str, src_str, tgt_str, reified_str, type});
                }
                i = end + 2;
                continue;
            }
        }

        // Regular word
        size_t word_start = i;
        while (i < sentence.size() && !std::isspace(sentence[i]) &&
               !(i + 1 < sentence.size() && sentence[i] == '{' && sentence[i+1] == '{')) {
            i++;
        }
        if (i > word_start) {
            tokens.push_back({sentence.substr(word_start, i - word_start), "", "", "", "", TokenType::PlainText});
        }
    }
    return tokens;
}

// Convert tokens back to template string
std::string tokens_to_template(const std::vector<SentenceToken>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) result += " ";
        if (tokens[i].type == TokenType::Concept) {
            result += "[[Q" + tokens[i].binding_symbol;
            if (!tokens[i].reified_symbol.empty()) result += ":" + tokens[i].reified_symbol;
            continue;
        }
        if (tokens[i].type == TokenType::ConceptEnd) {
            result += "]]";
            continue;
        }
        if (tokens[i].is_binding()) {
            std::string spec;
            switch (tokens[i].type) {
                case TokenType::Relationship: {
                    std::string src = !tokens[i].source_symbol.empty() ? tokens[i].source_symbol : "*";
                    std::string tgt = !tokens[i].target_symbol.empty() ? tokens[i].target_symbol : "*";
                    if (!tokens[i].reified_symbol.empty()) {
                        // Reified format: R (src:tgt)reified
                        spec = "R (" + src + ":" + tgt + ")" + tokens[i].reified_symbol;
                    } else {
                        // Non-reified format: R src:tgt
                        spec = "R " + src + ":" + tgt;
                    }
                    break;
                }
                default:
                    spec = !tokens[i].binding_symbol.empty() ? tokens[i].binding_symbol : "*";
                    // A lemma annotation (kind-promoted member) rides after ':'.
                    if (tokens[i].type == TokenType::Entity
                        && !tokens[i].reified_symbol.empty()) {
                        spec += ":" + tokens[i].reified_symbol;
                    }
                    break;
            }
            result += "{{" + tokens[i].text + ", " + spec + "}}";
        } else {
            result += tokens[i].text;
        }
    }
    return result;
}

// Word annotation selector - highlights words for annotation
// Stored message data for annotation
struct StoredMessage {
    std::string sentence_template;
    flecs::entity parent_entity;
    std::vector<flecs::entity> ui_entities;
    std::vector<flecs::entity> selection_entities;
    int token_count = 0;
};

// Global list of annotatable messages
std::vector<StoredMessage> g_annotatable_messages;
int g_current_message_idx = -1;  // -1 means no message selected

// Single annotation selector - moves between messages
struct WordAnnotationSelector {
    std::string sentence_template;              // Template string with {{entity, n}} bindings
    std::vector<flecs::entity> ui_entities;     // Current UI entities (recreated from template)
    std::vector<flecs::entity> selection_entities; // Maps selection index to entity for bounds (may differ from ui_entities for sub-parts)
    flecs::entity parent_entity;                // Parent for UI entities
    int start_index;                            // Start of selection range (0-based, selection index)
    int end_index;                              // End of selection range (inclusive, selection index)
    int token_count;                            // Number of selectable positions
    bool active;                                // Whether annotation mode is active
    bool dirty;                                 // Needs UI entity recreation
    uint32_t highlight_color;                   // Color for the highlight rectangle
};

std::mutex known_entities_mutex;
std::vector<KnownEntity> known_entities = {
    {"wesley", "Wesley", "6df0ff", "w"},
    {"heonae", "Heonae", "ff75ba", "h"}
};
int next_entity_number = 1;  // Global counter for entity display numbers

// Previous sentences for context (last N sentences)
std::mutex previous_sentences_mutex; // WTF?
std::vector<std::string> previous_sentences;
const int MAX_PREVIOUS_SENTENCES = 10;  // Keep last N sentences for context


// Superseded by the VirtualList component in components.h, which is what the
// Entities panel uses. This one never worked: Populate()'s element-creation
// body is commented out, so `entities` stays empty, page_max resolves to 0 and
// Refresh() iterates nothing -- the Condensate file list scans the directory
// and then renders none of it. Kept only because Condensate still calls it.
// TODO: port Condensate onto the VirtualList component and delete this.
class LegacyVirtualList
{
public:
    size_t index = 0;
    // The page count to display should be dynamically calculated based on the list element size
    // and the parent of the list
    size_t page_max = 64;
    std::vector<flecs::entity> entities;
    virtual void Populate(flecs::world*) = 0;
    virtual flecs::entity CreateNextElement(flecs::world*) = 0;
    virtual flecs::entity CreatePrevElement(flecs::world*) = 0;
};

class FileNavVirtualList : public LegacyVirtualList {
public:
    std::string path_to_scan;
    std::vector<std::string> file_names;
    flecs::entity container;

    FileNavVirtualList(std::string path, flecs::entity parent) 
        : path_to_scan(path), container(parent) {}

    virtual void Populate(flecs::world* world) override {
        auto UIElement = world->lookup("UIElement");
        file_names.clear();
        std::error_code ec;

        for (const auto& entry : std::filesystem::directory_iterator(path_to_scan, ec)) {
            std::string name = entry.path().filename().string();
            if (entry.is_directory()) name += "/";
            file_names.push_back(name);
        }

        float probe_height = 0.0f;
        for (size_t i = 0; i < file_names.size(); ++i) { // file_names.size()
            probe_height += 16.0f;
            page_max = entities.size();
            // if (probe_height < container.parent().get<RoundedRectRenderable>().height)
            // {
            //     auto e = world->entity()
            //         .is_a(UIElement)
            //         .child_of(container)
            //         .set<TextRenderable>({"", "JetBrainsMono", 16.0f, 0xFFFFFFFF})
            //         .set<ZIndex>({14});
                    
            //         entities.push_back(e);
            // } else{
            //     break;
            // }
        }
        
        Refresh(); // Fill the text for the first time
    }

    void Refresh() {
        for (size_t i = 0; i < page_max; ++i) {
            size_t data_idx = index + i;
            if (data_idx < file_names.size()) {
                entities[i].ensure<TextRenderable>().text = file_names[data_idx];
            } else {
                entities[i].ensure<TextRenderable>().text = ""; // Clear unused slots
            }
        }
    }

    virtual flecs::entity CreateNextElement(flecs::world* world) override {
        if (index + page_max < file_names.size()) {
            index++;
            Refresh();
        }
        return entities.back();
    }

    virtual flecs::entity CreatePrevElement(flecs::world* world) override {
        if (index > 0) {
            index--;
            Refresh();
        }
        return entities.front();
    }
};

flecs::world* world = nullptr;

// ============================================================================
// LAYOUT SYSTEM PHASE HELPERS
// ============================================================================

// Propagate world positions from an entity down to all descendants
// Called after setting a child's local position to update its world position
void propagate_world_positions(flecs::entity entity) {
    const Position* local = entity.try_get<Position, Local>();
    const Position* parent_world = nullptr;

    flecs::entity parent = entity.parent();
    if (parent.is_valid()) {
        parent_world = parent.try_get<Position, World>();
    }

    Position& world = entity.ensure<Position, World>();
    world.x = local ? local->x : 0.0f;
    world.y = local ? local->y : 0.0f;
    if (parent_world) {
        world.x += parent_world->x;
        world.y += parent_world->y;
    }

    // Recursively update children
    entity.children([](flecs::entity child) {
        propagate_world_positions(child);
    });
}

// Calculate bounding box of children using Position Local + UIElementSize
// Returns min/max in LOCAL coordinates relative to parent
void calculate_children_local_bounds(flecs::entity parent, float& min_x, float& min_y, float& max_x, float& max_y, bool& found) {
    parent.children([&](flecs::entity child) {
        if (child.has<Expand>()) return; // Skip expand entities

        const Position* local = child.try_get<Position, Local>();
        const UIElementSize* size = child.try_get<UIElementSize>();

        if (local && size && size->width > 0 && size->height > 0) {
            float child_xmin = local->x;
            float child_ymin = local->y;
            float child_xmax = local->x + size->width;
            float child_ymax = local->y + size->height;

            if (!found) {
                min_x = child_xmin;
                min_y = child_ymin;
                max_x = child_xmax;
                max_y = child_ymax;
                found = true;
            } else {
                min_x = std::min(min_x, child_xmin);
                min_y = std::min(min_y, child_ymin);
                max_x = std::max(max_x, child_xmax);
                max_y = std::max(max_y, child_ymax);
            }
        }
    });
}

// Forward declaration for mutual recursion
void process_layout_recursive(flecs::entity e, float available_width = 10000.0f);

// Update UIContainer size from children's local bounds
// Children are assumed to already be positioned (e.g., by LayoutBox or manually)
// Padding is applied: left/top padding is implicit in children positions,
// right/bottom padding is added to max extent
void update_ui_container_size(flecs::entity e) {
    UIContainer* container = e.try_get_mut<UIContainer>();
    if (!container) return;

    float min_x = 0.0f, min_y = 0.0f, max_x = 0.0f, max_y = 0.0f;
    bool found = false;
    calculate_children_local_bounds(e, min_x, min_y, max_x, max_y, found);

    if (found) {
        // Container encompasses children extent plus padding on right/bottom
        // Left/top padding is expected to be in children's positions already
        UIElementSize& size = e.ensure<UIElementSize>();
        size.width = max_x + container->pad_horizontal;
        size.height = max_y + container->pad_vertical;

        // Update renderable if present
        if (RoundedRectRenderable* rr = e.try_get_mut<RoundedRectRenderable>()) {
            rr->width = size.width;
            rr->height = size.height;
        }
        if (CustomRenderable* cr = e.try_get_mut<CustomRenderable>()) {
            cr->width = size.width;
            cr->height = size.height;
        }
    }
}

// Process FlowLayoutBox - positions children with wrapping
void process_flow_layout(flecs::entity e, float container_width) {
    FlowLayoutBox& box = e.ensure<FlowLayoutBox>();
    UIElementSize& container_size = e.ensure<UIElementSize>();

    box.x_progress = 0.0f;
    box.y_progress = 0.0f;
    box.line_height = 0.0f;

    float max_width = 0.0f;

    // Collect children info and determine line breaks
    struct ChildInfo {
        flecs::entity entity;
        float width;
        float height;
    };
    std::vector<std::vector<ChildInfo>> lines;
    std::vector<ChildInfo> current_line;
    float current_line_width = 0.0f;
    float current_line_height = 0.0f;

    e.children([&](flecs::entity child) {
        const UIElementSize* child_size = child.try_get<UIElementSize>();
        if (!child_size || child_size->width <= 0 || child_size->height <= 0) return;

        float child_width = child_size->width;
        float child_height = child_size->height;

        float needed_width = current_line_width + child_width;
        if (!current_line.empty()) {
            needed_width += box.padding;
        }

        if (!current_line.empty() && needed_width > container_width) {
            lines.push_back(current_line);
            current_line.clear();
            current_line_width = 0.0f;
            current_line_height = 0.0f;
        }

        current_line.push_back({child, child_width, child_height});
        if (current_line.size() > 1) {
            current_line_width += box.padding;
        }
        current_line_width += child_width;
        current_line_height = std::max(current_line_height, child_height);
    });

    if (!current_line.empty()) {
        lines.push_back(current_line);
    }

    // Position children with vertical centering
    box.y_progress = 0.0f;
    for (const auto& line : lines) {
        float line_height = 0.0f;
        for (const auto& child_info : line) {
            line_height = std::max(line_height, child_info.height);
        }

        box.x_progress = 0.0f;
        for (const auto& child_info : line) {
            Position& pos = child_info.entity.ensure<Position, Local>();
            pos.x = box.x_progress;
            float y_offset = (line_height - child_info.height) * 0.5f;
            pos.y = box.y_progress + y_offset;

            propagate_world_positions(child_info.entity);

            box.x_progress += child_info.width + box.padding;
        }

        max_width = std::max(max_width, box.x_progress - box.padding);
        box.y_progress += line_height + box.line_spacing;
    }

    // Update container size
    const Expand* expand = e.try_get<Expand>();
    if (!expand || !expand->x_enabled) {
        container_size.width = max_width;
    }
    if (!expand || !expand->y_enabled) {
        float total_height = box.y_progress;
        if (!lines.empty()) {
            total_height -= box.line_spacing;
        }
        container_size.height = total_height;
    }
}

// Recursively process layout bottom-up: children first, then parent
// Handles LayoutBox, UIContainer, and FlowLayoutBox in correct order
void process_layout_recursive(flecs::entity e, float available_width) {
    // For FlowLayoutBox with Expand, get width from parent bounds if available
    float my_width = available_width;
    if (e.has<FlowLayoutBox>()) {
        const Expand* expand = e.try_get<Expand>();
        if (expand && expand->x_enabled) {
            // Use parent bounds for available width
            flecs::entity parent = e.parent();
            if (parent.is_valid()) {
                const UIElementBounds* parent_bounds = parent.try_get<UIElementBounds>();
                if (parent_bounds && parent_bounds->xmax > parent_bounds->xmin) {
                    my_width = (parent_bounds->xmax - parent_bounds->xmin) * expand->x_percent;
                }
            }
        }
    }

    // First, recursively process all children (bottom-up)
    e.children([my_width](flecs::entity child) {
        process_layout_recursive(child, my_width);
    });

    // An entity is either Yoga-managed or legacy-managed, never both: Yoga
    // already positioned this entity's children and sized the entity itself
    // back at PreFrame, so the legacy layout must not touch it. Note we still
    // recursed above -- a Yoga subtree may contain legacy LayoutBox subtrees.
    // UIYogaLegacyLeaf is the deliberate exception: Yoga only positions it, so
    // its own UIContainer/LayoutBox sizing must still run here.
    if (e.has<UIYoga>() && !e.has<UIYogaLegacyLeaf>()) return;

    // Then update this entity's size based on its type
    if (e.has<LayoutBox>()) {
        LayoutBox& box = e.ensure<LayoutBox>();
        UIElementSize& container_size = e.ensure<UIElementSize>();

        float main_progress = 0.0f;
        float cross_max = 0.0f;
        bool horiz = (box.dir == LayoutBox::Horizontal);

        e.children([&](flecs::entity child) {
            const UIElementSize* child_size = child.try_get<UIElementSize>();
            if (!child_size || child_size->width <= 0 || child_size->height <= 0) return;

            Position& local_pos = child.ensure<Position, Local>();

            float child_main = horiz ? child_size->width : child_size->height;
            float child_cross = horiz ? child_size->height : child_size->width;

            if (box.move_dir < 0) {
                main_progress -= (child_main + box.padding);
            }

            if (horiz) {
                local_pos.x = main_progress;
            } else {
                local_pos.y = main_progress;
            }

            if (box.move_dir > 0) {
                main_progress += child_main + box.padding;
            }

            cross_max = std::max(cross_max, child_cross);
        });

        const Expand* expand = e.try_get<Expand>();
        float main_size = std::abs(main_progress);

        if (horiz) {
            if (!expand || !expand->x_enabled) container_size.width = main_size;
            if (!expand || !expand->y_enabled) container_size.height = cross_max;
        } else {
            if (!expand || !expand->y_enabled) container_size.height = main_size;
            if (!expand || !expand->x_enabled) container_size.width = cross_max;
        }
    }
    else if (e.has<FlowLayoutBox>()) {
        process_flow_layout(e, my_width);
    }
    else if (e.has<UIContainer>()) {
        update_ui_container_size(e);
    }
}

// Phase 3 helper: Propagate bounds containment from children to parent
void propagate_bounds_containment(flecs::entity parent) {
    UIElementBounds* parent_bounds = parent.try_get_mut<UIElementBounds>();
    if (!parent_bounds) return;

    parent.children([&](flecs::entity child) {
        // First recurse to ensure children's bounds are set
        propagate_bounds_containment(child);

        // Skip Expand entities (they fill parent, not contribute to size)
        if (child.has<Expand>()) return;

        const UIElementBounds* child_bounds = child.try_get<UIElementBounds>();
        if (!child_bounds) return;

        // Expand parent bounds to contain child
        parent_bounds->xmin = std::min(parent_bounds->xmin, child_bounds->xmin);
        parent_bounds->ymin = std::min(parent_bounds->ymin, child_bounds->ymin);
        parent_bounds->xmax = std::max(parent_bounds->xmax, child_bounds->xmax);
        parent_bounds->ymax = std::max(parent_bounds->ymax, child_bounds->ymax);
    });
}

// ============================================================================
// Yoga (flexbox) layout
// ----------------------------------------------------------------------------
// The per-entity steps below have two callers: the PreFrame systems that lay out
// every Yoga tree once per frame, and yoga_layout_now(), which lays out a single
// subtree synchronously. The latter exists because UI is frequently built inside
// an input handler -- a popup menu, say -- and has to be correct, drawable and
// hit-testable at that instant, not one frame later.
// ============================================================================

// nanovg context for text measurement and image queries. Yoga's measure
// callback signature has no room for user data beyond the node, so this is
// stashed at file scope; it is assigned once the Graphics singleton exists.
static NVGcontext* g_yoga_vg = nullptr;

static flecs::entity yoga_node_entity(YGNodeConstRef node)
{
    return world->entity((flecs::entity_t)(uintptr_t)YGNodeGetContext(node));
}

// Text leaves have no intrinsic size as far as Yoga is concerned, so they get a
// measure func that calls back into nanovg during layout. Without it, text in a
// flex container collapses to zero and the container can't size to its label.
static YGSize yoga_measure_text(YGNodeConstRef node,
                                float width, YGMeasureMode widthMode,
                                float height, YGMeasureMode heightMode)
{
    YGSize out{0.0f, 0.0f};
    if (!g_yoga_vg) return out;

    flecs::entity e = yoga_node_entity(node);
    if (!e.is_valid() || !e.is_alive()) return out;

    const TextRenderable* text = e.try_get<TextRenderable>();
    if (!text) return out;

    nvgFontSize(g_yoga_vg, text->fontSize);
    nvgFontFace(g_yoga_vg, text->fontFace.c_str());

    // Honour an explicit wrapWidth; otherwise wrap to the width Yoga is
    // offering, when it gives us a bound to work with.
    float wrap = text->wrapWidth;
    if (wrap <= 0.0f && widthMode != YGMeasureModeUndefined && width > 0.0f
        && !std::isnan(width)) {
        wrap = width;
    }

    TextSize ts = measureText(g_yoga_vg, text->text, wrap);
    out.width = ts.w;
    out.height = ts.h * text->scaleY;
    return out;
}

// Step 0: a Yoga root that fills a legacy-laid-out parent takes its UISize from
// that parent's UIElementSize.
static void yoga_sync_fill_parent(flecs::entity e)
{
    const UIFillParent* fill = e.try_get<UIFillParent>();
    if (!fill) return;
    flecs::entity parent = e.parent();
    if (!parent.is_valid()) return;
    const UIElementSize* parent_size = parent.try_get<UIElementSize>();
    if (!parent_size) return;

    UISize& sz = e.ensure<UISize>();
    sz.w = std::max(0.0f, parent_size->width  - fill->pad_left - fill->pad_right);
    sz.h = std::max(0.0f, parent_size->height - fill->pad_top  - fill->pad_bottom);
}

// The content box available to a Yoga entity's children: the parent's resolved
// size less its padding. Reads the parent's UISize first (Yoga's own input,
// already written by yoga_sync_fill_parent this pass) and falls back to the
// size the rest of the pipeline computed. Returns false if neither is known.
static bool yoga_parent_content_size(flecs::entity e, float& out_w, float& out_h)
{
    flecs::entity parent = e.parent();
    if (!parent.is_valid()) return false;

    float w = YGUndefined, h = YGUndefined;
    if (const UISize* sz = parent.try_get<UISize>()) { w = sz->w; h = sz->h; }
    if (std::isnan(w) || std::isnan(h)) {
        if (const UIElementSize* sz = parent.try_get<UIElementSize>()) {
            if (std::isnan(w)) w = sz->width;
            if (std::isnan(h)) h = sz->height;
        }
    }
    if (std::isnan(w) || std::isnan(h) || w <= 0.0f || h <= 0.0f) return false;

    if (const UIPadding* p = parent.try_get<UIPadding>()) {
        w -= p->left + p->right;
        h -= p->top + p->bottom;
    }
    out_w = std::max(0.0f, w);
    out_h = std::max(0.0f, h);
    return true;
}

// Step 1: make the Yoga node's child list match the flecs hierarchy. Entities
// without UIYoga are skipped, so they stay invisible to the Yoga tree even when
// their flecs parent is a Yoga node.
static void yoga_sync_topology(flecs::entity e, YogaNode& yn)
{
    if (!yn.node) return;
    uint32_t desired_idx = 0;
    e.children([&](flecs::entity child) {
        if (!child.has<UIYoga>()) return;
        const YogaNode* cyn = child.try_get<YogaNode>();
        if (!cyn || !cyn->node) return;
        YGNodeRef cur_owner = YGNodeGetOwner(cyn->node);
        YGNodeRef at_idx = (desired_idx < YGNodeGetChildCount(yn.node))
            ? YGNodeGetChild(yn.node, desired_idx)
            : nullptr;
        if (cur_owner != yn.node || at_idx != cyn->node) {
            if (cur_owner) YGNodeRemoveChild(cur_owner, cyn->node);
            // Yoga asserts when inserting into a node that measures itself; an
            // entity that gains Yoga children stops being a measured leaf.
            if (YGNodeHasMeasureFunc(yn.node))
                YGNodeSetMeasureFunc(yn.node, nullptr);
            YGNodeInsertChild(yn.node, cyn->node, desired_idx);
        }
        desired_idx++;
    });
}

// Step 2: push UI* component values into the YGNode. Cheap -- Yoga detects
// no-op writes and won't dirty the node.
static void yoga_sync_style(flecs::entity e, YogaNode& yn)
{
    if (!yn.node) return;
    YGNodeRef n = yn.node;

    // Absolute edge positioning (e.g. "fill parent with padding").
    if (const UIAbsoluteEdges* ae = e.try_get<UIAbsoluteEdges>()) {
        YGNodeStyleSetPositionType(n, YGPositionTypeAbsolute);
        if (!std::isnan(ae->left))   YGNodeStyleSetPosition(n, YGEdgeLeft, ae->left);
        if (!std::isnan(ae->top))    YGNodeStyleSetPosition(n, YGEdgeTop, ae->top);
        if (!std::isnan(ae->right))  YGNodeStyleSetPosition(n, YGEdgeRight, ae->right);
        if (!std::isnan(ae->bottom)) YGNodeStyleSetPosition(n, YGEdgeBottom, ae->bottom);
    } else {
        YGNodeStyleSetPositionType(n, YGPositionTypeRelative);
    }

    if (const UISize* sz = e.try_get<UISize>()) {
        if (!std::isnan(sz->w)) YGNodeStyleSetWidth(n, sz->w);
        if (!std::isnan(sz->h)) YGNodeStyleSetHeight(n, sz->h);
    }
    if (const UIMaxSize* mx = e.try_get<UIMaxSize>()) {
        if (!std::isnan(mx->w)) YGNodeStyleSetMaxWidth(n, mx->w);
        if (!std::isnan(mx->h)) YGNodeStyleSetMaxHeight(n, mx->h);
    }
    if (const UIMinSize* mn = e.try_get<UIMinSize>()) {
        if (!std::isnan(mn->w)) YGNodeStyleSetMinWidth(n, mn->w);
        if (!std::isnan(mn->h)) YGNodeStyleSetMinHeight(n, mn->h);
    }
    if (const UIAspectRatio* ar = e.try_get<UIAspectRatio>()) {
        if (!std::isnan(ar->ratio) && ar->ratio > 0.0f)
            YGNodeStyleSetAspectRatio(n, ar->ratio);
    }
    if (const UIFlexContainer* fc = e.try_get<UIFlexContainer>()) {
        YGNodeStyleSetFlexDirection(n, fc->direction);
        YGNodeStyleSetFlexWrap(n, fc->wrap);
        YGNodeStyleSetJustifyContent(n, fc->justify);
        YGNodeStyleSetAlignItems(n, fc->items);
        YGNodeStyleSetGap(n, YGGutterAll, fc->gap);
    }
    if (const UIFlexItem* fi = e.try_get<UIFlexItem>()) {
        YGNodeStyleSetFlexGrow(n, fi->grow);
        YGNodeStyleSetFlexShrink(n, fi->shrink);
        YGNodeStyleSetAlignSelf(n, fi->alignSelf);
    }
    if (const UIMargin* m = e.try_get<UIMargin>()) {
        YGNodeStyleSetMargin(n, YGEdgeTop, m->top);
        YGNodeStyleSetMargin(n, YGEdgeRight, m->right);
        YGNodeStyleSetMargin(n, YGEdgeBottom, m->bottom);
        YGNodeStyleSetMargin(n, YGEdgeLeft, m->left);
    }
    if (const UIPadding* p = e.try_get<UIPadding>()) {
        YGNodeStyleSetPadding(n, YGEdgeTop, p->top);
        YGNodeStyleSetPadding(n, YGEdgeRight, p->right);
        YGNodeStyleSetPadding(n, YGEdgeBottom, p->bottom);
        YGNodeStyleSetPadding(n, YGEdgeLeft, p->left);
    }
}

// Step 3: intrinsic sizing for leaves Yoga cannot measure by itself.
static void yoga_apply_intrinsic(flecs::entity e, YogaNode& yn)
{
    if (!yn.node) return;
    YGNodeRef n = yn.node;

    // A legacy leaf is sized by the old pipeline: mirror whatever size it
    // computed into the Yoga style so the flex parent can place it, and never
    // measure or descend into it.
    if (e.has<UIYogaLegacyLeaf>()) {
        float lw = YGUndefined, lh = YGUndefined;
        if (const RoundedRectRenderable* rr = e.try_get<RoundedRectRenderable>()) {
            lw = rr->width;  lh = rr->height;
        } else if (const CustomRenderable* cr = e.try_get<CustomRenderable>()) {
            lw = cr->width;  lh = cr->height;
        } else if (const RectRenderable* rect = e.try_get<RectRenderable>()) {
            lw = rect->width; lh = rect->height;
        } else if (const ImageRenderable* img = e.try_get<ImageRenderable>()) {
            lw = img->width; lh = img->height;
        }
        // TextRenderable leaves have no renderable size of their own, so fall
        // back to the measured size the legacy sizing system wrote.
        if (e.has<TextRenderable>()) {
            if (const UIElementSize* sz = e.try_get<UIElementSize>()) {
                lw = sz->width; lh = sz->height;
            }
        }
        if (!std::isnan(lw) && lw > 0.0f) YGNodeStyleSetWidth(n, lw);
        if (!std::isnan(lh) && lh > 0.0f) YGNodeStyleSetHeight(n, lh);
        if (YGNodeHasMeasureFunc(n)) YGNodeSetMeasureFunc(n, nullptr);
        return;
    }

    // Yoga forbids a measure func on a node with children, so only childless
    // text entities get one.
    bool wants_measure = (YGNodeGetChildCount(n) == 0) && e.has<TextRenderable>();

    if (wants_measure) {
        if (!YGNodeHasMeasureFunc(n)) YGNodeSetMeasureFunc(n, yoga_measure_text);

        // Yoga caches measure results, so only dirty the node when the text (or
        // its styling) actually changed. Dirtying every frame would force a full
        // relayout of every ancestor.
        const TextRenderable& t = e.get<TextRenderable>();
        YogaTextCache& cache = e.ensure<YogaTextCache>();
        if (cache.text != t.text || cache.fontFace != t.fontFace
            || cache.fontSize != t.fontSize || cache.wrapWidth != t.wrapWidth
            || cache.scaleY != t.scaleY) {
            cache = {t.text, t.fontFace, t.fontSize, t.wrapWidth, t.scaleY};
            YGNodeMarkDirty(n);
        }
        return;
    }
    if (YGNodeHasMeasureFunc(n)) YGNodeSetMeasureFunc(n, nullptr);

    // Images: publish native aspect ratio so a known width yields the right
    // height (and vice versa). This must come from nvgImageSize, NOT from
    // ImageRenderable::width/height -- readback overwrites those with the
    // laid-out size, so feeding them back would let the ratio drift a little
    // further every frame.
    if (const ImageRenderable* img = e.try_get<ImageRenderable>()) {
        if (g_yoga_vg && img->imageHandle != -1) {
            int nw = 0, nh = 0;
            nvgImageSize(g_yoga_vg, img->imageHandle, &nw, &nh);
            if (nw > 0 && nh > 0) {
                float base_w = nw * img->scaleX;
                float base_h = nh * img->scaleY;

                // "Contain": fit inside the parent's content box, preserving
                // proportions. Both axes are written explicitly, so no aspect
                // ratio is published -- publishing one and letting a max
                // constraint clamp a single axis is what skews the image.
                if (const UIContainParent* fit = e.try_get<UIContainParent>()) {
                    float aw = 0.0f, ah = 0.0f;
                    if (yoga_parent_content_size(e, aw, ah)
                        && base_w > 0.0f && base_h > 0.0f) {
                        aw = std::max(0.0f, aw - fit->pad_left - fit->pad_right);
                        ah = std::max(0.0f, ah - fit->pad_top - fit->pad_bottom);
                        float scale = std::min(aw / base_w, ah / base_h);
                        if (!fit->allow_upscale) scale = std::min(scale, 1.0f);
                        YGNodeStyleSetWidth(n, base_w * scale);
                        YGNodeStyleSetHeight(n, base_h * scale);
                    }
                    return;
                }

                if (!e.has<UIAspectRatio>())
                    YGNodeStyleSetAspectRatio(n, (float)nw / (float)nh);
                // Opt-in: pin the node to the image's native size.
                if (e.has<UINativeImageSize>() && !e.has<UISize>()) {
                    YGNodeStyleSetWidth(n, nw * img->scaleX);
                    YGNodeStyleSetHeight(n, nh * img->scaleY);
                }
                // Opt-in: allow shrinking below 1:1 but never above it.
                if (e.has<UIMaxNativeImageSize>()) {
                    YGNodeStyleSetMaxWidth(n, nw * img->scaleX);
                    YGNodeStyleSetMaxHeight(n, nh * img->scaleY);
                }
            }
        }
    }
}

// True when this entity owns its Yoga tree (its flecs parent isn't a Yoga node).
static bool yoga_is_root(flecs::entity e)
{
    flecs::entity parent = e.parent();
    return !(parent && parent.has<UIYoga>());
}

// Step 4: run layout for a Yoga root. Roots with a UISize get a fixed size;
// roots without one auto-size to their content (e.g. a popup list).
static void yoga_calculate(flecs::entity e, YogaNode& yn)
{
    if (!yn.node) return;
    float rw = YGUndefined, rh = YGUndefined;
    if (const UISize* sz = e.try_get<UISize>()) {
        if (!std::isnan(sz->w)) rw = sz->w;
        if (!std::isnan(sz->h)) rh = sz->h;
    }
    YGNodeCalculateLayout(yn.node, rw, rh, YGDirectionLTR);
}

// Step 5: pull Yoga's computed layout back into flecs. Writes Position<Local>
// for non-root nodes plus the renderable width/height, so the existing render
// queue (which reads those directly) keeps working unchanged.
static void yoga_readback(flecs::entity e, YogaNode& yn)
{
    if (!yn.node) return;
    float left = YGNodeLayoutGetLeft(yn.node);
    float top  = YGNodeLayoutGetTop(yn.node);
    float w    = YGNodeLayoutGetWidth(yn.node);
    float h    = YGNodeLayoutGetHeight(yn.node);

    // Roots keep whatever Position<Local> their owner set (panel code, scroll
    // systems, ...); only non-roots are positioned by Yoga.
    if (!yoga_is_root(e)) {
        Position& p = e.ensure<Position, Local>();
        p.x = left;
        p.y = top;
    }

    // A legacy leaf owns its own size -- Yoga only placed it. Writing the
    // computed size back would fight the calibrated layout inside.
    if (e.has<UIYogaLegacyLeaf>()) return;

    if (e.has<UIElementSize>()) {
        UIElementSize& usz = e.ensure<UIElementSize>();
        usz.width = w;
        usz.height = h;
    }
    if (e.has<RectRenderable>()) {
        RectRenderable& rect = e.ensure<RectRenderable>();
        rect.width = w;  rect.height = h;
    }
    if (e.has<RoundedRectRenderable>()) {
        RoundedRectRenderable& rect = e.ensure<RoundedRectRenderable>();
        rect.width = w;  rect.height = h;
    }
    if (e.has<CustomRenderable>()) {
        CustomRenderable& custom = e.ensure<CustomRenderable>();
        custom.width = w; custom.height = h;
    }
    if (e.has<ImageRenderable>()) {
        ImageRenderable& img = e.ensure<ImageRenderable>();
        img.width = w;   img.height = h;
    }
}

// Apply steps 1-3 to a Yoga subtree, top-down.
static void yoga_prepare_subtree(flecs::entity e)
{
    if (!e.has<UIYoga>()) return;
    if (YogaNode* yn = e.try_get_mut<YogaNode>()) {
        yoga_sync_fill_parent(e);
        yoga_sync_topology(e, *yn);
        yoga_sync_style(e, *yn);
        yoga_apply_intrinsic(e, *yn);
    }
    e.children([](flecs::entity child) { yoga_prepare_subtree(child); });
}

// Apply step 5 to a Yoga subtree, top-down.
static void yoga_readback_subtree(flecs::entity e)
{
    if (!e.has<UIYoga>()) return;
    if (YogaNode* yn = e.try_get_mut<YogaNode>()) yoga_readback(e, *yn);
    e.children([](flecs::entity child) { yoga_readback_subtree(child); });
}

// Refresh UIElementBounds from Position<World> + UIElementSize for a subtree.
// Mirrors boundsCalculationSystem, but runs now instead of at OnLoad.
static void update_bounds_recursive(flecs::entity e)
{
    const Position* world_pos = e.try_get<Position, World>();
    const UIElementSize* size = e.try_get<UIElementSize>();
    if (world_pos && size) {
        UIElementBounds& bounds = e.ensure<UIElementBounds>();
        bounds.xmin = world_pos->x;
        bounds.ymin = world_pos->y;
        bounds.xmax = world_pos->x + ((size->width  > 0) ? size->width  : 0.0f);
        bounds.ymax = world_pos->y + ((size->height > 0) ? size->height : 0.0f);
    }
    e.children([](flecs::entity child) { update_bounds_recursive(child); });
}

// Lay out a Yoga subtree immediately, so UI created inside an input handler is
// correct in the very frame it appears -- position, size, and hit-test bounds --
// rather than snapping into place after the next PreFrame pass.
void yoga_layout_now(flecs::entity root)
{
    if (!root.is_valid() || !root.is_alive() || !root.has<UIYoga>()) return;

    yoga_prepare_subtree(root);
    if (YogaNode* yn = root.try_get_mut<YogaNode>()) yoga_calculate(root, *yn);
    yoga_readback_subtree(root);

    // Sizes changed, so world positions and bounds must be recomputed for the
    // subtree before anything reads them (Align, hit-testing, the render queue).
    propagate_world_positions(root);
    update_bounds_recursive(root);
    propagate_bounds_containment(root);
}

// VNC Stream

// VNC callback: Resize framebuffer
static rfbBool vnc_resize_callback(rfbClient* client) {
    std::cout << "[VNC] Resize callback - width: " << client->width
              << ", height: " << client->height
              << ", depth: " << client->format.bitsPerPixel << std::endl;

    int width = client->width;
    int height = client->height;
    int depth = client->format.bitsPerPixel;

    // Free old surface
    SDL_Surface* oldSurface = (SDL_Surface*)rfbClientGetClientData(client, (void*)VNC_SURFACE_TAG);
    if (oldSurface) {
        std::cout << "[VNC] Freeing old surface" << std::endl;
        SDL_FreeSurface(oldSurface);
    }

    // Create new surface for framebuffer
    SDL_Surface* surface = SDL_CreateRGBSurface(0, width, height, depth, 0, 0, 0, 0);
    if (!surface) {
        std::cerr << "[VNC ERROR] Failed to create surface: " << SDL_GetError() << std::endl;
        return FALSE;
    }

    std::cout << "[VNC] Created new surface: " << width << "x" << height
              << " @ " << depth << "bpp" << std::endl;

    // Store surface in client data
    rfbClientSetClientData(client, (void*)VNC_SURFACE_TAG, surface);

    // Configure framebuffer
    // Note: Don't modify client->width based on pitch - keep actual width
    // The pitch may be larger due to alignment, but width should be display width
    client->frameBuffer = (uint8_t*)surface->pixels;

    // Set pixel format
    client->format.bitsPerPixel = depth;
    client->format.redShift = surface->format->Rshift;
    client->format.greenShift = surface->format->Gshift;
    client->format.blueShift = surface->format->Bshift;
    client->format.redMax = surface->format->Rmask >> client->format.redShift;
    client->format.greenMax = surface->format->Gmask >> client->format.greenShift;
    client->format.blueMax = surface->format->Bmask >> client->format.blueShift;

    std::cout << "[VNC] Pixel format - R shift: " << client->format.redShift
              << ", G shift: " << client->format.greenShift
              << ", B shift: " << client->format.blueShift << std::endl;

    SetFormatAndEncodings(client);
    std::cout << "[VNC] Resize complete" << std::endl;

    return TRUE;
}

static void vnc_update_callback(rfbClient* client, int x, int y, int w, int h) {
    // Mark texture region for update in ECS - find the texture for this specific client
    // This callback runs on LibVNC's internal thread, so use lock-free queue
    VNCClient* vnc = (VNCClient*)rfbClientGetClientData(client, (void*)VNC_CLIENT_TAG);
    if (vnc) {
        // Push to lock-free queue instead of holding mutex during ECS query
        vnc->dirtyRectQueue->push({x, y, w, h});
        vnc->needsUpdate.store(true, std::memory_order_release);
        LOG_TRACE(LogCategory::VNC_CLIENT, "VNC callback added dirty rect: ({},{}) {}x{}", x, y, w, h);
    }
}


rfbClient* connectToTurboVNC(const char* host, int port) {
    std::cout << "[VNC] Connecting to " << host << ":" << port << std::endl;

    rfbClient* client = rfbGetClient(8, 3, 4);

    // Set callbacks
    client->MallocFrameBuffer = vnc_resize_callback;
    client->canHandleNewFBSize = TRUE;
    client->GotFrameBufferUpdate = vnc_update_callback;

    // Enable TurboVNC/TurboJPEG compression
    // client->appData.encodingsString = "tight copyrect";
    client->appData.encodingsString = "tight copyrect";
    client->appData.compressLevel = 0;
    client->appData.qualityLevel = 10;
    client->appData.enableJPEG = FALSE;

    client->serverHost = strdup(host);
    client->serverPort = port;

    std::cout << "[VNC] Initializing client..." << std::endl;
    if (!rfbInitClient(client, NULL, NULL)) {
        std::cerr << "[VNC ERROR] Failed to initialize client" << std::endl;
        return NULL;
    }

    std::cout << "[VNC] Connected successfully!" << std::endl;
    std::cout << "[VNC] Desktop: " << client->desktopName << std::endl;
    std::cout << "[VNC] Size: " << client->width << "x" << client->height << std::endl;

    return client;
}

// VNC async connection function - runs on network thread
void vnc_connect_async(VNCClient* vnc) {
    vnc->connectionState = VNCClient::CONNECTING;

    LOG_INFO(LogCategory::VNC_CLIENT, "Connecting to {}:{}...", vnc->host, vnc->port);

    rfbClient* client = rfbGetClient(8, 3, 4);
    client->MallocFrameBuffer = vnc_resize_callback;
    client->canHandleNewFBSize = TRUE;
    client->GotFrameBufferUpdate = vnc_update_callback;

    client->appData.encodingsString = "tight copyrect";
    client->appData.compressLevel = 0;
    client->appData.qualityLevel = 10;
    client->appData.enableJPEG = FALSE;

    client->serverHost = strdup(vnc->host.c_str());
    client->serverPort = vnc->port;

    if (!rfbInitClient(client, NULL, NULL)) {
        std::lock_guard<std::mutex> lock(vnc->errorMutex);
        vnc->errorMessage = "Failed to initialize VNC client";
        vnc->connectionState = VNCClient::ERROR;
        LOG_ERROR(LogCategory::VNC_CLIENT, "{}", vnc->errorMessage);
        return;
    }

    vnc->client = client;
    vnc->connected = true;
    vnc->connectionState = VNCClient::CONNECTED;

    // Store VNCClient pointer in rfbClient for callbacks
    rfbClientSetClientData(client, (void*)VNC_CLIENT_TAG, vnc);

    SDL_Surface* surface = (SDL_Surface*)rfbClientGetClientData(client, (void*)VNC_SURFACE_TAG);
    vnc->surface = surface;
    vnc->width = client->width;
    vnc->height = client->height;

    LOG_INFO(LogCategory::VNC_CLIENT, "Connected to {} - {}x{}",
        vnc->toString(), vnc->width, vnc->height);
}

// VNC message processing thread - handles all network I/O
void vnc_message_thread(VNCClientHandle vnc_handle) {
    VNCClient* vnc = vnc_handle.get();
    LOG_INFO(LogCategory::VNC_CLIENT, "VNC thread started for {}", vnc->toString());

    // First connect, then run message loop
    vnc_connect_async(vnc);

    vnc->threadRunning = true;

    // Main message loop
    while (!vnc->threadShouldStop) {
        // 1. Process pending input events (send to VNC server)
        {
            std::unique_lock<std::mutex> lock(vnc->inputQueueMutex);
            vnc->inputQueueCV.wait_for(lock, std::chrono::milliseconds(10),
                [vnc]() { return !vnc->inputQueue.empty() || vnc->threadShouldStop; });

            while (!vnc->inputQueue.empty()) {
                auto event = vnc->inputQueue.front();
                vnc->inputQueue.pop_front();
                lock.unlock();

                // Send outside lock to avoid blocking input submission
                if (event.type == InputEvent::POINTER) {
                    SendPointerEvent(vnc->client,
                        event.data.pointer.x,
                        event.data.pointer.y,
                        event.data.pointer.buttonMask);
                } else if (event.type == InputEvent::KEY) {
                    SendKeyEvent(vnc->client,
                        event.data.key.keysym,
                        event.data.key.down);
                }

                lock.lock();
            }
        }

        // 2. Process VNC server messages (non-blocking check)
        if (vnc->connected && vnc->client) {
            // WaitForMessage with 0 timeout = non-blocking check
            while (WaitForMessage(vnc->client, 0) > 0) {
                if (!HandleRFBServerMessage(vnc->client)) {
                    LOG_ERROR(LogCategory::VNC_CLIENT, "VNC message handling failed for {}",
                        vnc->toString());
                    vnc->connected = false;
                    vnc->connectionState = VNCClient::DISCONNECTING;
                    break;
                }
            }
        } else {
            // No active connection, sleep to avoid busy-wait
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 3. Check if connection is still alive
        if (vnc->connected && vnc->client && vnc->client->sock < 0) {
            LOG_WARN(LogCategory::VNC_CLIENT, "VNC socket closed for {}", vnc->toString());
            vnc->connected = false;
            vnc->connectionState = VNCClient::ERROR;
        }
    }

    vnc->threadRunning = false;
    LOG_INFO(LogCategory::VNC_CLIENT, "VNC thread stopped for {}", vnc->toString());
}

// End VNC Stream

#include <algorithm>

// Helper to convert hex RGBA (0xRRGGBBAA) to NVGcolor
NVGcolor nvgRGBA8(uint32_t hex) {
    return nvgRGBA(
        (uint8_t)(hex >> 24), 
        (uint8_t)(hex >> 16), 
        (uint8_t)(hex >> 8), 
        (uint8_t)(hex)
    );
}

// Helper to scale a hex color (RGBA) by a factor
uint32_t scale_color(uint32_t hex, float factor, uint8_t override_alpha = 0) {
    uint8_t r = (uint8_t)((hex >> 24) & 0xFF);
    uint8_t g = (uint8_t)((hex >> 16) & 0xFF);
    uint8_t b = (uint8_t)((hex >> 8) & 0xFF);
    uint8_t a = (override_alpha > 0) ? override_alpha : (uint8_t)(hex & 0xFF);

    auto clamp = [](float val) { 
        return (uint8_t)std::min(255.0f, std::max(0.0f, val)); 
    };

    return (clamp(r * factor) << 24) | 
           (clamp(g * factor) << 16) | 
           (clamp(b * factor) << 8)  | 
           a;
}

inline NVGcolor uintToNvgColor(uint32_t hex) {
    return nvgRGBA(
        (uint8_t)((hex >> 24) & 0xFF), // Red
        (uint8_t)((hex >> 16) & 0xFF), // Green
        (uint8_t)((hex >> 8)  & 0xFF), // Blue
        (uint8_t)(hex & 0xFF)         // Alpha
    );
}

flecs::entity create_popup(flecs::entity parent)
{
    flecs::entity UIElement = world->lookup("UIElement");

    // UIElementSiz parent_size = parent.try_get<UIElementSize>();

    flecs::entity ui_popup = world->entity()
        .is_a(UIElement)
        .child_of(parent)
        // TODO: We should add a tag to indicate that we don't want to bubble up bounds...
        .set<Align>({0.0f, 0.0f, 0.0f, 1.0f})
        .set<RectRenderable>({64.0f, 96.0f, true, 0xFFFFFFFF})
        .set<ZIndex>({100});

    return ui_popup;
}

struct RenderCommand {
    Position pos;
    std::variant<RoundedRectRenderable, RectRenderable, TextRenderable, ImageRenderable, LineRenderable, QuadraticBezierRenderable, CustomRenderable> renderData;
    RenderType type;
    int zIndex;

    ecs_entity_t scissorEntity;
    
    bool useGradient;
    RenderGradient gradient;

    // Hierarchy depth, as a tiebreaker within a layer. Nested UI draws above
    // what contains it without every widget having to hand-allocate itself a
    // higher ZIndex than its parent -- which does not scale, since a badge's
    // layers are fixed constants and nesting can go arbitrarily deep.
    int depth = 0;

    bool operator<(const RenderCommand& other) const {
        if (zIndex != other.zIndex) return zIndex < other.zIndex;
        return depth < other.depth;
    }
};


// Whether `text` would render wider than `width` on one line -- i.e. whether a
// badge holding it is about to become a tall block rather than a wide one.
static bool badge_text_exceeds(const std::string& text, float width)
{
    if (!g_yoga_vg || text.empty()) return false;
    nvgFontSize(g_yoga_vg, 16.0f);
    nvgFontFace(g_yoga_vg, "CharisSIL");
    return measureText(g_yoga_vg, text, 0.0f).w > width;
}

// Longest prefix of `text` whose rendered width fits `max_width`, with an
// ellipsis appended when anything was dropped.
//
// A badge is as wide as its label, so a label with no ceiling makes a badge with
// no ceiling -- one long entity name pushes everything beside it off the panel.
// Callers that can afford the vertical space use create_longform_badge instead,
// which wraps rather than truncating; this is for the single-line contexts, a
// list row above all, where the row pitch is fixed and wrapping is not an option.
static std::string ellipsize_text(const std::string& text, const char* font_face,
                                  float font_size, float max_width)
{
    if (!g_yoga_vg || max_width <= 0.0f || text.empty()) return text;

    nvgFontSize(g_yoga_vg, font_size);
    nvgFontFace(g_yoga_vg, font_face);

    if (measureText(g_yoga_vg, text, 0.0f).w <= max_width) return text;

    const std::string suffix = "...";
    auto fits = [&](size_t n) {
        return measureText(g_yoga_vg, text.substr(0, n) + suffix, 0.0f).w <= max_width;
    };

    // Largest prefix length that still fits once the ellipsis is added.
    // Binary search rather than a character walk, so a pasted paragraph costs
    // log(n) measurements instead of n.
    size_t lo = 0, hi = text.size();          // lo always fits, hi may not
    while (lo < hi) {
        size_t mid = lo + (hi - lo + 1) / 2;
        // Never cut inside a UTF-8 sequence.
        while (mid > lo && (static_cast<unsigned char>(text[mid]) & 0xC0) == 0x80) mid--;
        if (mid == lo) break;                 // no boundary left to test
        if (fits(mid)) lo = mid; else hi = mid - 1;
    }

    return text.substr(0, lo) + suffix;
}

// Calibrated badge height. A badge that nests another grows by NEST_PAD on the
// top and bottom, so the inner block reads as contained rather than as exactly
// filling its parent.
static constexpr float BADGE_HEIGHT = 25.0f;

// Alpha for badge sprites. The sprite's black ground should read as a
// semitransparent darkening of the badge fill underneath -- the surface
// showing through is what makes the glyph look inset rather than pasted.
// Full alpha lands it as a harsh opaque black square.
static constexpr unsigned char BADGE_SPRITE_ALPHA = 0xC6;

// Extra opacity for sprites on bright badges, scaled by the tint's luminance.
// A fixed alpha reads weaker on light fills -- the brighter the surface, the
// more visible the bleed-through -- so Wesley's cyan and Heonae's pink need a
// darker ground than a navy badge to look equally inset.
static constexpr float BADGE_SPRITE_ALPHA_BOOST = 0x32;
static constexpr float BADGE_NEST_PAD = 4.0f;

// Space between a relation's label and the block nested beside it.
static constexpr float BADGE_NEST_GAP = 6.0f;

// The reification marker rides above the predicate, so it is drawn smaller
// than the source and target glyphs flanking it.
static constexpr float TRIPLE_REIFIED_SCALE = 0.6f;

// Height the reified node used to occupy in-flow. Kept as top padding when the
// node moves out of flow to straddle the border, so the block's size increase
// on reification is unchanged.
static constexpr float TRIPLE_NODE_HEIGHT = 15.0f;

// Horizontal padding a badge's content row reserves (xPad on each side).
static constexpr float BADGE_LONGFORM_INSET = 12.0f;

// Width of each arrowhead on a relationship badge. Shared by the geometry below
// and by the horizontal padding create_badge_impl reserves, so a label can never
// run into a tip.
static constexpr float BADGE_ARROW_TIP = 12.0f;

// Ceiling on a badge label's rendered width. Past this the label is ellipsized,
// so no badge can grow without bound and shove its neighbours off the panel.
static constexpr float BADGE_MAX_TEXT_WIDTH = 190.0f;

// The comprehension frame's shape: a trapezoid, top edge inset so the sides
// slope outward -- a pedestal the members stand in, star hovering above the
// narrow top. Sharp corners on purpose: every other container is rounded, so
// the silhouette alone says "query". Stroke mode runs the member gradient
// horizontally, first member's colour to last's; fill mode is the dark ground.
static constexpr float CONCEPT_TRAPEZOID_INSET = 12.0f;

// How far the star+id stack rises above the frame's top edge (39px star +
// ~25px digit, less the overlap that keeps its foot on the border).
static constexpr float CONCEPT_MARKER_RISE = 29.0f;

void draw_concept_trapezoid(NVGcontext* vg, const RenderCommand* cmd, const CustomRenderable& data)
{
    const float x = cmd->pos.x;
    const float y = cmd->pos.y;
    const float w = data.width;
    const float h = data.height;
    if (w <= 0.0f || h <= 0.0f) return;

    const float inset = std::min(CONCEPT_TRAPEZOID_INSET, w * 0.25f);

    nvgBeginPath(vg);
    nvgMoveTo(vg, x + inset, y);
    nvgLineTo(vg, x + w - inset, y);
    nvgLineTo(vg, x + w, y + h);
    nvgLineTo(vg, x, y + h);
    nvgClosePath(vg);

    if (data.stroke) {
        uint32_t start = data.gradient_start ? data.gradient_start : data.color;
        uint32_t end   = data.gradient_end   ? data.gradient_end   : data.color;
        nvgStrokePaint(vg, nvgLinearGradient(vg, x, y, x + w, y,
                                             uintToNvgColor(start), uintToNvgColor(end)));
        nvgStrokeWidth(vg, 1.5f);
        nvgStroke(vg);
    } else {
        nvgFillColor(vg, uintToNvgColor(data.color));
        nvgFill(vg);
    }
}

void draw_double_arrow(NVGcontext* vg, const RenderCommand* cmd, const CustomRenderable& data)
{
    float x = cmd->pos.x;
    float y = cmd->pos.y;

    float midY = y + data.height / 2.0f;

    // A triangle at each end, both pointing outward -- so the shape reads as
    // relating what is on its left to what is on its right, symmetrically.
    //
    // The previous form notched the left end *inward*, which made it a flag:
    // one point, one bitten-out tail, and a direction it did not actually mean.
    // It also drew half a height to the left of x, outside the box Yoga
    // measured; both tips now sit inside [x, x + width], so the drawn shape and
    // the laid-out shape are the same rectangle.
    //
    // The tip width is a constant rather than a fraction of height, so a block
    // that grows to hold a nested badge keeps the same arrowheads instead of
    // sprouting larger ones. Clamped so a very narrow badge degenerates to a
    // plain diamond rather than turning itself inside out.
    float tip = std::min(BADGE_ARROW_TIP, data.width * 0.5f);
    float bodyLeft = x + tip;
    float bodyRight = x + data.width - tip;

    nvgBeginPath(vg);

    nvgMoveTo(vg, x, midY);                        // left tip
    nvgLineTo(vg, bodyLeft, y);
    nvgLineTo(vg, bodyRight, y);
    nvgLineTo(vg, x + data.width, midY);           // right tip
    nvgLineTo(vg, bodyRight, y + data.height);
    nvgLineTo(vg, bodyLeft, y + data.height);

    nvgClosePath(vg);

    if (data.stroke) {
        // Stroke with outline color
        unsigned char r = (data.color >> 24) & 0xFF;
        unsigned char g = (data.color >> 16) & 0xFF;
        unsigned char b = (data.color >> 8) & 0xFF;
        unsigned char a = (data.color) & 0xFF;
        nvgStrokeColor(vg, nvgRGBA(r, g, b, a));
        nvgStrokeWidth(vg, 1.0f);
        nvgStroke(vg);
    } else {
        // Fill with horizontal gradient if gradient colors are set
        if (data.gradient_start != 0 || data.gradient_end != 0) {
            uint32_t start = data.gradient_start ? data.gradient_start : data.color;
            uint32_t end = data.gradient_end ? data.gradient_end : data.color;
            NVGcolor startColor = nvgRGBA((start >> 24) & 0xFF, (start >> 16) & 0xFF, (start >> 8) & 0xFF, start & 0xFF);
            NVGcolor endColor = nvgRGBA((end >> 24) & 0xFF, (end >> 16) & 0xFF, (end >> 8) & 0xFF, end & 0xFF);
            NVGpaint gradient = nvgLinearGradient(vg, x, midY, x + data.width, midY, startColor, endColor);
            nvgFillPaint(vg, gradient);
        } else {
            unsigned char r = (data.color >> 24) & 0xFF;
            unsigned char g = (data.color >> 16) & 0xFF;
            unsigned char b = (data.color >> 8) & 0xFF;
            unsigned char a = (data.color) & 0xFF;
            nvgFillColor(vg, nvgRGBA(r, g, b, a));
        }
        nvgFill(vg);
    }
}

// `yoga` opts the sprite into the Yoga tree, pinned to its native pixel size.
// Callers still on the legacy layout (create_triple_block) leave it off, since
// a Yoga node inside a LayoutBox parent would be sized by a different pass.
//
// `preserve_case` keeps a lowercase glyph lowercase. Off by default because
// binding symbols are written lowercase but have always drawn as capitals; the
// sprite-font path turns it on, since there a string's own casing is the point.
flecs::entity create_slot_image(flecs::entity parent_entity, const std::string& symbol, uint32_t tint, bool yoga = false, bool preserve_case = false, float scale = 0.9f) {
    auto UIElement = world->lookup("UIElement");
    uint32_t tint_color = scale_color(tint, 1.3f);
    unsigned char r = (tint_color >> 24) & 0xFF;
    unsigned char g = (tint_color >> 16) & 0xFF;
    unsigned char b = (tint_color >> 8) & 0xFF;
    // The alpha multiplies the whole sprite in the image pattern, ground
    // included -- this is where the overlay stops being opaque black. Scaled
    // up with the tint's luminance so light badges get an equally dark inset.
    float lum = (0.299f * ((tint >> 24) & 0xFF)
               + 0.587f * ((tint >> 16) & 0xFF)
               + 0.114f * ((tint >> 8) & 0xFF)) / 255.0f;
    unsigned char a = (unsigned char)std::min(248.0f,
        BADGE_SPRITE_ALPHA + lum * BADGE_SPRITE_ALPHA_BOOST);

    std::string image_path;
    if (symbol.empty() || symbol == "*") {
        // Unbound, or explicitly a wildcard -- both mean "any entity", and an
        // unbound slot is the normal state of a relationship the moment it is
        // created. Without this the path becomes "letter_sets/set_01/.png",
        // which fails to load and leaves the slot invisible rather than showing
        // that there is something here to bind.
        image_path = "wildcard.png";
    }
    else if (symbol == "checkbox")
    {
        // TODO: Bind to boolean state
        image_path == "checkbox_true.png";
    } else if (symbol.length() == 1 && std::isdigit(static_cast<unsigned char>(symbol[0]))) {
        // Standard MNIST digit
        image_path = "mnist/set_0/" + symbol + ".png";
    } else {
        // Non-digit - letter_set sprite. The sheets ship both cases, so
        // preserve_case just decides which file to reach for.
        std::string letter = symbol;
        if (!preserve_case) {
            for (auto& c : letter) c = std::toupper(static_cast<unsigned char>(c));
        }
        image_path = "letter_sets/set_01/" + letter + ".png";
    }

    flecs::entity slot = world->entity()
        .is_a(UIElement)
        .child_of(parent_entity)
        .set<ImageCreator>({image_path, scale, scale, nvgRGBA(r, g, b, a)})
        .set<ZIndex>({25});

    if (yoga) {
        // UINativeImageSize reads back through ImageRenderable, which the
        // ImageCreator observer above has already produced -- it runs
        // synchronously on set, so the sprite has a size before Yoga measures.
        slot.add<UIYoga>().add<UINativeImageSize>();
    }

    return slot;
};

// A single character as a chip: the badge recipe at glyph scale. Vertical
// gradient in the given colour with the sprite on top, so the sprite's own
// black ground reads as a darker inset region of the chip -- the depth the
// entity badges get for free -- instead of vanishing into a dark panel.
flecs::entity create_letter_chip(flecs::entity parent, const std::string& symbol, uint32_t color)
{
    auto UIElement = world->lookup("UIElement");
    auto chip = world->entity()
        .is_a(UIElement)
        .child_of(parent)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyCenter, YGAlignCenter, 0.0f})
        .set<UIPadding>({2.0f, 2.0f, 2.0f, 2.0f})
        .set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, false, 0x000000FF})
        .set<RenderGradient>({color, scale_color(color, 0.2f)})
        .set<ZIndex>({11});

    // EMNIST has no punctuation classes, so those glyphs are hand-drawn --
    // assets/punctuation, made with scripts/draw_punctuation.py against the
    // family's own sprites as reference. The name table is the contract with
    // that script: filenames are names because '?' cannot be one. Marks
    // without a sprite fall back to a typeset glyph rather than vanishing --
    // the apostrophe in a genitive is a morpheme, and losing it is data loss.
    if (!symbol.empty() && !std::isalnum(static_cast<unsigned char>(symbol[0]))) {
        static const std::unordered_map<char, const char*> punct_names = {
            {'\'', "apostrophe"}, {'-', "hyphen"},    {'.', "period"},
            {',', "comma"},       {'?', "question"},  {'!', "exclaim"},
            {':', "colon"},       {';', "semicolon"}, {'"', "quote"},
            {'(', "lparen"},      {')', "rparen"},    {'&', "ampersand"},
            {'/', "slash"},       {'@', "at"},        {'#', "hash"},
            {'%', "percent"},     {'+', "plus"},      {'=', "equals"},
            {'_', "underscore"},  {'~', "tilde"},
        };
        auto named = punct_names.find(symbol[0]);
        if (named != punct_names.end()) {
            std::string rel = std::string("punctuation/set_01/") + named->second + ".png";
            // Checked once per mark: a missing sprite means the set is
            // incomplete, and the typeset fallback below covers it.
            static std::unordered_map<char, bool> sprite_exists;
            auto cached = sprite_exists.find(symbol[0]);
            bool exists = (cached != sprite_exists.end())
                ? cached->second
                : (sprite_exists[symbol[0]] =
                       std::filesystem::exists("../assets/" + rel));
            if (exists) {
                uint32_t tinted = scale_color(color, 1.3f);
                world->entity()
                    .is_a(UIElement)
                    .child_of(chip)
                    .add<UIYoga>()
                    .add<UINativeImageSize>()
                    .set<ImageCreator>({rel, 0.9f, 0.9f,
                                        nvgRGBA((tinted >> 24) & 0xFF,
                                                (tinted >> 16) & 0xFF,
                                                (tinted >> 8) & 0xFF,
                                                BADGE_SPRITE_ALPHA)})
                    .set<ZIndex>({25});
                return chip;
            }
        }

        world->entity()
            .is_a(UIElement)
            .child_of(chip)
            .add<UIYoga>()
            .set<UISize>({14.0f, YGUndefined})
            .set<TextRenderable>({symbol, "CharisSIL", 18.0f, 0xFFFFFFFF})
            .set<ZIndex>({25});
        return chip;
    }

    create_slot_image(chip, symbol, color, true, true);
    return chip;
}

// Renders a string as a run of sprite-sheet glyphs -- a sprite font, one image
// entity per character, flowed by whatever row the caller passes as parent.
//
// This is what lets a symbol be a *string* rather than a single glyph: with more
// entities than there are letters, ids run to "AA", "AB" and so on, and each
// character has to come off the same sheet the one-letter ids do. Characters
// with no sprite (spaces, punctuation) are skipped rather than drawn as a
// missing-texture box.
void create_sprite_text(flecs::entity parent, const std::string& text, uint32_t tint, bool yoga = true)
{
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isspace(uc)) continue;
        create_letter_chip(parent, std::string(1, c), tint);
    }
}


// What a badge is standing for, which decides how it is drawn.
//
// Symbol is the default and the common case: the badge names something -- a
// word, a compound-word type, an instance of a physical object -- and reads as
// a solid token, filled with its identifying gradient.
//
// String is for a badge whose content *is* the text: a message, a name someone
// typed, a free-text value. Drawn as an outline rather than a filled gradient,
// because there is no symbol here to give a colour identity to; the colour is a
// border around content, not a fill standing in for a thing.
enum class BadgeStyle {
    Symbol,
    String,
};

// Makes a badge wrap its label at `width` instead of running on in one line.
// Shared by the automatic case (a label past the ceiling) and the deliberate
// one (create_longform_badge), so both produce the same shape.
static void badge_apply_wrap(flecs::entity badge, flecs::entity content,
                             flecs::entity label, float width)
{
    if (TextRenderable* text = label.try_get_mut<TextRenderable>()) {
        // Both Yoga's measure callback and the renderer key off wrapWidth.
        text->wrapWidth = width;
    }
    // Height now follows the line count rather than the calibrated single line.
    badge.ensure<UISize>().h = YGUndefined;
    badge.set<UIPadding>({BADGE_NEST_PAD, 0.0f, BADGE_NEST_PAD, 0.0f});
    // The single-line offset does not apply to wrapped text.
    if (UIMargin* margin = label.try_get_mut<UIMargin>()) margin->top = 0.0f;
    if (UIFlexContainer* row = content.try_get_mut<UIFlexContainer>()) {
        row->items = YGAlignCenter;
    }
}

static void badge_wrap_if_wide(flecs::entity badge, flecs::entity content,
                               flecs::entity label, const std::string& text, float width)
{
    if (!g_yoga_vg || text.empty()) return;
    nvgFontSize(g_yoga_vg, 16.0f);
    nvgFontFace(g_yoga_vg, "CharisSIL");
    if (measureText(g_yoga_vg, text, 0.0f).w <= width) return;
    badge_apply_wrap(badge, content, label, width);
}

// Forward declaration for vector-based version.
//
// `out_content` receives the badge's inner row. Handed back directly rather than
// left to be read off the BadgeContent component, because a caller that wants to
// nest something needs the row in the same breath as the badge -- and reading a
// component straight back after setting it only works while the world is not
// deferred, which is not a condition a widget factory should impose on callers.
flecs::entity create_badge_impl(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule, bool is_double_arrow,
                           const std::vector<std::string>& prefix_ids,
                           const std::vector<uint32_t>& prefix_tints,
                           const std::vector<std::string>& postfix_ids,
                           const std::vector<uint32_t>& postfix_tints,
                           flecs::entity* out_content = nullptr,
                           BadgeStyle style = BadgeStyle::Symbol);

// A relationship token: source glyph, predicate, target glyph inside one arrow
// block -- and, when the relationship is reified, a 3d-node marker and the
// reified entity's own glyph stacked above the predicate.
//
// Rebuilt on the badge path rather than the legacy LayoutBox/UIContainer
// pipeline it used to use. That pipeline runs at PostLoad, a phase *after*
// Yoga's, so the block was placed from a size computed on the previous frame --
// and on the frame it first appeared there was no previous size at all, leaving
// it at its 100x25 placeholder wherever stale layout had put it.
//
// The reification marker is the point of the widget: it is how a relationship
// becomes an entity in its own right, so it has to be visible on the
// relationship rather than hidden behind a selection.
// `joined_src` / `joined_tgt` are entity tokens adjacent to this relationship
// in the sentence whose symbols match its slots. Their full badges render
// *inside* the block -- the entity visually joins the relationship -- instead
// of a standalone badge next to a sprite repeating the same symbol.
flecs::entity create_triple_block(flecs::entity parent, const SentenceToken& token,
                                  flecs::entity* out_content = nullptr,
                                  const SentenceToken* joined_src = nullptr,
                                  const SentenceToken* joined_tgt = nullptr) {
    auto UIElement = world->lookup("UIElement");

    // The bridge takes a colour from both ends, since it belongs to neither.
    uint32_t src_color = get_entity_color(token.source_symbol, "");
    uint32_t tgt_color = get_entity_color(token.target_symbol, "");
    uint32_t bridge_color = (scale_color(src_color, 0.5f) & 0xFFFFFF00)
                          | (scale_color(tgt_color, 0.5f) & 0xFFFFFF00) | 0xFF;

    // Outlined: a relationship frames the two things it relates, and a filled
    // gradient would compete with the glyphs sitting inside it.
    flecs::entity content = flecs::entity::null();
    flecs::entity block = create_badge_impl(parent, UIElement, "", bridge_color,
                                            false, true, {}, {}, {}, {},
                                            &content, BadgeStyle::String);
    if (!content.is_valid()) return block;

    // create_badge_impl seeds the row with its own label; this widget lays out
    // its three parts itself, so the placeholder goes.
    std::vector<flecs::entity> seeded;
    content.children([&seeded](flecs::entity child) { seeded.push_back(child); });
    for (flecs::entity child : seeded) child.destruct();

    // A full set, not ensure-and-mutate. Under deferral, ensure() on a
    // component whose own set is still queued hands back a default-constructed
    // payload -- and UIFlexContainer's default direction is Column, which is
    // how a relationship block ends up stacking its parts vertically. A
    // complete set states every field, so whichever order commands land in,
    // the final value is this one.
    content.set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                  YGJustifyFlexStart, YGAlignCenter, 2.0f});

    if (joined_src) {
        create_badge_impl(content, UIElement, joined_src->text.c_str(), src_color,
                          false, false, {}, {},
                          {joined_src->binding_symbol}, {src_color});
    } else {
        create_slot_image(content, token.source_symbol, src_color, true);
    }

    // A reified block has two storeys -- the marker row above the predicate --
    // so its height must come from the content rather than the calibrated
    // single-line 25px, or the marker draws outside the arrow. The vertical
    // padding keeps the top storey off the arrow's edge.
    if (!token.reified_symbol.empty()) {
        block.ensure<UISize>().h = YGUndefined;
        // The node no longer sits in the centre column's flow, so its height
        // is held open as padding -- same total growth as before, with the
        // upper storey now empty space the straddling node overlaps.
        block.set<UIPadding>({BADGE_NEST_PAD + TRIPLE_NODE_HEIGHT, 0.0f,
                              BADGE_NEST_PAD, 0.0f});

        // The src/tgt sprites keep the digit-and-predicate row's baseline
        // rather than centring against both storeys -- only the 3d-node lives
        // on the upper storey, so centring would float the sprites between
        // lines that don't exist.
        content.set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                      YGJustifyFlexStart, YGAlignFlexEnd, 2.0f});
    }

    // The centre is always a column, reified or not. Collapsing to a plain row
    // when there is no marker would move the predicate every time reification
    // is toggled, and the slot the marker occupies has to already be there for
    // the block's geometry to stay put.
    flecs::entity centre = world->entity()
        .is_a(UIElement)
        .child_of(content)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                               YGJustifyCenter, YGAlignCenter, 0.0f});

    // Reified: the 3d-node rides alone above, and the assigned MNIST digit sits
    // inline in front of the predicate -- the digit reads as part of naming the
    // relationship, the node as the mark that it has become a thing.
    flecs::entity label_parent = centre;
    if (!token.reified_symbol.empty()) {
        // Tinted with the reified entity's own colour when it has one, so the
        // marker reads as that entity rather than as decoration.
        uint32_t reified_color = 0xFFFFFFFF;
        auto cached = entity_color_cache.find(token.reified_symbol);
        if (cached != entity_color_cache.end()) reified_color = cached->second;

        // Straddles the block's upper line at its horizontal midpoint: an
        // absolute strip pinned across the top edge, shifted up by half the
        // node's height, centring the node within it. Absolute, so the block's
        // measured size never includes it -- the growth is all padding.
        auto node_strip = world->entity()
            .is_a(UIElement)
            .child_of(block)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIAbsoluteEdges>({0.0f, -TRIPLE_NODE_HEIGHT * 0.5f, 0.0f, YGUndefined})
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                   YGJustifyCenter, YGAlignFlexStart, 0.0f});

        world->entity()
            .is_a(UIElement)
            .child_of(node_strip)
            .add<UIYoga>()
            .add<UINativeImageSize>()
            .set<ImageCreator>({"3d_node.png", TRIPLE_REIFIED_SCALE, TRIPLE_REIFIED_SCALE,
                                uintToNvgColor(reified_color)})
            .set<ZIndex>({26});

        label_parent = world->entity()
            .is_a(UIElement)
            .child_of(centre)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                   YGJustifyCenter, YGAlignCenter, 2.0f});

        // Full size, unlike the node above it: the digit is the entity's
        // glyph, the same rank as the src/tgt sprites beside it.
        create_slot_image(label_parent, token.reified_symbol, reified_color, true);
    }

    // Same treatment a badge gives its own label: the 1.2 y-scale and the
    // white-to-tint gradient. The predicate is a label like any other, and
    // without these it reads as flatter, smaller text sitting inside a badge
    // rather than as the badge's own word.
    flecs::entity label = world->entity()
        .is_a(UIElement)
        .child_of(label_parent)
        .add<UIYoga>()
        .set<TextRenderable>({token.text, "CharisSIL", 16.0f, 0xFFFFFFFF, 1.2f})
        .set<RenderGradient>({0xFFFFFFFF, scale_color(bridge_color, 1.3f)})
        .set<ZIndex>({25});

    if (joined_tgt) {
        create_badge_impl(content, UIElement, joined_tgt->text.c_str(), tgt_color,
                          false, false, {}, {},
                          {joined_tgt->binding_symbol}, {tgt_color});
    } else {
        create_slot_image(content, token.target_symbol, tgt_color, true);
    }

    block.set<BadgeContent>({content, label});

    // The whole reason the caller passes this in: the selection map is built
    // from the content row's children. create_badge_impl wrote the *seeded* row
    // into it earlier in this function, but the row has been rebuilt since --
    // handing back anything less than the final row leaves the caller's
    // selection slots falling back to the block, which outlines the full
    // triple at every index.
    if (out_content) *out_content = content;
    return block;
}

uint32_t entity_type_color(const std::string& name);
static void badge_becomes_container(flecs::entity block, flecs::entity slot, bool stacked);

// Opens a comprehension frame: an outlined block whose slot the member badges
// and relationship blocks render into, prefixed by the mag-2 star at native
// size. The frame contains its members Scratch-style -- they keep their own
// bindings and renderings; only their parent changes.
flecs::entity create_concept_frame(flecs::entity parent, const std::string& symbol,
                                   const std::string& ext_symbol,
                                   flecs::entity* out_slot,
                                   flecs::entity* out_outline,
                                   flecs::entity* out_ext_node)
{
    auto UIElement = world->lookup("UIElement");
    // The symbol's own colour, same lookup the slot glyphs use -- a relationship
    // slot bound to this concept's digit tints identically to the frame, so the
    // reference reads by colour before the arc is even drawn.
    uint32_t color = get_entity_color(symbol, "");

    flecs::entity slot = flecs::entity::null();
    flecs::entity frame = create_badge_impl(parent, UIElement, "", color,
                                            false, false, {}, {}, {}, {},
                                            &slot, BadgeStyle::String);
    if (!slot.is_valid()) { if (out_slot) *out_slot = slot; return frame; }

    std::vector<flecs::entity> seeded;
    slot.children([&seeded](flecs::entity child) { seeded.push_back(child); });
    for (flecs::entity child : seeded) child.destruct();

    // Members can be taller than a single badge line (a reified triple), so
    // the frame derives its height from what it holds.
    badge_becomes_container(frame, slot, false);

    // The rounded fill the badge factory supplied doesn't match a trapezoid
    // outline; the body becomes the trapezoid's own fill.
    frame.remove<RoundedRectRenderable>();
    frame.set<CustomRenderable>({0.0f, 0.0f, false, 0x141414E0, 0, 0,
                                 draw_concept_trapezoid});

    // Star and id stacked above the narrow top, centred at the midpoint, the
    // stack's foot overlapping the top edge -- same straddle convention as the
    // reified node. Absolute, so the members' layout never accounts for it.
    auto marker = world->entity()
        .is_a(UIElement)
        .child_of(frame)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIAbsoluteEdges>({0.0f, -CONCEPT_MARKER_RISE, 0.0f, YGUndefined})
        // One line, reading order: the intension pair first, the extension
        // pair to its right -- sense before reference.
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyCenter, YGAlignCenter, 8.0f});

    // Intension pair, leftmost: the 3d-node marks entity-hood -- the query
    // itself as a thing, same mark reification gives a triple -- tinted with
    // the concept's identity colour, beside the intension's id.
    auto intension_row = world->entity()
        .is_a(UIElement)
        .child_of(marker)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyCenter, YGAlignCenter, 2.0f});

    world->entity()
        .is_a(UIElement)
        .child_of(intension_row)
        .add<UIYoga>()
        .add<UINativeImageSize>()
        .set<ImageCreator>({"3d_node.png", TRIPLE_REIFIED_SCALE, TRIPLE_REIFIED_SCALE,
                            uintToNvgColor(scale_color(color, 1.3f))})
        .set<ZIndex>({26});

    create_slot_image(intension_row, symbol, color, true);

    // Extension pair, to its right: the star is the reference -- it shines
    // with what falls under the concept in the world. Tint arrives at the
    // closing marker as the composite of the member colours (the set is made
    // of what matches them), and magnitude can later carry cardinality.
    if (!ext_symbol.empty()) {
        auto extension_row = world->entity()
            .is_a(UIElement)
            .child_of(marker)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                   YGJustifyCenter, YGAlignCenter, 2.0f});

        // Star and id share the extension's own colour -- they are one
        // referent, and matching them is what lets a slot bound to this digit
        // read back to this star by colour alone.
        uint32_t ext_color = get_entity_color(ext_symbol, "");
        flecs::entity star = world->entity()
            .is_a(UIElement)
            .child_of(extension_row)
            .add<UIYoga>()
            .add<UINativeImageSize>()
            .set<ImageCreator>({"stellar/mag_2.PNG", 1.0f, 1.0f,
                                uintToNvgColor(scale_color(ext_color, 1.3f))})
            .set<ZIndex>({26});
        if (out_ext_node) *out_ext_node = star;

        create_slot_image(extension_row, ext_symbol, ext_color, true);
    }

    // Swap the stock flat outline for the gradient stroke. The member colours
    // aren't known yet -- the span renders after this returns -- so the caller
    // fills in the gradient stops at the closing marker.
    frame.children([&](flecs::entity child) {
        if (!child.has<UIAbsoluteEdges>() || !child.has<RoundedRectRenderable>()) return;
        uint32_t base = (scale_color(color, 1.3f) & 0xFFFFFF00) | 0xC0;
        child.remove<RoundedRectRenderable>();
        child.set<CustomRenderable>({0.0f, 0.0f, true, base, 0, 0, draw_concept_trapezoid});
        if (out_outline) *out_outline = child;
    });

    if (out_slot) *out_slot = slot;
    return frame;
}

// Backward-compatible overload with single prefix/postfix
flecs::entity create_badge(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule = false, bool is_double_arrow = false,
                           std::string postfix_symbol = "", std::string prefix_symbol = "",
                           uint32_t prefix_tint = 0, uint32_t postfix_tint = 0) {
    std::vector<std::string> prefix_ids;
    std::vector<uint32_t> prefix_tints;
    std::vector<std::string> postfix_ids;
    std::vector<uint32_t> postfix_tints;

    if (!prefix_symbol.empty()) {
        prefix_ids.push_back(prefix_symbol);
        prefix_tints.push_back(prefix_tint);
    }
    if (!postfix_symbol.empty()) {
        postfix_ids.push_back(postfix_symbol);
        postfix_tints.push_back(postfix_tint);
    }

    return create_badge_impl(parent, UIElement, text, base_color, is_capsule, is_double_arrow,
                             prefix_ids, prefix_tints, postfix_ids, postfix_tints);
}

// Vector-based version for sets of sources/targets
flecs::entity create_badge(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule, bool is_double_arrow,
                           const std::vector<std::string>& prefix_ids,
                           const std::vector<uint32_t>& prefix_tints,
                           const std::vector<std::string>& postfix_ids,
                           const std::vector<uint32_t>& postfix_tints) {
    return create_badge_impl(parent, UIElement, text, base_color, is_capsule, is_double_arrow,
                             prefix_ids, prefix_tints, postfix_ids, postfix_tints);
}

flecs::entity create_badge_impl(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule, bool is_double_arrow,
                           const std::vector<std::string>& prefix_ids,
                           const std::vector<uint32_t>& prefix_tints,
                           const std::vector<std::string>& postfix_ids,
                           const std::vector<uint32_t>& postfix_tints,
                           flecs::entity* out_content,
                           BadgeStyle style) {

    const bool outlined = (style == BadgeStyle::String);

    // --- 1. Color Logic (matching comp_gen.py) ---
    uint32_t dark = base_color;
    uint32_t very_dark = scale_color(base_color, 0.2f);
    uint32_t light = scale_color(base_color, 1.3f); // Clamped to 255 in scale_color
    uint32_t white = 0xFFFFFFFF;

    // Outline: Light variation with 50% alpha (128)
    uint32_t outline_color = (light & 0xFFFFFF00) | 0x80;

    // --- 2. Dimensions & Shape ---
    float corner_radius = 4.0f;
    float badge_height = BADGE_HEIGHT;

    // Distance from the badge's top edge to the top of the text's line box.
    // Calibrated by eye against the sprites, which sit at y=0.
    constexpr float BADGE_TEXT_TOP = 6.0f;

    if (is_capsule) {
        corner_radius = badge_height / 2.0f;
    }

    // --- 3. Create Entities ---

    float xPad = 6.0f;
    if (is_double_arrow) xPad += BADGE_ARROW_TIP;   // room for the arrow tips

    // The badge is a Yoga flex row all the way down, not a UIYogaLegacyLeaf.
    //
    // The legacy pass runs at PostLoad, a phase *after* Yoga's PreFrame pass, so
    // a legacy-sized badge was always placed using the size computed on the
    // previous frame. A badge built this frame -- a list row scrolling in --
    // therefore rendered its sprites at a stale offset for one frame, which
    // read as the sprite popping in at the left edge and then jumping into
    // place. Sizing and placing the whole subtree in one PreFrame pass removes
    // the lag entirely.
    //
    // Height stays pinned to the calibrated 25px; width is left auto so Yoga
    // derives it from the content, which is what UIContainer used to do.
    //
    // The horizontal padding lives on the inner content row, not here, so that
    // the outline's zero insets are unambiguously the badge's own edges --
    // absolute children resolve against the padding box, and a padded badge
    // would inset the outline by xPad on each side.
    flecs::entity badge = world->entity()
        .is_a(UIElement)
        .child_of(parent)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UISize>({YGUndefined, badge_height})
        .set<UIFlexContainer>({
            YGFlexDirectionRow, YGWrapNoWrap,
            YGJustifyFlexStart, YGAlignStretch, 0.0f})
        .set<ZIndex>({20});

    if (!outlined) {
        badge.set<RenderGradient>({dark, very_dark}); // Vertical gradient
    }

    if (is_double_arrow)
    {
        badge.set<CustomRenderable>({100.0f, badge_height, true, outline_color, 0, 0, draw_double_arrow});
    } else
    {
        // A string badge keeps a dark ground so its text stays legible against
        // whatever is behind it, but no gradient -- the colour lives in the
        // border instead.
        badge.set<RoundedRectRenderable>({100.0f, badge_height, corner_radius, false,
                                          outlined ? 0x141414E0u : 0x000000FFu});

        // Outline overlay: out of flow, so it tracks the badge's auto width
        // without being measured as content the way the old Expand fill was.
        // Drawn at full strength for a string badge, since it is the only thing
        // carrying the colour.
        world->entity()
            .is_a(UIElement)
            .child_of(badge)
            .add<UIYoga>()
            .set<UIAbsoluteEdges>({0.0f, 0.0f, 0.0f, 0.0f})
            .set<RoundedRectRenderable>({100.0f, badge_height, corner_radius, true,
                                         outlined ? (light | 0xFF) : outline_color})
            .set<ZIndex>({22});
    }

    // Content row: carries the padding and holds prefix sprites, label and
    // postfix sprites in order. The badge stretches it to the full 25px so this
    // row's top *is* the badge's top, which is where the old badge_content sat.
    //
    // FlexStart, not centred. The old layout top-aligned the sprites at y=0 and
    // gave the text a hand-tuned 6px offset, and that offset is not what
    // centring produces: measureText reports the full line box (ascender to
    // descender, times scaleY), which is taller than the glyphs actually drawn,
    // so centring it rides the visible text too high. The margins below keep
    // the calibrated positions.
    flecs::entity content = world->entity()
        .is_a(UIElement)
        .child_of(badge)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexContainer>({
            YGFlexDirectionRow, YGWrapNoWrap,
            YGJustifyFlexStart, YGAlignFlexStart, 0.0f})
        .set<UIPadding>({0.0f, xPad, 0.0f, xPad});

    // Helper lambda to create MNIST digit or wildcard image
    // auto create_slot_image = [&](flecs::entity parent_entity, const std::string& symbol, uint32_t tint) {
    //     uint32_t tint_color = scale_color(tint, 1.3f);
    //     unsigned char r = (tint_color >> 24) & 0xFF;
    //     unsigned char g = (tint_color >> 16) & 0xFF;
    //     unsigned char b = (tint_color >> 8) & 0xFF;
    //     unsigned char a = (tint_color) & 0xFF;

    //     std::string image_path;
    //     if (symbol == "*") {
    //         // Wildcard - use wildcard.png
    //         image_path = "wildcard.png";
    //     }
    //     else if (symbol == "checkbox")
    //     {
    //         // TODO: Bind to boolean state
    //         image_path == "checkbox_true.png";
    //     } else if (symbol.length() == 1 && std::isdigit(static_cast<unsigned char>(symbol[0]))) {
    //         // Standard MNIST digit
    //         image_path = "mnist/set_0/" + symbol + ".png";
    //     } else {
    //         // Non-digit - use uppercase letter_set sprite
    //         std::string upper_symbol = symbol;
    //         for (auto& c : upper_symbol) c = std::toupper(static_cast<unsigned char>(c));
    //         image_path = "letter_sets/set_01/" + upper_symbol + ".png";
    //     }

    //     world->entity()
    //         .is_a(UIElement)
    //         .child_of(parent_entity)
    //         .set<ImageCreator>({image_path, 0.9f, 0.9f, nvgRGBA(r, g, b, a)})
    //         .set<ZIndex>({25});
    // };

    // Helper lambda to create text element. The 6px that used to be a
    // Position<Local> offset is now a top margin, so Yoga reproduces the
    // calibrated baseline instead of the layout deciding one for us.
    auto create_text_element = [&](flecs::entity parent_entity, const char* txt, uint32_t color_val) {
        world->entity()
            .is_a(UIElement)
            .child_of(parent_entity)
            .add<UIYoga>()
            .set<UIMargin>({BADGE_TEXT_TOP, 0.0f, 0.0f, 0.0f})
            .set<TextRenderable>({txt, "CharisSIL", 16.0f, color_val})
            .set<ZIndex>({25});
    };

    // Helper to render a set of IDs with optional curly braces
    auto render_id_set = [&](flecs::entity parent_entity, const std::vector<std::string>& ids, const std::vector<uint32_t>& tints) {
        if (ids.empty()) return;

        bool is_set = ids.size() > 1;

        if (is_set) {
            create_text_element(parent_entity, " {", white);
        }

        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) {
                create_text_element(parent_entity, ",", white);
            }
            uint32_t tint = (i < tints.size()) ? tints[i] : 0x888888ff;
            create_slot_image(parent_entity, ids[i], tint, true);
        }

        if (is_set) {
            create_text_element(parent_entity, "} ", white);
        }
    };

    // Published for later lookups, and handed straight back for callers nesting
    // something right now.
    if (out_content) *out_content = content;

    // Render prefix IDs (sources)
    render_id_set(content, prefix_ids, prefix_tints);

    // Text with Gradient. The full string, always -- a badge showing a value
    // must show the value. When it is too wide the badge grows *downwards*
    // instead, below.
    flecs::entity label = world->entity()
        .is_a(UIElement)
        .child_of(content)
        .add<UIYoga>()
        .set<UIMargin>({BADGE_TEXT_TOP, 0.0f, 0.0f, 0.0f})
        .set<TextRenderable>({text, "CharisSIL", 16.0f, white, 1.2f})
        .set<ZIndex>({25});

    if (!outlined) {
        label.set<RenderGradient>({white, light});
    }

    // Published for later lookups now that both parts exist.
    badge.set<BadgeContent>({content, label});

    // Render postfix IDs (targets)
    render_id_set(content, postfix_ids, postfix_tints);

    // A label wider than the ceiling wraps, turning a very wide badge into a
    // taller one. Nothing is dropped: truncation would be wrong here, since
    // most badges exist precisely to show a value. Callers that cannot spend
    // the height -- a list row, whose pitch is fixed -- shorten the string
    // themselves before calling, with ellipsize_text.
    badge_wrap_if_wide(badge, content, label, text, BADGE_MAX_TEXT_WIDTH);

    return badge;
}

void create_editor(flecs::entity leaf, EditorNodeArea& node_area, flecs::entity UIElement)
{
    leaf.set<EditorLeafData>({EditorType::Void});

    auto editor_visual = world->entity()
        .is_a(UIElement)
        .set<Position, Local>({1.0f, 1.0f})
        .set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, false, 0x010222})
        .child_of(leaf);

    leaf.add<EditorVisual>(editor_visual);

    auto editor_outline = world->entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        // .set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, true, 0x111222FF})
        .set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, true, 0x030303FF})
        .set<ZIndex>({5});

    leaf.add<EditorOutline>(editor_outline);

    auto editor_header = world->entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        // .set<RectRenderable>({node_area.width-2-8.0f, 27.0f, true, 0x00000000})
        .set<Expand>({true, 4.0f, 4.0f, 1.0f, false, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({8});

    leaf.add<EditorHeader>(editor_header);
    leaf.add<PanelState>();

    // Add 'expand to parent UIElement bounds with padding'
    auto editor_canvas = world->entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({4.0f, 23.0f})
        // .add<DebugRenderBounds>()
        .set<RectRenderable>({node_area.width-2-8.0f, node_area.height-2-23.0f, false, 0x121212FF})
        .set<Expand>({true, 4.0f, 4.0f, 1.0f, true, 27.0f, 0.0f, 1.0f})
        .set<ZIndex>({8});

    leaf.add<EditorCanvas>(editor_canvas);

    auto editor_header_bkg = world->entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({0.0f, 0.0f})
        // .set<RoundedRectRenderable>({0.0f, 22.0f, 4.0f, false, 0x282828FF})
        .set<RoundedRectRenderable>({0.0f, 22.0f, 4.0f, false, 0x050505FF})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0, 0, 0})
        .set<ZIndex>({2});

    { // Panel type dropdown
    auto editor_icon_bkg = world->entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({8.0f, 2.0f})
        .set<RoundedRectRenderable>({32.0f, 20.0f, 4.0f, false, 0x282828FF})
        .add<EditorCanvas>(editor_canvas)
        .add<EditorLeaf>(leaf)
        .add<AddTagOnLeftClick, ShowEditorPanels>()
        .set<ZIndex>({4});

    auto editor_icon = world->entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({2.0f, 0.0f})
        .set<ImageCreator>({"../assets/panel_type.png", 1.0f, 1.0f})
        .set<ZIndex>({12});

    auto editor_dropdown = world->entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({22.0f, 8.0f})
        .set<ImageCreator>({"../assets/arrow_down.png", 1.0f, 1.0f})
        .set<ZIndex>({12});
        
    world->entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<RoundedRectRenderable>({32.0f, 20.0f, 4.0f, true, 0x5f5f5fFF})
        .set<ZIndex>({6});
    }
    { // Panel state dropdown
    auto editor_icon_bkg = world->entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({46.0f, 2.0f})
        .set<RoundedRectRenderable>({32.0f, 20.0f, 4.0f, false, 0x282828FF})
        .add<EditorCanvas>(editor_canvas)
        .add<EditorLeaf>(leaf)
        .add<AddTagOnLeftClick, ShowPanelState>()
        .set<ZIndex>({4});

    auto editor_icon = world->entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({2.0f, 0.0f})
        .set<ImageCreator>({"../assets/embodiment.png", 1.0f, 1.0f})
        .set<ZIndex>({12});

    auto editor_dropdown = world->entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({22.0f, 8.0f})
        .set<ImageCreator>({"../assets/arrow_down.png", 1.0f, 1.0f})
        .set<ZIndex>({12});
        
    world->entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<RoundedRectRenderable>({32.0f, 20.0f, 4.0f, true, 0x5f5f5fFF})
        .set<ZIndex>({6});
    }

    // TODO: Editor type entities...

    // if (leaf.has<EditorLeafData>() && leaf.get<EditorLeafData>().editor_type == EditorType::Bookshelf)
    // {
    // For bookshelf, we should create the horizontal layout with all the books

}

// Index-aligned with EditorType: the type selector adds (EditorType)index as a
// tag, so a new panel has to be appended to both lists at the same position.
// TODO: load this from config file
std::vector<std::string> editor_types =
{
    "Void",
    // "ECS Graph", // Entity component relationship
    "Peach Core",
    "Condensate",
    "Interlocutor", // Queue or Stream
    "VNC Stream",
    "Healthbar",
    // "Gynoid",
    "In-Silico Lab",
    "Droid",
    "Vision",
    "Hearing",
    "Memory",
    "Bookshelf",
    "Episodic",
    "Ontology",
    "Scene Graph",
    "Data Fusion",
    "Entities",
    "Lexicon",
    "Transducer"
};

struct VNCData
{
    flecs::entity vnc_stream;
    VNCClientHandle client;  // Stable handle that survives ECS moves
};

VNCData get_vnc_source(flecs::entity user_state_leaf, const std::string& host, int port) {
    std::string addr = host + ":" + std::to_string(port);

    // 1. Check if we already have this connection
    auto existing = world->lookup(addr.c_str());
    if (existing && existing.has<VNCClientHandle>()) {
        auto& handle = existing.ensure<VNCClientHandle>();
        handle->reference_count++;
        return {existing, handle};
    }

    // 2. Create VNCClient on heap with shared_ptr for thread safety
    auto client_handle = std::make_shared<VNCClient>();
    client_handle->host = host;
    client_handle->port = port;
    client_handle->reference_count = 1;
    client_handle->surfaceMutex = std::make_shared<std::mutex>();
    client_handle->dirtyRectQueue = std::make_shared<DirtyRectQueue>();
    client_handle->connectionState = VNCClient::CONNECTING;
    client_handle->user_state_leaf = user_state_leaf;

    // Create OpenGL texture (will be resized when connected)
    glGenTextures(1, &client_handle->vncTexture);
    glBindTexture(GL_TEXTURE_2D, client_handle->vncTexture);
    // Allocate empty texture initially (will resize when connected)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Create PBO for async texture upload
    glGenBuffers(1, &client_handle->pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, client_handle->pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 1920 * 1080 * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    client_handle->nvgHandle = nvglCreateImageFromHandleGL2(world->try_get<Graphics>()->vg,
        client_handle->vncTexture, 1920, 1080, 0);

    // Create entity with VNCClientHandle (shared_ptr stays stable)
    flecs::entity created = world->entity(addr.c_str()).set<VNCClientHandle>(client_handle);

    // Start network thread with shared_ptr (thread keeps VNCClient alive)
    client_handle->threadShouldStop = false;
    client_handle->messageThread = std::thread(vnc_message_thread, client_handle);

    return {created, client_handle};
}
// TODO: Make it transitive

void draw_diamond(NVGcontext* vg, const RenderCommand* cmd, const CustomRenderable& data) {
    // Assuming (x, y) is the top-left of the bounding box
    float x = cmd->pos.x; 
    float y = cmd->pos.y;
    
    float midX = x + data.width / 2.0f;
    float midY = y + data.height / 2.0f;

    nvgBeginPath(vg);
    
    // Move to Top Center
    nvgMoveTo(vg, midX, y); 
    // Line to Right Center
    nvgLineTo(vg, x + data.width, midY); 
    // Line to Bottom Center
    nvgLineTo(vg, midX, y + data.height); 
    // Line to Left Center
    nvgLineTo(vg, x, midY); 
    
    nvgClosePath(vg);

    // Convert uint32_t color to NVGcolor (assuming ARGB or RGBA)
    // This example assumes RGBA: 0xRRGGBBAA
    unsigned char r = (data.color >> 24) & 0xFF;
    unsigned char g = (data.color >> 16) & 0xFF;
    unsigned char b = (data.color >> 8) & 0xFF;
    unsigned char a = (data.color) & 0xFF;
    NVGcolor color = nvgRGBA(r, g, b, a);

    if (data.stroke) {
        nvgStrokeColor(vg, color);
        nvgStrokeWidth(vg, 1.0f);
        nvgStroke(vg);
    } else {
        nvgFillColor(vg, color);
        nvgFill(vg);
    }
}

std::unordered_map<std::string, Position> load_layout(const std::string& path) {
    std::unordered_map<std::string, Position> coords;
    std::ifstream file(path);
    std::string name, sx, sy;

    // Assumes format: Name [TAB] X [TAB] Y
    while (std::getline(file, name, '\t') && 
           std::getline(file, sx, '\t')   && 
           std::getline(file, sy)) {
        try {
            // Store directly as your Position type
            coords[name] = { std::stof(sx), std::stof(sy) };
            std::cout << name << " at " << std::stof(sx) << std::endl;
        } catch (...) { continue; } 
    }
    return coords;
}

// --- Entity Creation ---
flecs::entity create_ontology_sprite(const std::string& sprite_path, 
                                     flecs::entity parent, 
                                     flecs::entity UIElement, 
                                     Position pos) 
{  
    return world->entity()
        .is_a(UIElement)
        .child_of(parent)
        .set<Position, Local>(pos)
        .set<ImageCreator>({"../assets/bfo/" + sprite_path + ".png", 1.0f, 1.0f})
        .add<BFOSprite>()
        .add<AddTagOnHoverEnter, HighlightBFOInheritanceHierarchy>()
        .add<AddTagOnHoverExit, ResetBFOSprites>()
        .set<ZIndex>({50});
}

// --- Main Setup Logic ---
void setup_bfo_hierarchy(flecs::entity bfo_editor, flecs::entity UIElement) {
    // 1. Load layout coordinates from your Krita export
    auto layout_map = load_layout("../assets/config/bfo_layout.txt");

    // 2. Define the BFO hierarchy: { Child -> Parent }
    std::vector<std::pair<std::string, std::string>> hierarchy = {
        {"continuant", "entity"},
        {"occurrent", "entity"},
        
        // Occurrent branch
        {"process", "occurrent"},
        {"process_boundary", "occurrent"},
        {"spatiotemporal_region", "occurrent"},
        {"temporal_interval", "occurrent"},
        
        // Continuant branch
        {"independent_continuant", "continuant"},
        {"specifically_dependent_continuant", "continuant"},
        {"generically_dependent_continuant", "continuant"},
        
        {"quality", "specifically_dependent_continuant"},
        {"realizable_entity", "specifically_dependent_continuant"},

        {"role", "realizable_entity"},
        {"disposition", "realizable_entity"},
        {"function", "realizable_entity"},

        // Independent Continuant branch
        {"material_entity", "independent_continuant"},
        {"immaterial_entity", "independent_continuant"},
        
        // Material Entity branch
        {"object", "material_entity"},
        {"object_aggregate", "material_entity"},
        {"fiat_object_part", "material_entity"},
        
        // Immaterial Entity branch
        {"site", "immaterial_entity"},
        {"spatial_region", "immaterial_entity"},
        {"continuant_fiat_boundary", "immaterial_entity"}
    };

    std::unordered_map<std::string, flecs::entity> entities;

    // 3. Helper lambda to resolve position and create entity
    auto create_with_coords = [&](const std::string& name, flecs::entity p) -> flecs::entity {
        Position pos = {0.0f, 0.0f};

        // Krita often exports layers with the extension in the name (e.g. "entity.png")
        if (layout_map.count(name)) {
            pos = layout_map[name];
        } else if (layout_map.count(name + ".png")) {
            pos = layout_map[name + ".png"];
        }

        return create_ontology_sprite(name, p, UIElement, pos);
    };

    // 4. Create the root 'entity'
    entities["entity"] = create_with_coords("entity", bfo_editor);

    // 5. Build the tree
    for (const auto& [child_name, parent_name] : hierarchy) {
        // Ensure parent exists in the entities map
        if (entities.find(parent_name) == entities.end()) {
            entities[parent_name] = create_with_coords(parent_name, bfo_editor);
        }

        // Create child and create a Flecs relationship to the parent
        flecs::entity child = create_with_coords(child_name, bfo_editor);
        entities[parent_name].add<ParentClass>(child);
        
        entities[child_name] = child;
    }

    std::cout << "BFO Hierarchy initialized with " << entities.size() << " sprites." << std::endl;
}

void create_process_segment(flecs::entity UIElement, flecs::entity leaf, flecs::entity channel, TimeInterval interval)
{
    flecs::entity process_start = world->entity() 
        .is_a(UIElement)
        .child_of(channel)
        .set<Position, Local>({128.0f, 2.0f}); // TODO: This should start at the rightmost pos of the channel and behind everything
    // TODO: Binding processes to particular timestamps and channels, reflected in the UI
    world->entity()
        .is_a(UIElement)
        .child_of(process_start)
        .set<RectRenderable>({100.0f, 20.0f, false, 0xf039e044 })
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .set<ZIndex>({15});

    world->entity()
        .is_a(UIElement)
        .child_of(process_start)
        .set<TimeInterval>(interval)
        .add<TimeIntervalToSpaceSegment>()
        .set<RectRenderable>({100.0f, 20.0f, true, 0xf039e0ff })
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .set<ZIndex>({18});

    world->entity()
        .is_a(UIElement)
        .child_of(process_start)
        // .set<ImageCreator>({("../assets/checkbox_.png").c_str(), 1.0f, 1.0f})
        .set<ImageCreator>({"../assets/process_overlay.png", 1.0f, 1.0f})
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .set<ZIndex>({20});
}

void clicked_minilm(flecs::entity e)
{
    std::cout << "clicked_minilm" << std::endl;
    // Lands centred along the icon's bottom edge: the server icon is a column
    // flex container with JustifyFlexEnd / AlignCenter.
    world->entity()
    .is_a(e.world().lookup("UIElement"))
    .child_of(e)
    .add<UIYoga>()
    .add<UINativeImageSize>()
    .set<ImageCreator>({"../assets/server_dot.png", 1.0f, 1.0f})
    .set<ZIndex>({16});
}

// ============================================================================
// Entities panel
// ----------------------------------------------------------------------------
// Rows come from a separate, headless flecs world, reached through the
// entity_query bridge (scripts/entity_query.py). The panel holds no entity data
// of its own -- everything below reads the panel's EntitySet, so what is on
// screen is whatever that world currently answers with.

// The query a freshly opened panel runs. "Person" is what the python_query demo
// world is richest in; any flecs expression the far world understands works,
// including range and spatial filters.
static const char* ENTITY_DEFAULT_QUERY = "Person";

// Empty set for panels that have no EntitySet yet, so callers can hold a
// reference without checking for null at every use.
static const EntitySet& entity_set_of(flecs::entity leaf)
{
    static const EntitySet empty;
    const EntitySet* set = leaf.try_get<EntitySet>();
    return set ? *set : empty;
}

// Distinct badge colour per row, derived locally. get_entity_color() would be
// the semantically right call, but it round-trips a unix socket to a Python
// embedding server per lookup -- 100 of those during panel construction would
// stall startup for minutes on a cold cache.
//
// Golden-ratio hue stepping: successive rows land ~137.5 degrees apart on the
// wheel, so no two neighbours read as the same colour and the set stays evenly
// spread however long the list gets. The multiply is done in double because the
// fractional part is the whole result -- a float mantissa loses it once the
// product grows past 2^24.
uint32_t entity_badge_color(size_t index)
{
    float hue = (float)std::fmod((double)index * 0.6180339887498949, 1.0) * 6.0f;
    float sat = 0.62f, val = 0.78f;

    int sector = (int)hue;
    float frac = hue - sector;
    float p = val * (1.0f - sat);
    float q = val * (1.0f - sat * frac);
    float t = val * (1.0f - sat * (1.0f - frac));

    float r, g, b;
    switch (sector) {
        case 0:  r = val; g = t;   b = p;   break;
        case 1:  r = q;   g = val; b = p;   break;
        case 2:  r = p;   g = val; b = t;   break;
        case 3:  r = p;   g = q;   b = val; break;
        case 4:  r = t;   g = p;   b = val; break;
        default: r = val; g = p;   b = q;   break;
    }

    return ((uint32_t)(r * 255.0f) << 24) | ((uint32_t)(g * 255.0f) << 16)
         | ((uint32_t)(b * 255.0f) << 8)  | 0xFF;
}

// Uniform row height. The list's index <-> y arithmetic is only exact because
// every row is this tall regardless of what the row builder puts inside it.
static constexpr float ENTITY_ROW_PITCH = 28.0f;

// Inset of the rows inside the list viewport, and the fixed height the
// inspector subpanel takes off the top of the panel.
static constexpr float ENTITY_LIST_PAD = 6.0f;
static constexpr float ENTITY_INSPECTOR_HEIGHT = 44.0f;
static constexpr float ENTITY_QUERY_HEIGHT = 26.0f;

// Past this many characters a name stops being spelled out in letter sprites
// and becomes wrapped text -- one sprite per character does not scale to a
// sentence.
static constexpr size_t ENTITY_SPRITE_NAME_LIMIT = 24;

// Brings a list's realized window in line with where it is scrolled to.
//
// Cost is proportional to the visible row count, never to item_count: the first
// visible index is a division, and rows outside the window are simply never
// built. Rebuilding the whole window rather than recycling a row pool keeps the
// order Yoga reads out of OrderedChildren correct without any shuffling, and is
// affordable because it only happens when the window actually moves -- a
// fractional scroll within one row just slides the container.
void update_virtual_list(flecs::entity list_entity, VirtualList& list)
{
    if (!list.viewport.is_valid() || !list.build_row) return;

    const UIElementSize* viewport_size = list.viewport.try_get<UIElementSize>();
    if (!viewport_size) return;   // no measured height yet; try again next frame

    // The visible band is the viewport inset by pad top and bottom -- the same
    // rectangle the rows are cropped to. Scrolling to the end therefore lands
    // the last row on the inset edge, not on the viewport's own edge.
    float view_height = std::max(0.0f, viewport_size->height - list.pad * 2.0f);
    if (view_height <= 0.0f) return;

    float content_height = (float)list.item_count * list.row_pitch;
    float max_scroll = std::max(0.0f, content_height - view_height);
    list.scroll = std::clamp(list.scroll, 0.0f, max_scroll);

    size_t first = (size_t)std::floor(list.scroll / list.row_pitch);
    float remainder = list.scroll - (float)first * list.row_pitch;   // sub-row offset

    // Exactly the rows that intersect the visible band: the band plus whatever
    // of the first row is already scrolled past the top. No slack row -- every
    // realized row beyond this is drawn outside the crop and thrown away.
    size_t count = (size_t)std::ceil((view_height + remainder) / list.row_pitch);
    count = std::min(count, list.item_count - std::min(first, list.item_count));

    if (list.dirty || !list.realized
        || first != list.first_realized || count != list.realized_count) {
        // Collected before destructing: mutating the child list while iterating
        // it is not safe even though the ops themselves are deferred.
        std::vector<flecs::entity> stale;
        list_entity.children([&](flecs::entity row) { stale.push_back(row); });
        for (flecs::entity row : stale) row.destruct();

        for (size_t i = 0; i < count; ++i) {
            list.build_row(list_entity, first + i);
        }

        list.first_realized = first;
        list.realized_count = count;
        list.realized = true;
        list.dirty = false;
    }

    // Sub-row scrolling, expressed as the container's absolute top inset rather
    // than Position<Local>: the container sits inside a Yoga parent now, and
    // readback owns Position<Local> for every non-root node. Yoga applies this
    // later in the same PreFrame pass, so the wheel moves the list this frame.
    list_entity.ensure<UIAbsoluteEdges>().top = list.pad - remainder;
}

// Stable colour for a type name -- a tag, a component, a relation. Same
// golden-ratio spacing as the row colours, but keyed by the name so a component
// keeps its colour across panels, queries and sessions instead of shifting with
// whatever row it happened to appear on.
//
// The hash is reduced modulo a small number first: entity_badge_color multiplies
// by the golden ratio and keeps the fraction, and a raw size_t hash is far past
// the point where that product has any fraction left.
// True when an entity's identity is prose rather than a symbol -- it carries a
// Text component, or is tagged Chat.
//
// The distinction is the entity's, not the call site's: the same entity has to
// read the same way in a list row and in the inspector, so both ask this rather
// than each deciding for itself.
static bool entity_is_prose(const EntitySet& set, size_t index)
{
    if (index < set.tags.size()) {
        const std::vector<std::string>& tags = set.tags[index];
        if (std::find(tags.begin(), tags.end(), "Chat") != tags.end()) return true;
    }
    if (index < set.components.size()) {
        for (const std::string& entry : set.components[index]) {
            // Entries are "Type<TAB>member<TAB>value"; only the type is compared.
            size_t sep = entry.find('\t');
            if (entry.substr(0, sep) == "Text") return true;
        }
    }
    return false;
}

uint32_t entity_type_color(const std::string& name)
{
    return entity_badge_color(std::hash<std::string>{}(name) % 977);
}


// Width a longform badge wraps to. Fixed rather than derived: the text has no
// natural width, so something has to choose one, and a column of messages that
// all wrap to the same measure is far easier to read than one that ragged-edges
// to each message's length.
static constexpr float BADGE_LONGFORM_WIDTH = 300.0f;

// A badge for a sentence or a paragraph rather than a label.
//
// Every other badge is one calibrated line tall and as wide as its text. This
// one inverts that: width is fixed, the text wraps inside it, and the height
// follows from however many lines that takes. Used for entities whose identity
// *is* prose -- a chat message has no short name to show instead.
flecs::entity create_longform_badge(flecs::entity parent, flecs::entity UIElement,
                                    const std::string& text, uint32_t color)
{
    flecs::entity slot = flecs::entity::null();
    flecs::entity badge = create_badge_impl(parent, UIElement, text.c_str(), color,
                                            false, false, {}, {}, {}, {}, &slot,
                                            BadgeStyle::String);
    if (!slot.is_valid()) return badge;

    // Forced, not conditional: prose gets one consistent measure even when a
    // particular message happens to be short enough to fit on a line. A column
    // of messages all wrapping to the same width reads far better than one that
    // ragged-edges to each message's length.
    flecs::entity label = flecs::entity::null();
    slot.children([&label](flecs::entity child) {
        if (child.has<TextRenderable>()) label = child;
    });
    if (label.is_valid()) {
        badge_apply_wrap(badge, slot, label, BADGE_LONGFORM_WIDTH - BADGE_LONGFORM_INSET);
    }

    return badge;
}

// Turns a badge into a container for other badges.
//
// Every badge that holds something needs the same four things, so they live
// here rather than being re-derived per call site: height from content instead
// of the calibrated single-line height, an even inset all round, contents
// centred against each other with a real gap, and the label's top offset
// cleared -- that offset is calibrated for a badge that top-aligns its
// contents, and a container centres them.
// Re-cuts a container's silhouette so its label sits in a tab across the top.
//
// The plain rounded rect is replaced by a tab-shaped path, and the absolutely
// positioned outline child is re-cut to match, so fill and border stay the same
// shape. Colours are read back off the badge rather than passed in: they were
// derived from the base colour inside create_badge_impl, and re-deriving them
// here would be a second place to keep in step.
static void badge_becomes_container(flecs::entity block, flecs::entity slot,
                                    bool stacked)
{
    block.ensure<UISize>().h = YGUndefined;
    block.set<UIPadding>({BADGE_NEST_PAD, 0.0f, BADGE_NEST_PAD, 0.0f});

    UIFlexContainer& row = slot.ensure<UIFlexContainer>();
    row.gap = BADGE_NEST_GAP;

    if (stacked) {
        // Wrapped text makes a block far taller than its own label, and a row
        // centres that label against the whole height -- so a paragraph gets a
        // header floating in the middle of a tall empty column. Stacking turns
        // the label into a tab along the top instead: read down, not across.
        row.direction = YGFlexDirectionColumn;
        row.items = YGAlignFlexStart;
    } else {
        row.direction = YGFlexDirectionRow;
        row.items = YGAlignCenter;
    }

    slot.children([](flecs::entity child) {
        if (UIMargin* margin = child.try_get_mut<UIMargin>()) margin->top = 0.0f;
    });
}

// Values are all one colour rather than one per value: a value is data, and
// colouring each differently would imply a distinction that is not there.
static constexpr uint32_t BADGE_VALUE_COLOR = 0x3f6d8aff;

// Prose gets one neutral grey rather than a per-entity hue. A symbol's colour
// identifies the thing it names -- Person is always the same colour, which is
// what makes it recognisable at a glance. A message names nothing, so a unique
// colour per message would be noise dressed as meaning, and a wall of messages
// in twenty different hues is harder to read than a wall in one.
// Note this is the *base*: a string badge's visible border is scale_color(base,
// 1.3), so the drawn outline lands a step lighter than the value here.
static constexpr uint32_t BADGE_STRING_COLOR = 0x5c5c5cff;

// A parameter's own box is structure, not identity. Hashing its name to a hue
// gave every member of every component a different colour, which reads as though
// the colours mean something -- they don't, and the resulting confetti competes
// with the component around it and the value inside it. One mid grey instead.
static constexpr uint32_t BADGE_PARAM_COLOR = 0x707070ff;

// A component rendered as a block holding one block per member, each holding
// its value: Salary ( amount ( 100000 ) ).
//
// Nested rather than flattened to "amount: 100000" text because a member's
// value is a thing in its own right -- something to select, edit, or eventually
// bind to another entity -- and containment says which value belongs to which
// member without leaning on punctuation to carry it.
flecs::entity create_component_badge(flecs::entity parent, flecs::entity UIElement,
                                     const std::string& type_name,
                                     const std::vector<std::pair<std::string, std::string>>& members)
{
    uint32_t type_color = entity_type_color(type_name);

    // Outlined, not filled: a component with parameters is a frame around its
    // members, and a gradient fill behind them competes with the badges it
    // contains. The type's colour moves to a bar along the top instead.
    flecs::entity slot = flecs::entity::null();
    flecs::entity block = create_badge_impl(parent, UIElement, type_name.c_str(),
                                            type_color, false, false,
                                            {}, {}, {}, {}, &slot,
                                            BadgeStyle::String);

    if (members.empty() || !slot.is_valid()) return block;

    bool stacked_component = false;

    for (const auto& member : members) {
        const std::string& name = member.first;
        const std::string& value = member.second;

        if (name.empty()) {
            // A component with no reflected member names still has a value.
            // It goes straight into the component block, unlabelled.
            flecs::entity bare = create_badge_impl(slot, UIElement, value.c_str(),
                                                   BADGE_VALUE_COLOR, true, false,
                                                   {}, {}, {}, {}, nullptr,
                                                   BadgeStyle::String);
            continue;
        }

        uint32_t member_color = BADGE_PARAM_COLOR;
        flecs::entity value_slot = flecs::entity::null();
        flecs::entity member_block = create_badge_impl(slot, UIElement, name.c_str(),
                                                       member_color, false, false,
                                                       {}, {}, {}, {}, &value_slot,
                                                       BadgeStyle::String);

        // The value is a string, not a symbol naming something -- outlined, so a
        // glance separates "this is data" from "this is a type".
        flecs::entity value_badge = create_badge_impl(value_slot, UIElement, value.c_str(),
                                                      BADGE_VALUE_COLOR, true, false,
                                                      {}, {}, {}, {}, nullptr,
                                                      BadgeStyle::String);

        // A value that wrapped makes this member -- and the component holding
        // it -- a tall block, so both stack their labels rather than centring
        // them against that height.
        bool tall = badge_text_exceeds(value, BADGE_MAX_TEXT_WIDTH);
        badge_becomes_container(member_block, value_slot, tall);
        if (tall) stacked_component = true;

    }

    badge_becomes_container(block, slot, stacked_component);
    return block;
}

// A relationship rendered as a block that *contains* its target, rather than as
// two badges sitting next to each other.
//
// "LocatedIn -> Room102" placed side by side reads as two separate facts; the
// nested form reads as one, which is what a pair actually is. It is also the
// shape natural language wants when these get composed -- a relation whose slot
// holds an entity is a phrase, and a phrase can be dropped into another slot.
flecs::entity create_relation_badge(flecs::entity parent, flecs::entity UIElement,
                                    const std::string& relation, const std::string& target)
{
    flecs::entity slot = flecs::entity::null();
    flecs::entity block = create_badge_impl(parent, UIElement, relation.c_str(),
                                            entity_type_color(relation), false, true,
                                            {}, {}, {}, {}, &slot);

    if (target.empty() || !slot.is_valid()) return block;

    uint32_t target_color = entity_type_color(target);
    std::vector<std::string> postfix_ids{target.substr(0, 1)};
    std::vector<uint32_t> postfix_tints{target_color};

    // Capsule, matching how the Interlocutor draws an entity in a relationship.
    flecs::entity nested = create_badge(slot, UIElement, target.c_str(), target_color,
                                        true, false, {}, {}, postfix_ids, postfix_tints);

    badge_becomes_container(block, slot, badge_text_exceeds(target, BADGE_MAX_TEXT_WIDTH));
    return block;
}

// The colour a character carries as an entity in its own right: 'a' is the
// same colour in every word, in every panel. Index-stepped rather than hashed
// because the alphabet is small and closed -- golden-ratio spacing over 36
// indices keeps every letter maximally far from its neighbours, where a hash
// would waste separation on a space it never fills.
uint32_t letter_color(char c)
{
    unsigned char lower = std::tolower(static_cast<unsigned char>(c));
    size_t index;
    if (lower >= 'a' && lower <= 'z')      index = lower - 'a';
    else if (lower >= '0' && lower <= '9') index = 26 + (lower - '0');
    else return 0xAAAAAAFF;
    return entity_badge_color(index);
}

void consume_ui_click();

void clicked_entity_badge(flecs::entity e)
{
    const EntityBadge* badge = e.try_get<EntityBadge>();
    if (!badge || !badge->panel.is_valid() || !badge->panel.is_alive()) return;

    EntitySelection& selection = badge->panel.ensure<EntitySelection>();
    if (selection.selected == badge->index) return;

    selection.selected = badge->index;

    // The rows themselves show which one is selected, and the realized window
    // hasn't moved, so it has to be told to rebuild.
    if (selection.list.is_valid() && selection.list.is_alive()) {
        selection.list.ensure<VirtualList>().dirty = true;
    }

    // Nothing underneath should also act on this click -- the row sits over the
    // list background, which is a click target in its own right.
    consume_ui_click();
}

// Applies a result set returned by the entity_query bridge.
//
// Like every tradewinds response this arrives without context, so the panel is
// recovered by matching on the query that was in flight. That is stricter than
// the ChatPanel pattern it follows: two panels with different queries will not
// take each other's rows.
void entity_query_response(std::map<std::string, msgpack::object>& res_map)
{
    std::string status = res_map.count("status")
        ? res_map["status"].as<std::string>() : std::string("ERROR");
    std::string error = res_map.count("error")
        ? res_map["error"].as<std::string>() : std::string();

    using StringLists = std::vector<std::vector<std::string>>;
    std::vector<std::string> names;
    std::string unknown;
    StringLists tags, components, relations;

    if (status == "OK") {
        try {
            if (res_map.count("unknown"))    unknown    = res_map["unknown"].as<std::string>();
            if (res_map.count("names"))      names      = res_map["names"].as<std::vector<std::string>>();
            if (res_map.count("tags"))       tags       = res_map["tags"].as<StringLists>();
            if (res_map.count("components")) components = res_map["components"].as<StringLists>();
            if (res_map.count("relations"))  relations  = res_map["relations"].as<StringLists>();
        } catch (const std::exception& e) {
            status = "ERROR";
            error = std::string("bad payload: ") + e.what();
            names.clear(); tags.clear(); components.clear(); relations.clear();
        }
    }

    world->query<EntitySet>()
        .each([&](flecs::entity leaf, EntitySet& set) {
            if (set.requested.empty()) return;   // not waiting on anything

            set.status = status;
            set.error = unknown.empty() ? error : ("unknown: " + unknown);
            set.expr = set.requested;
            set.requested.clear();

            if (status != "OK") return;

            set.names = names;
            set.tags = tags;
            set.components = components;
            set.relations = relations;
            // Resized rather than trusted: a short list from the bridge would
            // otherwise make every lookup below a bounds check.
            set.tags.resize(set.names.size());
            set.components.resize(set.names.size());
            set.relations.resize(set.names.size());
            set.loaded = true;

            // The rows are entirely different entities now, so nothing about the
            // old selection or ordering survives.
            EntitySelection& selection = leaf.ensure<EntitySelection>();
            selection.selected = SIZE_MAX;
            selection.ranked_for = SIZE_MAX;
            selection.ranked_query.clear();
            leaf.ensure<EntityOrder>().order.clear();

            if (selection.list.is_valid() && selection.list.is_alive()) {
                VirtualList& list = selection.list.ensure<VirtualList>();
                list.item_count = set.names.size();
                list.scroll = 0.0f;
                list.dirty = true;
            }
        });
}

// Result of creating an entity in the headless world.
//
// On success every loaded panel is invalidated rather than patched: the new
// entity may or may not match a given panel's query, and the world is the
// authority on that -- re-asking is both simpler and correct.
void entity_create_response(std::map<std::string, msgpack::object>& res_map)
{
    std::string status = res_map.count("status")
        ? res_map["status"].as<std::string>() : std::string("ERROR");

    if (status != "OK") {
        std::string error = res_map.count("error")
            ? res_map["error"].as<std::string>() : std::string();
        std::cerr << "[entities] create failed: " << status << " " << error << std::endl;
        return;
    }

    world->query<EntitySet>().each([](flecs::entity leaf, EntitySet& set) {
        if (set.loaded) set.expr.clear();   // forces a re-query
    });
}

// Creates an entity of `kind` carrying `text` in the headless world.
void create_world_entity(const std::string& kind, const std::string& text)
{
    if (text.empty()) return;

    flecs::entity client = world->lookup("EntityCreateClient");
    if (!client.is_valid() || client.has<AwaitResponse>()) return;

    client.set<SendMapRequest>({{
        {"type", "create"},
        {"kind", kind},
        {"text", text},
    }});
    client.set<AwaitResponse>({entity_create_response});
}

// Sends the panel's flecs query to the bridge when it differs from what is
// loaded. One request in flight at a time, for the same REQ/REP reason as the
// ranking client.
void request_entity_query(flecs::entity leaf, EntitySet& set)
{
    if (!set.requested.empty() || set.wanted.empty()) return;
    // `status` guards the first attempt: expr matches wanted only after a reply
    // has been recorded, so a failed query is not retried in a tight loop.
    if (set.expr == set.wanted && !set.status.empty()) return;

    flecs::entity client = world->lookup("EntityQueryClient");
    if (!client.is_valid() || client.has<AwaitResponse>()) return;

    set.requested = set.wanted;
    client.set<SendMapRequest>({{
        {"type", "query"},
        {"expr", set.wanted},
    }});
    client.set<AwaitResponse>({entity_query_response});
}

// Applies a ranking returned by the entity semantics server.
//
// No context travels with a tradewinds response, so the panel is recovered by
// querying for it -- the same shape interlocutor_response uses. With more than
// one Entities panel open this would deliver to all of them; the ordering is a
// pure function of the selected name, so that is wrong only if their selections
// differ, and fixing it properly means routing context through AwaitResponse.
void entity_semantics_response(std::map<std::string, msgpack::object>& res_map)
{
    std::string status = res_map.count("status")
        ? res_map["status"].as<std::string>() : std::string("ERROR");

    world->query<EntitySelection>()
        .each([&](flecs::entity leaf, EntitySelection& selection) {
            selection.rank_pending = false;

            // LOADING means the model is still coming up. Leaving ranked_for
            // untouched is what makes the request system try again next frame.
            if (status != "OK" || !res_map.count("order")) return;

            std::vector<size_t> order;
            try {
                order = res_map["order"].as<std::vector<size_t>>();
            } catch (const std::exception& e) {
                std::cerr << "[entity_semantics] bad order payload: " << e.what() << std::endl;
                return;
            }

            // A short or malformed permutation would silently hide entities,
            // so an ordering that does not cover the data is refused outright.
            if (order.size() != entity_set_of(leaf).names.size()) {
                std::cerr << "[entity_semantics] order covers " << order.size()
                          << " of " << entity_set_of(leaf).names.size()
                          << " entities; ignoring" << std::endl;
                return;
            }

            leaf.ensure<EntityOrder>().order = std::move(order);
            selection.ranked_for = selection.selected;
            selection.ranked_query = selection.relation;

            if (selection.list.is_valid() && selection.list.is_alive()) {
                VirtualList& list = selection.list.ensure<VirtualList>();
                list.dirty = true;
                // The selected entity ranks nearest to itself, so it is now the
                // top row -- scrolling back up is what makes that visible.
                list.scroll = 0.0f;
            }
        });
}

// Asks the semantics server to rank the corpus against the current selection,
// under whatever relation has been typed into the panel's query bar.
//
// Only ever one request in flight: the socket is REQ/REP, so a second send
// before the reply arrives is silently dropped by ZeroMQ.
void request_entity_ranking(flecs::entity leaf, EntitySelection& selection)
{
    if (selection.rank_pending) return;

    const EntitySet& set = entity_set_of(leaf);
    std::string instruct = selection.relation;
    bool has_selection = selection.selected < set.names.size();

    // Either half is enough to rank: an entity alone gives nearest neighbours,
    // typed text alone is a free-text search, and both together read the text
    // as a relation anchored on that entity.
    if (!has_selection && instruct.empty()) {
        // Neither. Fall back to the data's own order rather than leaving the
        // list frozen in whatever ranking produced it.
        if (selection.ranked_for == SIZE_MAX && selection.ranked_query.empty()) return;

        leaf.ensure<EntityOrder>().order.clear();
        selection.ranked_for = SIZE_MAX;
        selection.ranked_query.clear();
        if (selection.list.is_valid() && selection.list.is_alive()) {
            VirtualList& list = selection.list.ensure<VirtualList>();
            list.dirty = true;
            list.scroll = 0.0f;
        }
        return;
    }

    if (selection.ranked_for == selection.selected
        && selection.ranked_query == instruct) return;

    flecs::entity client = world->lookup("EntitySemanticsClient");
    if (!client.is_valid() || client.has<AwaitResponse>()) return;

    if (set.names.empty()) return;

    std::string corpus;
    for (size_t i = 0; i < set.names.size(); ++i) {
        if (i) corpus += "\n";
        corpus += set.names[i];
    }

    selection.rank_pending = true;
    client.set<SendMapRequest>({{
        {"type", "rank"},
        // Empty when nothing is selected -- the server reads that as a search
        // with no anchor rather than as a missing field.
        {"query", has_selection ? set.names[selection.selected] : std::string()},
        {"instruct", instruct},
        {"corpus", corpus},
    }});
    client.set<AwaitResponse>({entity_semantics_response});
}

// Rebuilds the inspector subpanel when the selection no longer matches what is
// displayed. Comparing selected against shown means this converges from any
// state, including a selection made in the same frame the panel was built.
void update_entity_inspector(flecs::entity leaf, EntitySelection& selection)
{
    if (selection.shown == selection.selected) return;
    if (!selection.inspector.is_valid() || !selection.inspector.is_alive()) return;

    auto UIElement = world->lookup("UIElement");

    std::vector<flecs::entity> stale;
    selection.inspector.children([&](flecs::entity child) { stale.push_back(child); });
    for (flecs::entity child : stale) child.destruct();

    // The clip is declared on the direct children rather than on the inspector
    // itself: ScissorContainer is a transitive relationship, so a self-target is
    // a cycle and flecs asserts on it. resolve_scissor walks up from each
    // renderable, so tagging the children covers everything beneath them --
    // which is what keeps a badge wider than the subpanel from spilling onto the
    // list below.
    // Nothing at all when nothing is selected -- the subpanel is empty, not
    // captioned.
    const EntitySet& set = entity_set_of(leaf);
    if (selection.selected < set.names.size()) {
        const std::string& name = set.names[selection.selected];
        uint32_t color = entity_badge_color(selection.selected);
        if (entity_is_prose(set, selection.selected)) color = BADGE_STRING_COLOR;

        // A Chat entity's identity is its message, which is prose -- spelling a
        // paragraph out in letter sprites would be unreadable, so it gets the
        // longform badge instead.
        // Prose gets the wrapping badge; a short name gets sprite letters. Chat
        // qualifies by type, but so does anything simply too long to spell out
        // -- 24 letter sprites is not a label, it is a wall.
        bool is_prose = entity_is_prose(set, selection.selected)
                        || name.size() > ENTITY_SPRITE_NAME_LIMIT;
        if (is_prose) {
            create_longform_badge(selection.inspector, UIElement, name, color)
                .add<ScissorContainer>(selection.inspector);
        }

        // The selected entity's name spelled out left to right in letter_set
        // sprites, one glyph per character.
        auto letters = is_prose ? flecs::entity::null() : world->entity()
            .is_a(UIElement)
            .child_of(selection.inspector)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .add<ScissorContainer>(selection.inspector)
            .set<UIFlexContainer>({
                YGFlexDirectionRow, YGWrapNoWrap,
                YGJustifyFlexStart, YGAlignFlexStart, 0.0f});

        if (letters.is_valid()) create_sprite_text(letters, name, color);

        // The numeric id, in the same face and size as the Interlocutor chat input.
        world->entity()
            .is_a(UIElement)
            .child_of(selection.inspector)
            .add<UIYoga>()
            .add<ScissorContainer>(selection.inspector)
            .set<TextRenderable>({"#" + std::to_string(selection.selected),
                                  "JetBrainsMono", 16.0f, 0xFFFFFFFF})
            .set<ZIndex>({25});

        // What the far world says this entity is, as badges rather than text.
        auto badge_of = [&](const std::string& label, bool capsule) {
            uint32_t color = entity_type_color(label);
            std::vector<std::string> postfix_ids{label.substr(0, 1)};
            std::vector<uint32_t> postfix_tints{color};
            create_badge(selection.inspector, UIElement, label.c_str(), color,
                         capsule, false, {}, {}, postfix_ids, postfix_tints)
                .add<ScissorContainer>(selection.inspector);
        };

        // A tag is a bare badge -- it has no data, which is the whole point of
        // it being a tag rather than a component.
        for (const std::string& tag : set.tags[selection.selected]) badge_of(tag, false);

        // A component is a block holding its members. The wire form is
        // "Type<TAB>member<TAB>value<TAB>member<TAB>value" -- flat, so it
        // survives msgpack without depending on nested-container conversion.
        for (const std::string& entry : set.components[selection.selected]) {
            std::vector<std::string> fields;
            size_t start = 0;
            while (start <= entry.size()) {
                size_t sep = entry.find('\t', start);
                if (sep == std::string::npos) { fields.push_back(entry.substr(start)); break; }
                fields.push_back(entry.substr(start, sep - start));
                start = sep + 1;
            }
            if (fields.empty()) continue;

            // Trailing odd field would be a member with no value; dropped
            // rather than rendered as an empty badge.
            std::vector<std::pair<std::string, std::string>> members;
            for (size_t i = 1; i + 1 < fields.size(); i += 2) {
                members.emplace_back(fields[i], fields[i + 1]);
            }

            create_component_badge(selection.inspector, UIElement, fields[0], members)
                .add<ScissorContainer>(selection.inspector);
        }

        // Relations are one block holding their target, not two badges in a
        // row. The subject is the selected entity, already spelled out in
        // sprites to the left, so the block reads as the rest of the sentence.
        for (const std::string& entry : set.relations[selection.selected]) {
            size_t sep = entry.find('\t');
            std::string relation = entry.substr(0, sep);
            std::string target = (sep == std::string::npos) ? std::string()
                                                            : entry.substr(sep + 1);

            create_relation_badge(selection.inspector, UIElement, relation, target)
                .add<ScissorContainer>(selection.inspector);
        }
    }

    selection.shown = selection.selected;
}

// Factory function to populate editor content, whether the panel is initialized or changed
// TODO: Refactor editor creation to unique files
void create_editor_content(flecs::entity leaf, EditorType editor_type, flecs::entity UIElement)
{
    PanelState& state = leaf.ensure<PanelState>();
    state.options.clear();
    std::cout << "Change panel type to " << editor_types[(int)editor_type] << std::endl;
    if (editor_type == EditorType::PeachCore)
    {

        auto canvas = leaf.target<EditorCanvas>();

        // Server icon strip. A Yoga row filling the panel: each icon stretches
        // to the row height, takes its width from the image's native aspect
        // ratio, and shrinks uniformly if the strip would overflow.
        auto server_hud = world->entity("ServerHUD")
        .is_a(UIElement)
        .child_of(canvas)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFillParent>({4.0f, 4.0f, 4.0f, 4.0f})
        .set<Position, Local>({4.0f, 4.0f})
        .set<UIFlexContainer>({
            YGFlexDirectionRow, YGWrapNoWrap,
            YGJustifyCenter, YGAlignStretch, 0.0f});

        // Hover description backdrop: its own Yoga root covering the panel.
        auto panel_overlay = world->entity()
        .is_a(UIElement)
        .child_of(canvas)
        .add<UIYoga>()
        .set<UIFillParent>({})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x000000DD})
        .add<ServerDescription>()
        .set<ZIndex>({15});

        std::vector<std::string> server_icons = {"aeri_memory", "zmq", "flecs", "x11", "parakeet", "chatterbox", "doctr", "opencv", "minilm", "dino2", "alpaca", "modal", "borg"}; // , "autodistill", "yolo",
        // std::vector<std::string> server_icons = {"peach_core"};

        for (const auto& icon : server_icons)
        {
            // .set<ServerScript>({"chatterbox", "chatterbox", "../chatter_server"})
            // .add(ServerStatus::Offline)
            // .add<AddTagOnLeftClick, SelectServer>();
            flecs::entity server_icon = world->entity()
            .is_a(UIElement)
            .child_of(server_hud)
            .add<UIYoga>()
            // Height comes from the row's AlignStretch, width from the image
            // aspect ratio; shrink lets the whole strip scale down to fit.
            // Capped at 1:1 so a tall panel doesn't upscale the icons.
            .add<UIMaxNativeImageSize>()
            .set<UIFlexItem>({0.0f, 1.0f, YGAlignAuto})
            // Column so the status dot sits centred along the bottom edge.
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexEnd, YGAlignCenter, 0.0f})
            .set<UIPadding>({0.0f, 0.0f, 4.0f, 0.0f})
            .add<ServerHUDOverlay>(panel_overlay)
            // TODO: Replace these events with callback functions?
            .add<AddTagOnHoverEnter, ShowServerHUDOverlay>()
            .add<AddTagOnHoverExit, HideServerHUDOverlay>()
            .set<ImageCreator>({"../assets/server_hud/" + icon + ".png", 1.0f, 1.0f})
            .set<ZIndex>({10});

            if (icon == "minilm")
            {
                server_icon.set<CallbackOnLeftClick>({clicked_minilm});
                // TODO: OnClick
            } else
            {
                world->entity()
                .is_a(UIElement)
                .child_of(server_icon)
                .add<UIYoga>()
                .add<UINativeImageSize>()
                .set<ImageCreator>({"../assets/server_dot.png", 1.0f, 1.0f})
                .set<ZIndex>({12});
            }
        }

    } else if (editor_type == EditorType::Condensate)
    {
        // TODO: Generate list of files/directories
        auto dir_list = world->entity()
            .is_a(UIElement)
            .child_of(leaf.target<EditorCanvas>())
            .set<LayoutBox>({LayoutBox::Vertical, 0.0f})
            .add<ScissorContainer>(leaf.target<EditorCanvas>())
            .add(flecs::OrderedChildren);

        auto file_nav = FileNavVirtualList("/home/wesxdz/Downloads/", dir_list);
        file_nav.Populate(world);

        auto scrollbar_right = world->entity()
            .is_a(UIElement)
            .child_of(leaf.target<EditorCanvas>())
            .set<RectRenderable>({8.0f, 100.0f, false, 0x555555ff})
            .set<Align>({-1.0f, 0.0f, 1.0f, 0.0f})
            .set<Expand>({0, 0, 0, 0, 1, 0, 0, 1.0f})
            .set<ZIndex>({20});

        auto scrollbar_slider = world->entity()
            .is_a(UIElement)
            .child_of(scrollbar_right)
            .set<RectRenderable>({8.0f, 10.0f, false, 0x888888ff})
            .set<Expand>({1, 0, 0, 1, 1, 0, 0, 0.5f})
            .set<ZIndex>({20});
    }
    else if (editor_type == EditorType::Memory)
    {
        float diurnal_pos = get_time_of_day_normalized();
        std::cout << diurnal_pos << std::endl;
        int segments = 24;
        for (size_t i = 0; i < segments; ++i) {
            world->entity()
                .is_a(UIElement)
                .child_of(leaf.target<EditorCanvas>())
                .set<DiurnalHour>({i})
                .set<QuadraticBezierRenderable>(get_hour_segment(i, (0.5f * M_PI) - (diurnal_pos) * (2.0f * M_PI)))
                .set<ZIndex>({10});
        }
    } else if (editor_type == EditorType::Healthbar)
    {
        // Healthbar base level
        // Threshold at 0.5
        world->entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<LineRenderable>({0.0f, 0.0f, 100.0f, 0.0f, 2.0f, 0x00FF00FF})
        .set<Align>({0.0f, 0.0f, 0.0f, 0.5f})
        .set<Expand>({true, 0, 0, 1.0, false, 0, 0, 0})
        .set<ZIndex>({10});

        auto threshold_line = world->entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<RectRenderable>({0.0f, 0.0f, false, 0x00FF0055})
        .set<RenderStatus>({})
        .set<Align>({0.0f, 0.0f, 0.0f, 0.5f})
        .set<Expand>({true, 0, 0, 1.0, true, 0, 0, 0.5})
        .set<ZIndex>({10});

        auto endowment_monthly_income = world->entity()
        .is_a(UIElement)
        .child_of(threshold_line)
        .set<Position, Local>({4.0f, -4.0f})
        .set<TextRenderable>({"0.5%", "ATARISTOCRAT", 16.0f, 0xFFFFFFFF})
        .set<ZIndex>({30});
        
        auto threshold_amount = world->entity()
        .is_a(UIElement)
        .child_of(threshold_line)
        .set<Position, Local>({4.0f, 12.0f})
        .set<TextRenderable>({"$1200", "ATARISTOCRAT", 16.0f, 0x00FF00FF})
        .set<ZIndex>({30});

        auto badges = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .set<Position, Local>({84.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        create_badge(badges, UIElement, "Healthbar", 0x61c300ff);
    }
    else if (editor_type == EditorType::InSilicoLab)
    {
        auto canvas = leaf.target<EditorCanvas>();

        auto grey_bkg = world->entity()
        .is_a(UIElement)
        .child_of(canvas)
        .add<UIYoga>()
        .set<UIFillParent>({})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x868485ff})
        .set<ZIndex>({9});

        // Portrait frame: a Yoga root filling the panel that centres the image.
        // Replaces Expand + Align + Constrain, all of which read bounds that
        // aren't final until a later phase -- the source of the flicker.
        auto profile_frame = world->entity()
        .is_a(UIElement)
        .child_of(canvas)
        .add<UIYoga>()
        .set<UIFillParent>({})
        .set<UIFlexContainer>({
            YGFlexDirectionColumn, YGWrapNoWrap,
            YGJustifyCenter, YGAlignCenter, 0.0f})
        .set<ZIndex>({10});

        auto profile = world->entity()
        .is_a(UIElement)
        .child_of(profile_frame)
        .add<UIYoga>()
        // allow_upscale: the old Constrain path scaled up to fill the panel too.
        .set<UIContainParent>({0.0f, 0.0f, 0.0f, 0.0f, true})
        .set<ImageCreator>({"../assets/heonae_profile.png", 1.0f, 1.0f})
        .set<ZIndex>({10});

        auto badges = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .set<Position, Local>({84.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        
        // create_badge(badges, UIElement, "Heonae", 0xc72783ff);
        // create_badge(badges, UIElement, "Kahlo", 0x782910ff);
        create_badge(badges, UIElement, "Landaree", 0xebda59, false, false, {}, {0}, {"L"}, {0xebda59});
        create_badge(badges, UIElement, "Virtual", 0x619393ff);
        // create_badge(badges, UIElement, "Physical", 0x619393ff);
        
    }
    else if (editor_type == EditorType::Droid)
    {
        auto canvas = leaf.target<EditorCanvas>();

        auto grey_bkg = world->entity()
        .is_a(UIElement)
        .child_of(canvas)
        .add<UIYoga>()
        .set<UIFillParent>({})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x3b3b3bff})
        .set<ZIndex>({9});

        // The droid is a live render3d scene rather than a portrait bitmap: the
        // texture behind this ImageRenderable is the offscreen viewport in
        // panel3d.cpp, redrawn each frame before nanovg flushes. It fills the
        // canvas, and Panel3DSyncSystem keeps the render resolution matched to
        // whatever size Yoga gives it.
        //
        // UIAspectRatio with an undefined ratio opts out of the intrinsic
        // aspect ratio images normally publish -- the viewport is whatever
        // shape the panel is, not whatever shape the texture happens to be.
        auto viewport = world->entity()
        .is_a(UIElement)
        .child_of(canvas)
        .add<UIYoga>()
        .set<UIFillParent>({})
        .add<UIAspectRatio>()
        .add<Panel3DViewport>()
        .set<ImageRenderable>({panel3d::image_handle(), 1.0f, 1.0f, 0.0f, 0.0f})
        .set<ZIndex>({10});

        world->entity()
        .is_a(UIElement)
        .set<ImageCreator>({"../assets/mnist_version.png", 1.0f, 1.0f})
        // .set<Align>({-0.5f, -0.5f, 1.0f, 0.0f})
        .set<ZIndex>({15})
        .child_of(grey_bkg);

        auto badges = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .set<Position, Local>({84.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());


        // create_badge(badges, UIElement, "Heonae", 0xff75baff, false, false, {}, {0}, {"H"}, {0xff75baff});
        create_badge(badges, UIElement, "Aeri Peach", 0xe66c25ff, false, false, {}, {0}, {"A"}, {0xe66c25ff});
        create_badge(badges, UIElement, "Physical", 0x619393ff);
        
    }
    else if (editor_type == EditorType::ImaginaryInterlocutor)
    {
        // TODO: Message text size...
        state.options.push_back({"Show Messages", true, "Show messages as generically dependent continuants"});
        state.options.push_back({"Show Interpretations", true, "Show in-situ entity/relationship badges"});
        state.options.push_back({"Show BFO", true, "Show basic formal ontology annotation sprites"});
        
        auto canvas = leaf.target<EditorCanvas>();

        auto annotator = world->entity("InterlocutorAnnotator");

        auto black_bkg = world->entity()
        .is_a(UIElement)
        .child_of(canvas)
        .add<UIYoga>()
        .set<UIFillParent>({})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x000000FF})
        .set<ZIndex>({9});

        // Chat column: message history takes all the space the fixed-height
        // input bar doesn't. One Yoga root fills the panel; everything below
        // is laid out by flex, so nothing needs manual per-frame sizing.
        auto chat_root = world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFillParent>({})
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexStart, YGAlignStretch, 8.0f})
            .set<UIPadding>({8.0f, 8.0f, 8.0f, 8.0f})
            .set<ZIndex>({5});

        auto messages_panel = world->entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexItem>({1.0f, 1.0f, YGAlignAuto})   // absorbs leftover height
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexEnd, YGAlignStretch, 4.0f})
            .set<UIPadding>({16.0f, 12.0f, 12.0f, 12.0f})
            .set<RoundedRectRenderable>({100.0f, 100.0f, 4.0f, false, 0x050505FF})
            .set<ZIndex>({11});

        // Messages stack upward from the bottom of the history panel.
        // AlignStretch (not FlexStart) is what keeps each message row bound to
        // the panel width -- a FlexStart column gives children no definite
        // cross size, so they size to their content and overrun the frame.
        auto message_list = world->entity()
            .is_a(UIElement)
            .child_of(messages_panel)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexItem>({1.0f, 1.0f, YGAlignAuto})
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexEnd, YGAlignStretch, 4.0f});

        // Sibling *after* message_list so the badge row always sits below the
        // transcript. Inside message_list it would be pinned above every
        // message, since messages are appended to it later.
        auto meta_input = world->entity()
        .is_a(UIElement)
        .child_of(messages_panel)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
        .set<UIFlexContainer>({
            YGFlexDirectionRow, YGWrapWrap,
            YGJustifyFlexEnd, YGAlignCenter, 2.0f});

        auto input_panel = world->entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .add<UIYoga>()
            .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
            .set<UISize>({YGUndefined, 72.0f})   // taller input for multi-line text
            .set<UIPadding>({8.0f, 8.0f, 8.0f, 8.0f})
            .set<RoundedRectRenderable>({100.0f, 36.0f, 2.0f, true, 0x555555FF})
            .add<AddTagOnLeftClick, FocusChatInput>()
            .set<ZIndex>({11});

        auto input_bkg = world->entity()
            .is_a(UIElement)
            .child_of(input_panel)
            .add<UIYoga>()
            .set<UIAbsoluteEdges>({0.0f, 0.0f, 0.0f, 0.0f})
            .set<RoundedRectRenderable>({10.0f, 10.0f, 2.0f, false, 0x222327FF})
            .set<ZIndex>({10});

        auto input_text = world->entity()
            .is_a(UIElement)
            .child_of(input_panel)
            .add<UIYoga>()
            .set<TextRenderable>({"", "JetBrainsMono", 16.0f, 0xFFFFFFFF})
            .set<ZIndex>({12});

        create_badge(meta_input, UIElement, "Wesley", 0x6df0ffff, true, false, {}, {0}, {"W"}, {0x6df0ffff});
        create_badge(meta_input, UIElement, "advises", 0xa34d1aff, false, true);
        create_badge(meta_input, UIElement, "Heonae", 0xff75baff, true, false, {}, {0}, {"H"}, {0xff75baff});



        leaf.set<ChatPanel>({messages_panel, input_panel, input_text, message_list});
    }
    else if (editor_type == EditorType::Entities)
    {
        auto canvas = leaf.target<EditorCanvas>();

        world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .add<UIYoga>()
            .set<UIFillParent>({})
            .set<RectRenderable>({0.0f, 0.0f, false, 0x050505FF})
            .set<ZIndex>({9});

        // Column filling the panel: the inspector takes a fixed height off the
        // top and the list absorbs whatever is left, so selecting something
        // never costs the list its visibility.
        auto column = world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFillParent>({8.0f, 8.0f, 8.0f, 8.0f})
            .set<Position, Local>({8.0f, 8.0f})
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexStart, YGAlignStretch, 8.0f})
            .set<ZIndex>({10});

        // Inspector subpanel. Its contents are owned by update_entity_inspector,
        // which rebuilds them whenever the selection changes.
        auto inspector = world->entity()
            .is_a(UIElement)
            .child_of(column)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
            // Height from content, not fixed. An entity with several components
            // plus a relationship block is wider than one line, and clipping it
            // takes the rightmost thing first -- which is exactly the nested
            // target. Wrapping keeps every badge on screen; the min-size stops
            // the subpanel collapsing to nothing when the selection is cleared.
            .set<UIMinSize>({YGUndefined, ENTITY_INSPECTOR_HEIGHT})
            .set<UIFlexContainer>({
                YGFlexDirectionRow, YGWrapWrap,
                YGJustifyFlexStart, YGAlignCenter, 6.0f})
            .set<UIPadding>({8.0f, 10.0f, 8.0f, 10.0f})
            .set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, false, 0x161616FF})
            .set<ZIndex>({11});

        // Two input bars, identical machinery, different effect on Enter: the
        // first re-queries the headless world for a new set of rows, the second
        // reorders the rows already loaded.
        auto make_query_bar = [&](EntityQuery::Kind kind, const char* placeholder,
                                  const std::string& initial) {
            auto bar = world->entity()
                .is_a(UIElement)
                .child_of(column)
                .add(flecs::OrderedChildren)
                .add<UIYoga>()
                .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
                .set<UISize>({YGUndefined, ENTITY_QUERY_HEIGHT})
                .set<UIFlexContainer>({
                    YGFlexDirectionRow, YGWrapNoWrap,
                    YGJustifyFlexStart, YGAlignCenter, 0.0f})
                .set<UIPadding>({0.0f, 8.0f, 0.0f, 8.0f})
                .set<RoundedRectRenderable>({0.0f, 0.0f, 2.0f, false, 0x222327FF})
                .add<AddTagOnLeftClick, FocusEntityQuery>()
                .set<ZIndex>({11});

            auto text = world->entity()
                .is_a(UIElement)
                .child_of(bar)
                .add<UIYoga>()
                .add<ScissorContainer>(bar)
                .set<TextRenderable>({"", "JetBrainsMono", 16.0f, 0xFFFFFFFF})
                .set<ZIndex>({25});

            EntityQuery query;
            query.panel = leaf;
            query.input_text = text;
            query.draft = initial;
            query.applied = initial;
            query.kind = kind;
            query.placeholder = placeholder;
            bar.set<EntityQuery>(query);
            return bar;
        };

        make_query_bar(EntityQuery::Flecs, "flecs query...", ENTITY_DEFAULT_QUERY);
        make_query_bar(EntityQuery::Semantic, "search entities, or relate to selection...", "");

        // Seeded with the default query, which request_entity_query sends as
        // soon as the bridge client is reachable.
        EntitySet set;
        set.wanted = ENTITY_DEFAULT_QUERY;
        leaf.set<EntitySet>(set);

        // List viewport: absorbs the remaining height and is what the rows are
        // clipped to. It is also the height the virtual list measures itself
        // against, so the realized window tracks the space actually left over.
        auto viewport = world->entity()
            .is_a(UIElement)
            .child_of(column)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexItem>({1.0f, 1.0f, YGAlignAuto})
            .set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, false, 0x0B0B0BFF})
            .set<ZIndex>({11});

        // Draws nothing; it exists only to define the crop rectangle, inset from
        // the viewport on every edge. Clipping to the viewport itself would let
        // a partially scrolled row reach the viewport's very top edge, which
        // sits just under the inspector -- the rows would visibly crowd it. This
        // keeps a constant margin of background there no matter where the list
        // is scrolled to.
        auto clip = world->entity()
            .is_a(UIElement)
            .child_of(viewport)
            .add<UIYoga>()
            .set<UIAbsoluteEdges>({ENTITY_LIST_PAD, ENTITY_LIST_PAD, ENTITY_LIST_PAD, ENTITY_LIST_PAD});

        // The row container. Absolutely positioned inside the viewport so the
        // scroll offset is a style value Yoga applies, and so its height is its
        // own content rather than the viewport's -- the rows are meant to
        // overflow and be cropped. Left/right insets stand in for padding,
        // which would otherwise shift the absolute origin ambiguously.
        auto entity_list = world->entity()
            .is_a(UIElement)
            .child_of(viewport)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIAbsoluteEdges>({ENTITY_LIST_PAD, ENTITY_LIST_PAD, ENTITY_LIST_PAD, YGUndefined})
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexStart, YGAlignStretch, 0.0f})
            .add<ScissorContainer>(clip)
            .set<ZIndex>({12});

        // shown is seeded to something selected can never equal, so the very
        // first inspector pass builds the empty-state placeholder.
        leaf.set<EntitySelection>({inspector, entity_list, SIZE_MAX, SIZE_MAX - 1});

        VirtualList list;
        list.viewport = viewport;
        list.row_pitch = ENTITY_ROW_PITCH;
        list.pad = ENTITY_LIST_PAD;
        list.item_count = 0;   // set by entity_query_response once rows arrive
        list.build_row = [UIElement, leaf](flecs::entity parent, size_t row_position) {
            // `row_position` is a slot in the list; everything below keys off the
            // data index it currently maps to, so a re-ranking moves rows around
            // without disturbing selection or the per-entity colour.
            const EntityOrder* ordering = leaf.try_get<EntityOrder>();
            size_t index = ordering ? ordering->data_index(row_position) : row_position;

            const EntitySet& set = entity_set_of(leaf);
            if (index >= set.names.size()) return;

            const std::string& name = set.names[index];
            bool prose = entity_is_prose(set, index);
            uint32_t color = prose ? BADGE_STRING_COLOR : entity_badge_color(index);

            const EntitySelection* selection = leaf.try_get<EntitySelection>();
            bool is_selected = selection && selection->selected == index;

            // Fixed-height wrapper. The badge sizes itself from its text, but
            // the list's arithmetic only holds if the row pitch is uniform, so
            // the badge is centred inside a row of exactly ENTITY_ROW_PITCH
            // rather than being allowed to set the row height itself.
            auto row = world->entity()
                .is_a(UIElement)
                .child_of(parent)
                .add(flecs::OrderedChildren)
                .add<UIYoga>()
                .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
                .set<UISize>({YGUndefined, ENTITY_ROW_PITCH})
                .set<UIFlexContainer>({
                    YGFlexDirectionRow, YGWrapNoWrap,
                    YGJustifyFlexStart, YGAlignCenter, 0.0f})
                .set<ZIndex>({15});

            // Selected rows are marked on the row, not the badge: recolouring
            // the badge would fight the per-entity colour that identifies it.
            if (is_selected) {
                row.set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, false, 0x2A2A2AFF});
            }

            // A message is not named by its first letter, so the symbol sprite
            // is dropped for prose -- it would be labelling content as if it
            // were an identifier.
            std::vector<std::string> postfix_ids;
            std::vector<uint32_t> postfix_tints;
            if (!prose) {
                postfix_ids.push_back(name.substr(0, 1));
                postfix_tints.push_back(color);
            }

            // The only place truncation is right: the list's index<->y
            // arithmetic needs every row exactly ENTITY_ROW_PITCH tall, so a
            // row cannot grow to fit its text. Selecting the entity shows it
            // whole in the inspector.
            std::string row_label = ellipsize_text(name, "CharisSIL", 16.0f,
                                                   BADGE_MAX_TEXT_WIDTH);

            create_badge_impl(row, UIElement, row_label.c_str(), color,
                              false, false, {}, {}, postfix_ids, postfix_tints, nullptr,
                              prose ? BadgeStyle::String : BadgeStyle::Symbol)
                .set<EntityBadge>({name, index, leaf})
                .set<CallbackOnLeftClick>({clicked_entity_badge});
        };

        entity_list.set<VirtualList>(list);
    }
    else if (editor_type == EditorType::Lexicon)
    {
        auto canvas = leaf.target<EditorCanvas>();

        world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .add<UIYoga>()
            .set<UIFillParent>({})
            .set<RectRenderable>({0.0f, 0.0f, false, 0x050505FF})
            .set<ZIndex>({9});

        auto column = world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFillParent>({10.0f, 10.0f, 10.0f, 10.0f})
            .set<Position, Local>({10.0f, 10.0f})
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexStart, YGAlignFlexStart, 8.0f})
            .set<ZIndex>({10});

        // The sync system owns everything inside the column: the word's
        // character strip, and beneath it the transducer's derived row when a
        // suffix rule has been applied.
        leaf.set<LexiconPanel>({column, ""});
    }
    else if (editor_type == EditorType::Transducer)
    {
        auto canvas = leaf.target<EditorCanvas>();

        world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .add<UIYoga>()
            .set<UIFillParent>({})
            .set<RectRenderable>({0.0f, 0.0f, false, 0x050505FF})
            .set<ZIndex>({9});

        auto column = world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFillParent>({10.0f, 10.0f, 10.0f, 10.0f})
            .set<Position, Local>({10.0f, 10.0f})
            .set<UIFlexContainer>({
                YGFlexDirectionColumn, YGWrapNoWrap,
                YGJustifyFlexStart, YGAlignFlexStart, 6.0f})
            .set<ZIndex>({10});

        leaf.set<TransducerPanel>({column, "unrendered"});
    }
    else if (editor_type == EditorType::Bookshelf)
    {
        auto bookshelf_layer = world->entity()
            .is_a(UIElement)
            .child_of(leaf.target<EditorCanvas>())
            .set<LayoutBox>({LayoutBox::Horizontal, 8.0f})
            .add<FitChildren>()
            .set<Expand>({true, 4.0f, 4.0f, 1.0f, true, 4.0f, 4.0f, 1.0f});
        
        // 2. Add books with 'Constrain' and 'cap_to_intrinsic'
        // This ensures they fit within the parent bounds while maintaining aspect ratios
        std::vector<std::string> covers = {
            // "cover_james.jpg", 
            // "cover_cognitive_theory.jpg", 
            // "cover_soar.jpg", 
            // "cover_readings_in_kr.jpg"
            // "image00187.jpg",
            "image00188.jpg",
            "image00191.jpg"
        };

        for (const auto& cover : covers) {
            world->entity()
                .is_a(UIElement)
                .child_of(bookshelf_layer)
                .set<ImageCreator>({"../assets/" + cover, 1.0f, 1.0f})
                // Use Constrain{true, true} to force the image to fit the parent 
                // width/height without overflowing the EditorCanvas
                .set<Constrain>({true, true}) 
                .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f, true})
                .set<ZIndex>({10});
        }

        // TODO: Papers
        // world->entity()
        // .is_a(UIElement)
        // .child_of(bookshelf_layer)
        // .set<ImageCreator>({"../assets/cuct.png", 1.0f, 1.0f})
        // .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        // .set<ZIndex>({10});
    } 
    else if (editor_type == EditorType::Hearing)
    {
        std::cout << "Creating Hearing editor..." << std::endl;

        auto badges = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .set<Position, Local>({84.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());

        create_badge(badges, UIElement, "System", 0x1361b0ff, false);
        create_badge(badges, UIElement, "Microphone", 0xc43131ff, false);

        // Create vertical layout for mel spectrograms
        auto hearing_layer = world->entity()
            .is_a(UIElement)
            .child_of(leaf.target<EditorCanvas>())
            .set<LayoutBox>({LayoutBox::Vertical, 4.0f})
            .set<Align>({-0.5f, -0.5f, 0.5f, 0.5f});
        // Look up the mel spec renderer entities
        auto micRenderer = world->lookup("MelSpecRenderer");
        auto sysAudioRenderer = world->lookup("SystemAudioRenderer");

        std::cout << "MicRenderer exists: " << (micRenderer ? "yes" : "no") << std::endl;
        std::cout << "SysAudioRenderer exists: " << (sysAudioRenderer ? "yes" : "no") << std::endl;

        if (micRenderer && micRenderer.has<MelSpecRender>())
        {
            auto melSpec = micRenderer.get<MelSpecRender>();
            std::cout << "Mic texture handle: " << melSpec.nvgTextureHandle
                      << " size: " << melSpec.width << "x" << melSpec.height << std::endl;

            // Create microphone mel spec display
            world->entity()
                .is_a(UIElement)
                .child_of(hearing_layer)
                .set<ImageRenderable>({melSpec.nvgTextureHandle, 1.0f, 1.0f, (float)melSpec.width, (float)melSpec.height})
                // .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 0.5f, false})
                // .set<Constrain>({true, true})
                .set<ZIndex>({10});

            // Add label for microphone
            // world->entity()
            //     .is_a(UIElement)
            //     .child_of(hearing_layer)
            //     .set<Position, Local>({8.0f, 0.0f})
            //     .set<TextRenderable>({"Microphone", "ATARISTOCRAT", 12.0f, 0xAAAAAAFF})
            //     .set<ZIndex>({20});
        }

        if (sysAudioRenderer && sysAudioRenderer.has<MelSpecRender>())
        {
            auto melSpec = sysAudioRenderer.get<MelSpecRender>();
            std::cout << "Sys texture handle: " << melSpec.nvgTextureHandle
                      << " size: " << melSpec.width << "x" << melSpec.height << std::endl;

            // Create system audio mel spec display
            world->entity()
                .is_a(UIElement)
                .child_of(hearing_layer)
                .set<ImageRenderable>({melSpec.nvgTextureHandle, 1.0f, 1.0f, (float)melSpec.width, (float)melSpec.height})
                // .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 0.5f, false})
                // .set<Constrain>({true, true})
                .set<ZIndex>({10});

            // Add label for system audio
            // world->entity()
            //     .is_a(UIElement)
            //     .child_of(hearing_layer)
            //     .set<Position, Local>({8.0f, 0.0f})
            //     .set<TextRenderable>({"System Audio", "ATARISTOCRAT", 12.0f, 0xAAAAAAFF})
            //     .set<ZIndex>({20});
        }
    }
    else if (editor_type == EditorType::VNCStream)
    {
        state.options.push_back({"Passthrough Keyboard", true, "Keys pressed in Thornfield are sent to the VNC Stream"});
        state.options.push_back({"Passthrough Mouse", true, "Mouse events in Thornfield are sent to the VNC Stream"});

        auto badges = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .set<Position, Local>({84.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());

        create_badge(badges, UIElement, "Docker", 0x1d60e6ff);
        create_badge(badges, UIElement, "plasma-productivity-1", 0x9740f6ff, true);
        create_badge(badges, UIElement, "192.168.1.104", 0xa7a7a7ff, true);
        create_badge(badges, UIElement, "5901", 0xf64242ff, true);
        create_badge(badges, UIElement, "Kubuntu", 0xe2521fff);
        create_badge(badges, UIElement, "22.04", 0xe2521fff, true);
        // flecs::entity capture = create_badge(badges, UIElement, "Capture", 0x888888ff, false, {}, {}, {"checkbox"});
        

        const char* vnc_host = getenv("VNC_SERVER_HOST");
        if (!vnc_host) {
            vnc_host = "localhost";
            vnc_host = "192.168.1.104";
        }
        std::cout << "[VNC] VNC server host: " << vnc_host << " (ports 5901-5904)" << std::endl;
        
        int port = 5901;
        std::string host_string = std::string(vnc_host) + ":" + std::to_string(port);

        VNCData data = get_vnc_source(leaf, vnc_host, port);
        // const VNCClient* vncClient = vncStreamSource.try_get<VNCClient>();

        flecs::entity vnc_active_outline_indicator = world->entity()
        .is_a(UIElement)
        .set<ZIndex>({100})
        .set<RectRenderable>({100, 1, true, 0x00ff00ff})
        .set<Align>({0.0f, -1.0f, 0.0f, 1.0f})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0, 0, 0.0f})
        .child_of(leaf.target<EditorCanvas>());

        flecs::entity vnc_entity = world->entity()
        .is_a(UIElement)
        .add<ActiveIndicator>(vnc_active_outline_indicator)
        .add<IsStreamingFrom>(data.vnc_stream)
        .set<ImageRenderable>({data.client->nvgHandle, 1.0f, 1.0f, (float)data.client->width, (float)data.client->height})
        .set<ZIndex>({9})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0, 0, 0})
        .set<Constrain>({true, true})
        .set<Align>({-0.5f, -0.5f, 0.5f, 0.5f})
        // .add<DebugRenderBounds>()
        .add<X11Container>()
        .child_of(leaf.target<EditorCanvas>());


    } else if(editor_type == EditorType::Episodic)
    {

        auto badges = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .set<Position, Local>({84.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());

        auto messageBfoSprite = world->entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ZIndex>({20})
        .set<ImageCreator>({"../assets/bfo/temporal_interval.png", 1.0f, 1.0f});
        create_badge(badges, UIElement, "24 seconds", 0xf039e0ff, true);

        // Modulus rows

        // TODO: Fractal granularity navigation
        auto channels = world->entity()
        .is_a(UIElement)
        .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        .set<LayoutBox>({LayoutBox::Vertical})
        .add(flecs::OrderedChildren)
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .child_of(leaf.target<EditorCanvas>());

        // auto channels_2 = world->entity()
        // .is_a(UIElement)
        // .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        // .set<LayoutBox>({LayoutBox::Vertical})
        // .add(flecs::OrderedChildren)
        // .add<ScissorContainer>(leaf.target<EditorCanvas>())
        // .child_of(leaf.target<EditorCanvas>());

        // auto channels_3 = world->entity()
        // .is_a(UIElement)
        // .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        // .set<LayoutBox>({LayoutBox::Vertical})
        // .add(flecs::OrderedChildren)
        // .add<ScissorContainer>(leaf.target<EditorCanvas>())
        // .child_of(leaf.target<EditorCanvas>());

        auto frameChannel = world->entity()
        .is_a(UIElement)
        .set<FilmstripData>({
            .frame_limit = 8,
            .frames = {},
            .mode = FilmstripMode::Uniform,  // Change to FilmstripMode::Uniform for fixed 3s intervals
            .spike_threshold = 0.15f,          // Capture when cosDiff >= 0.15 (significant visual change)
            .spike_cooldown = 1.0f             // Min 1 second between spike captures
        })
        .set<TimeEventRowChannel>({8})
        // Custom scroll layout - no LayoutBox, positions set directly by FilmstripScrollSystem
        .set<Expand>({true, 0, 0, 1.0f, false, 0, 0, 1.0})
        // .add<DebugRenderBounds>()
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .child_of(channels);

        // TODO: Framespacer should have a component which copies the height of the maximum frameChannel child...
        auto frameSpacer = world->entity()
        .is_a(UIElement)
        .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        .set<RectRenderable>({10.0f, 100.0f, false, 0xFFFF00FF })
        .add<CopyChildHeight>(frameChannel)
        // .add<DebugRenderBounds>()
        .child_of(channels);

        leaf.target<EditorCanvas>().add<FilmstripChannel>(frameChannel);

        // Mel spectrogram channel (24 seconds of system audio)
        // auto melSpecChannel = world->entity()
        // .is_a(UIElement)
        // .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        // .set<LayoutBox>({LayoutBox::Horizontal})
        // .add(flecs::OrderedChildren)
        // .child_of(channels)
        // .add<ScissorContainer>(leaf.target<EditorCanvas>());
        

        // Look up the system audio mel spec renderer
        auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
        std::cout << "[Episodic] SystemAudioRenderer lookup: " << (sysAudioRenderer ? "found" : "NOT FOUND") << std::endl;
        if (sysAudioRenderer && sysAudioRenderer.has<MelSpecRender>())
        {
            auto melSpec = sysAudioRenderer.get<MelSpecRender>();
            std::cout << "[Episodic] MelSpec texture handle: " << melSpec.nvgTextureHandle
                      << " size: " << melSpec.width << "x" << melSpec.height << std::endl;

            // Create mel spec display element - starts at right, slides left as it fills
            world->entity()
                .is_a(UIElement)
                .child_of(channels)
                .set<ImageRenderable>({melSpec.nvgTextureHandle, 1.0f, 1.0f, (float)melSpec.width, (float)melSpec.height, nvgRGBA(255,255,255,255)})
                .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 0.0f})
                .add<ScissorContainer>(leaf.target<EditorCanvas>())
                .add<MelSpecSource>(sysAudioRenderer)
                .set<ZIndex>({18});
            std::cout << "[Episodic] Created mel spec display in channel (with fill-based positioning)" << std::endl;
        }
        else
        {
            std::cout << "[Episodic] SystemAudioRenderer not found or missing MelSpecRender component!" << std::endl;
        }

        // DINO cosine similarity line chart channel
        // Capacity: ~260 samples - 240 for 24 seconds visible + 20 extra for smooth scrolling
        // Extra samples extend past the right edge so new data scrolls in smoothly
        {
            LineChartData chartData;
            chartData.capacity = 260;
            chartData.min_value = 0.0f;
            chartData.max_value = 1.0f;
            chartData.fill_color = 0xFFFFFF30;  // Semi-transparent white fill
            chartData.line_color = 0xFFFFFFFF;  // Solid white line
            chartData.sample_interval = 0.1f;   // Sample every 100ms
            chartData.processing_lag = 0.15f;   // ~150ms DINO inference lag

            world->entity("DinoSimilarityChart")
                .is_a(UIElement)
                .child_of(channels)
                .set<LineChartData>(chartData)
                .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 0.0f})
                .set<UIElementSize>({100.0f, 48.0f})  // Fixed height for the chart
                .add<ScissorContainer>(leaf.target<EditorCanvas>())
                .add<LineChartChannel>()
                .set<ZIndex>({19});
            std::cout << "[Episodic] Created DINO similarity line chart channel" << std::endl;
        }

        // TODO: Implement scissors/vertical scrollbar

        for (size_t i = 0; i < 12; i++)
        {
            // TODO: Interface/method to convert an ordinary event channel to a frame/melspec stream...
            flecs::entity channel = world->entity()
                .is_a(UIElement)
                .child_of(channels)
                .set<RectRenderable>({10.0f, 24.0f, false, i % 2 == 0 ? 0x222327FF : 0x121212FF })
                .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
                .add<ScissorContainer>(leaf.target<EditorCanvas>())
                .add<TimeEventRowChannel>()
                .set<ZIndex>({12});

            if (i == 0)
            {
                create_process_segment(UIElement, leaf, channel, {0, 0});
            }
        }
            
        // for (size_t i = 0; i < 4; i++)
        // {
        //     world->entity()
        //     .is_a(UIElement)
        //     .set<Align>({0.0f, 0.0f, 0.8f, 0.0f})
        //     .set<CustomRenderable>({24*3, 24*3, false, i % 2 == 1 ? 0x222327FF : 0x121212FF, 0, 0, draw_diamond})
        //     .set<ZIndex>({14})
        //     .add<ScissorContainer>(leaf.target<EditorCanvas>())
        //     .child_of(channels_2);
        //     // .child_of(leaf.target<EditorCanvas>());

        //     world->entity()
        //     .is_a(UIElement)
        //     .child_of(channels_3)
        //     .set<Align>({0.0f, 0.0f, 0.8f, 0.0f})
        //     // .add<DebugRenderBounds>()
        //     .set<RectRenderable>({10.0f, 24.0f*3, false, i % 2 == 1 ? 0x222327FF : 0x121212FF })
        //     .add<ScissorContainer>(leaf.target<EditorCanvas>())
        //     .set<Expand>({true, 24*1.5f, 0, 0.2f, false, 0, 0, 0})
        //     .set<ZIndex>({14});
        // }
    } else if (editor_type == EditorType::BFO)
    {
        auto bfo_editor = world->entity()
        .is_a(UIElement)
        // .set<LayoutBox>({LayoutBox::Horizontal})
        // .add(flecs::OrderedChildren)
        .set<RectRenderable>({156.0f, 195.0f, false, 0x000000FF})
        .set<ZIndex>({30})
        .child_of(leaf.target<EditorCanvas>());

        auto bfo_editor_outline = world->entity()
        .is_a(UIElement)
        .set<RectRenderable>({156.0f, 195.0f, true, 0xc92b23FF})
        .set<ZIndex>({32})
        .child_of(leaf.target<EditorCanvas>());
    
        setup_bfo_hierarchy(bfo_editor, UIElement);
        

        // std::vector<std::string> bfo_categories = {"entity", "continuant", "occurrent", "process_boundary", "process", "spatiotemporal_region", "temporal_interval", "independent_continuant", "material"entity, "fiat_object_part", "specifically_dependent_continuant", "generically_dependent_continuant", "object_aggregate", "object", "site", "spatial_region", "immaterial_entity"};
    } else if (editor_type == EditorType::SceneGraph)
    {
        flecs::entity root = world->entity()
        .is_a(UIElement)
        .set<RectRenderable>({0.0f, 24.0f, false, 0x222327FF })
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0, 0, 0})
        .set<ZIndex>({15})
        .child_of(leaf.target<EditorCanvas>());

        // TODO: Sprite?
    }
    else
    {
        // auto editor_icon_bkg_square = world->entity()
        // .is_a(UIElement)
        // .child_of(leaf.target<EditorCanvas>())
        // .set<Position, Local>({4.0f, 12.0f})
        // .set<TextRenderable>({editor_types[(int)editor_type].c_str(), "Inter", 16.0f, 0xFFFFFFFF})
        // .set<ZIndex>({1000});
    }
}

void replace_editor_content(flecs::entity leaf, EditorType editor_type, flecs::entity UIElement)
{
    // Remove per-panel state so systems don't access destroyed entities.
    if (leaf.has<ChatPanel>()) {
        leaf.remove<ChatPanel>();
    }
    if (leaf.has<EntitySelection>()) {
        leaf.remove<EntitySelection>();
    }
    if (leaf.has<EntityOrder>()) {
        leaf.remove<EntityOrder>();
    }
    // EntityQuery lives on the bar entities, which die with the canvas; only
    // the panel-level set has to be cleared here.
    if (leaf.has<EntitySet>()) {
        leaf.remove<EntitySet>();
    }
    if (leaf.has<LexiconPanel>()) {
        leaf.remove<LexiconPanel>();
    }
    if (leaf.has<TransducerPanel>()) {
        leaf.remove<TransducerPanel>();
    }
    leaf.target<EditorHeader>().children([&](flecs::entity child)
    {
        child.destruct();
    });
    leaf.target<EditorCanvas>().children([&](flecs::entity child)
    {
        child.destruct();
    });
    leaf.set<EditorLeafData>({editor_type});
    create_editor_content(leaf, editor_type, UIElement);
}

void merge_editor(flecs::entity non_leaf, flecs::entity UIElement)
{
    non_leaf.remove<PanelSplit>();
    non_leaf.remove<LeftNode>(flecs::Wildcard);
    non_leaf.remove<RightNode>(flecs::Wildcard);
    non_leaf.remove<UpperNode>(flecs::Wildcard);
    non_leaf.remove<LowerNode>(flecs::Wildcard);
    non_leaf.children([](flecs::entity child) 
    {
        child.destruct();
    });
    EditorNodeArea& intermediate_area = non_leaf.ensure<EditorNodeArea>();
    create_editor(non_leaf, intermediate_area, UIElement);
}

void split_editor(PanelSplit split, flecs::entity leaf, flecs::entity UIElement)
{
    // Destroy any existing visual
    flecs::entity existing_visual = leaf.target<EditorVisual>();
    if (existing_visual.is_valid() && existing_visual.is_alive())
    {
        leaf.remove<EditorLeafData>();
        existing_visual.destruct();
    }
    leaf.set<PanelSplit>(split);
    const EditorNodeArea* node_area = leaf.try_get<EditorNodeArea>();
    // We want to reuse the existing editor_area by moving it from the leaf to the left_editor...
    if (split.dim == PanelSplitType::Horizontal)
    {   
        EditorNodeArea left_node_area = {node_area->width*split.percent, node_area->height};
        auto left_editor_leaf = world->entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(left_node_area)
            .add(flecs::OrderedChildren);
        create_editor(left_editor_leaf, left_node_area, UIElement);
        leaf.add<LeftNode>(left_editor_leaf);

        EditorNodeArea right_node_area = {node_area->width*(1.0f-split.percent), node_area->height};
        auto right_editor_leaf = world->entity()
            .child_of(leaf)
            .set<Position, Local>({node_area->width*split.percent, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(right_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(right_editor_leaf, right_node_area, UIElement);
        leaf.add<RightNode>(right_editor_leaf);
    }
    if (split.dim == PanelSplitType::Vertical)
    {   
        EditorNodeArea upper_node_area = {node_area->width, node_area->height*split.percent};
        auto upper_editor_leaf = world->entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(upper_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(upper_editor_leaf, upper_node_area, UIElement);
        leaf.add<UpperNode>(upper_editor_leaf);
        
        EditorNodeArea lower_node_area = {node_area->width, node_area->height*(1-split.percent)};
        auto lower_editor_leaf = world->entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, node_area->height*split.percent})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(lower_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(lower_editor_leaf, lower_node_area, UIElement);
        leaf.add<LowerNode>(lower_editor_leaf);
    }
}

struct RenderQueue {
    std::vector<RenderCommand> commands;

    void clear() {
        commands.clear();
    }

    void addRectCommand(const Position& pos, const RectRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Rectangle, zIndex});
    }

    void addRoundedRectCommand(const Position& pos, const RoundedRectRenderable& renderable, int zIndex, bool useGradient = false, RenderGradient renderGradient = {0, 0}, ecs_entity_t scissorEntity = 0, int depth = 0) {
        commands.push_back({pos, renderable, RenderType::RoundedRectangle, zIndex, scissorEntity, useGradient, renderGradient, depth});
    }

    void addTextCommand(const Position& pos, const TextRenderable& renderable, int zIndex, bool useGradient = false, RenderGradient renderGradient = {0, 0}, ecs_entity_t scissorEntity = 0, int depth = 0) {
        commands.push_back({pos, renderable, RenderType::Text, zIndex, scissorEntity, useGradient, renderGradient, depth});
    }

    void addImageCommand(const Position& pos, const ImageRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Image, zIndex});
    }

    void addLineCommand(const Position& pos, const LineRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Line, zIndex});
    }

    void addQuadraticBezierCommand(const Position& pos, const QuadraticBezierRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::QuadraticBezier, zIndex});
    }

    void addCustomCommand(const Position& pos, const CustomRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::CustomRenderable, zIndex});
    }

    void sort() {
        // Stable, so commands that tie on both layer and depth keep the order
        // they were queued in. std::sort would reorder them arbitrarily, and
        // since entity iteration order shifts whenever tables change -- which
        // is every time the virtual list rebuilds a row -- that showed up as
        // overlapping elements flickering between frames.
        std::stable_sort(commands.begin(), commands.end());
    }
};

void error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    Window& window_comp = world->lookup("GLFWState").ensure<Window>();
    window_comp.width = width;
    window_comp.height = height;
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

bool point_in_bounds(float x, float y, UIElementBounds bounds)
{
    return (x >= bounds.xmin && x <= bounds.xmax && y >= bounds.ymin && y <= bounds.ymax);
}

// The rect a renderable is clipped to: its own ScissorContainer target if it
// carries one, otherwise the nearest ancestor's.
//
// Resolving up the hierarchy is what lets a panel clip a whole subtree by
// tagging one container. Without it every entity would have to be tagged
// individually, which is impossible for subtrees built by shared factories like
// create_badge() -- the caller never sees the internals it would need to tag.
static ecs_entity_t resolve_scissor(flecs::entity e)
{
    for (flecs::entity c = e; c.is_valid(); c = c.parent()) {
        if (c.has<ScissorContainer>(flecs::Wildcard)) {
            flecs::entity target = c.target<ScissorContainer>();
            if (target.is_valid()) return target;
        }
    }
    return 0;
}

// The rectangle a ScissorContainer target clips to: its own layout box, taken
// from Position<World> and UIElementSize.
//
// Read with try_get rather than ecs_ensure because the render queue runs inside
// a system, where the world is deferred: ecs_ensure would hand back a pointer
// into the command queue's temporary storage instead of the live component, and
// the resulting rect does not clip anything.
static bool scissor_rect(flecs::entity clip, float& x, float& y, float& w, float& h)
{
    if (!clip.is_valid() || !clip.is_alive()) return false;
    const Position* world_pos = clip.try_get<Position, World>();
    const UIElementSize* size = clip.try_get<UIElementSize>();
    if (!world_pos || !size || size->width <= 0.0f || size->height <= 0.0f) return false;

    x = world_pos->x;
    y = world_pos->y;
    w = size->width;
    h = size->height;
    return true;
}

// Whether a click at (x, y) can reach `e` -- inside its own bounds, and inside
// the rect it is clipped to if it is clipped at all. A virtualized list keeps
// one row hanging past the bottom of its panel so scrolling has something to
// pull up; that row is invisible, and without this check it would still swallow
// clicks landing on whatever panel is drawn below.
static bool click_reaches(flecs::entity e, const UIElementBounds& bounds, float x, float y)
{
    if (!point_in_bounds(x, y, bounds)) return false;

    ecs_entity_t scissor = resolve_scissor(e);
    if (!scissor) return true;

    float cx, cy, cw, ch;
    if (!scissor_rect(world->entity(scissor), cx, cy, cw, ch)) return true;
    return point_in_bounds(x, y, {cx, cy, cx + cw, cy + ch});
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // A VirtualList under the cursor claims the wheel. Otherwise it falls
    // through to the Droid panel's orbit camera, which was the only consumer
    // before lists could scroll and which applies the delta on the next frame.
    const CursorState& cursor = world->lookup("GLFWState").ensure<CursorState>();

    static flecs::query<VirtualList> lists = world->query<VirtualList>();
    bool consumed = false;
    lists.each([&](flecs::entity e, VirtualList& list) {
        if (consumed || !list.viewport.is_valid()) return;
        const UIElementBounds* bounds = list.viewport.try_get<UIElementBounds>();
        if (!bounds || !point_in_bounds(cursor.x, cursor.y, *bounds)) return;

        // Clamped by VirtualListSystem, which is the only place that knows the
        // current viewport height and therefore the maximum scroll.
        list.scroll -= (float)yoffset * list.row_pitch * 3.0f;
        consumed = true;
    });

    if (!consumed) panel3d::add_scroll(yoffset);
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    // TODO: Move to Observer?
    CursorState& cursor_state = world->lookup("GLFWState").ensure<CursorState>();
    // TODO: Query for hoverable UIElement 
    flecs::query hoverable_elements = world->query_builder<AddTagOnHoverEnter, AddTagOnHoverExit, UIElementBounds>()
    .term_at(0).second(flecs::Wildcard).optional()
    .term_at(1).second(flecs::Wildcard).optional()
    .build();

    hoverable_elements.each([&](flecs::entity ui_element, AddTagOnHoverEnter, AddTagOnHoverExit, UIElementBounds& bounds) {
        bool in_bounds_prior = point_in_bounds(cursor_state.x, cursor_state.y, bounds);
        bool in_bounds_post = point_in_bounds(xpos, ypos, bounds);
        
        if (in_bounds_prior && !in_bounds_post && ui_element.has<AddTagOnHoverExit>(flecs::Wildcard))
        {
            // TODO: Store hover state...
            world->event<HoverExitEvent>()
            .id<UIElementBounds>()
            .entity(ui_element)
            .enqueue();  
        } else if (!in_bounds_prior && in_bounds_post && ui_element.has<AddTagOnHoverEnter>(flecs::Wildcard))
        {   
            world->event<HoverEnterEvent>()
            .id<UIElementBounds>()
            .entity(ui_element)
            .enqueue();
        }
    });

    cursor_state.x = xpos;
    cursor_state.y = ypos;
}

void drop_callback(GLFWwindow* window, int count, const char** paths)
{
    // 1. Get cursor state
    flecs::entity glfw_state = world->lookup("GLFWState");
    const CursorState* cursorState = glfw_state.try_get<CursorState>();

    // 2. Find VNC panel under mouse
    flecs::entity target_vnc_entity = flecs::entity::null();

    auto vnc_query = world->query_builder<Position, ImageRenderable>()
        .with<IsStreamingFrom>(flecs::Wildcard)
        .term_at(0).second<World>()
        .build();

    vnc_query.each([&](flecs::entity e, Position& pos, ImageRenderable& img) {
        flecs::entity vnc_entity = e.target<IsStreamingFrom>();
        const VNCClientHandle* handle = vnc_entity.try_get<VNCClientHandle>();

        if (handle && *handle && (*handle)->eventPassthroughEnabled && (*handle)->connected) {
            target_vnc_entity = vnc_entity;
        }
    });

    // 5. Queue file transfers
    for (int i = 0; i < count; i++) {
        std::string local_path = paths[i];
        std::string filename = local_path.substr(local_path.find_last_of("/\\") + 1);
        std::string remote_path = "/home/grok/Downloads/" + filename;
    }

}

// Track mouse button state for VNC
static int g_vncButtonMask = 0;

// Called from a click handler to stop the event propagating to lower-priority
// elements under the same cursor position. See UIInputDispatch.
void consume_ui_click()
{
    world->ensure<UIInputDispatch>().consumed = true;
}

// Depth in the ChildOf hierarchy; deeper elements draw over their ancestors, so
// they get the click first when ZIndex ties.
static int ui_hierarchy_depth(flecs::entity e)
{
    int depth = 0;
    for (flecs::entity p = e.parent(); p.is_valid(); p = p.parent())
    {
        if (++depth > 64) break; // guard against cycles
    }
    return depth;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    flecs::entity glfw_state = world->lookup("GLFWState");
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            flecs::query interactive_elements = world->query_builder<AddTagOnLeftClick, UIElementBounds>()
            .term_at(0).second(flecs::Wildcard)
            .build();
            // TODO: Eventually this should use a more efficient partition bound check as the first layer
            const CursorState* cursor_state = world->lookup("GLFWState").try_get<CursorState>();

            // Gather every hit element, then dispatch in priority order so that
            // a handler can consume the click and stop it reaching whatever is
            // underneath. Emitting to all hits unordered lets a popup's option
            // row and the dropdown icon beneath it both fire, which closed the
            // menu the moment you clicked an option.
            struct ClickTarget {
                flecs::entity entity;
                int z;
                int depth;
                bool has_callback;
            };
            std::vector<ClickTarget> targets;

            interactive_elements.each([&](flecs::entity ui_element, AddTagOnLeftClick, UIElementBounds& bounds) {
                if (click_reaches(ui_element, bounds, cursor_state->x, cursor_state->y))
                {
                    const ZIndex* z = ui_element.try_get<ZIndex>();
                    targets.push_back({ui_element, z ? z->layer : 0,
                                       ui_hierarchy_depth(ui_element), false});
                }
            });

            flecs::query callback_elements = world->query_builder<CallbackOnLeftClick, UIElementBounds>()
            .build();
            callback_elements.each([&](flecs::entity ui_element, CallbackOnLeftClick& callback, UIElementBounds& bounds) {
                if (click_reaches(ui_element, bounds, cursor_state->x, cursor_state->y))
                {
                    // An element can carry both a tag and a callback; fold them
                    // into one target so it is only visited once.
                    for (auto& t : targets) {
                        if (t.entity == ui_element) { t.has_callback = true; return; }
                    }
                    const ZIndex* z = ui_element.try_get<ZIndex>();
                    targets.push_back({ui_element, z ? z->layer : 0,
                                       ui_hierarchy_depth(ui_element), true});
                }
            });

            std::stable_sort(targets.begin(), targets.end(),
                [](const ClickTarget& a, const ClickTarget& b) {
                    if (a.z != b.z) return a.z > b.z;         // topmost layer first
                    return a.depth > b.depth;                  // then innermost
                });

            UIInputDispatch& dispatch = world->ensure<UIInputDispatch>();
            dispatch.consumed = false;

            for (const ClickTarget& target : targets)
            {
                // A previous handler may have destroyed this entity (menus close
                // by destructing their subtree), so re-check before dispatching.
                if (!target.entity.is_alive()) continue;

                if (target.entity.has<AddTagOnLeftClick>(flecs::Wildcard))
                {
                    world->event<LeftClickEvent>()
                    .id<UIElementBounds>()
                    .entity(target.entity)
                    .emit();
                }

                if (target.has_callback && target.entity.is_alive())
                {
                    if (const CallbackOnLeftClick* cb = target.entity.try_get<CallbackOnLeftClick>())
                        cb->onClicked(target.entity);
                }

                if (world->ensure<UIInputDispatch>().consumed) break;
            }

            world->event<LeftClickEvent>()
            .id<CursorState>()
            .entity(glfw_state)
            .emit();


        } else if (action == GLFW_RELEASE)
        {
            world->event<LeftReleaseEvent>()
            .id<CursorState>()
            .entity(glfw_state)
            .emit();
        }
    }
    
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);

    // Update button mask based on action
    int buttonBit = 0;
    if (button == GLFW_MOUSE_BUTTON_LEFT) buttonBit = rfbButton1Mask;
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE) buttonBit = rfbButton2Mask;
    else if (button == GLFW_MOUSE_BUTTON_RIGHT) buttonBit = rfbButton3Mask;

    if (action == GLFW_PRESS) {
        g_vncButtonMask |= buttonBit;  // Set button bit
    } else if (action == GLFW_RELEASE) {
        g_vncButtonMask &= ~buttonBit;  // Clear button bit
    }

    // Find the active VNC client
    auto query = world->query<VNCClientHandle, Position, ImageRenderable>();
    query.each([&](flecs::entity e, VNCClientHandle& handle, Position& pos, ImageRenderable& img) {
        VNCClient& vnc = *handle;
        if (vnc.connected && vnc.client) {
            // Convert mouse coordinates from window space to VNC space
            int win_w, win_h;
            glfwGetWindowSize(window, &win_w, &win_h);

            float scale_w = img.width / vnc.width;
            float scale_h = img.height / vnc.height;

            int offset_x = (int)pos.x;
            int offset_y = (int)pos.y;

            // Convert to VNC coordinates
            int vnc_x = (int)((mouseX - offset_x) / scale_w);
            int vnc_y = (int)((mouseY - offset_y) / scale_h);

            // Clamp to VNC bounds
            if (vnc_x < 0) vnc_x = 0;
            if (vnc_y < 0) vnc_y = 0;
            if (vnc_x >= vnc.width) vnc_x = vnc.width - 1;
            if (vnc_y >= vnc.height) vnc_y = vnc.height - 1;

            // Send pointer event with updated button mask
            SendPointerEvent(vnc.client, vnc_x, vnc_y, g_vncButtonMask);
            std::cout << "[VNC INPUT] Mouse " << (action == GLFW_PRESS ? "press" : "release")
                        << " at VNC coords (" << vnc_x << "," << vnc_y << ") mask=" << g_vncButtonMask << std::endl;
        }
    });
}

// The Entities panel whose relation bar currently holds the keyboard, if any.
static flecs::entity focused_entity_query()
{
    static flecs::query<EntityQuery> queries = world->query<EntityQuery>();
    flecs::entity focused = flecs::entity::null();
    queries.each([&](flecs::entity leaf, EntityQuery& query) {
        if (!focused && query.focused) focused = leaf;
    });
    return focused;
}

static void char_callback(GLFWwindow* window, unsigned int codepoint)
{
    if (codepoint < 32 || codepoint >= 127) return;

    // While annotation is active it owns the keyboard outright. Binding keys
    // (digits, letters) are handled in key_callback; without this, a chat draft
    // that quietly kept focus turns every binding keystroke into typed text --
    // the same key both fails to bind and appears in the Interlocutor input.
    {
        static flecs::query<WordAnnotationSelector> annotators =
            world->query<WordAnnotationSelector>();
        bool annotating = false;
        annotators.each([&](flecs::entity, WordAnnotationSelector& selector) {
            if (selector.active) annotating = true;
        });
        if (annotating) return;
    }

    // Checked ahead of the chat draft: focus is exclusive between the two, and
    // whichever field the user last clicked owns the keystroke.
    if (flecs::entity leaf = focused_entity_query())
    {
        leaf.ensure<EntityQuery>().draft.push_back(static_cast<char>(codepoint));
        return;
    }

    // Multiple chat query... only one active
    ChatState* chat = world->try_get_mut<ChatState>();
    if (chat && chat->input_focused)
    {
        chat->draft.push_back(static_cast<char>(codepoint));
    }
}

rfbKeySym glfw_key_to_rfb_keysym(int key, int mods) {
    // 1. Handle non-printable / special keys first
    switch (key) {
        case GLFW_KEY_BACKSPACE:    return XK_BackSpace;
        case GLFW_KEY_TAB:          return XK_Tab;
        case GLFW_KEY_ENTER:        return XK_Return;
        case GLFW_KEY_PAUSE:        return XK_Pause;
        case GLFW_KEY_ESCAPE:       return XK_Escape;
        case GLFW_KEY_DELETE:       return XK_Delete;
        case GLFW_KEY_KP_0:         return XK_KP_0;
        case GLFW_KEY_KP_1:         return XK_KP_1;
        case GLFW_KEY_KP_2:         return XK_KP_2;
        case GLFW_KEY_KP_3:         return XK_KP_3;
        case GLFW_KEY_KP_4:         return XK_KP_4;
        case GLFW_KEY_KP_5:         return XK_KP_5;
        case GLFW_KEY_KP_6:         return XK_KP_6;
        case GLFW_KEY_KP_7:         return XK_KP_7;
        case GLFW_KEY_KP_8:         return XK_KP_8;
        case GLFW_KEY_KP_9:         return XK_KP_9;
        case GLFW_KEY_KP_DECIMAL:   return XK_KP_Decimal;
        case GLFW_KEY_KP_DIVIDE:    return XK_KP_Divide;
        case GLFW_KEY_KP_MULTIPLY:  return XK_KP_Multiply;
        case GLFW_KEY_KP_SUBTRACT:  return XK_KP_Subtract;
        case GLFW_KEY_KP_ADD:       return XK_KP_Add;
        case GLFW_KEY_KP_ENTER:     return XK_KP_Enter;
        case GLFW_KEY_KP_EQUAL:     return XK_KP_Equal;
        case GLFW_KEY_UP:           return XK_Up;
        case GLFW_KEY_DOWN:         return XK_Down;
        case GLFW_KEY_RIGHT:        return XK_Right;
        case GLFW_KEY_LEFT:         return XK_Left;
        case GLFW_KEY_INSERT:       return XK_Insert;
        case GLFW_KEY_HOME:         return XK_Home;
        case GLFW_KEY_END:          return XK_End;
        case GLFW_KEY_PAGE_UP:      return XK_Page_Up;
        case GLFW_KEY_PAGE_DOWN:    return XK_Page_Down;
        case GLFW_KEY_F1:           return XK_F1;
        case GLFW_KEY_F2:           return XK_F2;
        case GLFW_KEY_F3:           return XK_F3;
        case GLFW_KEY_F4:           return XK_F4;
        case GLFW_KEY_F5:           return XK_F5;
        case GLFW_KEY_F6:           return XK_F6;
        case GLFW_KEY_F7:           return XK_F7;
        case GLFW_KEY_F8:           return XK_F8;
        case GLFW_KEY_F9:           return XK_F9;
        case GLFW_KEY_F10:          return XK_F10;
        case GLFW_KEY_F11:          return XK_F11;
        case GLFW_KEY_F12:          return XK_F12;
        case GLFW_KEY_NUM_LOCK:     return XK_Num_Lock;
        case GLFW_KEY_CAPS_LOCK:    return XK_Caps_Lock;
        case GLFW_KEY_SCROLL_LOCK:  return XK_Scroll_Lock;
        case GLFW_KEY_RIGHT_SHIFT:  return XK_Shift_R;
        case GLFW_KEY_LEFT_SHIFT:   return XK_Shift_L;
        case GLFW_KEY_RIGHT_CONTROL:return XK_Control_R;
        case GLFW_KEY_LEFT_CONTROL: return XK_Control_L;
        case GLFW_KEY_RIGHT_ALT:    return XK_Alt_R;
        case GLFW_KEY_LEFT_ALT:     return XK_Alt_L;
        case GLFW_KEY_RIGHT_SUPER:  return XK_Super_R;
        case GLFW_KEY_LEFT_SUPER:   return XK_Super_L;
        case GLFW_KEY_MENU:         return XK_Menu;
        case GLFW_KEY_PRINT_SCREEN: return XK_Print;
    }

    // 2. Handle Letters (A-Z) with Shift/Caps awareness
    if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z) {
        bool shift = (mods & GLFW_MOD_SHIFT) != 0;
        bool caps = (mods & GLFW_MOD_CAPS_LOCK) != 0;
        // If Shift and CapsLock are both on, they cancel out for letters
        return (shift != caps) ? key : (key + 32); 
    }

    // 3. Handle Numbers and Symbols with Shift awareness
    if (mods & GLFW_MOD_SHIFT) {
        switch (key) {
            case GLFW_KEY_1: return XK_exclam;
            case GLFW_KEY_2: return XK_at;
            case GLFW_KEY_3: return XK_numbersign;
            case GLFW_KEY_4: return XK_dollar;
            case GLFW_KEY_5: return XK_percent;
            case GLFW_KEY_6: return XK_asciicircum;
            case GLFW_KEY_7: return XK_ampersand;
            case GLFW_KEY_8: return XK_asterisk;
            case GLFW_KEY_9: return XK_parenleft;
            case GLFW_KEY_0: return XK_parenright;
            case GLFW_KEY_MINUS: return XK_underscore;
            case GLFW_KEY_EQUAL: return XK_plus;
            case GLFW_KEY_LEFT_BRACKET:  return XK_braceleft;
            case GLFW_KEY_RIGHT_BRACKET: return XK_braceright;
            case GLFW_KEY_BACKSLASH:     return XK_bar;
            case GLFW_KEY_SEMICOLON:     return XK_colon;
            case GLFW_KEY_APOSTROPHE:    return XK_quotedbl;
            case GLFW_KEY_COMMA:         return XK_less;
            case GLFW_KEY_PERIOD:        return XK_greater;
            case GLFW_KEY_SLASH:         return XK_question;
            case GLFW_KEY_GRAVE_ACCENT:  return XK_asciitilde;
        }
    } else {
        // Not shifted, just return the character key as is
        // (GLFW keys like GLFW_KEY_COMMA are already equal to the ASCII code for ',')
        if (key >= GLFW_KEY_SPACE && key <= GLFW_KEY_GRAVE_ACCENT) {
            // Special check: GLFW uses uppercase for letters in this range.
            // But we already handled letters above.
            return key;
        }
    }

    return 0;
}

// Editor config serialization
struct EditorConfigNode {
    bool is_leaf;
    EditorType editor_type; // Only valid if is_leaf
    float split_percent;    // Only valid if !is_leaf
    PanelSplitType split_dim; // Only valid if !is_leaf
    std::unique_ptr<EditorConfigNode> child_a; // Left/Upper
    std::unique_ptr<EditorConfigNode> child_b; // Right/Lower
};

std::unique_ptr<EditorConfigNode> save_editor_config_recursive(flecs::entity node) {
    auto config = std::make_unique<EditorConfigNode>();

    const PanelSplit* split = node.try_get<PanelSplit>();
    if (split) {
        config->is_leaf = false;
        config->split_percent = split->percent;
        config->split_dim = split->dim;

        if (split->dim == PanelSplitType::Horizontal) {
            flecs::entity left = node.target<LeftNode>();
            flecs::entity right = node.target<RightNode>();
            if (left.is_valid()) config->child_a = save_editor_config_recursive(left);
            if (right.is_valid()) config->child_b = save_editor_config_recursive(right);
        } else {
            flecs::entity upper = node.target<UpperNode>();
            flecs::entity lower = node.target<LowerNode>();
            if (upper.is_valid()) config->child_a = save_editor_config_recursive(upper);
            if (lower.is_valid()) config->child_b = save_editor_config_recursive(lower);
        }
    } else {
        config->is_leaf = true;
        const EditorLeafData* leaf_data = node.try_get<EditorLeafData>();
        config->editor_type = leaf_data ? leaf_data->editor_type : EditorType::VNCStream;
    }

    return config;
}

void write_editor_config(std::ofstream& out, const EditorConfigNode& node, int depth = 0) {
    std::string indent(depth * 2, ' ');
    if (node.is_leaf) {
        out << indent << "leaf " << static_cast<int>(node.editor_type) << "\n";
    } else {
        out << indent << "split " << node.split_percent << " "
            << (node.split_dim == PanelSplitType::Horizontal ? "h" : "v") << "\n";
        if (node.child_a) write_editor_config(out, *node.child_a, depth + 1);
        if (node.child_b) write_editor_config(out, *node.child_b, depth + 1);
    }
}

void save_editor_layout(flecs::entity editor_root) {
    auto config = save_editor_config_recursive(editor_root);
    std::ofstream out("editor_layout.cfg");
    if (out.is_open()) {
        write_editor_config(out, *config);
        std::cout << "Editor layout saved to editor_layout.cfg" << std::endl;
    }
}

std::unique_ptr<EditorConfigNode> read_editor_config(std::ifstream& in) {
    std::string type;
    if (!(in >> type)) return nullptr;

    auto node = std::make_unique<EditorConfigNode>();
    if (type == "leaf") {
        node->is_leaf = true;
        int editor_type_int;
        in >> editor_type_int;
        node->editor_type = static_cast<EditorType>(editor_type_int);
    } else if (type == "split") {
        node->is_leaf = false;
        char dim;
        in >> node->split_percent >> dim;
        node->split_dim = (dim == 'h') ? PanelSplitType::Horizontal : PanelSplitType::Vertical;
        node->child_a = read_editor_config(in);
        node->child_b = read_editor_config(in);
    }
    return node;
}

void apply_editor_config(const EditorConfigNode& config, flecs::entity node, flecs::entity UIElement) {
    if (config.is_leaf) {
        EditorNodeArea& area = node.ensure<EditorNodeArea>();
        create_editor(node, area, UIElement);
        // Override the default Void type and create content
        node.set<EditorLeafData>({config.editor_type});
        create_editor_content(node, config.editor_type, UIElement);
    } else {
        split_editor({config.split_percent, config.split_dim}, node, UIElement);

        if (config.split_dim == PanelSplitType::Horizontal) {
            flecs::entity left = node.target<LeftNode>();
            flecs::entity right = node.target<RightNode>();
            // Remove the default editor content created by split_editor so we can apply config
            if (left.is_valid() && config.child_a) {
                flecs::entity visual = left.target<EditorVisual>();
                if (visual.is_valid()) visual.destruct();
                left.remove<EditorLeafData>();
                apply_editor_config(*config.child_a, left, UIElement);
            }
            if (right.is_valid() && config.child_b) {
                flecs::entity visual = right.target<EditorVisual>();
                if (visual.is_valid()) visual.destruct();
                right.remove<EditorLeafData>();
                apply_editor_config(*config.child_b, right, UIElement);
            }
        } else {
            flecs::entity upper = node.target<UpperNode>();
            flecs::entity lower = node.target<LowerNode>();
            if (upper.is_valid() && config.child_a) {
                flecs::entity visual = upper.target<EditorVisual>();
                if (visual.is_valid()) visual.destruct();
                upper.remove<EditorLeafData>();
                apply_editor_config(*config.child_a, upper, UIElement);
            }
            if (lower.is_valid() && config.child_b) {
                flecs::entity visual = lower.target<EditorVisual>();
                if (visual.is_valid()) visual.destruct();
                lower.remove<EditorLeafData>();
                apply_editor_config(*config.child_b, lower, UIElement);
            }
        }
    }
}

bool load_editor_layout(flecs::entity editor_root, flecs::entity UIElement) {
    std::ifstream in("editor_layout.cfg");
    if (!in.is_open()) return false;

    auto config = read_editor_config(in);
    if (!config) return false;

    apply_editor_config(*config, editor_root, UIElement);
    std::cout << "Editor layout loaded from editor_layout.cfg" << std::endl;
    return true;
}

flecs::entity create_message(flecs::entity& UIElement, ChatPanel& chat_panel, flecs::entity speaker, std::string message)
{
    bool is_self = (speaker.name() == "Wesley"); // TODO: Obviously...

    // The row spans the full panel width (message_list stretches it) and
    // justifies its bubble to one side, so own messages sit right and replies
    // left without either being able to escape the frame.
    auto messageBox = world->entity()
    .is_a(UIElement)
    .child_of(chat_panel.message_list)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFlexContainer>({
        YGFlexDirectionRow, YGWrapNoWrap,
        is_self ? YGJustifyFlexEnd : YGJustifyFlexStart, YGAlignCenter, 4.0f});

    // The background bubble is the text's flex container, so it auto-sizes to
    // the message plus padding. flexShrink lets it give way at the panel edge,
    // which is what forces the text inside to wrap rather than overflow.
    auto example_message_bkg = world->entity()
        .is_a(UIElement)
        .child_of(messageBox) // Attached to the ChatPanel found via query
        .add<UIYoga>()
        .set<UIFlexItem>({0.0f, 1.0f, YGAlignAuto})
        .set<UIFlexContainer>({
            YGFlexDirectionColumn, YGWrapNoWrap,
            YGJustifyFlexStart, YGAlignFlexStart, 0.0f})
        .set<UIPadding>({6.0f, 8.0f, 6.0f, 8.0f})
        .set<RoundedRectRenderable>({100.0f, 16.0f, 2.0f, false, 0x121212FF})
        .set<ZIndex>({15});

    // Create the text content using the actual draft. No DynamicTextWrap here:
    // Yoga offers the measure func the width the bubble actually got, so the
    // wrap width follows the panel instead of being pushed in from outside.
    auto example_message_text = world->entity()
        .is_a(UIElement)
        .child_of(example_message_bkg)
        .add<UIYoga>()
        .set<TextRenderable>({message, "JetBrainsMono", 16.0f, 0xFFFFFFFF})
        .set<ZIndex>({17});

    if (is_self)
    {
        // I put it outside the message since it is a meta annotation
        // This might only need to be visible during certain 'entity binding' interface modes...
        auto messageBfoSprite = world->entity()
        .is_a(UIElement)
        .child_of(messageBox)
        .add<UIYoga>()
        .add<UINativeImageSize>()
        .set<ZIndex>({20})
        .set<ImageCreator>({"../assets/bfo/generically_dependent_continuant.png", 1.0f, 1.0f});

    } else if (speaker.name() == "Heonae")
    {
        example_message_bkg.set<RoundedRectRenderable>({100.0f, 16.0f, 2.0f, false, 0x12121200});
    }

    return messageBox;
}

void interlocutor_response(std::map<std::string, msgpack::object>& res_map)
{
    std::string reply = res_map["reply"].as<std::string>();
    std::cout << reply << std::endl;
    // TOOD: Include 'context' entity in the function signature of Tradewinds
    // to get the right panel and interlocutor
    world->query<ChatPanel>()
        .each([&](flecs::entity leaf, ChatPanel& chat_panel) {
            auto UIElement = world->lookup("UIElement");
            auto messageBox = create_message(UIElement, chat_panel, world->lookup("Heonae"), reply.c_str());
        });
}

// Appends the current annotation state to the training log.
//
// One JSONL line per template *state*, so the file captures edit trajectories
// -- the sequence of corrections is supervision for the binding model, not
// noise. Context rides along because the template alone is unlearnable: "I"
// binding to Wesley needs the speaker, "her" binding to 0 needs the discourse
// entity table, and both need the sentences that came before.
static void append_annotation_record(const WordAnnotationSelector& selector)
{
    if (selector.sentence_template.empty()) return;

    // Only log actual changes; navigation re-syncs would duplicate lines.
    static std::string last_logged;
    if (selector.sentence_template == last_logged) return;
    last_logged = selector.sentence_template;

    auto escape = [](const std::string& in) {
        std::string out;
        out.reserve(in.size() + 8);
        for (char c : in) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\t': out += "\\t"; break;
                default: out += c;
            }
        }
        return out;
    };

    // Runs from build/, so the log lands at the repo root.
    std::ofstream log("../annotations.jsonl", std::ios::app);
    if (!log) return;

    log << "{\"ts\": " << (long long)std::time(nullptr)
        << ", \"speaker\": \"Wesley\""
        << ", \"template\": \"" << escape(selector.sentence_template) << "\"";

    {
        std::lock_guard<std::mutex> lock(known_entities_mutex);
        log << ", \"entities\": [";
        for (size_t i = 0; i < known_entities.size(); ++i) {
            if (i) log << ", ";
            log << "{\"name\": \"" << escape(known_entities[i].label)
                << "\", \"symbol\": \"" << escape(known_entities[i].display_symbol) << "\"}";
        }
        log << "]";
    }
    {
        std::lock_guard<std::mutex> lock(previous_sentences_mutex);
        log << ", \"prior\": [";
        for (size_t i = 0; i < previous_sentences.size(); ++i) {
            if (i) log << ", ";
            log << "\"" << escape(previous_sentences[i]) << "\"";
        }
        log << "]";
    }
    log << "}\n";
}

// The transducer's current proposal for the word under inspection. One slot,
// not a cache: the panel only ever shows one word, and a stale proposal for a
// previous word must never dress up as an analysis of the current one.
static struct {
    std::string word;      // the word this proposal analyses
    std::string lemma;
    std::string suffix;
    std::string rule;
    std::string word_type;      // what the inspected word is (e.g. Plural)
    std::string variant;        // its counterpart form
    std::string variant_type;   // and what that one is
    float confidence = 0.0f;
    int support = 0;
} g_fst_proposal;
static std::string g_fst_requested;

// The rule series as last reported by the morphology server's dump: what the
// Transducer panel renders. Stale after any teach; refreshed through its own
// client so rule listing never contends with the lexicon's analyze requests.
static struct {
    bool valid = false;
    bool stale = true;
    int states = 0, arcs = 0, lexicon = 0;
    struct Rule { std::string in, out; int support; float confidence; };
    std::vector<Rule> rules;
    std::string signature;
} g_transducer;

void transducer_dump_response(std::map<std::string, msgpack::object>& res_map)
{
    g_transducer.rules.clear();
    g_transducer.valid = true;
    g_transducer.stale = false;
    g_transducer.states = res_map.count("states") ? (int)res_map["states"].as<int64_t>() : 0;
    g_transducer.arcs = res_map.count("arcs") ? (int)res_map["arcs"].as<int64_t>() : 0;
    g_transducer.lexicon = res_map.count("lexicon") ? (int)res_map["lexicon"].as<int64_t>() : 0;

    g_transducer.signature.clear();
    if (res_map.count("rules")) {
        try {
            auto rules = res_map["rules"].as<std::vector<std::map<std::string, msgpack::object>>>();
            for (auto& rule : rules) {
                g_transducer.rules.push_back({
                    rule.count("in") ? rule["in"].as<std::string>() : "",
                    rule.count("out") ? rule["out"].as<std::string>() : "",
                    rule.count("support") ? (int)rule["support"].as<int64_t>() : 0,
                    rule.count("confidence") ? (float)rule["confidence"].as<double>() : 0.0f,
                });
                auto& r = g_transducer.rules.back();
                g_transducer.signature += r.in + ">" + r.out + ":"
                    + std::to_string(r.support) + ":"
                    + std::to_string((int)(r.confidence * 1000)) + ";";
            }
        } catch (const std::exception&) {}
    }
    g_transducer.signature += "L" + std::to_string(g_transducer.lexicon);

    world->query<TransducerPanel>().each([](flecs::entity, TransducerPanel&) {});
}

// A demonstration waiting to reach the world -- kept until the bridge client
// is free, so an Enter pressed while a request is mid-flight teaches late
// rather than never.
static std::string g_pending_teach;   // "Type\tText\n..." awaiting the bridge
static std::pair<std::string, std::string> g_pending_false;   // rejected derivation

void teach_number_response(std::map<std::string, msgpack::object>& res_map)
{
    g_pending_teach.clear();
    // The lexicon just changed; the current word deserves a fresh analysis,
    // and the rule series shown in the Transducer panel is stale.
    g_fst_requested.clear();
    g_transducer.stale = true;
    world->query<LexiconPanel>().each([](flecs::entity, LexiconPanel& panel) {
        panel.dirty = true;
    });
}

static void try_send_false()
{
    if (g_pending_false.first.empty()) return;
    flecs::entity client = world->lookup("EntityCreateClient");
    if (!client.is_valid() || client.has<AwaitResponse>()) return;
    client.set<SendMapRequest>({{
        {"type", "teach_false"},
        {"word", g_pending_false.first},
        {"variant", g_pending_false.second},
    }});
    client.set<AwaitResponse>({[](std::map<std::string, msgpack::object>&) {
        g_pending_false = {};
        g_fst_requested.clear();       // re-analyze: the rejection changed things
        g_transducer.stale = true;
        world->query<LexiconPanel>().each([](flecs::entity, LexiconPanel& panel) {
            panel.dirty = true;
        });
    }});
}

static void try_send_teach()
{
    if (g_pending_teach.empty()) return;
    flecs::entity client = world->lookup("EntityCreateClient");
    if (!client.is_valid() || client.has<AwaitResponse>()) return;
    client.set<SendMapRequest>({{
        {"type", "teach_forms"},
        {"forms", g_pending_teach},
    }});
    client.set<AwaitResponse>({teach_number_response});
}

void morphology_response(std::map<std::string, msgpack::object>& res_map)
{
    g_fst_proposal.word.clear();
    if (res_map.count("lemma")) {
        g_fst_proposal.word = g_fst_requested;
        g_fst_proposal.lemma = res_map["lemma"].as<std::string>();
        g_fst_proposal.suffix = res_map.count("suffix_in")
            ? res_map["suffix_in"].as<std::string>() : std::string();
        g_fst_proposal.support = res_map.count("support")
            ? (int)res_map["support"].as<int64_t>() : 0;
        g_fst_proposal.rule = res_map.count("rule")
            ? res_map["rule"].as<std::string>() : std::string();
        g_fst_proposal.confidence = res_map.count("confidence")
            ? (float)res_map["confidence"].as<double>() : 0.0f;
        g_fst_proposal.word_type = res_map.count("word_type")
            ? res_map["word_type"].as<std::string>() : std::string("Plural");
        g_fst_proposal.variant = res_map.count("variant")
            ? res_map["variant"].as<std::string>() : g_fst_proposal.lemma;
        g_fst_proposal.variant_type = res_map.count("variant_type")
            ? res_map["variant_type"].as<std::string>() : std::string("Singular");
    }
    // Whether an analysis arrived or an empty OK, the panels re-render.
    world->query<LexiconPanel>().each([](flecs::entity, LexiconPanel& panel) {
        panel.dirty = true;
    });
}

// One line of plurality supervision: the user segmented a suffix by hand and
// typed the pair. This is the gold data the plural transducer trains on --
// surface form, lemma, the exact suffix, and which is which.
static void append_lexicon_record(const std::string& word, const std::string& lemma,
                                  const std::string& suffix)
{
    std::ofstream log("../lexicon.jsonl", std::ios::app);
    if (!log) return;
    auto escape = [](const std::string& in) {
        std::string out;
        for (char c : in) {
            if (c == '"' || c == '\\') out += '\\';
            out += c;
        }
        return out;
    };
    log << "{\"ts\": " << (long long)std::time(nullptr)
        << ", \"plural\": \"" << escape(word) << "\""
        << ", \"singular\": \"" << escape(lemma) << "\""
        << ", \"suffix\": \"" << escape(suffix) << "\""
        << ", \"rule\": \"strip-suffix\"}\n";
}

void sync_representation_grounding(WordAnnotationSelector& selector, const std::string& template_str)
{
    // This function destroys UI and immediately reads back the structure it
    // rebuilds -- content rows are walked to fill the selection map. Under
    // command deferral (key callbacks run inside world->progress()), those
    // reads see nothing: child_of pairs are still queued, ensure() hands out
    // default-constructed components (UIFlexContainer defaults to Column --
    // the "relationship suddenly lays out vertically" bug), and the selection
    // map falls back to whole-block slots. The R handler happened to suspend
    // deferral at its call site; the entity-tagging handlers did not, which is
    // why annotating an entity broke every relationship in the sentence.
    // Suspending here makes every caller safe.
    flecs::entity container = selector.parent_entity;
    if (!container.is_valid() || !container.is_alive()) return;

    // Suspend only genuine manual deferral (defer_begin, GLFW callbacks fired
    // mid-progress). Readonly staging also reports as deferred, but suspending
    // there is illegal -- the world cannot be written at all, and the attempt
    // asserts. Callers in that state must use immediate() systems instead.
    const bool was_deferred = world->is_deferred() && !world->is_readonly();
    if (was_deferred) world->defer_suspend();

    // Collected before destroying. Mutating the child list while iterating it
    // can leave entities behind, and with OrderedChildren the survivors keep
    // their slots -- so the rebuilt tokens append *after* the wreckage and the
    // sentence comes out reordered.
    std::vector<flecs::entity> stale;
    container.children([&stale](flecs::entity child) { stale.push_back(child); });
    for (flecs::entity child : stale) child.destruct();

    selector.selection_entities.clear(); 

    auto tokens = parse_sentence_template(template_str);
    auto UIElement = world->lookup("UIElement");

    // Slots for tokens rendered ahead of their turn. An entity that joins the
    // relationship next to it is drawn *inside* that block, so by the time the
    // loop reaches the relationship (or the joined target after it), its UI
    // already exists -- only the selection slots remain to be pushed, in token
    // order so the arrow-key indices stay aligned with the sentence.
    std::map<size_t, std::vector<flecs::entity>> preslots;

    // Where tokens currently render: the sentence container, or the open
    // comprehension frame's slot between [[Q and ]].
    flecs::entity insert_parent = container;
    flecs::entity open_outline = flecs::entity::null();
    flecs::entity open_ext_node = flecs::entity::null();
    std::vector<uint32_t> member_colors;

    // Where each symbol lives on screen (entity badges, joined sidecars,
    // reified blocks), and which slots refer to a symbol from afar. Joined
    // slots are excluded -- fusion already shows their co-reference.
    std::map<std::string, flecs::entity> symbol_anchors;
    std::vector<std::pair<std::string, flecs::entity>> referring_slots;

    for (size_t token_idx = 0; token_idx < tokens.size(); ++token_idx)
    {
        const auto& token = tokens[token_idx];

        auto pre = preslots.find(token_idx);
        if (pre != preslots.end()) {
            for (flecs::entity slot : pre->second) selector.selection_entities.push_back(slot);
            continue;
        }

        // A relationship group can begin one token early: an entity whose
        // symbol matches the next relationship's source joins that block
        // directly instead of standing beside a sprite repeating its symbol.
        size_t rel_idx = SIZE_MAX;
        bool src_joined = false;
        if (token.type == TokenType::Relationship) {
            rel_idx = token_idx;
        } else if (token.type == TokenType::Entity
                   && token_idx + 1 < tokens.size()
                   && relationship_src_joined(tokens, token_idx + 1)) {
            rel_idx = token_idx + 1;
            src_joined = true;
        }

        if (rel_idx != SIZE_MAX)
        {
            const auto& rel = tokens[rel_idx];
            bool tgt_joined = relationship_tgt_joined(tokens, rel_idx);
            const SentenceToken* joined_tgt = tgt_joined ? &tokens[rel_idx + 1] : nullptr;

            flecs::entity content = flecs::entity::null();
            flecs::entity block = create_triple_block(insert_parent, rel, &content,
                                                      src_joined ? &token : nullptr,
                                                      joined_tgt);

            std::vector<flecs::entity> parts;
            if (content.is_valid()) {
                content.children([&parts](flecs::entity slot) { parts.push_back(slot); });
            }
            flecs::entity src_slot = parts.size() > 0 ? parts[0] : block;
            flecs::entity tgt_slot = parts.size() > 2 ? parts[2] : block;

            if (insert_parent != container) {
                if (!rel.source_symbol.empty()) member_colors.push_back(get_entity_color(rel.source_symbol, ""));
                if (!rel.target_symbol.empty()) member_colors.push_back(get_entity_color(rel.target_symbol, ""));
            }

            if (src_joined) symbol_anchors[rel.source_symbol] = src_slot;
            else if (!rel.source_symbol.empty()) referring_slots.push_back({rel.source_symbol, src_slot});
            if (tgt_joined) symbol_anchors[rel.target_symbol] = tgt_slot;
            else if (!rel.target_symbol.empty()) referring_slots.push_back({rel.target_symbol, tgt_slot});
            // A reified triple is itself referable -- an arc from it to a slot
            // bound to its digit is the love->eat wiring made visible.
            if (!rel.reified_symbol.empty()) symbol_anchors[rel.reified_symbol] = block;

            // Slot layout mirrors token_selection_width exactly: a joined badge
            // is counted once, by its own entity token, and the relationship
            // sheds that stop. Any disagreement here desynchronises the arrow
            // keys from the sentence.
            if (src_joined) {
                selector.selection_entities.push_back(src_slot);   // entity's stop
                std::vector<flecs::entity> rel_slots = {block};
                if (!tgt_joined) rel_slots.push_back(tgt_slot);
                preslots[rel_idx] = rel_slots;
            } else {
                selector.selection_entities.push_back(src_slot);
                selector.selection_entities.push_back(block);
                if (!tgt_joined) selector.selection_entities.push_back(tgt_slot);
            }
            if (tgt_joined) {
                preslots[rel_idx + 1] = {tgt_slot};                // entity's stop
            }
            continue;
        }

        if (token.type == TokenType::Concept) {
            flecs::entity slot = flecs::entity::null();
            flecs::entity frame = create_concept_frame(container, token.binding_symbol,
                                                       token.reified_symbol,
                                                       &slot, &open_outline, &open_ext_node);
            member_colors.clear();
            selector.selection_entities.push_back(frame);
            // Both ids are referable: the intension's and the extension's arcs
            // and slot bindings land on the same frame.
            if (!token.binding_symbol.empty()) symbol_anchors[token.binding_symbol] = frame;
            if (!token.reified_symbol.empty()) symbol_anchors[token.reified_symbol] = frame;
            if (slot.is_valid()) insert_parent = slot;
            continue;
        }
        if (token.type == TokenType::ConceptEnd) {
            // The frame's outline runs first member's colour -> last member's,
            // matching their left-to-right order inside it.
            if (open_outline.is_valid() && !member_colors.empty()) {
                if (CustomRenderable* stroke = open_outline.try_get_mut<CustomRenderable>()) {
                    stroke->gradient_start = (scale_color(member_colors.front(), 1.3f) & 0xFFFFFF00) | 0xC0;
                    stroke->gradient_end   = (scale_color(member_colors.back(),  1.3f) & 0xFFFFFF00) | 0xC0;
                }
            }
            open_outline = flecs::entity::null();
            open_ext_node = flecs::entity::null();
            insert_parent = container;   // zero-width: no selection stop
            continue;
        }

        if (token.type == TokenType::Entity) {
            uint32_t color = get_entity_color(token.binding_symbol, token.text);
            // A lemma-annotated entity (a kind-promoted member) displays its
            // canonical form; the surface stays in the template so dissolving
            // the frame restores the sentence exactly.
            const std::string& display = token.reified_symbol.empty()
                ? token.text : token.reified_symbol;
            // An unbound entity wears the wildcard as its symbol -- the empty
            // string would suppress the sprite entirely, and a badge without
            // its glyph reads as a different kind of thing.
            flecs::entity badge = create_badge(insert_parent, UIElement, display.c_str(), color, 
                         false, false,
                         token.binding_symbol.empty() ? "*" : token.binding_symbol,
                         "", 0, color);
            
            if (!token.binding_symbol.empty()) symbol_anchors[token.binding_symbol] = badge;
            if (insert_parent != container) member_colors.push_back(color);
            selector.selection_entities.push_back(badge);
        } else {
            // Plain text
            flecs::entity text_ent = world->entity().is_a(UIElement).child_of(insert_parent)
                .add<UIYoga>()
                .add<UIYogaLegacyLeaf>()
                .set<TextRenderable>({token.text.c_str(), "CharisSIL", 16.0f, 0x777777FF})
                .set<ZIndex>({17});
                
            selector.selection_entities.push_back(text_ent);
        }
    }
    
    // Same symbol, two places, not fused: draw the identity as an arc.
    for (const auto& referring : referring_slots) {
        auto anchor = symbol_anchors.find(referring.first);
        if (anchor == symbol_anchors.end()) continue;

        world->entity()
            .is_a(UIElement)
            .child_of(container)
            .set<CorefArc>({anchor->second, referring.second})
            .set<QuadraticBezierRenderable>({0, 0, 0, 0, 0, 0, 1.5f,
                                             get_entity_color(referring.first, "")})
            .set<ZIndex>({28});
    }

    selector.token_count = selector.selection_entities.size();
    // Deliberately NOT set dirty here: the rebuild system consumes dirty by
    // calling this function, so re-raising it would rebuild every frame.
    selector.dirty = false;

    // Every template state is a training example -- this is where the dataset
    // accrues as a side effect of use.
    append_annotation_record(selector);

    if (was_deferred) world->defer_resume();
}

// TODO: The key callback needs to be refactored to callback functions or observers with semantic description
// Toggles real fullscreen, remembering the windowed geometry to restore to.
//
// The window is created at monitor size but windowed (monitor argument NULL),
// so this is what actually hands the display over and drops the decorations.
static void toggle_fullscreen(GLFWwindow* window)
{
    static bool fullscreen = false;
    static int windowed_x = 0, windowed_y = 0, windowed_w = 0, windowed_h = 0;

    if (!fullscreen) {
        glfwGetWindowPos(window, &windowed_x, &windowed_y);
        glfwGetWindowSize(window, &windowed_w, &windowed_h);

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        if (!monitor) return;
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (!mode) return;

        glfwSetWindowMonitor(window, monitor, 0, 0,
                             mode->width, mode->height, mode->refreshRate);
    } else {
        // Falling back to the monitor's size rather than zero, in case the
        // window went fullscreen before it had ever been given a size.
        int w = windowed_w > 0 ? windowed_w : 1280;
        int h = windowed_h > 0 ? windowed_h : 720;
        glfwSetWindowMonitor(window, nullptr, windowed_x, windowed_y, w, h, 0);
    }

    fullscreen = !fullscreen;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Before anything else, so it works regardless of which text field or panel
    // currently owns the keyboard.
    if (key == GLFW_KEY_F11 && action == GLFW_PRESS)
    {
        toggle_fullscreen(window);
        return;
    }

    // Blender-style numpad views for the Droid panel's 3D scene. Consumed only
    // while the pointer is over that panel, so the numpad still reaches text
    // fields and the VNC passthrough everywhere else. Repeats are honoured so
    // holding 4/6/8/2 orbits continuously.
    if ((action == GLFW_PRESS || action == GLFW_REPEAT) && panel3d::view_key(key, mods))
    {
        return;
    }

    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_S && mods & GLFW_MOD_CONTROL)
        {
            flecs::entity editor_root = world->lookup("editor_root");
            if (editor_root.is_valid()) {
                save_editor_layout(editor_root);
            }
            return;
        }

        // Lexicon character mode: Down descends from word-level annotation into
        // the selected word's characters -- altitude, the same axis the panels
        // split on. Arrows move over the chips (starting at the last character,
        // since suffixes are the point), S applies the strip-suffix
        // transduction and types the pair plural/singular, Up returns to the
        // sentence.
        {
            static flecs::query<LexiconPanel> lexicons = world->query<LexiconPanel>();
            bool char_mode = false;
            std::string lex_word;
            lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                if (panel.selecting) char_mode = true;
                if (lex_word.empty()) lex_word = panel.shown;
            });
            int chip_count = 0;
            for (char c : lex_word)
                if (!std::isspace(static_cast<unsigned char>(c))) chip_count++;

            bool annotating_now = false;
            static flecs::query<WordAnnotationSelector> lex_annotators =
                world->query<WordAnnotationSelector>();
            lex_annotators.each([&](flecs::entity, WordAnnotationSelector& sel) {
                if (sel.active) annotating_now = true;
            });

            if (!char_mode && annotating_now && key == GLFW_KEY_L
                && (mods & GLFW_MOD_CONTROL) && chip_count > 0) {
                lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                    if (panel.forms.empty() && !panel.shown.empty()) {
                        std::string chars;
                        for (char c : panel.shown)
                            if (!std::isspace(static_cast<unsigned char>(c))) chars += c;
                        panel.forms.push_back({chars, std::string()});
                    }
                    panel.selecting = true;
                    panel.row = 0;
                    int len = panel.forms.empty() ? 0 : (int)panel.forms[0].text.size();
                    panel.sel_start = panel.sel_end = std::max(0, len - 1);
                    panel.dirty = true;
                });
                return;
            }

            if (char_mode) {
                bool shift = (mods & GLFW_MOD_SHIFT) != 0;
                auto row_len = [](LexiconPanel& panel) {
                    return (panel.row < (int)panel.forms.size())
                        ? (int)panel.forms[panel.row].text.size() : 0;
                };

                if (key == GLFW_KEY_ESCAPE
                    || (key == GLFW_KEY_L && (mods & GLFW_MOD_CONTROL))) {
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (key == GLFW_KEY_ESCAPE && panel.typing_type) {
                            panel.typing_type = false;   // Escape peels one layer
                        } else {
                            panel.typing_type = false;
                            panel.selecting = false;     // L toggles all the way out
                        }
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_UP) {
                    // Up climbs the paradigm; from the surface row it returns
                    // to the sentence.
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        panel.typing_type = false;
                        if (panel.row > 0) {
                            panel.row--;
                            int len = row_len(panel);
                            panel.sel_end = std::clamp(panel.sel_end, 0, std::max(0, len - 1));
                            panel.sel_start = panel.sel_end;
                        } else {
                            panel.selecting = false;
                        }
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_DOWN) {
                    // Down walks into the variants; past the last row it spawns
                    // a new one -- a copy of the current form, ready to edit.
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        panel.typing_type = false;
                        if (panel.row + 1 < (int)panel.forms.size()) {
                            panel.row++;
                        } else if (panel.row < (int)panel.forms.size()) {
                            panel.forms.push_back({panel.forms[panel.row].text, std::string()});
                            panel.row = (int)panel.forms.size() - 1;
                        }
                        int len = row_len(panel);
                        panel.sel_end = std::clamp(panel.sel_end, 0, std::max(0, len - 1));
                        panel.sel_start = panel.sel_end;
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) {
                    // Focus is explicit now (L toggles), so the mode is
                    // strict: while the lexicon holds focus, arrows are
                    // chip-local -- plain moves, Shift extends -- and the
                    // sentence moves only after L out.
                    int step = (key == GLFW_KEY_RIGHT) ? 1 : -1;
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        int len = row_len(panel);
                        if (len <= 0) return;
                        int next = std::clamp(panel.sel_end + step, 0, len - 1);
                        panel.sel_end = next;
                        if (!shift) panel.sel_start = next;
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_TAB) {
                    // Field switch on the row: characters <-> type annotation.
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        panel.typing_type = !panel.typing_type;
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_X) {
                    // On a machine-proposed row, X is rejection: the guessed
                    // derivation is wrong, the row disappears, and the
                    // rejection is taught to the world as a NotLemmaOf pair
                    // so this exact derivation never comes back.
                    bool rejected = false;
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (panel.row <= 0 || panel.row >= (int)panel.forms.size()) return;
                        if (!panel.forms[panel.row].proposed) return;
                        g_pending_false = {panel.forms[0].text,
                                           panel.forms[panel.row].text};
                        panel.forms.erase(panel.forms.begin() + panel.row);
                        panel.row = std::clamp(panel.row - 1, 0,
                                               std::max(0, (int)panel.forms.size() - 1));
                        panel.dirty = true;
                        rejected = true;
                    });
                    if (rejected) { try_send_false(); return; }

                    // On the surface row, X derives: the deletion spawns a new
                    // variant carrying it -- the edit is the transduction, the
                    // new row its output -- and the surface stays faithful to
                    // the sentence. On a variant row, X edits in place.
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (panel.row >= (int)panel.forms.size()) return;

                        if (panel.row == 0) {
                            const std::string& surface = panel.forms[0].text;
                            int lo = std::clamp(std::min(panel.sel_start, panel.sel_end),
                                                0, (int)surface.size());
                            int hi = std::clamp(std::max(panel.sel_start, panel.sel_end),
                                                0, (int)surface.size() - 1);
                            if (lo > hi) return;
                            std::string variant = surface.substr(0, lo) + surface.substr(hi + 1);
                            panel.forms.push_back({variant, std::string()});
                            panel.row = (int)panel.forms.size() - 1;
                            panel.sel_start = panel.sel_end =
                                std::clamp(lo, 0, std::max(0, (int)variant.size() - 1));
                            panel.dirty = true;
                            return;
                        }

                        std::string& text = panel.forms[panel.row].text;
                        int lo = std::clamp(std::min(panel.sel_start, panel.sel_end), 0, (int)text.size());
                        int hi = std::clamp(std::max(panel.sel_start, panel.sel_end), 0, (int)text.size() - 1);
                        if (lo > hi) return;
                        text.erase(lo, hi - lo + 1);
                        panel.sel_start = panel.sel_end =
                            std::clamp(lo, 0, std::max(0, (int)text.size() - 1));
                        panel.dirty = true;
                    });
                    return;
                }
                if ((key >= GLFW_KEY_A && key <= GLFW_KEY_Z)
                    || (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
                    || key == GLFW_KEY_SPACE) {
                    char inserted;
                    if (key == GLFW_KEY_SPACE) inserted = ' ';
                    else if (key <= GLFW_KEY_9) inserted = (char)('0' + (key - GLFW_KEY_0));
                    else inserted = (char)((shift ? 'A' : 'a') + (key - GLFW_KEY_A));
                    // In the type field, letters spell the entity type name --
                    // arbitrary, case as typed. Any row may be typed,
                    // including the surface row.
                    bool any_typing = false;
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (!panel.typing_type) return;
                        any_typing = true;
                        if (panel.row < (int)panel.forms.size()) {
                            panel.forms[panel.row].type += inserted;
                            panel.forms[panel.row].proposed = false;   // adopted
                            panel.dirty = true;
                        }
                    });
                    if (any_typing) return;

                    // Otherwise: insert after the selected character; Shift
                    // inserts before, which is how a prepend at 0 is reached.
                    if (inserted == ' ') return;   // spaces belong to types only
                    inserted = (char)std::tolower(static_cast<unsigned char>(inserted));
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (panel.row == 0 || panel.row >= (int)panel.forms.size()) return;
                        std::string& text = panel.forms[panel.row].text;
                        int at = shift ? std::min(panel.sel_start, panel.sel_end)
                                       : panel.sel_end + 1;
                        at = std::clamp(at, 0, (int)text.size());
                        text.insert(text.begin() + at, inserted);
                        panel.forms[panel.row].proposed = false;   // adopted
                        panel.sel_start = panel.sel_end = at;   // land on the new letter
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_DELETE) {
                    // Delete removes the selected variant row outright. The
                    // surface row stays -- it is the word from the sentence,
                    // not ours to remove. Distinct from X, which rejects a
                    // machine proposal (teaching) or edits characters; Delete
                    // just discards, teaching nothing.
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (panel.row <= 0 || panel.row >= (int)panel.forms.size()) return;
                        panel.forms.erase(panel.forms.begin() + panel.row);
                        panel.row = std::clamp(panel.row - 1, 0,
                                               std::max(0, (int)panel.forms.size() - 1));
                        int len = (panel.row < (int)panel.forms.size())
                            ? (int)panel.forms[panel.row].text.size() : 0;
                        panel.sel_start = panel.sel_end = std::max(0, len - 1);
                        panel.typing_type = false;
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_BACKSPACE) {
                    bool any_typing = false;
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (!panel.typing_type) return;
                        any_typing = true;
                        if (panel.row < (int)panel.forms.size()
                            && !panel.forms[panel.row].type.empty()) {
                            panel.forms[panel.row].type.pop_back();
                            panel.dirty = true;
                        }
                    });
                    if (any_typing) return;

                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        if (panel.row == 0 || panel.row >= (int)panel.forms.size()) return;
                        std::string& text = panel.forms[panel.row].text;
                        if (text.empty()) return;
                        int at = std::clamp(panel.sel_end, 0, (int)text.size() - 1);
                        text.erase(text.begin() + at);
                        panel.sel_start = panel.sel_end =
                            std::clamp(at - 1, 0, std::max(0, (int)text.size() - 1));
                        panel.dirty = true;
                    });
                    return;
                }
                if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
                    // Commit the paradigm: every typed row becomes a Form
                    // entity of its annotated type in the world. Types are
                    // whatever was typed -- created on demand at the far end.
                    lexicons.each([&](flecs::entity, LexiconPanel& panel) {
                        std::string forms;
                        for (const LexiconForm& form : panel.forms) {
                            if (form.text.empty() || form.type.empty()) continue;
                            // "Plural or Singular" unfolds to one line per
                            // type; the bridge folds same-text lines back into
                            // one entity wearing every tag.
                            std::string current;
                            std::vector<std::string> tokens;
                            for (char c : form.type) {
                                if (c == ' ') {
                                    if (!current.empty()) tokens.push_back(current);
                                    current.clear();
                                } else current += c;
                            }
                            if (!current.empty()) tokens.push_back(current);
                            for (const std::string& token : tokens) {
                                std::string lower;
                                for (char c : token)
                                    lower += (char)std::tolower(static_cast<unsigned char>(c));
                                if (lower == "or") continue;   // operator, not a type
                                if (!forms.empty()) forms += "\n";
                                forms += token + "\t" + form.text;
                            }
                        }
                        if (forms.empty()) return;
                        g_pending_teach = forms;
                        try_send_teach();
                        panel.dirty = true;
                    });
                    return;
                }
                // Character mode owns the keyboard while focused.
                return;
            }
        }

        // Word annotation selector - Tab toggles, arrows navigate, shift+arrows expand selection
        static auto annotation_query = world->query<WordAnnotationSelector>();
        if (key == GLFW_KEY_TAB)
        {
            annotation_query.each([](flecs::entity e, WordAnnotationSelector& selector) {
                // Save full state before toggling
                if (selector.active && g_current_message_idx >= 0 && g_current_message_idx < (int)g_annotatable_messages.size()) {
                    auto& msg = g_annotatable_messages[g_current_message_idx];
                    msg.sentence_template = selector.sentence_template;
                    msg.ui_entities = selector.ui_entities;  // Keep copy, don't move - entities stay visible
                    msg.selection_entities = selector.selection_entities;
                    msg.token_count = selector.token_count;
                }

                selector.active = !selector.active;
                world->ensure<ChatState>().input_focused = !selector.active;
                if (selector.active && !selector.sentence_template.empty()) {
                    selector.start_index = 0;  // Reset to first word when activating
                    selector.end_index = 0;
                }
            });
            return;
        }
        else if (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT)
        {
            bool handled = false;
            bool shift_held = (mods & GLFW_MOD_SHIFT) != 0;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (selector.active && selector.token_count > 0) {
                    int max_idx = selector.token_count - 1;
                    if (key == GLFW_KEY_LEFT) {
                        if (shift_held) {
                            // Expand selection left, or shrink from right if can't
                            if (selector.start_index > 0) {
                                selector.start_index--;
                                handled = true;
                            } else if (selector.end_index > selector.start_index) {
                                selector.end_index--;
                                handled = true;
                            }
                        } else {
                            // Move selection left (collapse to single word)
                            if (selector.start_index > 0) {
                                selector.start_index--;
                                selector.end_index = selector.start_index;
                                handled = true;
                            }
                        }
                    } else if (key == GLFW_KEY_RIGHT) {
                        if (shift_held) {
                            // Expand selection right, or shrink from left if can't
                            if (selector.end_index < max_idx) {
                                selector.end_index++;
                                handled = true;
                            } else if (selector.start_index < selector.end_index) {
                                selector.start_index++;
                                handled = true;
                            }
                        } else {
                            // Move selection right (collapse to single word)
                            if (selector.end_index < max_idx) {
                                selector.end_index++;
                                selector.start_index = selector.end_index;
                                handled = true;
                            }
                        }
                    }
                }
            });
            if (handled) return;
        }

        // Up/Down arrows - cycle between message annotations
        if (key == GLFW_KEY_UP || key == GLFW_KEY_DOWN)
        {
            if (g_annotatable_messages.size() > 1) {
                annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                    if (!selector.active) return;

                    // Save current message state back to storage (don't move, just copy refs)
                    if (g_current_message_idx >= 0 && g_current_message_idx < (int)g_annotatable_messages.size()) {
                        auto& msg = g_annotatable_messages[g_current_message_idx];
                        msg.sentence_template = selector.sentence_template;
                        msg.ui_entities = selector.ui_entities;
                        msg.selection_entities = selector.selection_entities;
                        msg.token_count = selector.token_count;
                    }

                    // Calculate next message index
                    int next_idx;
                    if (key == GLFW_KEY_UP) {
                        next_idx = (g_current_message_idx - 1 + (int)g_annotatable_messages.size()) % (int)g_annotatable_messages.size();
                    } else {
                        next_idx = (g_current_message_idx + 1) % (int)g_annotatable_messages.size();
                    }

                    // Switch to new message - point to its state
                    g_current_message_idx = next_idx;
                    auto& new_msg = g_annotatable_messages[next_idx];
                    selector.sentence_template = new_msg.sentence_template;
                    selector.parent_entity = new_msg.parent_entity;
                    selector.ui_entities = new_msg.ui_entities;
                    selector.selection_entities = new_msg.selection_entities;
                    selector.token_count = new_msg.token_count;
                    selector.start_index = 0;
                    selector.end_index = 0;

                    // If no UI entities yet for this message, create them
                    if (selector.ui_entities.empty() && !selector.sentence_template.empty()) {
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                        // Save newly created entities back
                        new_msg.ui_entities = selector.ui_entities;
                        new_msg.selection_entities = selector.selection_entities;
                        new_msg.token_count = selector.token_count;
                    }
                });
                return;
            }
        }

        // Number keys (0-9) - assign digit to selected token/part (skip if shift is held for special chars)
        if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9 && !(mods & GLFW_MOD_SHIFT))
        {
            int digit = key - GLFW_KEY_0;
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (selector.active && !selector.sentence_template.empty()) {
                    // Parse current template into tokens
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find which token and sub-part the selection is on
                    int expanded_pos = 0;
                    int token_idx = -1;
                    int sub_part = 0;  // 0=source, 1=rel, 2=target (for relationships)
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            sub_part = selector.start_index - expanded_pos;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    // Check if selection spans multiple tokens - if so, combine into entity
                    int end_expanded_pos = 0;
                    int end_token_idx = -1;
                    expanded_pos = 0;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.end_index >= expanded_pos && selector.end_index < expanded_pos + width) {
                            end_token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }

                    if (token_idx != end_token_idx || tokens[token_idx].type == TokenType::PlainText) {
                        // Multiple tokens or plain text - combine into entity binding
                        std::string combined_text;
                        int start_tok = token_idx;
                        int end_tok = end_token_idx >= 0 ? end_token_idx : token_idx;
                        for (int i = start_tok; i <= end_tok; i++) {
                            if (!combined_text.empty()) combined_text += " ";
                            combined_text += tokens[i].text;
                        }

                        std::vector<SentenceToken> new_tokens;
                        int new_sel_start = 0;
                        for (int i = 0; i < start_tok; i++) {
                            new_sel_start += token_selection_width(tokens, i);
                            new_tokens.push_back(tokens[i]);
                        }
                        new_tokens.push_back({combined_text, std::to_string(digit), "", "", "", TokenType::Entity});
                        for (int i = end_tok + 1; i < (int)tokens.size(); i++) {
                            new_tokens.push_back(tokens[i]);
                        }

                        selector.sentence_template = tokens_to_template(new_tokens);
                        selector.dirty = true;
                        selector.start_index = new_sel_start;
                        selector.end_index = new_sel_start;
                        // recreate_annotation_entities(selector);
                    } else if (tokens[token_idx].type == TokenType::Relationship) {
                        // sub_part indexes the *remaining* stops, so it has to
                        // be translated through the roles a joined relationship
                        // still exposes -- with a joined source, stop 0 is the
                        // whole triple, not the source.
                        std::vector<int> roles = relationship_slot_roles(tokens, token_idx);
                        int role = (sub_part >= 0 && sub_part < (int)roles.size())
                                 ? roles[sub_part] : 1;
                        if (role == 0) {
                            tokens[token_idx].source_symbol = std::to_string(digit);
                        } else if (role == 2) {
                            tokens[token_idx].target_symbol = std::to_string(digit);
                        }
                        // the whole-triple stop doesn't accept digit assignment
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                    } else if (tokens[token_idx].type == TokenType::Entity) {
                        // Update entity binding symbol
                        tokens[token_idx].binding_symbol = std::to_string(digit);
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                    }

                    handled = true;
                }
            });
            if (handled) return;
        }

        // Q - Comprehension: wraps the selected span in a frame. The members --
        // words, entity badges, whole relationships -- keep their bindings and
        // renderings; only a [[Qn ... ]] pair is inserted around them.
        if (key == GLFW_KEY_Q)
        {
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (!selector.active || selector.sentence_template.empty()) return;
                auto tokens = parse_sentence_template(selector.sentence_template);
                if (tokens.empty()) return;

                int expanded_pos = 0;
                int start_tok = -1, end_tok = -1;
                for (size_t ti = 0; ti < tokens.size(); ti++) {
                    int width = token_selection_width(tokens, ti);
                    if (start_tok < 0 && selector.start_index >= expanded_pos
                        && selector.start_index < expanded_pos + width) {
                        start_tok = (int)ti;
                    }
                    if (selector.end_index >= expanded_pos
                        && selector.end_index < expanded_pos + width) {
                        end_tok = (int)ti;
                    }
                    expanded_pos += width;
                }
                if (start_tok < 0) return;
                if (end_tok < start_tok) end_tok = start_tok;

                // No nesting or overlap in v1: a marker anywhere in the span
                // means this comprehension would cross another's boundary.
                for (int i = start_tok; i <= end_tok; i++) {
                    if (tokens[i].type == TokenType::Concept
                        || tokens[i].type == TokenType::ConceptEnd) return;
                }

                // Two ids per comprehension: the intension (query) and the
                // extension (its output set) are both entities.
                std::set<std::string> used = collect_used_symbols(tokens);
                std::string symbol, ext_symbol;
                for (int d = 0; d <= 9; d++) {
                    std::string candidate = std::to_string(d);
                    if (used.count(candidate)) continue;
                    if (symbol.empty()) symbol = candidate;
                    else { ext_symbol = candidate; break; }
                }
                if (symbol.empty() || ext_symbol.empty()) return;

                std::vector<SentenceToken> new_tokens;
                int new_sel_start = 0;
                for (int i = 0; i < start_tok; i++) {
                    new_sel_start += token_selection_width(tokens, i);
                    new_tokens.push_back(tokens[i]);
                }
                new_tokens.push_back({"", symbol, "", "", ext_symbol, TokenType::Concept});
                for (int i = start_tok; i <= end_tok; i++) new_tokens.push_back(tokens[i]);
                new_tokens.push_back({"", "", "", "", "", TokenType::ConceptEnd});
                for (int i = end_tok + 1; i < (int)tokens.size(); i++) new_tokens.push_back(tokens[i]);

                selector.sentence_template = tokens_to_template(new_tokens);
                selector.dirty = true;
                // Land on the frame's own stop.
                selector.start_index = new_sel_start;
                selector.end_index = new_sel_start;
                handled = true;
            });
            if (handled) return;
        }

        // Letter keys (A-Z) - assign letter as binding symbol to selected token/part
        // Exclude E, X, R which have special annotation functions
        if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z &&
            key != GLFW_KEY_E && key != GLFW_KEY_X && key != GLFW_KEY_R &&
            key != GLFW_KEY_Q)
        {
            char letter = 'a' + (key - GLFW_KEY_A);  // lowercase letter
            std::string symbol(1, letter);
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (selector.active && !selector.sentence_template.empty()) {
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find which token and sub-part the selection is on
                    int expanded_pos = 0;
                    int token_idx = -1;
                    int sub_part = 0;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            sub_part = selector.start_index - expanded_pos;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    // Check if selection spans multiple tokens
                    int end_token_idx = -1;
                    expanded_pos = 0;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.end_index >= expanded_pos && selector.end_index < expanded_pos + width) {
                            end_token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }

                    if (token_idx != end_token_idx || tokens[token_idx].type == TokenType::PlainText) {
                        // Multiple tokens or plain text - combine into entity binding
                        std::string combined_text;
                        int start_tok = token_idx;
                        int end_tok = end_token_idx >= 0 ? end_token_idx : token_idx;
                        for (int i = start_tok; i <= end_tok; i++) {
                            if (!combined_text.empty()) combined_text += " ";
                            combined_text += tokens[i].text;
                        }

                        std::vector<SentenceToken> new_tokens;
                        int new_sel_start = 0;
                        for (int i = 0; i < start_tok; i++) {
                            new_sel_start += token_selection_width(tokens, i);
                            new_tokens.push_back(tokens[i]);
                        }
                        new_tokens.push_back({combined_text, symbol, "", "", "", TokenType::Entity});
                        for (int i = end_tok + 1; i < (int)tokens.size(); i++) {
                            new_tokens.push_back(tokens[i]);
                        }

                        selector.sentence_template = tokens_to_template(new_tokens);
                        selector.dirty = true;
                        selector.start_index = new_sel_start;
                        selector.end_index = new_sel_start;
                        sync_representation_grounding(selector, selector.sentence_template);
                        // recreate_annotation_entities(selector);
                    } else if (tokens[token_idx].type == TokenType::Relationship) {
                        // Same translation as the digit handler: sub_part
                        // indexes the remaining stops, and with a joined side
                        // the raw 0/1/2 layout no longer holds.
                        std::vector<int> roles = relationship_slot_roles(tokens, token_idx);
                        int role = (sub_part >= 0 && sub_part < (int)roles.size())
                                 ? roles[sub_part] : 1;
                        if (role == 0) {
                            tokens[token_idx].source_symbol = symbol;
                        } else if (role == 2) {
                            tokens[token_idx].target_symbol = symbol;
                        }
                        // badge part (1) doesn't accept symbol assignment
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                    } else if (tokens[token_idx].type == TokenType::Entity) {
                        // Update entity binding symbol
                        tokens[token_idx].binding_symbol = symbol;
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        sync_representation_grounding(selector, selector.sentence_template);
                        // recreate_annotation_entities(selector);
                    }

                    handled = true;
                }
            });
            if (handled) return;
        }

        // Asterisk key (Shift+8) - set wildcard for relationship source/target
        if (key == GLFW_KEY_8 && (mods & GLFW_MOD_SHIFT))
        {
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (selector.active && !selector.sentence_template.empty()) {
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find which token and sub-part the selection is on
                    int expanded_pos = 0;
                    int token_idx = -1;
                    int sub_part = 0;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            sub_part = selector.start_index - expanded_pos;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    if (tokens[token_idx].type == TokenType::Relationship) {
                        // Set wildcard (empty string) for source or target
                        // 0=source, 1=badge, 2=target
                        if (sub_part == 0) {
                            tokens[token_idx].source_symbol = "";
                        } else if (sub_part == 2) {
                            tokens[token_idx].target_symbol = "";
                        }
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                        handled = true;
                    }
                }
            });
            if (handled) return;
        }

        // E key - bind to entity with first available ID, or convert entity back to text
        if (key == GLFW_KEY_E)
        {
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                    std::cout << "Selector active " << selector.active << std::endl;
                    std::cout << selector.sentence_template << selector.active << std::endl;

                if (selector.active && !selector.sentence_template.empty()) {
                    std::cout << "Token parser" << std::endl;
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find token at selection
                    int expanded_pos = 0;
                    int token_idx = -1;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    // If already an entity, convert back to plain text
                    if (tokens[token_idx].type == TokenType::Entity) {
                        tokens[token_idx].type = TokenType::PlainText;
                        tokens[token_idx].binding_symbol = "";
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                        handled = true;
                        return;
                    }

                    // If relationship, bind source to closest entity before, target to closest entity after
                    if (tokens[token_idx].type == TokenType::Relationship) {
                        std::string source_symbol = "";
                        std::string target_symbol = "";

                        // Find closest entity with lower index
                        for (int i = token_idx - 1; i >= 0; i--) {
                            if (tokens[i].type == TokenType::Entity && !tokens[i].binding_symbol.empty()) {
                                source_symbol = tokens[i].binding_symbol;
                                break;
                            }
                        }

                        // Find closest entity with higher index
                        for (int i = token_idx + 1; i < (int)tokens.size(); i++) {
                            if (tokens[i].type == TokenType::Entity && !tokens[i].binding_symbol.empty()) {
                                target_symbol = tokens[i].binding_symbol;
                                break;
                            }
                        }

                        tokens[token_idx].source_symbol = source_symbol;
                        tokens[token_idx].target_symbol = target_symbol;
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                        handled = true;
                        return;
                    }

                    // The special case that closes the intension circle: a
                    // plural word denotes a kind, not an instance -- "robots"
                    // is Q(robot). When the transducer knows the selected word
                    // is plural, E promotes it to a comprehension frame rather
                    // than an entity binding. Knowledge comes from the same
                    // analysis the Lexicon panel requested; no analysis, no
                    // special case.
                    if (tokens[token_idx].type == TokenType::PlainText
                        && selector.start_index == selector.end_index) {
                        std::string lowered = tokens[token_idx].text;
                        for (char& c : lowered)
                            c = (char)std::tolower(static_cast<unsigned char>(c));
                        if (g_fst_proposal.word == lowered
                            && g_fst_proposal.word_type == "Plural"
                            && tokens[token_idx].type == TokenType::PlainText) {
                            std::set<std::string> used = collect_used_symbols(tokens);
                            std::string q_sym, e_sym;
                            for (int d = 0; d <= 9; d++) {
                                std::string candidate = std::to_string(d);
                                if (used.count(candidate)) continue;
                                if (q_sym.empty()) q_sym = candidate;
                                else { e_sym = candidate; break; }
                            }
                            if (!q_sym.empty() && !e_sym.empty()) {
                                std::vector<SentenceToken> new_tokens;
                                int new_sel_start = 0;
                                for (int i = 0; i < token_idx; i++) {
                                    new_sel_start += token_selection_width(tokens, i);
                                    new_tokens.push_back(tokens[i]);
                                }
                                new_tokens.push_back({"", q_sym, "", "", e_sym,
                                                      TokenType::Concept});
                                // Inside the frame sits the SINGULAR as a
                                // badge -- but the template keeps the SURFACE
                                // word, with the lemma riding as an annotation
                                // (in reified_symbol, unused for entities).
                                // Display shows "robot"; dissolving the frame
                                // restores "robots". Never destroy surface to
                                // show canonical.
                                new_tokens.push_back({tokens[token_idx].text,
                                                      "", "", "",
                                                      g_fst_proposal.variant,
                                                      TokenType::Entity});
                                new_tokens.push_back({"", "", "", "", "",
                                                      TokenType::ConceptEnd});
                                for (int i = token_idx + 1; i < (int)tokens.size(); i++)
                                    new_tokens.push_back(tokens[i]);

                                selector.sentence_template = tokens_to_template(new_tokens);
                                selector.dirty = true;
                                selector.start_index = new_sel_start;
                                selector.end_index = new_sel_start;
                                handled = true;
                                return;
                            }
                        }
                    }

                    // Collect all used symbols
                    std::set<std::string> used_symbols;
                    for (const auto& tok : tokens) {
                        if (tok.type == TokenType::Entity && !tok.binding_symbol.empty()) {
                            used_symbols.insert(tok.binding_symbol);
                        }
                        if (tok.type == TokenType::Relationship) {
                            if (!tok.source_symbol.empty()) used_symbols.insert(tok.source_symbol);
                            if (!tok.target_symbol.empty()) used_symbols.insert(tok.target_symbol);
                        }
                    }

                    // Find first available digit (0-9)
                    int available_digit = -1;
                    for (int d = 0; d <= 9; d++) {
                        if (used_symbols.find(std::to_string(d)) == used_symbols.end()) {
                            available_digit = d;
                            break;
                        }
                    }
                    if (available_digit < 0) return; // All digits used

                    int end_token_idx = -1;
                    expanded_pos = 0;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.end_index >= expanded_pos && selector.end_index < expanded_pos + width) {
                            end_token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }

                    // Combine selected tokens into entity binding
                    std::string combined_text;
                    int start_tok = token_idx;
                    int end_tok = end_token_idx >= 0 ? end_token_idx : token_idx;
                    for (int i = start_tok; i <= end_tok; i++) {
                        if (!combined_text.empty()) combined_text += " ";
                        combined_text += tokens[i].text;
                    }

                    std::vector<SentenceToken> new_tokens;
                    int new_sel_start = 0;
                    for (int i = 0; i < start_tok; i++) {
                        new_sel_start += token_selection_width(tokens, i);
                        new_tokens.push_back(tokens[i]);
                    }
                    new_tokens.push_back({combined_text, std::to_string(available_digit), "", "", "", TokenType::Entity});
                    for (int i = end_tok + 1; i < (int)tokens.size(); i++) {
                        new_tokens.push_back(tokens[i]);
                    }

                    selector.sentence_template = tokens_to_template(new_tokens);
                    selector.dirty = true;
                    selector.start_index = new_sel_start;
                    selector.end_index = new_sel_start;

                    flecs::entity interlocutorAnnotator = world->lookup("InterlocutorAnnotator");
                    sync_representation_grounding(selector, selector.sentence_template);
                    // recreate_annotation_entities(selector);
                    handled = true;
                }
            });
            if (handled) return;
        }

        // X key - convert entity or relationship back to plain text
        if (key == GLFW_KEY_X)
        {
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (selector.active && !selector.sentence_template.empty()) {
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find token at selection
                    int expanded_pos = 0;
                    int token_idx = -1;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    if (tokens[token_idx].type == TokenType::Relationship) {
                        // On a bound slot, X releases the binding back to a
                        // wildcard. Only on the whole-triple stop does it
                        // dissolve the relationship itself -- clearing one end
                        // should never cost the whole annotation.
                        int sub_part = selector.start_index - expanded_pos;
                        std::vector<int> roles = relationship_slot_roles(tokens, token_idx);
                        int role = (sub_part >= 0 && sub_part < (int)roles.size())
                                 ? roles[sub_part] : 1;

                        if (role == 0 && !tokens[token_idx].source_symbol.empty()) {
                            tokens[token_idx].source_symbol = "";
                        } else if (role == 2 && !tokens[token_idx].target_symbol.empty()) {
                            tokens[token_idx].target_symbol = "";
                        } else {
                            tokens[token_idx].type = TokenType::PlainText;
                            tokens[token_idx].binding_symbol = "";
                            tokens[token_idx].source_symbol = "";
                            tokens[token_idx].target_symbol = "";
                        }
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        handled = true;
                    }
                    else if (tokens[token_idx].type == TokenType::Concept) {
                        // Dissolve the frame: remove the marker pair. Members
                        // that only made sense IN the frame revert -- a
                        // lemma-annotated unbound entity was the kind's mark,
                        // and without the frame it is just the surface word
                        // again, plural and all.
                        int close = -1;
                        for (size_t i = token_idx + 1; i < tokens.size(); i++) {
                            if (tokens[i].type == TokenType::ConceptEnd) { close = (int)i; break; }
                        }
                        int limit = close >= 0 ? close : (int)tokens.size();
                        for (int i = token_idx + 1; i < limit; i++) {
                            if (tokens[i].type == TokenType::Entity
                                && tokens[i].binding_symbol.empty()
                                && !tokens[i].reified_symbol.empty()) {
                                tokens[i].type = TokenType::PlainText;
                                tokens[i].reified_symbol = "";
                            }
                        }
                        if (close >= 0) tokens.erase(tokens.begin() + close);
                        tokens.erase(tokens.begin() + token_idx);
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        handled = true;
                    }
                    else if (tokens[token_idx].type == TokenType::Entity) {
                        // A joined sidecar un-joins first: clearing the
                        // relationship's matching symbol moves the badge back
                        // out to stand alone and returns the slot it occupied
                        // to a wildcard. The entity itself is only dissolved
                        // when it isn't part of a join -- X peels one layer at
                        // a time.
                        if (token_idx + 1 < tokens.size()
                            && relationship_src_joined(tokens, token_idx + 1)) {
                            tokens[token_idx + 1].source_symbol = "";
                        } else if (token_idx > 0
                            && relationship_tgt_joined(tokens, token_idx - 1)) {
                            tokens[token_idx - 1].target_symbol = "";
                        } else {
                            tokens[token_idx].type = TokenType::PlainText;
                            tokens[token_idx].binding_symbol = "";
                            tokens[token_idx].source_symbol = "";
                            tokens[token_idx].target_symbol = "";
                        }
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        handled = true;
                    }
                }
            });
            if (handled) return;
        }

        // R key - create relationship from selected text
        if (key == GLFW_KEY_R)
        {
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (selector.active && !selector.sentence_template.empty()) {
                    // Parse current template into tokens
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find token at selection
                    int expanded_pos = 0;
                    int token_idx = -1;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }

                    // R on an existing relationship reifies it: the triple
                    // itself becomes an entity, marked by the 3d-node and a
                    // freshly assigned MNIST digit above the predicate.
                    // Pressed again, it un-reifies -- same toggle as Shift+7.
                    if (token_idx >= 0 && tokens[token_idx].type == TokenType::Relationship) {
                        if (!tokens[token_idx].reified_symbol.empty()) {
                            tokens[token_idx].reified_symbol = "";
                        } else {
                            // The next digit no other binding in the sentence
                            // is using; derived per press, so un-reifying
                            // returns the digit to the pool.
                            std::set<std::string> used_symbols = collect_used_symbols(tokens);
                            for (int d = 0; d <= 9; d++) {
                                if (used_symbols.find(std::to_string(d)) == used_symbols.end()) {
                                    tokens[token_idx].reified_symbol = std::to_string(d);
                                    break;
                                }
                            }
                        }
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        handled = true;
                        return;
                    }

                    // Calculate expanded selection range (accounting for relationship width)
                    expanded_pos = 0;
                    int token_start = -1, token_end = -1;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (token_start < 0 && expanded_pos + width > selector.start_index) {
                            token_start = (int)ti;
                        }
                        if (expanded_pos + width > selector.end_index && token_end < 0) {
                            token_end = (int)ti;
                        }
                        expanded_pos += width;
                    }
                    if (token_start < 0) token_start = 0;
                    if (token_end < 0) token_end = (int)tokens.size() - 1;

                    // Combine selected tokens into relationship label
                    std::string combined_text;
                    for (int i = token_start; i <= token_end; i++) {
                        if (!combined_text.empty()) combined_text += " ";
                        combined_text += tokens[i].text;
                    }

                    // Build new token list: before + [relationship with wildcards] + after
                    std::vector<SentenceToken> new_tokens;
                    int new_sel_start = 0;
                    for (int i = 0; i < token_start; i++) {
                        new_sel_start += token_selection_width(tokens, i);
                        new_tokens.push_back(tokens[i]);
                    }

                    // Add single relationship token with wildcard source/target
                    new_tokens.push_back({combined_text, "", "", "", "", TokenType::Relationship});

                    for (int i = token_end + 1; i < (int)tokens.size(); i++) {
                        new_tokens.push_back(tokens[i]);
                    }

                    world->defer_suspend();
                    // Update template and recreate UI entities
                    selector.sentence_template = tokens_to_template(new_tokens);
                    selector.dirty = true;
                    // Select source + relationship (first 2 of 3 parts)
                    selector.start_index = new_sel_start;
                    selector.end_index = new_sel_start + 1;
                    // recreate_annotation_entities(selector);
                    sync_representation_grounding(selector, selector.sentence_template);
                    world->defer_resume();

                    handled = true;
                }
            });
            if (handled) return;
        }

        // Shift+7 (&) - toggle reification on relationship
        if (key == GLFW_KEY_7 && (mods & GLFW_MOD_SHIFT))
        {
            bool handled = false;
            annotation_query.each([&](flecs::entity e, WordAnnotationSelector& selector) {
                if (selector.active && !selector.sentence_template.empty()) {
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find token at selection
                    int expanded_pos = 0;
                    int token_idx = -1;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = token_selection_width(tokens, ti);
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    // Only toggle reification on relationships
                    if (tokens[token_idx].type == TokenType::Relationship) {
                        if (!tokens[token_idx].reified_symbol.empty()) {
                            // Already reified - remove reification
                            tokens[token_idx].reified_symbol = "";
                        } else {
                            // Reify: find first available symbol
                            std::set<std::string> used_symbols = collect_used_symbols(tokens);
                            // Find first available digit
                            for (int d = 0; d <= 9; d++) {
                                if (used_symbols.find(std::to_string(d)) == used_symbols.end()) {
                                    tokens[token_idx].reified_symbol = std::to_string(d);
                                    break;
                                }
                            }
                        }
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        // recreate_annotation_entities(selector);
                        handled = true;
                    }
                }
            });
            if (handled) return;
        }

    // The Entities relation bar owns the keyboard when it is focused, so its
    // editing keys are handled before the chat draft ever sees them.
    if (flecs::entity query_leaf = focused_entity_query())
    {
        EntityQuery& query = query_leaf.ensure<EntityQuery>();

        if (key == GLFW_KEY_BACKSPACE)
        {
            if (!query.draft.empty()) query.draft.pop_back();
            return;
        }
        if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER)
        {
            // Submitting is what acts; typing alone must not, since either kind
            // of submit is a round trip to a server.
            query.applied = query.draft;

            if (query.panel.is_valid() && query.panel.is_alive()) {
                if (query.kind == EntityQuery::Flecs) {
                    query.panel.ensure<EntitySet>().wanted = query.applied;
                } else {
                    query.panel.ensure<EntitySelection>().relation = query.applied;
                }
            }
            return;
        }
        if (key == GLFW_KEY_ESCAPE)
        {
            query.focused = false;
            return;
        }
    }

    // While the annotation outline is up, every keystroke belongs to
    // annotation. Keys its handlers claim have already returned above; keys
    // they don't claim must die here rather than leak into the chat draft --
    // a stray Backspace eating the draft or Enter sending a half-typed message
    // from inside annotation mode is worse than the key doing nothing.
    {
        bool annotating = false;
        annotation_query.each([&](flecs::entity, WordAnnotationSelector& selector) {
            if (selector.active) annotating = true;
        });
        if (annotating) return;
    }

    // Retrieve the singleton ChatState
    ChatState* chat = world->try_get_mut<ChatState>();
    if (chat)
    {
    if (key == GLFW_KEY_BACKSPACE)
    {
        if (!chat->draft.empty()) chat->draft.pop_back();
    }
    else if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER)
    {
        if (mods & GLFW_MOD_SHIFT)
        {
            // Shift+Enter: insert newline
            chat->draft.push_back('\n');
        }
        else if (!chat->draft.empty())
        {
            // Enter: send message
            chat->messages.push_back({"You", chat->draft});

            // Query for the ChatPanel entity to attach the UI element to
            world->query<ChatPanel>()
                .each([&](flecs::entity leaf, ChatPanel& chat_panel) {

                    auto UIElement = world->lookup("UIElement");
                    auto messageBox = create_message(UIElement, chat_panel, world->lookup("Wesley"), chat->draft.c_str());
                    
                    // The interlocutor client might have some submodel dependencies...
                    // which might be logical to run on another server because other parts of Thornfield depend on it

                    auto i_client = world->lookup("InterlocutorClient");
                    i_client.set<SendMapRequest>({ { {"type", "message"}, {"content", chat->draft} } });
                    i_client.set<AwaitResponse>({interlocutor_response});

                    // The message is also an entity. It lands in the headless
                    // world tagged Chat, so it is queryable and selectable
                    // alongside everything else rather than living only in this
                    // panel's scrollback.
                    create_world_entity("Chat", chat->draft);
                    
                    // TODO: Create thinking indicators...

                    world->defer_suspend();
                    // Annotation badges for the message just sent: a wrapping
                    // Yoga row, so they flow onto further lines instead of
                    // running off the edge of the history panel.
                    auto representation_ux = world->entity()
                        .is_a(UIElement)
                        // .add<DebugRenderBounds>()
                        .add(flecs::OrderedChildren)
                        .add<UIYoga>()
                        .set<UIFlexContainer>({
                            YGFlexDirectionRow, YGWrapWrap,
                            YGJustifyFlexStart, YGAlignCenter, 2.0f})
                        .child_of(chat_panel.message_list);

                        // Register message in global list
                        
                        g_annotatable_messages.push_back({chat->draft, representation_ux});
                        int msg_idx = (int)g_annotatable_messages.size() - 1;

                        flecs::entity interlocutorAnnotator = world->lookup("InterlocutorAnnotator");

                        // TODO: Only set sentence_template?
                        interlocutorAnnotator.set<WordAnnotationSelector>({
                            chat->draft,           // sentence_template
                            {},                       // ui_entities
                            {},                       // selection_entities
                            representation_ux,       // parent_entity
                            0, 0,                     // start_index, end_index
                            0,                        // token_count
                            true,                    // active
                            true,                     // dirty
                            0x4488FFAA                // highlight_color
                        })
                        .set<Position, World>({0, 0})
                        .set<RectRenderable>({0, 0, false, 0x4488FFAA})
                        .set<RenderStatus>({true})
                        .set<ZIndex>({15});
                        world->defer_resume();
                        
                        sync_representation_grounding(interlocutorAnnotator.ensure<WordAnnotationSelector>(), chat->draft);
                });

            chat->draft.clear();
        }
    }
    } 
    }
    // TODO: Check for focused VNC Stream...
    // Use static cached query to avoid creating new query on every key press
    static auto vnc_query = world->query<VNCClientHandle>();
    vnc_query.each([&](flecs::entity e, VNCClientHandle& handle) {
        VNCClient& vnc = *handle;
        PanelState& state = vnc.user_state_leaf.ensure<PanelState>();
        // TODO: Keyboard passthrough on Leaf State...
        if (vnc.connected && vnc.client && vnc.eventPassthroughEnabled) {
            rfbKeySym keysym = glfw_key_to_rfb_keysym(key, mods);

            // Queue keyboard event instead of sending directly
            InputEvent event;
            event.type = InputEvent::KEY;
            event.data.key.keysym = keysym;
            event.data.key.down = (action == GLFW_PRESS) ? TRUE : FALSE;

            if (action == GLFW_PRESS || action == GLFW_RELEASE) {
                std::lock_guard<std::mutex> lock(vnc.inputQueueMutex);
                vnc.inputQueue.push_back(event);
                vnc.inputQueueCV.notify_one();
            }
        }
    });
}

float point_distance_to_edge(Position p, Position a, Position b)
{
    // 1. Get vector from line start (a) to point (p)
    Vector2 v = Vector2Subtract(p, a);
    // 2. Get vector of the edge itself (a -> b)
    Vector2 edge = Vector2Subtract(b, a);
    // 3. Project v onto edge (returns normalized position t along the line)
    float edge_len_sq = Vector2LengthSqr(edge);
    if (edge_len_sq == 0.0f) return Vector2Distance(p, a); // Safety check
    float t = Vector2DotProduct(v, edge) / edge_len_sq;
    // 4. Clamp t to ensure we stay on the segment (0 to 1)
    t = Clamp(t, 0.0f, 1.0f);
    // 5. Calculate closest point: a + (edge * t)
    Vector2 closest_point = Vector2Add(a, Vector2Scale(edge, t));
    // 6. Return distance
    return Vector2Distance(p, closest_point);
}

thread_local std::stack<TracyCZoneCtx> zone_stack;

void trace_push(const char *file, size_t line, const char *name) {
    uint64_t srcloc = ___tracy_alloc_srcloc_name(
        (uint32_t)line,
        file, strlen(file),       // Source file
        name, strlen(name),       // Function name
        name, strlen(name),       // Zone name
        0                         // Color
    );

    TracyCZoneCtx ctx = ___tracy_emit_zone_begin_alloc(srcloc, 1);
    zone_stack.push(ctx);
}

void trace_pop(const char *file, size_t line, const char *name) {
    if (!zone_stack.empty()) {
        TracyCZoneEnd(zone_stack.top());
        zone_stack.pop();
    }
}

int main(int, char *[]) {

    ecs_os_set_api_defaults();
    ecs_os_api_t os_api = ecs_os_get_api();
    os_api.perf_trace_push_ = trace_push;
    os_api.perf_trace_pop_ = trace_pop;
    ecs_os_set_api(&os_api);

    flecs::world world_instance;
    world = &world_instance;

    world->entity("Wesley");
    world->entity("Heonae");

    world->import<tradewinds::module>();

    world->set<ZMQContext>({ std::make_shared<zmq::context_t>(2) });

    // TODO: Multiple chats
    flecs::entity interlocutor_server = world->entity("InterlocutorServer")
    .set<SpawnRequest>({"python3", {"../scripts/interlocutor.py"}});
    flecs::entity interlocutor_client = world->entity("InterlocutorClient")
        .set<ZMQClient>({ "ipc:///tmp/thornfield_interlocutor_socket", zmq::socket_type::req });

    // Embeddings for ordering the Entities panel by semantic proximity.
    flecs::entity entity_semantics_server = world->entity("EntitySemanticsServer")
        .set<SpawnRequest>({"python3", {"../scripts/entity_semantics.py"}});
    flecs::entity entity_semantics_client = world->entity("EntitySemanticsClient")
        .set<ZMQClient>({ "ipc:///tmp/thornfield_entity_semantics_socket", zmq::socket_type::req });

    // OpenFst plural transducer, induced live from lexicon.jsonl -- the S
    // gesture in the Lexicon panel teaches it.
    flecs::entity morphology_server = world->entity("MorphologyServer")
        .set<SpawnRequest>({"python3", {"../scripts/morphology.py"}});
    flecs::entity morphology_client = world->entity("MorphologyClient")
        .set<ZMQClient>({ "ipc:///tmp/thornfield_morphology_socket", zmq::socket_type::req });
    flecs::entity morphology_rules_client = world->entity("MorphologyRulesClient")
        .set<ZMQClient>({ "ipc:///tmp/thornfield_morphology_socket", zmq::socket_type::req });

    // Bridge to a separate headless flecs world; the Entities panel's rows come
    // from whatever that world answers with.
    flecs::entity entity_query_server = world->entity("EntityQueryServer")
        .set<SpawnRequest>({"python3", {"../scripts/entity_query.py"}});
    flecs::entity entity_query_client = world->entity("EntityQueryClient")
        .set<ZMQClient>({ "ipc:///tmp/thornfield_entity_query_socket", zmq::socket_type::req });

    // Its own socket rather than sharing the query client's: REQ/REP allows one
    // request in flight per socket, and a message submitted while a panel is
    // mid-query would otherwise be silently dropped.
    flecs::entity entity_create_client = world->entity("EntityCreateClient")
        .set<ZMQClient>({ "ipc:///tmp/thornfield_entity_query_socket", zmq::socket_type::req });

    // Initialize spatial index manager
    spatial::SpatialIndexManager spatial_manager(world);

    // Register std::string as an opaque type for serialization
    world->component<std::string>()
        .opaque(flecs::String)
        .serialize([](const flecs::serializer *s, const std::string *data) {
            const char *str = data->c_str();
            return s->value(flecs::String, &str);
        })
        .assign_string([](std::string *data, const char *value) {
            *data = value;
        });

    // TODO: Register spatial data

    // TODO: Refactor to ZeroMQ
    query_server::initialize(world, &spatial_manager);

    // TODO: query_server::register_spatial_handler
    // Can this be moved elsewhere?

    // Start socket server
    int port = 8005;
    query_server::start_server(port);

    glfwSetErrorCallback(error_callback);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Initialize fpng for fast PNG encoding
    fpng::fpng_init();

    // Start background PNG save queue
    g_pngSaveQueue.start();

    // Load DINO embedding model
    const char* dinoModelPath = getenv("DINO_MODEL_PATH");
    if (!dinoModelPath) {
        dinoModelPath = "models/dinov2-small.gguf";
    }
    if (g_dinoEmbedder.loadModel(dinoModelPath, 4)) {
        std::cout << "[DINO] Embedder loaded successfully" << std::endl;
    } else {
        std::cout << "[DINO] Failed to load model from " << dinoModelPath << " (embeddings disabled)" << std::endl;
    }

    // Initialize libssh2
    int rc = libssh2_init(0);
    if (rc) {
        std::cerr << "libssh2 initialization failed (" << rc << ")" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    // Start in fullscreen mode
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Thornfield", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetCharCallback(window, char_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetDropCallback(window, drop_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, mode->width, mode->height);
    NVGcontext* vg = nvgCreateGL2(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
    if (vg == NULL) {
        std::cerr << "Failed to initialize NanoVG" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Initialize SDL for VNC texture creation
    std::cout << "[SDL] Initializing SDL..." << std::endl;
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "[SDL ERROR] Failed to initialize: " << SDL_GetError() << std::endl;
        glfwTerminate();
        return -1;
    }
    std::cout << "[SDL] SDL initialized successfully" << std::endl;
    
    world->component<Position>()
    .member<float>("x")
    .member<float>("y");
    world->component<Velocity>();
    world->component<RectRenderable>();
    world->component<CustomRenderable>();
    world->component<TextRenderable>();
    world->component<ImageCreator>();
    world->component<ImageRenderable>();
    world->component<LineChartData>();
    world->component<LineChartChannel>();
    world->component<ZIndex>()
    .member<int>("layer");

    world->component<TimeInterval>();
    world->component<TimeIntervalToSpaceSegment>();

    world->component<PanelState>();

    world->component<RenderGradient>();

    world->component<Window>();
    world->component<CursorState>();
    world->component<Graphics>().add(flecs::Singleton);
    world->component<RenderQueue>();
    world->component<UIElementBounds>()
    .member<float>("xmin")
    .member<float>("ymin")
    .member<float>("xmax")
    .member<float>("ymax");
    ECS_COMPONENT(*world, UIElementBounds);
    world->component<UIElementSize>();
    world->component<UIContainer>();

    world->component<ScissorContainer>().add(flecs::Transitive);

    world->component<EditorNodeArea>();
    world->component<PanelSplit>();

    world->component<Align>();
    world->component<Expand>();
    world->component<Constrain>();
    world->component<BFOSprite>();

    world->component<ParentClass>().add(flecs::Transitive);

    world->component<LayoutBox>();
    world->component<FlowLayoutBox>();

    // ------------------------------------------------------------------
    // Yoga (flexbox) layout components + node lifecycle.
    // Registered here (rather than next to the Yoga systems further down)
    // because UI entities are created during setup *above* the system
    // registrations; the OnAdd observer must already exist by then or those
    // entities would never get a YogaNode.
    // ------------------------------------------------------------------
    world->component<UIYoga>();
    world->component<YogaNode>();
    world->component<UISize>();
    world->component<UIMaxSize>();
    world->component<UIMinSize>();
    world->component<UIAbsoluteEdges>();
    world->component<UIAspectRatio>();
    world->component<UIFlexContainer>();
    world->component<UIFlexItem>();
    world->component<UIMargin>();
    world->component<UIPadding>();
    world->component<UIFillParent>();
    world->component<UINativeImageSize>();
    world->component<UIMaxNativeImageSize>();
    world->component<UIContainParent>();
    world->component<UIYogaLegacyLeaf>();
    world->component<YogaTextCache>();
    world->component<UIInputDispatch>().add(flecs::Singleton);
    world->set<UIInputDispatch>({false});

    // Any entity tagged with UIYoga participates in a Yoga layout tree. When
    // the tag is added we allocate a YGNode and stash the entity id on it so
    // measure funcs / readback can recover the entity. When the backing
    // YogaNode is removed we detach from its Yoga parent, orphan its Yoga
    // children (their own observers will free them), and free the node.
    world->observer<UIYoga>()
        .event(flecs::OnAdd)
        .each([](flecs::entity e, UIYoga) {
            if (e.has<YogaNode>()) return;
            YGNodeRef n = YGNodeNew();
            YGNodeSetContext(n, (void*)(uintptr_t)e.id());
            e.set<YogaNode>({n});
        });

    world->observer<YogaNode>()
        .event(flecs::OnRemove)
        .each([](flecs::entity e, YogaNode& yn) {
            if (!yn.node) return;
            YGNodeRef owner = YGNodeGetOwner(yn.node);
            if (owner) YGNodeRemoveChild(owner, yn.node);
            while (YGNodeGetChildCount(yn.node) > 0) {
                YGNodeRemoveChild(yn.node, YGNodeGetChild(yn.node, 0));
            }
            YGNodeFree(yn.node);
            yn.node = nullptr;
        });

    world->component<DiurnalHour>();

    world->component<ChatMessage>();
    world->component<ChatMessageView>();
    world->component<ChatState>().add(flecs::Singleton);
    world->component<ChatPanel>();
    world->component<FocusChatInput>();
    world->component<SendChatMessage>();
    world->set<ChatState>({std::vector<ChatMessage>{}, "", false});

    world->component<PanelState>();

    // Registered up front because their first use is inside a deferred callback
    // -- EntityOrder lands in the semantics server's response handler, which
    // runs from a system, and registering a component while the world is in
    // readonly mode is a fatal assert.
    world->component<EntityBadge>();
    world->component<EntitySelection>();
    world->component<EntityOrder>();
    world->component<EntityQuery>();
    world->component<EntitySet>();
    world->component<BadgeContent>();
    world->component<CorefArc>();
    world->component<LexiconPanel>();
    world->component<TransducerPanel>();
    world->component<FocusEntityQuery>();
    world->component<VirtualList>();

    world->component<DragContext>().add(flecs::Singleton);
    world->set<DragContext>({false, flecs::entity::null(), PanelSplitType::Horizontal, 0.0f});

    world->observer<ImageCreator, Graphics>()
    .event(flecs::OnSet)
    .each([&](flecs::entity e, ImageCreator& img, Graphics& graphics)
    {
        // Memoized by path: nvgCreateImage decodes the file and uploads a
        // texture on every call, and nothing ever calls nvgDeleteImage. A list
        // that rebuilds its rows as it scrolls sets ImageCreator over and over
        // on the same handful of sprites, so uncached this would re-decode the
        // same PNG thousands of times and leak a texture on each one. Tint
        // lives on ImageRenderable, not in the texture, so sharing is safe.
        static std::unordered_map<std::string, int> image_cache;

        int imgHandle;
        auto cached = image_cache.find(img.path);
        if (cached != image_cache.end()) {
            imgHandle = cached->second;
        } else {
            imgHandle = nvgCreateImage(graphics.vg, ("../assets/" + img.path).c_str(), 0);
            if (imgHandle == -1) {
                std::cerr << "Failed to load " << img.path << std::endl;
            }
            image_cache[img.path] = imgHandle;
        }

        e.set<ImageRenderable>({imgHandle, img.scaleX, img.scaleY, 0.0f, 0.0f, img.tint});
        e.remove<ImageCreator>();
    });

    world->observer<ImageRenderable, Graphics>()
    .event(flecs::OnSet)
    .each([&](flecs::entity e, ImageRenderable& img, Graphics& graphics)
    {
        int nativeWidth, nativeHeight;
        nvgImageSize(graphics.vg, img.imageHandle, &nativeWidth, &nativeHeight);
        img.width = nativeWidth * img.scaleX;
        img.height = nativeHeight * img.scaleY;
    });

    double cursorXPos, cursorYPos;
    glfwGetCursorPos(window, &cursorXPos, &cursorYPos);

    auto glfwStateEntity = world->entity("GLFWState")
        .set<Window>({window, 800, 600})
        .set<CursorState>({cursorXPos, cursorYPos});

    auto graphicsEntity = world->entity("Graphics")
        .set<Graphics>({vg});

    // Initialize 3D rendering for plane
    Graphics& graphics = graphicsEntity.ensure<Graphics>();
    initialize3DRendering(graphics, 1200, 800);

    // The Droid panel's 3D scene. Imports render3d and compiles its shaders, so
    // it needs the GL context and nanovg, both of which exist by now. The
    // starting size is a placeholder -- Panel3DSyncSystem resizes the viewport
    // to the panel the first time one is laid out.
    if (!panel3d::init(*world, vg, window, "../assets/truncated_frame.stl", 512, 512)) {
        std::cerr << "[panel3d] disabled; the Droid panel will show only its background" << std::endl;
    }

    auto renderQueueEntity = world->entity("RenderQueue")
        .set<RenderQueue>({});

    // Initialize debug logger module (MUST BE FIRST!)
    DebugLogModule(*world);

    X11OutlineModule(*world);

    // Initialize mel spectrogram rendering module (must be after Graphics entity is created)
    MelSpecRenderModule(*world);

    auto UIElement = world->prefab("UIElement")
        .set<Position, Local>({0.0f, 0.0f})
        .set<Position, World>({0.0f, 0.0f})
        .set<UIElementBounds>({0, 0, 0, 0})
        .set<UIElementSize>({0.0f, 0.0f})
        .set<RenderStatus>({true})
        .set<ZIndex>({0});




    // TODO: Text search field
    // auto FieldEntry = world->prefab("FieldEntry")

    world->observer<UIElementBounds, AddTagOnHoverEnter>()
    .term_at(1).second<ShowServerHUDOverlay>()
    .event<HoverEnterEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverEnter)
    {
        e.set<ZIndex>({18});
        auto overlay = e.target<ServerHUDOverlay>();
        overlay.set<ServerDescription>({e});
        // create_popup(e);
        // overlay.set<ZIndex>({15});
    });
    
    world->observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<HideServerHUDOverlay>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        auto overlay = e.target<ServerHUDOverlay>();
        auto desc = overlay.ensure<ServerDescription>();
        if (desc.selected == e)
        {
            overlay.set<ServerDescription>({0});
        }
        e.set<ZIndex>({10});
    });

    world->observer<UIElementBounds, AddTagOnHoverEnter>()
    .term_at(1).second<HighlightBFOInheritanceHierarchy>()
    .event<HoverEnterEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverEnter)
    {
        auto bfo_sprite_query = world->query_builder<ZIndex>()
        .with<BFOSprite>()
        .without<ParentClass>(e)
        .build();

        bfo_sprite_query.each([e](flecs::entity non_parent, ZIndex& z_index)
        {
            if (non_parent != e)
            {
                z_index.layer = 0;    
            }
        });
    });

    world->observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<ResetBFOSprites>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        auto bfo_sprite_query = world->query_builder<ZIndex>()
        .with<BFOSprite>()
        .build();

        bfo_sprite_query.each([](flecs::entity non_parent, ZIndex& z_index)
        {
            z_index.layer = 50;
        });
    });

    world->system<ServerDescription, ZIndex>()
    .each([&](flecs::entity e, ServerDescription& desc, ZIndex& index) 
    {
        if (ecs_is_valid(*world, desc.selected))
        {
            e.set<ZIndex>({15});
        } 
        else
        {
            e.set<ZIndex>({0});
        }
    });
    
    world->observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<CloseEditorSelector>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        std::cout << "Hover exit editor selector region" << std::endl;
        e.destruct();
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<CloseEditorSelector>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        std::cout << "Select a particular panel..." << std::endl;
        e.destruct();
    });

    world->observer<UIElementBounds, AddTagOnHoverEnter>()
    .term_at(1).second<SetMenuHighlightColor>()
    .event<HoverEnterEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverEnter)
    {
        RoundedRectRenderable& bkg = e.ensure<RoundedRectRenderable>();
        bkg.color = 0x585858FF;
    });

    world->observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<SetMenuStandardColor>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        RoundedRectRenderable& bkg = e.ensure<RoundedRectRenderable>();
        bkg.color = 0x383838FF;
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<SetPanelEditorType>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        replace_editor_content(e.target<EditorLeaf>(), e.get_constant<EditorType>(), UIElement);
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<ToggleBooleanOption>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        // Toggle PanelState variable, switch checkbox to the opposite!
        PanelState& state = e.target<EditorLeaf>().ensure<PanelState>();
        PanelOption& option = state.options[e.ensure<PanelOptionData>().index];
        option.active = !option.active;
        std::string active = option.active ? "true" : "false";
        // std::string active = option.active ? "true" : "false";
        e.target<CheckBox>().set<ImageCreator>({("../assets/checkbox_" + active + ".png").c_str(), 1.0f, 1.0f});

        // Keep the menu open: without consuming, this click would also reach the
        // dropdown icon underneath (whose bubbled-up bounds contain the popup)
        // and immediately toggle the popup shut.
        consume_ui_click();
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<FocusChatInput>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds&, AddTagOnLeftClick)
    {
        ChatState& chat = world->ensure<ChatState>();
        chat.input_focused = true;
        // Only one text field can own the keyboard.
        world->query<EntityQuery>().each([](flecs::entity, EntityQuery& q) { q.focused = false; });
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<FocusEntityQuery>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds&, AddTagOnLeftClick)
    {
        // EntityQuery lives on the bar itself, so the clicked entity *is* the
        // one to focus. Everything else loses focus, including the
        // Interlocutor's chat draft.
        world->ensure<ChatState>().input_focused = false;
        world->query<EntityQuery>().each([&](flecs::entity bar, EntityQuery& query) {
            query.focused = (bar == e);
        });
        consume_ui_click();
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<SendChatMessage>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds&, AddTagOnLeftClick)
    {
        ChatState& chat = world->ensure<ChatState>();
        if (!chat.draft.empty())
        {
            chat.messages.push_back({"You", chat.draft});
            chat.draft.clear();
        }
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<SelectServer>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        const ServerScript* script = e.try_get<ServerScript>();
        if (script)
        {

        }
        std::cout << "Start chatter server here" << std::endl;
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<ShowPanelState>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        std::cout << "Left mouse click event on " << e.id() << std::endl;
        // Popup a menu to selector editor type (or close it)
        bool has_close_child = false;
        e.children([&](flecs::entity child) 
        {
            if (child.has<AddTagOnHoverExit, CloseEditorSelector>())
            {
                child.destruct();
                has_close_child = true;
            }
        });
        if (has_close_child) return;

        auto editor_hover_region = world->entity()
        .is_a(UIElement)
        .child_of(e)
        // .add<DebugRenderBounds>()
        .add<AddTagOnHoverExit, CloseEditorSelector>(); // TODO Invisible padding....
        // .add<AddTagOnLeftClick, CloseEditorSelector>();

        auto editor_icon_bkg_square = world->entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .set<Position, Local>({-1.0f, 10.0f})
        .set<RectRenderable>({32.0f, 12.0f, false, 0x282828FF})
        .set<ZIndex>({7});

        // Square corner hides the list's rounded top-left where the popup meets
        // the dropdown icon. ZIndex 31 > list (30) so it draws on top.
        auto editor_type_selector_square_corner = world->entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .set<Position, Local>({-1.0f, 19.0f})
        .set<RectRenderable>({16.0f, 16.0f, false, 0x282828FF})
        .set<ZIndex>({31});

        // The list is its own Yoga root: a vertical flex column that auto-sizes
        // to its option rows plus padding, and draws the popup background
        // itself. That replaces the old fixed-256px backing rect that had to be
        // stretched to fit with Expand.
        auto editor_type_list = world->entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexContainer>({
            YGFlexDirectionColumn, YGWrapNoWrap,
            YGJustifyFlexStart, YGAlignFlexStart, 2.0f})
        .set<UIPadding>({4.0f, 6.0f, 4.0f, 6.0f})
        .set<Position, Local>({-1.0f, 19.0f})
        .set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, false, 0x282828FF})
        .set<ZIndex>({30});

        flecs::entity leaf = e.parent().parent();
        PanelState& state = leaf.ensure<PanelState>();
        size_t i = 0;
        for (PanelOption& option : state.options)
        {
            auto editor_option_btn = world->entity()
            .is_a(UIElement)
            .child_of(editor_type_list)
            .add<UIYoga>()
            .set<UISize>({196.0f-12.0f, 28.0f})
            // Row so the checkbox is placed by Yoga against the right edge and
            // vertically centred. It used to rely on Align, which reads bounds
            // that aren't computed until a later phase -- so the checkbox
            // snapped into position a frame after the popup appeared.
            .set<UIFlexContainer>({
                YGFlexDirectionRow, YGWrapNoWrap,
                YGJustifyFlexEnd, YGAlignCenter, 0.0f})
            .set<UIPadding>({0.0f, 10.0f, 0.0f, 0.0f})
            .set<RoundedRectRenderable>({196.0f-12.0f, 28.0f, 2.0f, false, 0x383838FF})
            .add<AddTagOnHoverEnter, SetMenuHighlightColor>()
            .add<AddTagOnHoverExit, SetMenuStandardColor>()
            .add<AddTagOnLeftClick, ToggleBooleanOption>()
            .add<EditorLeaf>(leaf)
            .set<ZIndex>({38});

            world->entity()
            .is_a(UIElement)
            .child_of(editor_option_btn)
            .set<TextRenderable>({option.name, "JetBrainsMono", 12.0f, 0xFFFFFFFF})
            .set<Position, Local>({4.0f, 8.0f})
            .set<ZIndex>({40});
            
            std::string active = option.active ? "true" : "false";

            flecs::entity checkbox = world->entity()
            .is_a(UIElement)
            .child_of(editor_option_btn)
            .add<UIYoga>()
            .add<UINativeImageSize>()
            .set<ImageCreator>({("../assets/checkbox_" + active + ".png").c_str(), 1.0f, 1.0f})
            .set<ZIndex>({42});

            editor_option_btn.add<CheckBox>(checkbox);
            editor_option_btn.set<PanelOptionData>({i});
            i++;

        }

        // Built inside a click handler, so lay it out now: the popup is drawn
        // and hit-tested this frame at its final size, with no settling pass.
        yoga_layout_now(editor_type_list);
    });

    world->observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<ShowEditorPanels>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        std::cout << "Left mouse click event on " << e.id() << std::endl;
        // Popup a menu to selector editor type (or close it)
        bool has_close_child = false;
        e.children([&](flecs::entity child) 
        {
            if (child.has<AddTagOnHoverExit, CloseEditorSelector>())
            {
                child.destruct();
                has_close_child = true;
            }
        });
        if (has_close_child) return;

        auto editor_hover_region = world->entity()
        .is_a(UIElement)
        .child_of(e)
        // .add<DebugRenderBounds>()
        .add<AddTagOnHoverExit, CloseEditorSelector>()
        .add<AddTagOnLeftClick, CloseEditorSelector>();

        auto editor_icon_bkg_square = world->entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .set<Position, Local>({-1.0f, 10.0f})
        .set<RectRenderable>({32.0f, 12.0f, false, 0x282828FF})
        .set<ZIndex>({7});

        // Square corner hides the list's rounded top-left where the popup meets
        // the dropdown icon. ZIndex 31 > list (30) so it draws on top.
        auto editor_type_selector_square_corner = world->entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .set<Position, Local>({-1.0f, 19.0f})
        .set<RectRenderable>({16.0f, 16.0f, false, 0x282828FF})
        .set<ZIndex>({31});

        // Self-sizing Yoga column: height follows the number of editor types
        // instead of the old fixed 256px backing rect.
        auto editor_type_list = world->entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexContainer>({
            YGFlexDirectionColumn, YGWrapNoWrap,
            YGJustifyFlexStart, YGAlignFlexStart, 2.0f})
        .set<UIPadding>({4.0f, 6.0f, 4.0f, 6.0f})
        .set<Position, Local>({-1.0f, 19.0f})
        .set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, false, 0x282828FF})
        .set<ZIndex>({30});

        size_t editor_type_index = 0;
        for (std::string editor_type_name : editor_types)
        {
            // When you click on these elements,
            // change the EditorType
            // Remove any existing type scene content
            // and load the new default scene...
            auto edtior_type_btn = world->entity()
            .is_a(UIElement)
            .child_of(editor_type_list)
            .add<UIYoga>()
            .set<UISize>({196.0f-12.0f, 20.0f})
            .set<RoundedRectRenderable>({196.0f-12.0f, 20.0f, 2.0f, false, 0x383838FF})
            .add<AddTagOnHoverEnter, SetMenuHighlightColor>()
            .add<AddTagOnHoverExit, SetMenuStandardColor>()
            .add<AddTagOnLeftClick, SetPanelEditorType>()
            .add<EditorLeaf>(e.target<EditorLeaf>())
            .add((EditorType)editor_type_index)
            .add<EditorCanvas>(e.target<EditorCanvas>())
            .set<ZIndex>({38});


            world->entity()
            .is_a(UIElement)
            .child_of(edtior_type_btn)
            .set<TextRenderable>({editor_type_name.c_str(), "JetBrainsMono", 12.0f, 0xFFFFFFFF})
            .set<Position, Local>({4.0f, 5.0f})
            .set<ZIndex>({40});
            
            editor_type_index++;
        }

        // Built inside a click handler, so lay it out now: the popup is drawn
        // and hit-tested this frame at its final size, with no settling pass.
        yoga_layout_now(editor_type_list);
    });

    auto movementSystem = world->system<Position, Velocity>()
    .term_at(0).second<Local>()
        .each([](flecs::iter& it, size_t i, Position& pos, Velocity& vel) {
            float deltaTime = it.delta_system_time();

            pos.x += vel.dx * deltaTime;
            pos.y += vel.dy * deltaTime;
        });

    // ========================================================================
    // PHASE 1: Initialize sizes and propagate world positions
    // ========================================================================

    // Phase 1a: Hierarchical positioning - computes world positions from local positions
    auto hierarchicalQuery = world->query_builder<const Position, const Position*, Position>()
        .term_at(0).second<Local>()      // Local position
        .term_at(1).second<World>()      // Parent world position
        .term_at(2).second<World>()      // This entity's world position
        .term_at(1).parent().cascade()   // Get parent position in breadth-first order
        .build();

    auto hierarchicalSystem = world->system()
        .kind(flecs::OnLoad)  // Run after layout systems to compute world positions
        .each([&]() {
            // std::cout << "Update hierarchy" << std::endl;
            hierarchicalQuery.each([](const Position& local, const Position* parentWorld, Position& worldPos) {
                worldPos.x = local.x;
                worldPos.y = local.y;
                if (parentWorld) {
                    worldPos.x += parentWorld->x;
                    worldPos.y += parentWorld->y;
                }
            });
        });

    // Phase 1b: Initial bounds calculation from world position + size
    auto boundsCalculationSystem = world->system<Position, UIElementBounds, UIElementSize>()
        .term_at(0).second<World>()
        .kind(flecs::OnLoad)
        .each([&](flecs::entity e, Position& worldPos, UIElementBounds& bounds, UIElementSize& size) {
            ZoneScoped;
            // Set bounds from world position + size
            bounds.xmin = worldPos.x;
            bounds.ymin = worldPos.y;
            bounds.xmax = worldPos.x;
            bounds.ymax = worldPos.y;

            if (size.width > 0 && size.height > 0)
            {
                bounds.xmax += size.width;
                bounds.ymax += size.height;
            }
        });

    int editor_padding = 3.0f;
    int editor_edge_hover_dist = 8.0f;

    auto editor_root = world->entity("editor_root")
        .set<Position, Local>({0.0f, 28.0f})
        .set<Position, World>({0.0f, 0.0f})
        .set<EditorNodeArea>({800.0f, 600.0f - 28.0f}) // TOOD: Observer to update root to window width/height updates
        .add<EditorRoot>()
        .add(flecs::OrderedChildren);

    auto editor_header = world->entity()
        .set<Position, Local>({0.0f, -28.0f})
        .set<EditorNodeArea>({800.0f, 28.0f})
        .is_a(UIElement)
        .child_of(editor_root);

    auto editor_header_elements = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 16.0f})
        .add(flecs::OrderedChildren)
        .child_of(editor_header);

    auto cogarc_module_logo = world->entity()
        .is_a(UIElement)
        .set<ImageCreator>({"../assets/ecs_header.png", 1.0f, 1.0f})
        .set<ZIndex>({5})
        .child_of(editor_header_elements);

    auto editor_version = world->entity()
        .is_a(UIElement)
        .set<Position, Local>({0.0f, 8.0f})
        .set<TextRenderable>({"The heaven is high and the emperor is far away. The monologic is dead. Long live the swarm.", "JetBrainsMono", 12.0f, 0x888888FF})
        // .set<TextRenderable>({"Pearl Magi, Whalefall Abyss", "JetBrainsMono", 12.0f, 0x888888FF})
        .set<ZIndex>({5})
        .child_of(editor_header_elements);

    // Load saved editor config if it exists, otherwise use default layout
    if (!load_editor_layout(editor_root, UIElement)) {
        // Default layout
        split_editor({0.5, PanelSplitType::Horizontal}, editor_root, UIElement);
        auto right_node = editor_root.target<RightNode>();
        split_editor({0.35, PanelSplitType::Vertical}, right_node, UIElement);
        auto left_node = editor_root.target<LeftNode>();
        split_editor({0.25, PanelSplitType::Vertical}, left_node, UIElement);
    }

    float diurnal_pos = 0.0f;
    world->system<DiurnalHour, QuadraticBezierRenderable>()
    .interval(1)
    .run([&diurnal_pos](flecs::iter& it)
    {
        diurnal_pos = get_time_of_day_normalized();
        while (it.next()) {
            it.each();
        }
    }, [&diurnal_pos](flecs::entity e, DiurnalHour& hour, QuadraticBezierRenderable& renderable) {
        e.set<QuadraticBezierRenderable>(get_hour_segment(hour.segment, (0.5f * M_PI) - (diurnal_pos) * (2.0f * M_PI)));
    });

    world->system<Position, EditorNodeArea, PanelSplit, CursorState>()
        .kind(flecs::PreFrame)
        .term_at(0).second<World>()
        .term_at(3).src(glfwStateEntity)
        .with<Dragging>()
        .each([](flecs::entity e, Position& world_pos, EditorNodeArea& node_area, PanelSplit& panel_split, CursorState& cursor_state)
        {
            // TODO: Propagate percent update to children to keep them 'in place'
            if (panel_split.dim == PanelSplitType::Horizontal)
            {
                panel_split.percent = std::clamp(((float)cursor_state.x-world_pos.x)/node_area.width, 0.05f, 0.95f);
            } 
            else if (panel_split.dim == PanelSplitType::Vertical)
            {
                panel_split.percent = std::clamp(((float)cursor_state.y-world_pos.y)/node_area.height, 0.05f, 0.95f);
            }
            // std::cout << "Evaluate drag update" << std::endl;
        });

    auto propagateEditorRoot = world->system<Window, EditorNodeArea, EditorRoot>("EditorPropagate")
    .term_at(0).src(glfwStateEntity)
    .kind(flecs::PreFrame)
    .run([](flecs::iter& it)
    {
        while (it.next()) {
            auto window = it.field<const Window>(0);
            auto node_area = it.field<EditorNodeArea>(1);
            for (auto i : it) {
                node_area[i].width = window[i].width;
                node_area[i].height = window[i].height-28.0f; // minus the header...
            }
            flecs::entity editor_root = world->lookup("editor_root");

            std::stack<flecs::entity> editors_to_visit;
            editors_to_visit.push(editor_root);
            while (!editors_to_visit.empty())
            {
                flecs::entity editor = editors_to_visit.top();
                editors_to_visit.pop();

                bool is_leaf = editor.has<EditorVisual>(flecs::Wildcard);
                EditorNodeArea& node_area = editor.ensure<EditorNodeArea>();
                // Update EditorVisual/EditorOutline if the node is a leaf
                // This could be refactored to an independent system...
                if (is_leaf)
                {
                    flecs::entity visual = editor.target<EditorVisual>();
                    // TODO: Only modify first two params...
                    visual.set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, false, 0x010222});
                    flecs::entity outline = editor.target<EditorOutline>();
                    outline.set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, true, 0x191919FF});
                }

                PanelSplit* split = editor.try_get_mut<PanelSplit>();
                // Update child EditorNodeAreas if the panel is split
                if (split)
                {
                    if (split->dim == PanelSplitType::Horizontal)
                    {
                        flecs::entity left_node = editor.target<LeftNode>();
                        left_node.set<EditorNodeArea>({node_area.width*split->percent, node_area.height});
                        editors_to_visit.push(left_node);
                        flecs::entity right_node = editor.target<RightNode>();
                        right_node.set<EditorNodeArea>({node_area.width*(1-split->percent), node_area.height});
                        right_node.set<Position, Local>({node_area.width*split->percent, 0.0f});
                        editors_to_visit.push(right_node);
                    } else
                    {
                        flecs::entity upper_node = editor.target<UpperNode>();
                        upper_node.set<EditorNodeArea>({node_area.width, node_area.height*split->percent});
                        editors_to_visit.push(upper_node);
                        flecs::entity lower_node = editor.target<LowerNode>();
                        lower_node.set<EditorNodeArea>({node_area.width, node_area.height*(1-split->percent)});
                        lower_node.set<Position, Local>({0.0f, node_area.height*split->percent});
                        editors_to_visit.push(lower_node);
                    }
                }
            }
        }
    });


    // ========================================================================
    // PHASE 1b2: Yoga (flexbox) layout
    // ------------------------------------------------------------------------
    // Runs at PreFrame, registered *before* sizeCalculationSystem so that the
    // ordering within the phase is:
    //   topology sync -> style sync -> measure -> calculate -> readback
    //     -> sizeCalculationSystem (Phase 1c)
    // Readback writes Position<Local> plus renderable width/height, which
    // Phase 1c then lifts into UIElementSize and OnLoad turns into world
    // positions + UIElementBounds. Only entities tagged UIYoga participate,
    // so Yoga subtrees and legacy LayoutBox subtrees can coexist.
    // ========================================================================

    // The per-entity steps live at file scope (see yoga_sync_* above) so that
    // yoga_layout_now() can run the same pipeline synchronously for UI built
    // inside an input handler.
    g_yoga_vg = graphics.vg;

    // Virtualized lists realize their visible window before any of the layout
    // below runs, so rows built this frame are laid out this frame. immediate()
    // is what makes that true: a deferred create would not become a real entity
    // until the next sync point, leaving scrolling a frame behind the wheel.
    world->system<VirtualList>("VirtualListSystem")
        .kind(flecs::PreFrame)
        .immediate()
        .run([](flecs::iter& it) {
            // Handles are collected before any are updated: update_virtual_list
            // creates and destroys row entities, and in immediate mode that can
            // move tables out from under a live iterator.
            std::vector<flecs::entity> lists;
            while (it.next()) {
                for (auto i : it) lists.push_back(it.entity(i));
            }
            for (flecs::entity list : lists) {
                update_virtual_list(list, list.ensure<VirtualList>());
            }
        });

    // Rebuilds the annotation UI whenever a key handler has rewritten the
    // template. Every handler sets dirty; before this system existed only the
    // R handler called the rebuild directly, so binding a symbol to a slot
    // updated the template invisibly -- the sentence on screen never changed.
    // Keeps Transducer panels current with the morphology server's rule
    // series: requests a dump when stale, rebuilds the display when the
    // series actually changed. Rules render in the badge language -- the
    // rewrite as letter chips, evidence and confidence beside it.
    world->system<TransducerPanel>("TransducerSyncSystem")
        .kind(flecs::PreFrame)
        .immediate()
        .run([](flecs::iter& it) {
            std::vector<flecs::entity> panels;
            while (it.next()) {
                for (auto i : it) panels.push_back(it.entity(i));
            }
            if (panels.empty()) return;

            if (g_transducer.stale) {
                flecs::entity client = world->lookup("MorphologyRulesClient");
                if (client.is_valid() && !client.has<AwaitResponse>()) {
                    g_transducer.stale = false;   // one request per staleness
                    client.set<SendMapRequest>({{{"type", "dump"}}});
                    client.set<AwaitResponse>({transducer_dump_response});
                }
            }
            if (!g_transducer.valid) return;

            auto UIElement = world->lookup("UIElement");
            for (flecs::entity e : panels) {
                TransducerPanel& panel = e.ensure<TransducerPanel>();
                if (panel.shown_sig == g_transducer.signature) continue;
                if (!panel.list.is_valid() || !panel.list.is_alive()) continue;

                std::vector<flecs::entity> stale_children;
                panel.list.children([&](flecs::entity c) { stale_children.push_back(c); });
                for (flecs::entity c : stale_children) c.destruct();

                // The machine itself, in one line: what was learned, from how
                // much, compiled to what.
                world->entity()
                    .is_a(UIElement)
                    .child_of(panel.list)
                    .add<UIYoga>()
                    .set<TextRenderable>({
                        std::to_string(g_transducer.rules.size()) + " rules from "
                        + std::to_string(g_transducer.lexicon) + " demonstrations   "
                        + "OpenFst " + std::to_string(g_transducer.states) + " states / "
                        + std::to_string(g_transducer.arcs) + " arcs",
                        "JetBrainsMono", 12.0f, 0x9c9c9cFF})
                    .set<ZIndex>({12});

                for (const auto& rule : g_transducer.rules) {
                    auto row = world->entity()
                        .is_a(UIElement)
                        .child_of(panel.list)
                        .add(flecs::OrderedChildren)
                        .add<UIYoga>()
                        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                               YGJustifyFlexStart, YGAlignCenter, 3.0f})
                        .set<ZIndex>({12});

                    // The rewrite, in letter chips: consumed suffix, arrow,
                    // produced suffix (or the deletion mark when it produces
                    // nothing).
                    for (char c : rule.in)
                        create_letter_chip(row, std::string(1, c), letter_color(c));
                    world->entity()
                        .is_a(UIElement)
                        .child_of(row)
                        .add<UIYoga>()
                        .set<UIMargin>({0.0f, 2.0f, 0.0f, 2.0f})
                        .set<TextRenderable>({"\u2192", "CharisSIL", 16.0f, 0x9c9c9cFF})
                        .set<ZIndex>({13});
                    if (rule.out.empty()) {
                        world->entity()
                            .is_a(UIElement)
                            .child_of(row)
                            .add<UIYoga>()
                            .set<TextRenderable>({"\u2205", "CharisSIL", 16.0f, 0x6a6a6aFF})
                            .set<ZIndex>({13});
                    } else {
                        for (char c : rule.out)
                            create_letter_chip(row, std::string(1, c), letter_color(c));
                    }

                    world->entity()
                        .is_a(UIElement)
                        .child_of(row)
                        .add<UIYoga>()
                        .set<UIMargin>({0.0f, 0.0f, 0.0f, 10.0f})
                        .set<TextRenderable>({
                            std::to_string((int)(rule.confidence * 100.0f)) + "%   \u00d7"
                            + std::to_string(rule.support),
                            "JetBrainsMono", 12.0f, 0x4d6d8aFF})
                        .set<ZIndex>({13});
                }

                panel.shown_sig = g_transducer.signature;
            }
        });

    // Mirrors the annotator's current selection into any open Lexicon panels:
    // the selected word, one character glyph per entity, tinted with the
    // word's bound entity colour when it has one. immediate() because it
    // rebuilds UI. Each glyph is its own entity deliberately -- the rule
    // layers this panel exists for will attach to individual characters.
    world->system<LexiconPanel>("LexiconSyncSystem")
        .kind(flecs::PreFrame)
        .immediate()
        .run([](flecs::iter& it) {
            std::vector<flecs::entity> panels;
            while (it.next()) {
                for (auto i : it) panels.push_back(it.entity(i));
            }
            if (panels.empty()) return;

            // The word under the annotation cursor, from the active selector.
            std::string word;
            static flecs::query<WordAnnotationSelector> selectors =
                world->query<WordAnnotationSelector>();
            selectors.each([&](flecs::entity, WordAnnotationSelector& selector) {
                if (!selector.active || selector.sentence_template.empty()) return;
                auto tokens = parse_sentence_template(selector.sentence_template);
                int expanded_pos = 0;
                for (size_t ti = 0; ti < tokens.size(); ti++) {
                    int width = token_selection_width(tokens, ti);
                    if (selector.start_index >= expanded_pos
                        && selector.start_index < expanded_pos + width) {
                        word = tokens[ti].text;
                        return;
                    }
                    expanded_pos += width;
                }
            });

            try_send_teach();
            try_send_false();

            // Ask the transducer about a word it hasn't been asked about.
            // One request in flight (REQ/REP); LOADING never happens here --
            // the FST compiles in milliseconds -- so no retry machinery.
            if (!word.empty() && word != g_fst_requested) {
                flecs::entity client = world->lookup("MorphologyClient");
                if (client.is_valid() && !client.has<AwaitResponse>()) {
                    g_fst_requested = word;
                    client.set<SendMapRequest>({{{"type", "analyze"}, {"word", word}}});
                    client.set<AwaitResponse>({morphology_response});
                }
            }

            auto UIElement = world->lookup("UIElement");

            // Builds one character row; chips in [ring_lo, ring_hi] get the
            // selection ring. Each glyph stays its own entity -- rule layers
            // point at characters, not strings.
            auto build_row = [&](flecs::entity parent, const std::string& text,
                                 int ring_lo, int ring_hi, const char* type_label,
                                 const std::vector<uint8_t>* disabled = nullptr) {
                auto row = world->entity()
                    .is_a(UIElement)
                    .child_of(parent)
                    .add(flecs::OrderedChildren)
                    .add<UIYoga>()
                    .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                                           YGJustifyFlexStart, YGAlignCenter, 2.0f})
                    .set<ZIndex>({12});
                int idx = 0;
                for (char c : text) {
                    if (std::isspace(static_cast<unsigned char>(c))) continue;
                    bool off = disabled && idx < (int)disabled->size() && (*disabled)[idx];
                    flecs::entity chip = create_letter_chip(row, std::string(1, c),
                                                            off ? 0x2e2e2eFF : letter_color(c));
                    if (idx >= ring_lo && idx <= ring_hi) {
                        world->entity()
                            .is_a(UIElement)
                            .child_of(chip)
                            .add<UIYoga>()
                            .set<UIAbsoluteEdges>({-2.0f, -2.0f, -2.0f, -2.0f})
                            .set<RoundedRectRenderable>({0.0f, 0.0f, 5.0f, true, 0xFFFFFFE0})
                            .set<ZIndex>({14});
                    }
                    idx++;
                }
                if (type_label) {
                    world->entity()
                        .is_a(UIElement)
                        .child_of(row)
                        .add<UIYoga>()
                        .set<UIMargin>({0.0f, 0.0f, 0.0f, 8.0f})
                        .set<TextRenderable>({type_label, "JetBrainsMono", 12.0f, 0x9c9c9cFF})
                        .set<ZIndex>({13});
                }
                return row;
            };

            for (flecs::entity e : panels) {
                LexiconPanel& panel = e.ensure<LexiconPanel>();
                if (panel.shown == word && !panel.dirty) continue;
                if (!panel.glyph_row.is_valid() || !panel.glyph_row.is_alive()) continue;

                // A new word resets the paradigm to its surface row. Char
                // mode survives the move -- arrowing the sentence while
                // descended sweeps word to word with the cursor landing on
                // each word's last character, ready for suffix work.
                if (panel.shown != word) {
                    panel.row = 0;
                    panel.typing_type = false;
                    panel.prefilled = false;
                    panel.forms.clear();
                    if (!word.empty()) {
                        std::string chars;
                        for (char c : word)
                            if (!std::isspace(static_cast<unsigned char>(c))) chars += c;
                        panel.forms.push_back({chars, std::string()});
                        panel.sel_start = panel.sel_end =
                            std::max(0, (int)chars.size() - 1);
                    } else {
                        panel.selecting = false;
                    }
                }

                // The transducer's proposal pre-fills the paradigm once per
                // word -- but ONLY while the paradigm is untouched. The reply
                // is asynchronous and can land seconds late; a proposal must
                // never overwrite an annotation the user has already made.
                // The empty type on the surface row is the untouched test.
                if (!panel.prefilled && panel.forms.size() == 1
                    && panel.forms[0].type.empty()) {
                    std::string lowered = panel.forms[0].text;
                    for (char& c : lowered) c = std::tolower(static_cast<unsigned char>(c));
                    if (!lowered.empty() && g_fst_proposal.word == lowered) {
                        // An ambiguous form (pants) is ONE node with an OR
                        // of types; a paired form gets its variant as a
                        // proposed second row; a typed-but-unpaired form gets
                        // its type and nothing else.
                        if (g_fst_proposal.variant == panel.forms[0].text) {
                            panel.forms[0].type = g_fst_proposal.word_type + " or "
                                                + g_fst_proposal.variant_type;
                        } else {
                            panel.forms[0].type = g_fst_proposal.word_type;
                            if (!g_fst_proposal.variant.empty()) {
                                LexiconForm variant;
                                variant.text = g_fst_proposal.variant;
                                variant.type = g_fst_proposal.variant_type;
                                variant.proposed = true;
                                panel.forms.push_back(variant);
                            }
                        }
                        panel.prefilled = true;
                    }
                }

                std::vector<flecs::entity> stale;
                panel.glyph_row.children([&stale](flecs::entity c) { stale.push_back(c); });
                for (flecs::entity c : stale) c.destruct();

                for (int fi = 0; fi < (int)panel.forms.size(); fi++) {
                    const LexiconForm& form = panel.forms[fi];
                    bool selected_row = panel.selecting && fi == panel.row;
                    int lo = selected_row ? std::min(panel.sel_start, panel.sel_end) : -1;
                    int hi = selected_row ? std::max(panel.sel_start, panel.sel_end) : -1;
                    flecs::entity row_entity =
                        build_row(panel.glyph_row, form.text, lo, hi, nullptr);

                    // The annotation is a badge, because the type is an entity
                    // -- same visual grammar as every other entity in the
                    // system, identity colour included. While the type is
                    // being typed, the badge grows with a cursor.
                    bool typing_here = panel.typing_type && selected_row;
                    if (!form.type.empty() || typing_here) {
                        // Typed naturally: "Plural or Singular". Space-separated
                        // tokens; the word "or" between type names renders as
                        // the OR glyph, every other token as a type badge.
                        std::vector<std::string> tokens;
                        std::string current;
                        for (char c : form.type) {
                            if (c == ' ') {
                                if (!current.empty()) tokens.push_back(current);
                                current.clear();
                            } else current += c;
                        }
                        if (!current.empty()) tokens.push_back(current);

                        auto is_operator = [](const std::string& token) {
                            std::string lower;
                            for (char c : token)
                                lower += (char)std::tolower(static_cast<unsigned char>(c));
                            return lower == "or";   // surface word -> Or entity
                        };
                        bool first = true;
                        for (const std::string& token : tokens) {
                            if (is_operator(token)) {
                                // The operator is an entity like everything
                                // else, so it renders as its badge: outlined
                                // (an operator is structure), in the Or
                                // entity's own identity colour.
                                create_badge_impl(row_entity, UIElement, "or",
                                                  entity_type_color("Or"),
                                                  false, false, {}, {}, {}, {},
                                                  nullptr, BadgeStyle::String)
                                    .set<UIMargin>({0.0f, 0.0f, 0.0f, 4.0f});
                                continue;
                            }
                            create_badge(row_entity, UIElement, token.c_str(),
                                         entity_type_color(token))
                                .set<UIMargin>({0.0f, 0.0f, 0.0f, first ? 8.0f : 4.0f});
                            first = false;
                        }
                        if (typing_here) {
                            world->entity()
                                .is_a(UIElement)
                                .child_of(row_entity)
                                .add<UIYoga>()
                                .set<UIMargin>({0.0f, 0.0f, 0.0f, 2.0f})
                                .set<TextRenderable>({"|", "JetBrainsMono",
                                                      16.0f, 0xFFFFFFFF})
                                .set<ZIndex>({14});
                        }
                    }
                }
                if (panel.prefilled) {
                    world->entity()
                        .is_a(UIElement)
                        .child_of(panel.glyph_row)
                        .add<UIYoga>()
                        .set<TextRenderable>({"FST  " + g_fst_proposal.rule + "  "
                                              + std::to_string((int)(g_fst_proposal.confidence * 100.0f))
                                              + "%",
                                              "JetBrainsMono", 11.0f, 0x4d6d8aFF})
                        .set<ZIndex>({13});
                }

                panel.shown = word;
                panel.dirty = false;
            }
        });

    // Re-derives each co-reference arc's curve from the current bounds of the
    // two things it connects. Geometry lives here, not at creation: the badges
    // move whenever layout runs -- a wrap, a join, a reified block growing --
    // and an arc baked to yesterday's positions would point at empty space.
    world->system<CorefArc, QuadraticBezierRenderable>("CorefArcSystem")
        .kind(flecs::PreUpdate)
        .each([](flecs::entity e, CorefArc& arc, QuadraticBezierRenderable& curve) {
            // Top-centre of a connected element, from its bounds.
            auto anchor_top = [](flecs::entity ent, float& x, float& y) {
                if (!ent.is_valid() || !ent.is_alive()) return false;
                const UIElementBounds* b = ent.try_get<UIElementBounds>();
                if (!b || b->xmax <= b->xmin) return false;
                x = (b->xmin + b->xmax) * 0.5f;
                y = b->ymin;
                return true;
            };

            float fx, fy, tx, ty;
            if (!anchor_top(arc.from, fx, fy) || !anchor_top(arc.to, tx, ty)) {
                curve.thickness = 0.0f;   // nothing sane to draw yet
                return;
            }
            if (curve.thickness <= 0.0f) curve.thickness = 1.5f;

            // Rise with distance, capped: long arcs stay shallow enough to
            // read as connection rather than as a dome over the sentence.
            float span = std::fabs(tx - fx);
            float rise = std::clamp(span * 0.25f, 12.0f, 42.0f);

            // Curve coords are relative to the entity's world position.
            const Position* origin = e.try_get<Position, World>();
            float ox = origin ? origin->x : 0.0f;
            float oy = origin ? origin->y : 0.0f;

            curve.x1 = fx - ox;  curve.y1 = fy - oy;
            curve.cx = (fx + tx) * 0.5f - ox;
            curve.cy = std::min(fy, ty) - rise - oy;
            curve.x2 = tx - ox;  curve.y2 = ty - oy;
        });

    // immediate(), like VirtualListSystem: the rebuild creates and destroys UI
    // entities, which cannot happen inside the pipeline's readonly staging --
    // suspending deferral there is illegal and asserts EcsWorldReadonly the
    // moment anything structural runs. Handles are collected before any
    // rebuild, since rebuilding moves tables under a live iterator.
    world->system<WordAnnotationSelector>("AnnotationRebuildSystem")
        .kind(flecs::PreFrame)
        .immediate()
        .run([](flecs::iter& it) {
            std::vector<flecs::entity> dirty;
            while (it.next()) {
                auto selectors = it.field<WordAnnotationSelector>(0);
                for (auto i : it) {
                    if (selectors[i].dirty) dirty.push_back(it.entity(i));
                }
            }
            for (flecs::entity e : dirty) {
                WordAnnotationSelector& selector = e.ensure<WordAnnotationSelector>();
                sync_representation_grounding(selector, selector.sentence_template);
            }
        });

    // Same phase and same reasoning as the list above: the inspector's contents
    // are built before layout runs, so a freshly selected entity is positioned
    // in the frame it is selected rather than the one after.
    world->system<EntitySelection>("EntityInspectorSystem")
        .kind(flecs::PreFrame)
        .immediate()
        .run([](flecs::iter& it) {
            std::vector<flecs::entity> panels;
            while (it.next()) {
                for (auto i : it) panels.push_back(it.entity(i));
            }
            for (flecs::entity panel : panels) {
                update_entity_inspector(panel, panel.ensure<EntitySelection>());
            }
        });

    // Keeps the relation bar's text in sync with what has been typed, and shows
    // a prompt when it is empty so the field reads as something to type into
    // rather than as an empty strip.
    world->system<EntityQuery>("EntityQueryTextSystem")
        .kind(flecs::PreFrame)
        .each([](flecs::entity bar, EntityQuery& query) {
            if (!query.input_text.is_valid() || !query.input_text.is_alive()) return;
            TextRenderable* text = query.input_text.try_get_mut<TextRenderable>();
            if (!text) return;

            if (query.draft.empty() && !query.focused) {
                text->text = query.placeholder;
                text->color = 0x555555FF;
                return;
            }

            text->text = query.draft + (query.focused ? "|" : "");
            text->color = 0xFFFFFFFF;

            // The flecs bar is the one that can fail, so it carries the bridge's
            // verdict: a query the far world rejected, or a world that is not
            // running at all, has to be visible somewhere.
            if (query.kind == EntityQuery::Flecs && !query.focused
                && query.panel.is_valid() && query.panel.is_alive()) {
                const EntitySet* set = query.panel.try_get<EntitySet>();
                if (set && set->status == "OFFLINE") {
                    text->text = query.draft + "   [world offline]";
                    text->color = 0xB05A3CFF;
                } else if (set && set->status == "ERROR") {
                    text->text = query.draft + "   ["
                               + (set->error.empty() ? std::string("bad query") : set->error) + "]";
                    text->color = 0xB05A3CFF;
                } else if (set && set->loaded) {
                    // A name that matches nothing is reported alongside the
                    // count, since "0" alone reads as "none exist" when the
                    // truth is usually "that name does not exist".
                    std::string note = std::to_string(set->names.size());
                    if (!set->error.empty()) note += ", " + set->error;
                    text->text = query.draft + "   [" + note + "]";
                }
            }
        });

    // Loads the panel's rows from the headless world whenever the query it
    // should be showing has moved ahead of the one that produced them.
    world->system<EntitySet>("EntityQueryRequestSystem")
        .kind(flecs::OnUpdate)
        .each([](flecs::entity leaf, EntitySet& set) {
            request_entity_query(leaf, set);
        });

    // Issues the ranking request when the selection has moved ahead of the
    // ordering. Retrying every frame is what recovers from the server's LOADING
    // replies while the model is still warming up -- request_entity_ranking is
    // a no-op unless the socket is actually free.
    world->system<EntitySelection>("EntityRankingRequestSystem")
        .kind(flecs::OnUpdate)
        .each([](flecs::entity leaf, EntitySelection& selection) {
            request_entity_ranking(leaf, selection);
        });

    // 0. Yoga roots that fill a legacy-laid-out parent take their UISize from
    //    that parent's UIElementSize. The parent's size is finalised later in
    //    the frame (OnLoad/PostLoad), so this reads last frame's value -- one
    //    frame of lag on panel resize, same as the old Expand systems had.
    world->system<UIFillParent>("YogaFillParentSize")
        .kind(flecs::PreFrame)
        .each([](flecs::entity e, UIFillParent&) { yoga_sync_fill_parent(e); });

    // 1. Topology sync: ensure each Yoga node's parent + child order in the
    //    Yoga tree matches the flecs hierarchy. Entities without UIYoga are
    //    skipped, so they stay invisible to the Yoga tree even if their flecs
    //    parent is a Yoga node.
    world->system<YogaNode>("YogaTopologySync")
        .with<UIYoga>()
        .kind(flecs::PreFrame)
        .each([](flecs::entity e, YogaNode& yn) { yoga_sync_topology(e, yn); });

    // 2. Style sync: push UI* component values to the YGNode each frame.
    //    Cheap -- Yoga detects no-op writes and won't dirty the node.
    world->system<YogaNode>("YogaStyleSync")
        .with<UIYoga>()
        .kind(flecs::PreFrame)
        .each([](flecs::entity e, YogaNode& yn) { yoga_sync_style(e, yn); });

    // 3. Intrinsic sizing for leaves Yoga can't measure on its own: text leaves
    //    get a measure func, images publish their native aspect ratio.
    world->system<YogaNode>("YogaIntrinsicSize")
        .with<UIYoga>()
        .kind(flecs::PreFrame)
        .each([](flecs::entity e, YogaNode& yn) { yoga_apply_intrinsic(e, yn); });

    // 4. Calculate: for each Yoga *root* (a UIYoga entity whose flecs parent is
    //    not itself UIYoga), run layout. Roots with a UISize get a fixed size;
    //    roots without one auto-size to their content.
    world->system<YogaNode>("YogaCalculate")
        .with<UIYoga>()
        .kind(flecs::PreFrame)
        .each([](flecs::entity e, YogaNode& yn) {
            ZoneScoped;
            if (!yoga_is_root(e)) return;
            yoga_calculate(e, yn);
        });

    // 5. Readback: pull Yoga's computed layout into flecs.
    world->system<YogaNode>("YogaReadback")
        .with<UIYoga>()
        .kind(flecs::PreFrame)
        .each([](flecs::entity e, YogaNode& yn) { yoga_readback(e, yn); });

    // Phase 1c: Set UIElementSize from basic renderables
    auto sizeCalculationSystem = world->system<UIElementSize, Graphics>()
        .kind(flecs::PreFrame)
        .each([&](flecs::entity e, UIElementSize& size, Graphics& graphics) {

            if (e.has<RectRenderable>()) {
                auto rect = e.get<RectRenderable>();
                size.width = rect.width;
                size.height = rect.height;
            }
            else if (e.has<RoundedRectRenderable>()) {
                auto rect = e.get<RoundedRectRenderable>();
                if (!e.has<UIContainer>())
                {
                    size.width = rect.width;
                    size.height = rect.height;
                }
                //std::cout << "Setting size to" << size.width << std::endl;
            } else if (e.has<ImageRenderable>()) {
                auto img = e.get<ImageRenderable>();
                size.width = img.width;
                size.height = img.height;
            } else if (e.has<TextRenderable>()) {
                auto text = e.get<TextRenderable>();
                // Approximate text bounds
                nvgFontSize(graphics.vg, text.fontSize);
                nvgFontFace(graphics.vg, text.fontFace.c_str());
                float approxWidth = text.text.length() * text.fontSize * 0.6f;
                float approxHeight = text.fontSize;
                TextSize ts = measureText(graphics.vg, text.text, text.wrapWidth);
                size.width = ts.w;
                size.height = ts.h * text.scaleY;
            } else if (e.has<CustomRenderable>())
            {
                auto custom = e.get<CustomRenderable>();
                size.width = custom.width;
                size.height = custom.height;
            }
        });

    // Update Language Game chat UI each frame
    world->system<ChatPanel, EditorNodeArea>()
        .kind(flecs::PreFrame)
        .each([&](flecs::entity leaf, ChatPanel& panel, EditorNodeArea&)
        {
            if (!panel.messages_panel.is_alive() || !panel.input_panel.is_alive() ||
                !panel.input_text.is_alive())
            {
                leaf.remove<ChatPanel>();
                return;
            }
            // Panel geometry is Yoga's job now (chat_root is a UIFillParent
            // root, the history panel flex-grows, the input bar is fixed
            // height) -- this system only keeps the draft text in sync.
            ChatState& chat = world->ensure<ChatState>();
            std::string caret = chat.input_focused ? "|" : "";
            if (auto* input_tr = panel.input_text.try_get_mut<TextRenderable>())
            {
                input_tr->text = chat.draft + caret;
                // Wrap within the input panel, allowing for its padding.
                if (const UIElementSize* input_size = panel.input_panel.try_get<UIElementSize>())
                    input_tr->wrapWidth = std::max(0.0f, input_size->width - 16.0f);
            }

            const int kMaxMessages = 30;
            int total = (int)chat.messages.size();
            int start = std::max(0, total - kMaxMessages);

            flecs::query msg_views = world->query_builder<ChatMessageView, TextRenderable, Position>()
                .term_at(0).src(panel.messages_panel)
                .build();

            // msg_views.each([&](flecs::entity, ChatMessageView& view, TextRenderable& tr, Position& pos)
            // {
            //     int msg_index = start + view.index;
            //     if (msg_index < total)
            //     {
            //         const auto& msg = chat.messages[msg_index];
            //         tr.text = msg.author + ": " + msg.text;
            //         tr.wrapWidth = messages_rect.width - 12.0f;  // Wrap messages within panel
            //         pos.x = 6.0f;
            //         pos.y = 6.0f + view.index * 18.0f;
            //     }
            //     else
            //     {
            //         tr.text.clear();
            //     }
            // });
        });

    world->system<const UIElementBounds, LayoutBox, FitChildren, TimeEventRowChannel*, Graphics>()
    .kind(flecs::PreUpdate)
    .term_at(0).parent()
    .term_at(3).optional()
    .with<FitChildren>()
    .each([](flecs::entity e, const UIElementBounds& container_bounds, LayoutBox& box, FitChildren& fit, TimeEventRowChannel* timeEvent, Graphics& graphics) {
        float container_w = container_bounds.xmax - container_bounds.xmin;
        float container_h = container_bounds.ymax - container_bounds.ymin;
        
        float total_intrinsic_w = 0;
        int child_count = 0;

        // Pass 1: Sum up current natural widths
        e.children([&](flecs::entity child) {
            const ImageRenderable* img = child.try_get<ImageRenderable>();
            if (img) 
            {
                int w, h;
                nvgImageSize(graphics.vg, img->imageHandle, &w, &h);
                total_intrinsic_w += w;
                child_count++;
            }
        });

        if (child_count == 0 || total_intrinsic_w <= 0) return;

        if (timeEvent)
        {
            total_intrinsic_w *= timeEvent->scaleForMinimumCount/child_count;
            child_count = std::max(child_count, timeEvent->scaleForMinimumCount);
        }

        // Pass 2: Calculate Scale Factors
        float total_padding = box.padding * (child_count - 1);
        float available_w = container_w - total_padding;
        
        // This is our ideal horizontal scale factor
        fit.scale_factor = available_w / total_intrinsic_w;

        // Pass 3: Distribute Constraints with Y-Limit
        // e.children([&](flecs::entity child) {
        //     if (child.has<ImageRenderable>()) {
        //         ImageRenderable img = child.ensure<ImageRenderable>();
        //         int w, h;
        //         nvgImageSize(graphics.vg, img.imageHandle, &w, &h);
        //         float aspect = w / h;
        //         float target_w = w * fit.scale_factor;
        //         float target_h = target_w / aspect;

        //         // --- THE Y-OVERFLOW FIX ---
        //         // If the target width makes the book too tall, 
        //         // clamp based on container height instead.
        //         if (target_h > container_h) {
        //             target_h = container_h;
        //             target_w = target_h * aspect;
        //         }
        //         img.width = target_w;
        //         img.height = target_h;

        //         child.set<ProportionalConstraint>({ target_w, target_h });
        //     }
        // });
    });

    world->system<const UIElementBounds, LayoutBox, FitChildren, ImageRenderable, Graphics>()
    .kind(flecs::PreUpdate)
    .term_at(0).parent()
    .term_at(1).parent()
    .term_at(2).parent()
    .immediate()
    .each([](flecs::entity e, const UIElementBounds& container_bounds, LayoutBox& box, FitChildren& fit, ImageRenderable& img, Graphics& graphics)
    {
        float container_w = container_bounds.xmax - container_bounds.xmin;
        float container_h = container_bounds.ymax - container_bounds.ymin;

        int w, h;
        nvgImageSize(graphics.vg, img.imageHandle, &w, &h);

        // 1. Fix: Cast to float to avoid integer division
        float aspect = (float)w / (float)h; 

        // 2. Calculate initial dimensions
        float target_w = (float)w * fit.scale_factor;
        float target_h = (float)h * fit.scale_factor; // Direct calculation is more robust

        // --- THE Y-OVERFLOW FIX ---
        if (target_h > container_h) {
            target_h = container_h;
            target_w = target_h * aspect; // Aspect is now a correct float
        }

        img.width = target_w;
        img.height = target_h;

        e.set<ProportionalConstraint>({ target_w, target_h });
    });

    // ========================================================================
    // PHASE 2: Layout - Process LayoutBox and UIContainer bottom-up
    // Uses recursive function to ensure proper ordering:
    // children sizes computed before parents position them
    // ========================================================================
    world->system<EditorRoot>("LayoutSystem")
        .kind(flecs::PostLoad)
        .each([](flecs::entity root, EditorRoot&) {
            // Process entire UI hierarchy from editor root
            process_layout_recursive(root);
        });


    world->system<FlowLayoutBox, UIElementSize, const UIElementBounds*, UIElementBounds*>("ResetFlowProgress")
        .kind(flecs::PreUpdate)
        .term_at(2).parent()
        .term_at(3).optional()
        .each([](flecs::entity e, FlowLayoutBox& box, UIElementSize& container_size, const UIElementBounds* parent_bounds, UIElementBounds* own_bounds)
        {
            if (!parent_bounds) return;

            box.x_progress = 0.0f;
            box.y_progress = 0.0f;
            box.line_height = 0.0f;

            // Determine the available width for wrapping
            // Use parent bounds as the constraint for wrapping
            float container_width = parent_bounds->xmax - parent_bounds->xmin;
            float max_width = 0.0f;

            // First pass: collect children info and determine line breaks
            struct ChildInfo {
                flecs::entity entity;
                float width;
                float height;
            };
            std::vector<std::vector<ChildInfo>> lines;
            std::vector<ChildInfo> current_line;
            float current_line_width = 0.0f;
            float current_line_height = 0.0f;

            e.children([&](flecs::entity child) {
                const UIElementSize* child_size = child.try_get<UIElementSize>();
                if (!child_size) return;

                float child_width = child_size->width;
                float child_height = child_size->height;

                // Check if adding this child would exceed container width
                float needed_width = current_line_width + child_width;
                if (!current_line.empty()) {
                    needed_width += box.padding; // Add padding between items
                }

                // If this item doesn't fit and we have items on the line, start a new line
                if (!current_line.empty() && needed_width > container_width) {
                    lines.push_back(current_line);
                    current_line.clear();
                    current_line_width = 0.0f;
                    current_line_height = 0.0f;
                }

                // Add child to current line
                current_line.push_back({child, child_width, child_height});
                if (!current_line.empty() && current_line.size() > 1) {
                    current_line_width += box.padding;
                }
                current_line_width += child_width;
                current_line_height = std::max(current_line_height, child_height);
            });

            // Don't forget the last line
            if (!current_line.empty()) {
                lines.push_back(current_line);
            }

            // Second pass: position children with vertical centering
            box.y_progress = 0.0f;
            for (const auto& line : lines) {
                // Find max height for this line
                float line_height = 0.0f;
                for (const auto& child_info : line) {
                    line_height = std::max(line_height, child_info.height);
                }

                // Position children on this line with vertical centering
                box.x_progress = 0.0f;
                for (const auto& child_info : line) {
                    Position& pos = child_info.entity.ensure<Position, Local>();
                    pos.x = box.x_progress;

                    // Vertically center the child within the line
                    float y_offset = (line_height - child_info.height) * 0.5f;
                    pos.y = box.y_progress + y_offset;

                    // Propagate world positions to child and all descendants
                    propagate_world_positions(child_info.entity);

                    box.x_progress += child_info.width + box.padding;
                }

                max_width = std::max(max_width, box.x_progress - box.padding);
                box.y_progress += line_height + box.line_spacing;
            }

            // Update container size
            const Expand* expand = e.try_get<Expand>();
            if (!expand || !expand->x_enabled) {
                container_size.width = max_width;
            }
            if (!expand || !expand->y_enabled) {
                // Remove the last line spacing
                float total_height = box.y_progress;
                if (!lines.empty()) {
                    total_height -= box.line_spacing;
                }
                container_size.height = total_height;
            }

            // Immediately recalculate bounds if available
            if (own_bounds) {
                const Position& world_pos = e.get<Position, World>();
                own_bounds->xmin = world_pos.x;
                own_bounds->ymin = world_pos.y;
                if (!expand || !expand->x_enabled) {
                    own_bounds->xmax = world_pos.x + container_size.width;
                }
                if (!expand || !expand->y_enabled) {
                    own_bounds->ymax = world_pos.y + container_size.height;
                }
            }

            // Propagate wrapped height to ancestor vertical LayoutBoxes
            flecs::entity ancestor = e.parent();
            while (ancestor.is_valid()) {
                LayoutBox* layout = ancestor.try_get_mut<LayoutBox>();
                if (layout && layout->dir == LayoutBox::Vertical) {
                    // Re-layout children with updated sizes
                    float main_progress = 0.0f;
                    ancestor.children([&](flecs::entity child) {
                        const UIElementSize* child_size = child.try_get<UIElementSize>();
                        if (!child_size || child_size->width <= 0 || child_size->height <= 0) return;

                        Position& local_pos = child.ensure<Position, Local>();
                        float child_main = child_size->height;

                        if (layout->move_dir < 0) {
                            main_progress -= (child_main + layout->padding);
                        }

                        local_pos.y = main_progress;

                        if (layout->move_dir > 0) {
                            main_progress += child_main + layout->padding;
                        }
                    });
                }
                ancestor = ancestor.parent();
            }
        });

    auto cursorEvents = world->observer<CursorState, EditorRoot>()
        .event<LeftClickEvent>()
        .term_at(1).src(editor_root)
        .each([&UIElement](flecs::iter& it, size_t i, CursorState& cursor_state, EditorRoot& editor_root) {
            std::cout << "Left click at " << cursor_state.x << ", " << cursor_state.y << std::endl;

            bool in_modify_region = false;
            for (EditorModifyPartitionRegion& partition_region : editor_root.modify_partition_regions)
            {
                if (point_in_bounds((float)cursor_state.x, (float)cursor_state.y, partition_region.bounds))
                {
                    in_modify_region = true; 
                    split_editor({0.05, PanelSplitType::Horizontal}, partition_region.split_target, UIElement);
                    partition_region.split_target.add<Dragging>().add<DynamicPartition>();
                }
            }
            if (!in_modify_region)
            {
                for (EditorShiftRegion& shift_region : editor_root.shift_regions)
                {
                    // std::cout << "Check shift region" << std::endl;
                    if (point_in_bounds((float)cursor_state.x, (float)cursor_state.y, shift_region.bounds))
                    {
                        shift_region.split_target.add<Dragging>();
                    }
                }
            }
        });

    world->system<Position, EditorNodeArea, PanelSplit*, CursorState>()
        .term_at(0).second<World>()
        .term_at(2).optional()
        .term_at(3).src(glfwStateEntity)
        .with<Dragging>()
        .with<DynamicPartition>()
        .each([&UIElement](flecs::entity e, Position& world_pos, EditorNodeArea& node_area, PanelSplit* panel_split, CursorState& cursor_state)
        {
            // First, we need to determine if the dynamic partition hover is hovered over the same node area it was spawned in
            // If not, then we should merge it with its sibling
            if (!point_in_bounds(cursor_state.x, cursor_state.y, {world_pos.x, world_pos.y, world_pos.x + node_area.width, world_pos.y + node_area.height}))
            {
                if (!e.has<DynamicMerge>())
                {
                    e.add<DynamicMerge>();
                    merge_editor(e, UIElement);
                    // TODO: Temporary merger to allow reversal?
                    merge_editor(e.parent(), UIElement);
                }
            } else
            {
                if (e.has<DynamicMerge>())
                {
                    e.remove<DynamicMerge>();
                    split_editor({0.05, PanelSplitType::Horizontal}, e, UIElement);
                } else
                {
                // TODO: Consider the case of child panel splits of the same dimension as multiple siblings
                // 1. Define the boundaries of the node
                float left   = world_pos.x;
                float right  = world_pos.x + node_area.width;
                float top    = world_pos.y;
                float bottom = world_pos.y + node_area.height;

                // 2. Calculate distance to vertical edges (Left/Right)
                float dist_left  = std::abs(cursor_state.x - left);
                float dist_right = std::abs(cursor_state.x - right);
                float min_dist_x = std::min(dist_left, dist_right);

                // 3. Calculate distance to horizontal edges (Top/Bottom)
                float dist_top    = std::abs(cursor_state.y - top);
                float dist_bottom = std::abs(cursor_state.y - bottom);
                float min_dist_y  = std::min(dist_top, dist_bottom);

                // 4. Compare: Is the cursor physically closer to a side edge, or a top/bottom edge?
                if (min_dist_x < min_dist_y) {
                    // Closer to Left or Right edge -> We want to split vertically ( | )
                    if (panel_split->dim == PanelSplitType::Horizontal)
                    {
                        flecs::entity prev_left = e.target<LeftNode>();
                        flecs::entity prev_right = e.target<RightNode>();
                        e.remove<LeftNode>(prev_left);
                        e.remove<RightNode>(prev_right);
                        e.add<UpperNode>(prev_left);
                        e.add<LowerNode>(prev_right);
                        panel_split->dim = PanelSplitType::Vertical;
                    }
                } else {
                    // Closer to Top or Bottom edge -> We want to split horizontally ( - )
                    if (panel_split->dim == PanelSplitType::Vertical)
                    {
                        flecs::entity prev_upper = e.target<UpperNode>();
                        flecs::entity prev_lower = e.target<LowerNode>();
                        e.remove<UpperNode>(prev_upper);
                        e.remove<LowerNode>(prev_lower);
                        e.add<LeftNode>(prev_upper);
                        e.add<RightNode>(prev_lower);
                        panel_split->dim = PanelSplitType::Horizontal;
                    }
                }
            }
            }
        });

    auto query_dragging = world->query_builder<PanelSplit>()
    .with<Dragging>()
    .build();

    world->observer<CursorState, EditorRoot>()
        .event<LeftReleaseEvent>()
        .term_at(1).src(editor_root)
        .each([&query_dragging](flecs::entity e, CursorState& cursor_state, EditorRoot& editor_root) {
            std::cout << "Left mouse release" << std::endl;
            query_dragging.each([](flecs::iter& it, size_t row, PanelSplit& panel_split) {
                flecs::entity e_drag = it.entity(row);
                e_drag.remove<Dragging>();
                e_drag.remove<DynamicMerge>(); // Remove dynamics too if they exist
                e_drag.remove<DynamicPartition>();
                std::cout << "Remove dragging" << std::endl;
            });
        });

    auto bubbleUpBoundsQuery = world->query_builder<UIElementBounds, UIElementBounds*, RenderStatus*>()
        .term_at(1).parent().up()  // Parent UIElementBounds
        .term_at(2).optional()          // Optional RenderStatus
        .build();

    auto bubbleUpBoundsSystem = world->system<UIElementBounds, UIElementBounds*, UIElementSize, RenderStatus*>()
        .kind(flecs::PostLoad) 
        .term_at(1).parent().up()
        .term_at(2).optional()
        .each([&](flecs::entity e, UIElementBounds& bounds, UIElementBounds* parent_bounds, UIElementSize& size, RenderStatus* render) {
            if (parent_bounds && (!render || render->visible)) {
                
                const Expand* expand = e.try_get<Expand>();

                if (size.height == 0 || size.width == 0)
                {
                // Inherit bounds if the entity is 'non renderable'
                bounds.xmin = parent_bounds->xmin;
                bounds.xmax = parent_bounds->xmax;
                bounds.ymin = parent_bounds->ymin;
                bounds.ymax = parent_bounds->ymax;
                }

                flecs::entity parent = e.target(flecs::ChildOf);
                const Expand* parent_expand = parent.try_get<Expand>();
                
                bool parent_locked_x = parent_expand && parent_expand->x_enabled;
                bool parent_locked_y = parent_expand && parent_expand->y_enabled;

                // UIFillParent roots take their size *from* the parent, so
                // letting them bubble back into it would feed panel bounds into
                // their own sizing. This is the Yoga equivalent of the Expand
                // opt-out above. Auto-sizing Yoga entities (e.g. a popup list)
                // must still bubble, so containers without a renderable of
                // their own get usable bounds for hit-testing.
                if (!expand && !e.has<UIFillParent>()) {
                    // FlowLayoutBox only propagates height, not width (allows wrapping)
                    if (!e.has<FlowLayoutBox>()) {
                        parent_bounds->xmin = std::min(parent_bounds->xmin, bounds.xmin);
                        parent_bounds->xmax = std::max(parent_bounds->xmax, bounds.xmax);
                    }
                    parent_bounds->ymin = std::min(parent_bounds->ymin, bounds.ymin);
                    parent_bounds->ymax = std::max(parent_bounds->ymax, bounds.ymax);
                }
                }
        });

    // ========================================================================
    // PHASE 3: Bounds - Calculate bounds from world position + size
    // Then propagate containment from children to parents
    // ========================================================================

    auto bubbleUpBoundsSecondarySystem = world->system<UIElementBounds, UIElementBounds*, RenderStatus*>()
        .kind(flecs::PostLoad) 
        .term_at(1).parent().up()
        .term_at(2).optional()
        .each([&](flecs::entity e, UIElementBounds& bounds, UIElementBounds* parent_bounds, RenderStatus* render) {
            if (parent_bounds && (!render || render->visible)) {
                
                const Expand* expand = e.try_get<Expand>();

                if (bounds.xmax == 0)
                {
                    // Inherit bounds if non renderable
                    bounds.xmin = parent_bounds->xmin;
                    bounds.xmax = parent_bounds->xmax;
                    bounds.ymin = parent_bounds->ymin;
                    bounds.ymax = parent_bounds->ymax;
                }

                // See the primary bubble-up system: UIFillParent is sized by its
                // parent, so it must not size its parent back.
                if (!expand && !e.has<UIFillParent>()) {
                    // FlowLayoutBox only propagates height, not width (allows wrapping)
                    if (!e.has<FlowLayoutBox>()) {
                        parent_bounds->xmin = std::min(parent_bounds->xmin, bounds.xmin);
                        parent_bounds->xmax = std::max(parent_bounds->xmax, bounds.xmax);
                    }
                    parent_bounds->ymin = std::min(parent_bounds->ymin, bounds.ymin);
                    parent_bounds->ymax = std::max(parent_bounds->ymax, bounds.ymax);
                }
                }
        });

    world->system<Position, UIElementBounds*, UIElementSize, UIElementBounds, Align, Expand*>()
    .term_at(0).second<Local>()
    .term_at(1).parent()
    .term_at(5).optional()
    .kind(flecs::PreFrame)
    .each([&](flecs::entity e, Position& pos, UIElementBounds* parent_bounds, UIElementSize& ui_size, UIElementBounds& bounds, Align& align, Expand* expand)
    {
        const LayoutBox* parent_layout = e.parent().try_get<LayoutBox>();
        bool parent_controls_x = e.parent().has<FlowLayoutBox>() || (parent_layout && parent_layout->dir == LayoutBox::Horizontal);
        bool parent_controls_y = e.parent().has<FlowLayoutBox>() || (parent_layout && parent_layout->dir == LayoutBox::Vertical);

        if (!parent_controls_x)
        {
            pos.x = align.horizontal * (parent_bounds->xmax - parent_bounds->xmin) + ui_size.width * align.self_horizontal + (expand ? expand->pad_left : 0);
        }
        if (!parent_controls_y)
        {
            pos.y = align.vertical * (parent_bounds->ymax - parent_bounds->ymin) + ui_size.height * align.self_vertical;
        }
    });

    world->system<UIElementBounds*, RectRenderable, Expand>()
    .term_at(0).parent()
    .kind(flecs::PreUpdate)
    .immediate()
    .each([&](flecs::entity e, UIElementBounds* bounds, RectRenderable& rect, Expand& expand) {
        if (expand.x_enabled)
        {
            rect.width = (bounds->xmax - bounds->xmin)*expand.x_percent - (expand.pad_left + expand.pad_right);
        }
        if (expand.y_enabled)
        {
            rect.height = (bounds->ymax - bounds->ymin)*expand.y_percent- (expand.pad_top + expand.pad_bottom);
        }
    });

    world->system<UIElementBounds*, RoundedRectRenderable, Expand>()
    .term_at(0).parent()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, RoundedRectRenderable& rect, Expand& expand) {
        if (expand.x_enabled)
        {
            rect.width = (bounds->xmax - bounds->xmin - (expand.pad_left + expand.pad_right))*expand.x_percent;
        }
        if (expand.y_enabled)
        {
            rect.height = (bounds->ymax - bounds->ymin - (expand.pad_top + expand.pad_bottom))*expand.y_percent;
        }
    });

    world->system<UIElementBounds*, LineRenderable, Expand>()
    .term_at(0).parent()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, LineRenderable& line, Expand& expand) {
        if (expand.x_enabled)
        {
            line.x2 = (bounds->xmax - bounds->xmin - (expand.pad_left + expand.pad_right))*expand.x_percent;
        }
        if (expand.y_enabled)
        {
            line.y2 = (bounds->ymax - bounds->ymin - (expand.pad_top + expand.pad_bottom))*expand.y_percent;
        }
    });

    // What the fuck is this stupid fucking system?
    world->system<UIElementBounds*, ImageRenderable, Expand, Constrain*, Graphics>()
    .term_at(0).parent()
    .term_at(3).optional()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* parent_bounds, ImageRenderable& sprite, Expand& expand, Constrain* constrain, Graphics& graphics) {        
        if (!parent_bounds) return;

        int img_width, img_height;
        nvgImageSize(graphics.vg, sprite.imageHandle, &img_width, &img_height);
        if (img_width == 0 || img_height == 0) return;

        float aspect = (float)img_width / (float)img_height;
        float desired_w = sprite.width;
        float desired_h = sprite.height;

        // PRIORITY 1: Respect the Proportional Constraint if it exists
        const ProportionalConstraint* prop = e.try_get<ProportionalConstraint>();
        if (prop && prop->max_width > 0) {
            desired_w = prop->max_width;
            desired_h = prop->max_height; 
        } 
        else {
            // PRIORITY 2: Fallback to standard Expand/Constrain logic
            float avail_w = (parent_bounds->xmax - parent_bounds->xmin) - (expand.pad_left + expand.pad_right);
            float avail_h = (parent_bounds->ymax - parent_bounds->ymin) - (expand.pad_top + expand.pad_bottom);

            if (expand.x_enabled) {
                float target_w = avail_w;
                if (constrain && constrain->fit_y && (target_w / aspect) > avail_h) 
                    target_w = avail_h * aspect;
                
                desired_w = target_w * expand.x_percent;
                if (!expand.y_enabled) desired_h = desired_w / aspect;
            }
            
            if (expand.y_enabled) {
                float target_h = avail_h;
                if (!expand.x_enabled) {
                    if (constrain && constrain->fit_x && (target_h * aspect) > avail_w) 
                        target_h = avail_w / aspect;
                    desired_h = target_h * expand.y_percent;
                    desired_w = desired_h * aspect;
                } else {
                    if (constrain && (constrain->fit_x || constrain->fit_y)) {
                        float scale = std::min(avail_w / img_width, avail_h / img_height);
                        desired_w = img_width * scale * expand.x_percent;
                        desired_h = img_height * scale * expand.y_percent;
                    } else {
                        desired_h = avail_h * expand.y_percent;
                    }
                }
            }
        }

        // Apply global Intrinsic Caps
        if (expand.cap_to_intrinsic) {
            float max_w = img_width * sprite.scaleX;
            float max_h = img_height * sprite.scaleY;
            float cap = std::min({1.0f, max_w / desired_w, max_h / desired_h});
            desired_w *= cap;
            desired_h *= cap;
        }

        sprite.width = desired_w;
        sprite.height = desired_h;
    });

    // auto debugRenderBounds = world->system<RenderQueue, UIElementBounds, DebugRenderBounds>()
    // .term_at(0).src(renderQueueEntity)
    // .each([](flecs::entity e, RenderQueue& render_queue, UIElementBounds& bounds, DebugRenderBounds)
    // {
    //     RectRenderable debug_bound {bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin, true, 0xFFFF00FF};
    //     render_queue.addRectCommand({bounds.xmin, bounds.ymin}, debug_bound, 100);
    // });

    // Word annotation selector - updates bounds for selection highlight (entity creation happens in key handlers)
    auto wordAnnotationBoundsSystem = world->system<WordAnnotationSelector, Position, RectRenderable>()
    .term_at(1).second<World>()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, WordAnnotationSelector& selector, Position& pos, RectRenderable& rect) {
        if (!selector.active || selector.token_count == 0) {
            rect.width = 0;
            rect.height = 0;
            return;
        }

        // Clamp indices to valid token range
        int max_idx = selector.token_count - 1;
        if (selector.start_index < 0) selector.start_index = 0;
        if (selector.end_index < 0) selector.end_index = 0;
        if (selector.start_index > max_idx) selector.start_index = max_idx;
        if (selector.end_index > max_idx) selector.end_index = max_idx;
        if (selector.start_index > selector.end_index) {
            std::swap(selector.start_index, selector.end_index);
        }

        // Calculate combined bounds for all tokens in selection range
        float combined_xmin = FLT_MAX, combined_ymin = FLT_MAX;
        float combined_xmax = -FLT_MAX, combined_ymax = -FLT_MAX;
        bool has_valid_bounds = false;

        for (int i = selector.start_index; i <= selector.end_index; i++) {
            if (i >= (int)selector.selection_entities.size()) continue;

            flecs::entity ent = selector.selection_entities[i];
            if (!ent.is_valid()) continue;

            // Try UIElementBounds first (skip if bounds not yet computed)
            const UIElementBounds* bounds = ent.try_get<UIElementBounds>();
            if (bounds && (bounds->xmax > bounds->xmin) && (bounds->ymax > bounds->ymin)) {
                combined_xmin = std::min(combined_xmin, bounds->xmin);
                combined_ymin = std::min(combined_ymin, bounds->ymin);
                combined_xmax = std::max(combined_xmax, bounds->xmax);
                combined_ymax = std::max(combined_ymax, bounds->ymax);
                has_valid_bounds = true;
            } else if (!bounds) {
                // Fall back to Position<World> + size from renderable
                const Position* world_pos = ent.try_get<Position, World>();
                if (world_pos) {
                    float w = 0, h = 0;
                    const ImageRenderable* img = ent.try_get<ImageRenderable>();
                    if (img) {
                        w = img->width;
                        h = img->height;
                    }
                    const TextRenderable* txt = ent.try_get<TextRenderable>();
                    if (txt) {
                        // Estimate text size (rough approximation)
                        w = txt->text.length() * txt->fontSize * 0.6f;
                        h = txt->fontSize;
                    }
                    if (w > 0 && h > 0) {
                        combined_xmin = std::min(combined_xmin, world_pos->x);
                        combined_ymin = std::min(combined_ymin, world_pos->y);
                        combined_xmax = std::max(combined_xmax, world_pos->x + w);
                        combined_ymax = std::max(combined_ymax, world_pos->y + h);
                        has_valid_bounds = true;
                    }
                }
            }
        }

        if (has_valid_bounds) {
            // Set world position directly (annotation selector has no parent)
            // Add 1 pixel padding on all sides
            pos.x = combined_xmin - 1.0f;
            pos.y = combined_ymin - 1.0f;
            rect.width = combined_xmax - combined_xmin + 2.0f;
            rect.height = combined_ymax - combined_ymin + 2.0f;
            rect.stroke = true;
            rect.color = 0xFFFFFFFF;
        }
    });

    auto roundedRectQueueSystem = world->system<Position, RoundedRectRenderable, ZIndex, RenderGradient*>()
    .term_at(0).second<World>()
    .term_at(3).optional()
    .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, Position& pos, RoundedRectRenderable& renderable, ZIndex& zIndex, RenderGradient* rg) {
            RenderQueue& queue = world->ensure<RenderQueue>();
            queue.addRoundedRectCommand(pos, renderable, zIndex.layer, rg, rg ? *rg : RenderGradient{0, 0}, resolve_scissor(e), ui_hierarchy_depth(e));
        });


    auto rectQueueSystem = world->system<Position, RectRenderable, ZIndex, RenderStatus>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
        .each([&](flecs::entity e, Position& pos, RectRenderable& renderable, ZIndex& zIndex, RenderStatus& status) {
            if (status.visible)
            {
                RenderQueue& queue = world->ensure<RenderQueue>();
                queue.commands.push_back({pos, renderable, RenderType::Rectangle, zIndex.layer, resolve_scissor(e), false, {0, 0}, ui_hierarchy_depth(e)});
            }
        });

    world->system<TextRenderable, DynamicTextWrap>()
    .kind(flecs::PreUpdate)
    .with<DynamicTextWrapContainer>(flecs::Wildcard)
    .each([&](flecs::entity e, TextRenderable& text, DynamicTextWrap& data)
    {
        UIElementSize& uiElementSize = e.target<DynamicTextWrapContainer>().ensure<UIElementSize>();
        text.wrapWidth = uiElementSize.width - data.pad;
    });

    auto textQueueSystem = world->system<Position, TextRenderable, ZIndex, RenderGradient*>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .term_at(3).optional()
    .each([&](flecs::entity e, Position& pos, TextRenderable& renderable, ZIndex& zIndex, RenderGradient* rg) {
        RenderQueue& queue = world->ensure<RenderQueue>();
        queue.addTextCommand(pos, renderable, zIndex.layer, rg, rg ? *rg : RenderGradient{0, 0}, resolve_scissor(e), ui_hierarchy_depth(e));
    });

    auto imageQueueSystem = world->system<Position, ImageRenderable, ZIndex, RenderStatus>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, ImageRenderable& renderable, ZIndex& zIndex, RenderStatus& status) {
        RenderQueue& queue = world->ensure<RenderQueue>();
        if (status.visible)
        {
            queue.commands.push_back({pos, renderable, RenderType::Image, zIndex.layer, resolve_scissor(e), false, {0, 0}, ui_hierarchy_depth(e)});
        }
    });

    auto lineQueueSystem = world->system<Position, LineRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, LineRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world->ensure<RenderQueue>();
        queue.addLineCommand(pos, renderable, zIndex.layer);
    });

    auto quadraticBezierQueueSystem = world->system<Position, QuadraticBezierRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, QuadraticBezierRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world->ensure<RenderQueue>();
        queue.addQuadraticBezierCommand(pos, renderable, zIndex.layer);
    });

    auto customQueueSystem = world->system<Position, CustomRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, CustomRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world->ensure<RenderQueue>();
        queue.commands.push_back({pos, renderable, RenderType::CustomRenderable, zIndex.layer, resolve_scissor(e), false, {0, 0}, ui_hierarchy_depth(e)});
    });

    world->system<Position, EditorNodeArea, EditorLeafData, EditorRoot>()
    .term_at(0).second<World>()
    .term_at(3).src(editor_root)
    .run([](flecs::iter& it)
    {
        auto editor_root = world->lookup("editor_root").try_get_mut<EditorRoot>();
        editor_root->modify_partition_regions.clear();
        while (it.next()) {
            it.each();
        }
    }, [](flecs::entity e, Position& world_pos, EditorNodeArea& node_area, EditorLeafData& leaf_data, EditorRoot& editor_root) 
    {
        editor_root.modify_partition_regions.push_back({{world_pos.x, world_pos.y, world_pos.x + 8.0f, world_pos.y + 24.0f}, e});
    });

    int scale_region_dist = 8;
    
    world->system<Window, CursorState, EditorNodeArea, PanelSplit, Position, EditorRoot>()
    .term_at(0).src(glfwStateEntity)
    .term_at(1).src(glfwStateEntity)
    .term_at(4).second<World>()
    .term_at(5).src(editor_root)
    .run([scale_region_dist](flecs::iter& it)
    {
        auto editor_root = world->lookup("editor_root").try_get_mut<EditorRoot>();
        editor_root->shift_regions.clear();
        while (it.next()) {
            it.each();
        }
        
        auto window = it.field<Window>(0);
        auto cursor_state = it.field<CursorState>(1);

        // Hide cursor while in grid mode (3D triangle animation)
        const Graphics* graphics = world->lookup("Graphics").try_get<Graphics>();
        if (graphics && graphics->useGridMode) {
            glfwSetInputMode(window->handle, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            return;
        }
        glfwSetInputMode(window->handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        glfwSetCursor(window->handle, NULL);

        bool in_modify_region = false;
        for (EditorModifyPartitionRegion& partition_region : editor_root->modify_partition_regions)
        {
            if (point_in_bounds((float)cursor_state->x, (float)cursor_state->y, partition_region.bounds))
            {
                GLFWcursor* cursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
                glfwSetCursor(window->handle, cursor);
                in_modify_region = true;
            }
        }
        if (!in_modify_region)
        {
            for (EditorShiftRegion& shift_region : editor_root->shift_regions)
            {
                if (point_in_bounds((float)cursor_state->x, (float)cursor_state->y, shift_region.bounds))
                {
                    GLFWcursor* cursor = glfwCreateStandardCursor(shift_region.cursor_type);
                    glfwSetCursor(window->handle, cursor);
                }
            }
        }
        
        // TODO: If the cursor location is within one of those rectangles, then create a tag indicating the scale target node
        // and change the curorsor type
        
        // GLFWcursor* cursor = glfwCreateStandardCursor(GLFW_RESIZE_EW_CURSOR);
        // GLFWcursor* cursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
        // glfwSetCursor(window, cursor);
    },
    [scale_region_dist](flecs::entity e, Window& window, CursorState& cursor_state, EditorNodeArea& node_area, PanelSplit& panel_split, Position& world_pos, EditorRoot& editor_root) {
        UIElementBounds bounds;
        if (panel_split.dim == PanelSplitType::Horizontal)
        {
            float line_xpos = world_pos.x + panel_split.percent * node_area.width;
            bounds.xmin = line_xpos - scale_region_dist;
            bounds.ymin = world_pos.y;
            bounds.xmax = line_xpos + scale_region_dist;
            bounds.ymax = world_pos.y + node_area.height;
            editor_root.shift_regions.push_back({bounds, GLFW_RESIZE_EW_CURSOR, e});
        } else if (panel_split.dim == PanelSplitType::Vertical)
        {
            float line_ypos = world_pos.y + panel_split.percent * node_area.height;
            bounds.xmin = world_pos.x;
            bounds.ymin = line_ypos - scale_region_dist;
            bounds.xmax = world_pos.x + node_area.width;
            bounds.ymax = line_ypos + scale_region_dist;
            editor_root.shift_regions.push_back({bounds, GLFW_RESIZE_NS_CURSOR, e});
        }
    });

    // Particle animation system - updates triangle positions for spawn animation
    auto particleAnimationSystem = world->system<Graphics>()
        .kind(flecs::PostUpdate)
        .each([](flecs::iter& it, size_t i, Graphics& graphics) {
            float deltaTime = it.delta_time();

            // Update grid particles
            if (!graphics.particles.empty()) {
                updateParticles(graphics, deltaTime);
                uploadParticleVertices(graphics);

                // Check if all triangles are locked - switch to plane mode after delay
                if (graphics.useGridMode) {
                    bool allLocked = true;
                    for (const auto& p : graphics.particles) {
                        if (!p.locked) {
                            allLocked = false;
                            break;
                        }
                    }
                    if (allLocked) {
                        if (!graphics.allParticlesLocked) {
                            graphics.allParticlesLocked = true;
                            graphics.gridModeTransitionTimer = 0.0f;
                        }
                        graphics.gridModeTransitionTimer += deltaTime;
                        if (graphics.gridModeTransitionTimer >= 1.0f) {
                            graphics.useGridMode = false;
                        }
                    }
                }
            }

            // Update noise tetrahedrons (fly past continuously)
            if (!graphics.noiseParticles.empty()) {
                updateNoiseTetrahedrons(graphics, deltaTime);
                generateNoiseVertices(graphics);
                uploadNoiseVertices(graphics);
            }
        });

    auto renderExecutionSystem = world->system<RenderQueue, Graphics>()
        .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, RenderQueue& queue, Graphics& graphics) {
            queue.sort();
            for (const auto& cmd : queue.commands) {
                if (cmd.scissorEntity) {
                    float cx, cy, cw, ch;
                    if (scissor_rect(world->entity(cmd.scissorEntity), cx, cy, cw, ch)) {
                        nvgScissor(graphics.vg, cx, cy, cw, ch);
                    }
                }
                switch (cmd.type) {
                    case RenderType::CustomRenderable: {
                        const auto& custom = std::get<CustomRenderable>(cmd.renderData);
                        custom.render_function(graphics.vg, &cmd, custom);
                        break;
                    }
                    case RenderType::RoundedRectangle: {
                        const auto& rect = std::get<RoundedRectRenderable>(cmd.renderData);
                        nvgBeginPath(graphics.vg);
                        nvgRoundedRect(graphics.vg, cmd.pos.x, cmd.pos.y, rect.width, rect.height, rect.radius);

                        uint8_t r = (rect.color >> 24) & 0xFF;
                        uint8_t g = (rect.color >> 16) & 0xFF;
                        uint8_t b = (rect.color >> 8) & 0xFF;
                        uint8_t a = (rect.color >> 0) & 0xFF;
                        
                        if (rect.stroke)
                        {
                            nvgStrokeWidth(graphics.vg, 1.0f);
                            if (cmd.useGradient)
                            {
                                NVGpaint gradient = nvgLinearGradient(graphics.vg, cmd.pos.x, cmd.pos.y, cmd.pos.x, cmd.pos.y + rect.height, uintToNvgColor(cmd.gradient.start), uintToNvgColor(cmd.gradient.end));
                                nvgStrokePaint(graphics.vg, gradient);
                            } else
                            {
                                nvgStrokeColor(graphics.vg, nvgRGBA(r, g, b, a));
                            }
                            nvgStroke(graphics.vg);
                        } else
                        {
                            if (cmd.useGradient)
                            {
                                NVGpaint gradient = nvgLinearGradient(graphics.vg, cmd.pos.x, cmd.pos.y, cmd.pos.x, cmd.pos.y + rect.height, uintToNvgColor(cmd.gradient.start), uintToNvgColor(cmd.gradient.end));
                                nvgFillPaint(graphics.vg, gradient);
                            } else
                            {
                                nvgFillColor(graphics.vg, nvgRGBA(r, g, b, a));
                            }
                            nvgFill(graphics.vg);
                        }
                        break;
                    }
                    case RenderType::Rectangle: {
                        const auto& rect = std::get<RectRenderable>(cmd.renderData);
                        nvgBeginPath(graphics.vg);
                        nvgRect(graphics.vg, cmd.pos.x, cmd.pos.y, rect.width, rect.height);

                        uint8_t r = (rect.color >> 24) & 0xFF;
                        uint8_t g = (rect.color >> 16) & 0xFF;
                        uint8_t b = (rect.color >> 8) & 0xFF;
                        uint8_t a = (rect.color >> 0) & 0xFF;

                        if (rect.stroke)
                        {
                            nvgStrokeWidth(graphics.vg, 1.0f);
                            nvgStrokeColor(graphics.vg, nvgRGBA(r, g, b, a));
                            nvgStroke(graphics.vg);
                        } else
                        {
                            nvgFillColor(graphics.vg, nvgRGBA(r, g, b, a));
                            nvgFill(graphics.vg);
                        }
                        break;
                    }
                    case RenderType::Text: {
                        const auto& text = std::get<TextRenderable>(cmd.renderData);

                        nvgSave(graphics.vg);
                        nvgTranslate(graphics.vg, cmd.pos.x, cmd.pos.y);
                        nvgScale(graphics.vg, 1.0f, text.scaleY);

                        // it's not easy to render text with a gradient color
                        nvgFontSize(graphics.vg, text.fontSize);
                        nvgFontFace(graphics.vg, text.fontFace.c_str());
                        nvgTextAlign(graphics.vg, NVG_ALIGN_TOP | NVG_ALIGN_LEFT);

                        uint8_t r = (text.color >> 24) & 0xFF;
                        uint8_t g = (text.color >> 16) & 0xFF;
                        uint8_t b = (text.color >> 8) & 0xFF;

                        // NVGpaint gradient = nvgLinearGradient(graphics.vg, cmd.pos.x, cmd.pos.y, cmd.pos.x, cmd.pos.y + rect.height, cmd.gradient.start, cmd.gradient.end);
                        
                        if (cmd.useGradient)
                        {
                            NVGpaint gradient = nvgLinearGradient(graphics.vg, 0, 0, 0, 0 + text.fontSize, uintToNvgColor(cmd.gradient.start), uintToNvgColor(cmd.gradient.end));
                            nvgFillPaint(graphics.vg, gradient);
                        } else
                        {
                            nvgFillColor(graphics.vg, nvgRGB(r, g, b));
                        }

                        if (text.wrapWidth > 0.0f)
                        {
                            nvgTextBox(graphics.vg, 0, 0, text.wrapWidth, text.text.c_str(), nullptr);
                        }
                        else
                        {
                            nvgText(graphics.vg, 0, 0, text.text.c_str(), nullptr);
                        }
                        nvgRestore(graphics.vg);
                        break;
                    }
                    case RenderType::Image: {
                        const auto& image = std::get<ImageRenderable>(cmd.renderData);
                        if (image.imageHandle != -1) {
                            // Apply texture offset to pattern position (shifts UV sampling)
                            // Rect stays at cmd.pos, pattern is offset to shift texture content
                            NVGpaint imgPaint = nvgImagePattern(graphics.vg,
                                cmd.pos.x - image.texOffsetX,
                                cmd.pos.y - image.texOffsetY,
                                image.width, image.height, 0.0f,
                                image.imageHandle, 1.0);
                            nvgBeginPath(graphics.vg);
                            nvgRect(graphics.vg, cmd.pos.x, cmd.pos.y, image.width, image.height);
                            imgPaint.innerColor = imgPaint.outerColor = image.tint;
                            nvgFillPaint(graphics.vg, imgPaint);
                            nvgFill(graphics.vg);
                        }
                        break;
                    }
                    case RenderType::Line: {
                        const auto& line = std::get<LineRenderable>(cmd.renderData);    
                        nvgBeginPath(graphics.vg);
                        nvgMoveTo(graphics.vg, cmd.pos.x + line.x1, cmd.pos.y + line.y1);
                        // The parameters are: ctrl_x, ctrl_y, end_x, end_y
                        nvgLineTo(graphics.vg, cmd.pos.x + line.x2, cmd.pos.y + line.y2);
                        uint8_t r = (line.color >> 24) & 0xFF;
                        uint8_t g = (line.color >> 16) & 0xFF;
                        uint8_t b = (line.color >> 8) & 0xFF;

                        nvgStrokeColor(graphics.vg, nvgRGB(r, g, b));
                        nvgStrokeWidth(graphics.vg, line.thickness);
                        nvgStroke(graphics.vg);
                        break;
                    }
                    case RenderType::QuadraticBezier:
                        const auto& curve = std::get<QuadraticBezierRenderable>(cmd.renderData);    
                        nvgBeginPath(graphics.vg);
                        nvgMoveTo(graphics.vg, cmd.pos.x + curve.x1, cmd.pos.y + curve.y1);
                        // The parameters are: ctrl_x, ctrl_y, end_x, end_y
                        nvgQuadTo(graphics.vg, cmd.pos.x + curve.cx, cmd.pos.y + curve.cy, cmd.pos.x + curve.x2, cmd.pos.y + curve.y2);
                        uint8_t r = (curve.color >> 24) & 0xFF;
                        uint8_t g = (curve.color >> 16) & 0xFF;
                        uint8_t b = (curve.color >> 8) & 0xFF;

                        nvgStrokeColor(graphics.vg, nvgRGB(r, g, b));
                        nvgStrokeWidth(graphics.vg, curve.thickness);
                        nvgStroke(graphics.vg);
                        break;
                }
                nvgResetScissor(graphics.vg);
            }

            queue.clear();
        });

    // LineChart rendering system - renders filled polygon for streaming data visualization
    // Like mel spec: right edge is "now", data flows continuously from right to left
    // No discrete frame snapping - points have no width unlike filmstrip frames
    auto lineChartRenderSystem = world->system<LineChartData, UIElementBounds>("LineChartRenderSystem")
        .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, LineChartData& chart, UIElementBounds& bounds) {
            if (chart.capacity == 0) return;

            Graphics& graphics = world->ensure<Graphics>();
            if (!graphics.vg) return;

            // Get parent bounds for full width
            flecs::entity parent = e.parent();
            if (!parent.is_valid()) return;
            const UIElementBounds* parent_bounds = parent.try_get<UIElementBounds>();
            if (!parent_bounds) return;

            float parent_width = parent_bounds->xmax - parent_bounds->xmin;

            // Apply scissor if entity has ScissorContainer
            if (e.has<ScissorContainer>(flecs::Wildcard)) {
                flecs::entity scissorEntity = e.target<ScissorContainer>();
                if (scissorEntity.is_valid()) {
                    const UIElementBounds* scissorBounds = scissorEntity.try_get<UIElementBounds>();
                    if (scissorBounds) {
                        nvgScissor(graphics.vg, scissorBounds->xmin, scissorBounds->ymin,
                                   scissorBounds->xmax - scissorBounds->xmin,
                                   scissorBounds->ymax - scissorBounds->ymin);
                    }
                }
            }

            // Use parent width for chart, bounds height for vertical
            float width = parent_width;
            float height = bounds.ymax - bounds.ymin;

            // Time-based positioning using mel spec's actual timeline
            float base_x = parent_bounds->xmin;
            float y_bottom = bounds.ymax;

            // Use mel spec's timeline so line chart moves at exactly mel spec's rate
            double melSpecNow = glfwGetTime();  // Default to wall clock
            auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
            if (sysAudioRenderer.is_valid()) {
                const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
                if (melSpec && melSpec->fillProgress >= 1.0f && melSpec->scrollStartTime > 0) {
                    // Mel spec's elapsed time based on actual columns received
                    double melSpecElapsed = (double)melSpec->totalScrollCommands / MelSpecRender::COLUMNS_PER_SECOND;
                    melSpecNow = melSpec->scrollStartTime + melSpecElapsed;
                }
            }

            size_t num_points = chart.size();

            // If we have data, draw it
            if (num_points >= 1) {
                // Calculate x position for each point based on its capture time
                // Uses mel spec's timeline so everything moves at the same rate
                auto calcX = [&](size_t i) -> float {
                    LineChartPoint point = chart.get(i);
                    double age = melSpecNow - point.capture_time;
                    // Normalize: 1.0 = now (right), 0.0 = WINDOW_DURATION ago (left)
                    float normalizedTime = 1.0f - (float)(age / LineChartData::WINDOW_DURATION);
                    return base_x + normalizedTime * width;
                };

                auto calcY = [&](size_t i) -> float {
                    LineChartPoint point = chart.get(i);
                    float normalized = (point.value - chart.min_value) / (chart.max_value - chart.min_value);
                    normalized = std::max(0.0f, std::min(1.0f, normalized));
                    return y_bottom - normalized * height;
                };

                // Build polygon path: start at bottom of first point, trace line, close at bottom
                nvgBeginPath(graphics.vg);

                float first_x = calcX(0);
                nvgMoveTo(graphics.vg, first_x, y_bottom);

                // Draw line through all data points (from oldest to newest)
                for (size_t i = 0; i < num_points; i++) {
                    nvgLineTo(graphics.vg, calcX(i), calcY(i));
                }

                // Close polygon: line to bottom at last point, then back to start
                float last_x = calcX(num_points - 1);
                nvgLineTo(graphics.vg, last_x, y_bottom);
                nvgClosePath(graphics.vg);

                // Fill with semi-transparent color
                uint8_t fr = (chart.fill_color >> 24) & 0xFF;
                uint8_t fg = (chart.fill_color >> 16) & 0xFF;
                uint8_t fb = (chart.fill_color >> 8) & 0xFF;
                uint8_t fa = (chart.fill_color) & 0xFF;
                nvgFillColor(graphics.vg, nvgRGBA(fr, fg, fb, fa));
                nvgFill(graphics.vg);

                // Draw line on top
                nvgBeginPath(graphics.vg);
                for (size_t i = 0; i < num_points; i++) {
                    if (i == 0) {
                        nvgMoveTo(graphics.vg, calcX(i), calcY(i));
                    } else {
                        nvgLineTo(graphics.vg, calcX(i), calcY(i));
                    }
                }

                uint8_t lr = (chart.line_color >> 24) & 0xFF;
                uint8_t lg = (chart.line_color >> 16) & 0xFF;
                uint8_t lb = (chart.line_color >> 8) & 0xFF;
                uint8_t la = (chart.line_color) & 0xFF;
                nvgStrokeColor(graphics.vg, nvgRGBA(lr, lg, lb, la));
                nvgStrokeWidth(graphics.vg, 1.5f);
                nvgStroke(graphics.vg);
            }

            nvgResetScissor(graphics.vg);
        });

    auto vncInitSystem = world->system<VNCClientHandle>()
        .kind(flecs::PreUpdate)
        .each([](flecs::iter& it, size_t i, VNCClientHandle& handle) {
            VNCClient& vnc = *handle;
            if (!vnc.initialized && vnc.connected && vnc.client) {
                SendFramebufferUpdateRequest(vnc.client, 0, 0, vnc.client->width, vnc.client->height, FALSE);
                vnc.initialized = true;  // Mark as initialized to avoid repeated requests
                LOG_INFO(LogCategory::VNC_CLIENT, "Sent initial framebuffer update request for {}", vnc.toString());
            }
        });

    // VNC mouse tracking system - sends mouse position to active VNC client in interactive mode
    // Note: Message processing now handled by vnc_message_thread() on network thread
    auto vncMouseTrackingSystem = world->system<Position, ImageRenderable>()
        .with<IsStreamingFrom>(flecs::Wildcard)
        .term_at(0).second<World>()
        .kind(flecs::PreUpdate)
        .each([&](flecs::entity e, Position& pos, ImageRenderable& img) {
            auto* handle = e.target<IsStreamingFrom>().try_get<VNCClientHandle>();
            if (!handle || !*handle) return;
            VNCClient& vnc = **handle;
            if (!vnc.connected || !vnc.client) return;
            // Get cursor state
            auto glfw_state = world->lookup("GLFWState");
            if (!glfw_state.is_valid()) return;

            const CursorState* cursorState = glfw_state.try_get<CursorState>();
            if (!cursorState) return;

            // Convert mouse coordinates from window space to VNC space
            float scale_w = img.width / vnc.width;
            float scale_h = img.height / vnc.height;

            int offset_x = (int)pos.x;
            int offset_y = (int)pos.y;

            // Convert to VNC coordinates
            int vnc_x = (int)((cursorState->x - offset_x) / scale_w);
            int vnc_y = (int)((cursorState->y - offset_y) / scale_h);

            bool mouseOutOfBounds = vnc_x < 0 || vnc_y < 0 || cursorState->x >= offset_x + img.width || cursorState->y >= offset_y + img.height;

            std::cout << vnc_x << std::endl;
            vnc.eventPassthroughEnabled = !mouseOutOfBounds;

            if (!mouseOutOfBounds)
            {
                // Queue input event instead of sending directly
                InputEvent event;
                event.type = InputEvent::POINTER;
                event.data.pointer.x = vnc_x;
                event.data.pointer.y = vnc_y;
                event.data.pointer.buttonMask = g_vncButtonMask;

                {
                    std::lock_guard<std::mutex> lock(vnc.inputQueueMutex);
                    vnc.inputQueue.push_back(event);
                }
                vnc.inputQueueCV.notify_one();
            }
        });

    auto vncPassthroughIndicatorVisibility = world->system<ImageRenderable>()
        .with<IsStreamingFrom>(flecs::Wildcard)
        .kind(flecs::PreUpdate)
        .each([&](flecs::entity e, ImageRenderable& img) {
            auto* handle = e.target<IsStreamingFrom>().try_get<VNCClientHandle>();
            if (!handle || !*handle) return;
            VNCClient& vnc = **handle;
            if (!vnc.connected || !vnc.client) return;
            e.target<ActiveIndicator>().ensure<RenderStatus>().visible = vnc.eventPassthroughEnabled;
        });

    auto spaceframeSelector = world->system<FilmstripData>()
    .kind(flecs::PreFrame)
    // .immediate()
    .each([&](flecs::iter& it, size_t index, FilmstripData& data)
    {
        flecs::entity e = it.entity(index);
        float dt = it.delta_time();

        // Detect when a new frame has been added and reset scroll for seamless transition
        if (data.total_frames_added != data.last_seen_frame_count) {
            data.elapsed_time = 0.0f;  // Reset scroll to start
            data.last_seen_frame_count = data.total_frames_added;
        }

        // Update scroll timing
        data.elapsed_time += dt;

        // Time for one frame to scroll through = total duration / number of visible frames
        float time_per_frame = FilmstripData::SCROLL_DURATION / data.frame_limit;

        // Calculate scroll progress (0.0 to 1.0 for one frame width)
        // Clamp at 1.0 if no new frame arrives in time
        float scroll_progress = std::min(1.0f, data.elapsed_time / time_per_frame);

        // Detach all current children
        e.children([&](flecs::entity child)
        {
            child.remove(flecs::ChildOf, e);
            child.ensure<RenderStatus>().visible = false;
        });

        // Show frame_limit + 1 frames for smooth scrolling (extra frame on right side)
        int display_count = data.frame_limit + 1;
        int start_index = std::max(0, (int)data.frames.size() - display_count);
        for (int i = start_index; i < (int)data.frames.size(); i++)
        {
            data.frames[i].child_of(e);
            data.frames[i].ensure<RenderStatus>().visible = true;
        }

        // Store scroll offset to be applied after layout
        // Offset moves frames left over time (0 to 1 representing one frame width)
        // Resets to 0 when a new frame is added (seamless transition)
        data.scroll_offset = scroll_progress;
    });

    // Custom scroll layout for filmstrip - directly positions frames based on scroll offset
    // No LayoutBox - we handle all positioning here for smooth, stutter-free scrolling
    world->system<FilmstripData, UIElementBounds>("FilmstripScrollSystem")
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, FilmstripData& data, UIElementBounds& bounds)
    {
        Graphics& graphics = world->ensure<Graphics>();
        double currentTime = glfwGetTime();

        // Visible container width (what fits in the panel)
        float container_width = bounds.xmax - bounds.xmin;
        float container_height = bounds.ymax - bounds.ymin;
        // Each frame width = visible width / number of visible frames
        float frame_width = container_width / data.frame_limit;

        // Get mel spec render offset to sync filmstrip with mel spec timing
        // renderOffset > 0 means mel spec is behind wall clock, so shift filmstrip right to match
        // Note: renderOffset is already clamped to reasonable range in mel_spec_render.cpp
        float melSpecOffset = 0.0f;
        auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
        if (sysAudioRenderer.is_valid()) {
            const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
            if (melSpec && melSpec->fillProgress >= 1.0f) {
                melSpecOffset = melSpec->renderOffset;
            }
        }

        if (data.mode == FilmstripMode::Uniform) {
            // UNIFORM MODE: Position frames based on capture time with fixed widths
            // Use mel spec's actual timeline so filmstrip moves at exactly mel spec's rate

            // Calculate mel spec's "effective now" based on how many columns it's actually received
            double melSpecNow = currentTime;  // Default to wall clock
            auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
            if (sysAudioRenderer.is_valid()) {
                const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
                if (melSpec && melSpec->fillProgress >= 1.0f && melSpec->scrollStartTime > 0) {
                    // Mel spec's elapsed time based on actual columns received
                    double melSpecElapsed = (double)melSpec->totalScrollCommands / MelSpecRender::COLUMNS_PER_SECOND;
                    melSpecNow = melSpec->scrollStartTime + melSpecElapsed;
                }
            }

            e.children([&](flecs::entity child)
            {
                const FilmstripFrameTime* frameTime = child.try_get<FilmstripFrameTime>();
                if (!frameTime) return;

                // Calculate age using mel spec's timeline, not wall clock
                double age = melSpecNow - frameTime->capture_time;

                // Position based on age: 0 seconds ago = right edge, SCROLL_DURATION ago = left edge
                // Normalize to 0.0 (oldest/left) to 1.0 (newest/right)
                float normalizedTime = 1.0f - (float)(age / FilmstripData::SCROLL_DURATION);

                // Frame's LEFT edge aligns with capture time position
                // New frames spawn one frame width to the right (off-screen) and scroll in
                float x_pos = normalizedTime * container_width;

                // Hide frames that have scrolled off the left edge
                RenderStatus& renderStatus = child.ensure<RenderStatus>();
                if (normalizedTime < -0.5f) {
                    renderStatus.visible = false;
                } else {
                    renderStatus.visible = true;
                }

                Position& local = child.ensure<Position, Local>();
                local.x = x_pos;
                local.y = 0.0f;

                // Scale frame to fit within frame_width while maintaining aspect ratio
                ImageRenderable* img = child.try_get_mut<ImageRenderable>();
                if (img && img->imageHandle > 0) {
                    int native_w, native_h;
                    nvgImageSize(graphics.vg, img->imageHandle, &native_w, &native_h);

                    if (native_w > 0 && native_h > 0) {
                        float aspect = (float)native_w / (float)native_h;

                        // Fit to frame_width, scale height to maintain aspect ratio
                        float target_width = frame_width;
                        float target_height = target_width / aspect;

                        // If height exceeds container, fit to height instead
                        if (target_height > container_height) {
                            target_height = container_height;
                            target_width = target_height * aspect;
                        }

                        img->width = target_width;
                        img->height = target_height;

                        // Update UIElementSize to match
                        UIElementSize& size = child.ensure<UIElementSize>();
                        size.width = target_width;
                        size.height = target_height;
                    }
                }

                propagate_world_positions(child);
            });
        } else {
            // STEGOSAURUS MODE: Position frames based on capture time, allowing overlaps
            // Left edge = 24 seconds ago, right edge = now
            // Frame's left edge aligns with its capture time position

            e.children([&](flecs::entity child)
            {
                const FilmstripFrameTime* frameTime = child.try_get<FilmstripFrameTime>();
                if (!frameTime) return;

                // Calculate age of this frame (seconds ago it was captured)
                double age = currentTime - frameTime->capture_time;

                // Position based on age: 0 seconds ago = right edge, SCROLL_DURATION ago = left edge
                // Normalize to 0.0 (oldest/left) to 1.0 (newest/right)
                float normalizedTime = 1.0f - (float)(age / FilmstripData::SCROLL_DURATION);

                // Apply mel spec offset to sync with mel spec timing
                // If mel spec is behind (renderOffset > 0), shift filmstrip right (add to position)
                float x_pos = (normalizedTime + melSpecOffset) * container_width;

                // Hide frames that have scrolled off the left edge
                RenderStatus& renderStatus = child.ensure<RenderStatus>();
                if (normalizedTime < -0.5f) {
                    renderStatus.visible = false;
                } else {
                    renderStatus.visible = true;
                }

                Position& local = child.ensure<Position, Local>();
                local.x = x_pos;
                local.y = 0.0f;

                // Scale frame to fit within frame_width while maintaining aspect ratio
                ImageRenderable* img = child.try_get_mut<ImageRenderable>();
                if (img && img->imageHandle > 0) {
                    int native_w, native_h;
                    nvgImageSize(graphics.vg, img->imageHandle, &native_w, &native_h);

                    if (native_w > 0 && native_h > 0) {
                        float aspect = (float)native_w / (float)native_h;

                        // Fit to frame_width, scale height to maintain aspect ratio
                        float target_width = frame_width;
                        float target_height = target_width / aspect;

                        // If height exceeds container, fit to height instead
                        if (target_height > container_height) {
                            target_height = container_height;
                            target_width = target_height * aspect;
                        }

                        img->width = target_width;
                        img->height = target_height;

                        // Update UIElementSize to match
                        UIElementSize& size = child.ensure<UIElementSize>();
                        size.width = target_width;
                        size.height = target_height;
                    }
                }

                propagate_world_positions(child);
            });
        }
    });
    // DINO similarity score collection system - samples DINO cos diff and pushes to LineChartData
    // Also updates smooth scroll offset every frame for continuous scrolling
    world->system<LineChartData>("DinoScoreCollectionSystem")
    .kind(flecs::PreUpdate)
    .each([&](flecs::iter& it, size_t index, LineChartData& chart) {
        float dt = it.delta_time();
        chart.time_since_sample += dt;

        // Only sample DINO if loaded
        if (!g_dinoEmbedder.isLoaded()) return;

        // Sample at the specified interval (or every frame if interval is 0)
        if (chart.sample_interval > 0 && chart.time_since_sample < chart.sample_interval) {
            return;
        }

        float cosDiff = 0.0f;
        int frameCount = 0;
        if (g_dinoEmbedder.getResult(cosDiff, frameCount)) {
            // cosDiff is 1.0 - cosineSimilarity, so higher = more different
            // We want to show similarity, so invert: similarity = 1.0 - cosDiff
            float similarity = 1.0f - cosDiff;

            // Use mel spec timeline for timestamp (prevents drift from filmstrip/mel spec)
            // Fall back to wall clock if mel spec not ready
            double captureTime = glfwGetTime() - chart.processing_lag;
            auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
            if (sysAudioRenderer.is_valid()) {
                const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
                if (melSpec && melSpec->fillProgress >= 1.0f && melSpec->scrollStartTime > 0) {
                    double melSpecNow = melSpec->scrollStartTime + (double)melSpec->totalScrollCommands / MelSpecRender::COLUMNS_PER_SECOND;
                    captureTime = melSpecNow - chart.processing_lag;
                }
            }
            chart.push(similarity, captureTime);

            chart.time_since_sample = 0.0f;
        }
    });

    // Stegosaurus spike detection system - monitors DINO values and triggers filmstrip captures on spikes
    world->system<FilmstripData>("StegosaurusSpikeDetectionSystem")
    .kind(flecs::PreUpdate)
    .each([&](flecs::iter& it, size_t index, FilmstripData& data) {
        // Only run in Stegosaurus mode
        if (data.mode != FilmstripMode::Stegosaurus) return;

        float dt = it.delta_time();
        data.time_since_spike += dt;

        // Only check for spikes if DINO is loaded
        if (!g_dinoEmbedder.isLoaded()) return;

        // Get latest DINO result
        float cosDiff = 0.0f;
        int frameCount = 0;
        if (g_dinoEmbedder.getResult(cosDiff, frameCount)) {
            // Store the latest value for reference
            data.last_dino_value = cosDiff;

            // Check if this is a spike (exceeds threshold and past cooldown)
            if (cosDiff >= data.spike_threshold && data.time_since_spike >= data.spike_cooldown) {
                data.pending_capture = true;
                data.pending_spike_time = glfwGetTime();  // Record when spike was detected for accurate positioning
                std::cout << "[DINO SPIKE] cosDiff=" << cosDiff << " threshold=" << data.spike_threshold << std::endl;
            }
        }
    });

    // The 3D viewport is the one element whose *pixels* depend on its layout, so
    // the size Yoga gave it has to reach the renderer before the passes run.
    // Registered after boundsCalculationSystem (also OnLoad) so the bounds read
    // here are this frame's, and before Panel3DInput on PreUpdate so a drag is
    // applied in the frame it happened.
    //
    // With no Droid panel open the query is empty and the whole render3d chain
    // is switched off rather than drawing a scene nothing samples.
    auto panel3d_viewports = world->query_builder<const UIElementBounds>()
        .with<Panel3DViewport>()
        .cached()
        .build();

    world->system("Panel3DSyncSystem")
    .kind(flecs::OnLoad)
    .run([panel3d_viewports, window](flecs::iter& it)
    {
        flecs::world w = it.world();
        bool live = false;

        panel3d_viewports.each([&](flecs::entity e, const UIElementBounds& bounds)
        {
            if (live) return;   // one scene, so the first laid-out panel wins
            int panel_w = (int)std::lround(bounds.xmax - bounds.xmin);
            int panel_h = (int)std::lround(bounds.ymax - bounds.ymin);
            if (panel_w <= 0 || panel_h <= 0) return;
            live = true;

            panel3d::set_active(true);
            panel3d::resize(w, panel_w, panel_h);

            // The image handle is minted per viewport allocation, so a resize
            // invalidates whatever the panel was created with.
            ImageRenderable& img = e.ensure<ImageRenderable>();
            img.imageHandle = panel3d::image_handle();

            double cursor_x = 0.0, cursor_y = 0.0;
            if (const CursorState* cursor = w.lookup("GLFWState").try_get<CursorState>()) {
                cursor_x = cursor->x;
                cursor_y = cursor->y;
            }
            const bool left_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            panel3d::set_pointer(cursor_x, cursor_y, left_down,
                                 point_in_bounds(cursor_x, cursor_y, bounds));
        });

        if (!live) panel3d::set_active(false);
    });

    // Sync mel spec position during fill phase only
    // After fill, mel spec stays anchored at rightmost position (no texture offset)
    // Filmstrip scroll is offset instead to match mel spec timing
    world->system<ImageRenderable, UIElementBounds>("MelSpecSyncPositionSystem")
    .kind(flecs::PreUpdate)
    .with<MelSpecSource>(flecs::Wildcard)
    .each([&](flecs::entity e, ImageRenderable& img, UIElementBounds& bounds)
    {
        // Get the mel spec renderer entity this is linked to
        flecs::entity melSpecEntity = e.target<MelSpecSource>();
        if (!melSpecEntity.is_valid()) return;

        const MelSpecRender* melSpec = melSpecEntity.try_get<MelSpecRender>();
        if (!melSpec) return;

        // Get parent bounds for sizing
        flecs::entity parent = e.parent();
        if (!parent.is_valid()) return;
        const UIElementBounds* parent_bounds = parent.try_get<UIElementBounds>();
        if (!parent_bounds) return;

        float parent_width = parent_bounds->xmax - parent_bounds->xmin;

        if (melSpec->fillProgress < 1.0f) {
            // Initial fill phase: slide element in from right
            Position& local = e.ensure<Position, Local>();
            local.x = parent_width * (1.0f - melSpec->fillProgress);
            img.texOffsetX = 0.0f;
            propagate_world_positions(e);
        } else {
            // Fully filled: mel spec anchored at x=0 with no texture offset
            // Filmstrip handles sync offset instead
            Position& local = e.ensure<Position, Local>();
            local.x = 0.0f;
            img.texOffsetX = 0.0f;
            propagate_world_positions(e);
        }
    });

    world->system<RectRenderable>()
    .kind(flecs::PostLoad)
    .with<CopyChildHeight>(flecs::Wildcard)
    .each([&](flecs::entity e, RectRenderable& rect)
    {
        flecs::entity target = e.target<CopyChildHeight>();
        target.children([&](flecs::entity child)
        {
            UIElementBounds& bounds = child.ensure<UIElementBounds>();
            rect.height = bounds.ymax - bounds.ymin;
        });
    });

    auto vncTextureUpdateSystem = world->system<VNCClientHandle>()
        .kind(flecs::OnUpdate)
        .each([&](flecs::entity e, VNCClientHandle& handle) {
            VNCClient& vnc = *handle;
            if (!vnc.connected || !vnc.client) {
                return;
            }
            if (!vnc.needsUpdate.load(std::memory_order_acquire)) return;

            // Swap dirty rect queue (fast, non-blocking)
            auto newRects = vnc.dirtyRectQueue->swap();
            if (newRects.empty()) {
                // Double-check queue is still empty before clearing flag
                std::lock_guard<std::mutex> queueLock(vnc.dirtyRectQueue->mutex);
                if (vnc.dirtyRectQueue->pending.empty()) {
                    vnc.needsUpdate.store(false, std::memory_order_release);
                }
                return;
            }

            std::lock_guard<std::mutex> lock(*vnc.surfaceMutex);
            
            LOG_TRACE(LogCategory::VNC_CLIENT, "Updating OpenGL texture {} for quadrant {}", vnc.vncTexture, vnc.toString());

            SDL_Surface* surface = (SDL_Surface*)rfbClientGetClientData(vnc.client, (void*)VNC_SURFACE_TAG);
            VisionProcessingJob job;
            bool shouldCapture = false;
            if (surface && surface->pixels) {
                // Submit vision processing job to background thread instead of blocking
                job.quadrant = (int)e.raw_id(); // TODO: Fucking rename
                job.paletteFile = "../assets/palettes/resurrect-64.hex";
                job.outputPath = "/tmp/vision_" + std::to_string(e.raw_id()) + ".png";
                job.width = surface->w;
                job.height = surface->h;
                job.pitch = surface->pitch;

                // Copy pixel data to avoid race conditions with VNC updates
                size_t dataSize = surface->pitch * surface->h;
                job.pixelData.resize(dataSize);                // Send pointer event with current button mask (preserves button state during drags)
                memcpy(job.pixelData.data(), surface->pixels, dataSize);

                g_visionQueue.submit(job);
                LOG_TRACE(LogCategory::VNC_CLIENT, "Submitted vision processing job for quadrant {}", vnc.toString());

                LOG_TRACE(LogCategory::VNC_CLIENT, "Processing {} dirty rectangles", newRects.size());
                LOG_TRACE(LogCategory::VNC_CLIENT, "Surface info: {}x{}, format: {}", surface->w, surface->h, SDL_GetPixelFormatName(surface->format->format));

                // Bind PBO for async texture upload
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vnc.pbo);

                // Resize PBO if needed (connection established and size changed)
                size_t requiredSize = surface->pitch * surface->h;
                GLint currentSize = 0;
                glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &currentSize);
                if (currentSize < (GLint)requiredSize) {
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, requiredSize, nullptr, GL_STREAM_DRAW);
                }

                // Determine pixel format
                int bytesPerPixel = surface->format->BytesPerPixel;
                GLenum format = GL_BGRA;
                if (surface->format->Rmask == 0xFF) {
                    format = GL_RGBA;
                }

                // Map PBO for writing
                void* pboMem = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
                if (!pboMem) {
                    LOG_ERROR(LogCategory::VNC_CLIENT, "Failed to map PBO");
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                    vnc.needsUpdate = false;
                    return;
                }

                uint8_t* pboBytes = static_cast<uint8_t*>(pboMem);
                uint8_t* surfaceBytes = static_cast<uint8_t*>(surface->pixels);

                // Copy only dirty rectangles to PBO with alpha correction
                for (const auto& rectIn : newRects) {
                    // Clamp rect to surface bounds
                    int rx = std::max(0, rectIn.x);
                    int ry = std::max(0, rectIn.y);
                    int rw = std::min(rectIn.w, surface->w - rx);
                    int rh = std::min(rectIn.h, surface->h - ry);
                    if (rw <= 0 || rh <= 0) continue;

                    // Copy this dirty rect to PBO with alpha correction
                    for (int y = 0; y < rh; ++y) {
                        int srcY = ry + y;
                        uint8_t* srcRow = surfaceBytes + srcY * surface->pitch + rx * 4;
                        uint8_t* dstRow = pboBytes + srcY * surface->pitch + rx * 4;

                        for (int x = 0; x < rw; ++x) {
                            int idx = x * 4;
                            dstRow[idx + 0] = srcRow[idx + 0];  // R or B
                            dstRow[idx + 1] = srcRow[idx + 1];  // G
                            dstRow[idx + 2] = srcRow[idx + 2];  // B or R
                            dstRow[idx + 3] = 0xFF;              // A (force opaque)
                        }
                    }
                }

                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

                // Update OpenGL texture from PBO
                glBindTexture(GL_TEXTURE_2D, vnc.vncTexture);

                int rowLengthPixels = surface->pitch / bytesPerPixel;
                glPixelStorei(GL_UNPACK_ROW_LENGTH, rowLengthPixels);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

                LOG_TRACE(LogCategory::VNC_CLIENT, "Using OpenGL format: {}", (format == GL_BGRA ? "BGRA" : "RGBA"));

                // Upload each dirty rectangle from PBO to texture
                for (const auto& rectIn : newRects) {
                    int rx = std::max(0, rectIn.x);
                    int ry = std::max(0, rectIn.y);
                    int rw = std::min(rectIn.w, surface->w - rx);
                    int rh = std::min(rectIn.h, surface->h - ry);
                    if (rw <= 0 || rh <= 0) continue;

                    // Upload from PBO (GPU-side transfer, non-blocking)
                    size_t offset = ry * surface->pitch + rx * 4;
                    glTexSubImage2D(GL_TEXTURE_2D, 0, rx, ry, rw, rh, format, GL_UNSIGNED_BYTE, (void*)offset);
                }

                // Reset to defaults
                glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

                // Unbind PBO
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

                LOG_TRACE(LogCategory::VNC_CLIENT, "All rects updated successfully");

                // Check if more updates arrived while we were processing
                // Only clear needsUpdate if the queue is truly empty
                {
                    std::lock_guard<std::mutex> queueLock(vnc.dirtyRectQueue->mutex);
                    if (vnc.dirtyRectQueue->pending.empty()) {
                        vnc.needsUpdate.store(false, std::memory_order_release);
                    }
                }

                std::cout << "Updated VNC texture (async PBO)" << std::endl;

                // Frame capture - mode depends on FilmstripData settings
                double currentTime = glfwGetTime();

                // Check filmstrip mode to determine if we should capture
                flecs::query q_editor_panels_check = world->query_builder<EditorLeafData>()
                    .build();

                q_editor_panels_check.each([&](flecs::entity leaf, EditorLeafData& leaf_data)
                {
                    if (leaf_data.editor_type == EditorType::Episodic)
                    {
                        flecs::entity canvas = leaf.target<EditorCanvas>();
                        FilmstripData* filmstripData = canvas.target<FilmstripChannel>().try_get_mut<FilmstripData>();
                        if (filmstripData) {
                            if (filmstripData->mode == FilmstripMode::Uniform) {
                                // Uniform mode: capture when mel spec has scrolled one frame width
                                // This keeps filmstrip in sync with mel spec regardless of timing variations
                                auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
                                if (sysAudioRenderer.is_valid()) {
                                    const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
                                    if (melSpec && melSpec->fillProgress >= 1.0f) {
                                        // Calculate columns per frame width
                                        float frameWidthSeconds = FilmstripData::SCROLL_DURATION / filmstripData->frame_limit;
                                        size_t columnsPerFrame = (size_t)(frameWidthSeconds * MelSpecRender::COLUMNS_PER_SECOND);

                                        // Capture when mel spec has scrolled one frame width since last capture
                                        size_t scrolledSinceCapture = melSpec->totalScrollCommands - filmstripData->last_capture_scroll_commands;
                                        shouldCapture = (scrolledSinceCapture >= columnsPerFrame);
                                    }
                                }
                            } else if (filmstripData->mode == FilmstripMode::Stegosaurus) {
                                // Stegosaurus mode: capture when pending_capture is set
                                shouldCapture = filmstripData->pending_capture;
                                if (shouldCapture) {
                                    filmstripData->pending_capture = false;  // Clear the flag
                                    filmstripData->time_since_spike = 0.0f;  // Reset cooldown
                                }
                            }
                        }
                    }
                });

                if (shouldCapture) {
                    // Prepare RGBA data buffer
                    std::vector<uint8_t> rgbaData(surface->w * surface->h * 4);

                    // Convert surface data to RGBA format
                    uint8_t* src = static_cast<uint8_t*>(surface->pixels);
                    uint8_t* dst = rgbaData.data();

                    bool isBGRA = (surface->format->Rmask != 0xFF);

                    for (int y = 0; y < surface->h; ++y) {
                        for (int x = 0; x < surface->w; ++x) {
                            int srcIdx = y * surface->pitch + x * 4;
                            int dstIdx = (y * surface->w + x) * 4;

                            if (isBGRA) {
                                // BGRA -> RGBA
                                dst[dstIdx + 0] = src[srcIdx + 2];  // R
                                dst[dstIdx + 1] = src[srcIdx + 1];  // G
                                dst[dstIdx + 2] = src[srcIdx + 0];  // B
                                dst[dstIdx + 3] = 0xFF;             // A
                            } else {
                                // Already RGBA, just copy with opaque alpha
                                dst[dstIdx + 0] = src[srcIdx + 0];  // R
                                dst[dstIdx + 1] = src[srcIdx + 1];  // G
                                dst[dstIdx + 2] = src[srcIdx + 2];  // B
                                dst[dstIdx + 3] = 0xFF;             // A
                            }
                        }
                    }

                    // Create NVG image directly from pixel data (no PNG round-trip for display)
                    int imgHandle = nvgCreateImageRGBA(graphics.vg, surface->w, surface->h, 0, rgbaData.data());

                    if (imgHandle > 0) {
                        vnc.framesCaptured++;
                        vnc.lastCaptureTime = currentTime;

                        // Queue PNG save to background thread (non-blocking)
                        std::string frame_filename = "frame_" + std::to_string(vnc.framesCaptured - 1) + ".png";
                        g_pngSaveQueue.submit({std::move(rgbaData), surface->w, surface->h, frame_filename});

                        // Create frame entity with direct ImageRenderable (skip ImageCreator)
                        flecs::query q_editor_panels = world->query_builder<EditorLeafData>()
                        .build();

                        q_editor_panels.each([&](flecs::entity leaf, EditorLeafData& leaf_data)
                        {
                            if (leaf_data.editor_type == EditorType::Episodic)
                            {
                                flecs::entity canvas = leaf.target<EditorCanvas>();
                                FilmstripData& filmstripData = canvas.target<FilmstripChannel>().ensure<FilmstripData>();

                                // ZIndex: base 15, increment for each frame so newer ones render on top
                                int zIndex = 15 + (int)(filmstripData.total_frames_added % 100);

                                // Use spike detection time for Stegosaurus mode (aligns with chart),
                                // mel spec timeline for Uniform mode (prevents drift)
                                double frameTime = currentTime;  // Default to wall clock
                                if (filmstripData.mode == FilmstripMode::Stegosaurus) {
                                    frameTime = filmstripData.pending_spike_time;
                                } else if (filmstripData.mode == FilmstripMode::Uniform) {
                                    // Use mel spec's timeline so frame positioning matches
                                    auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
                                    if (sysAudioRenderer.is_valid()) {
                                        const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
                                        if (melSpec && melSpec->fillProgress >= 1.0f && melSpec->scrollStartTime > 0) {
                                            double melSpecElapsed = (double)melSpec->totalScrollCommands / MelSpecRender::COLUMNS_PER_SECOND;
                                            frameTime = melSpec->scrollStartTime + melSpecElapsed;
                                        }
                                    }
                                }

                                auto frame = world->entity()
                                .is_a(UIElement)
                                .set<ImageRenderable>({imgHandle, 1.0f, 1.0f, (float)surface->w, (float)surface->h})
                                .set<ZIndex>({zIndex})
                                .set<FilmstripFrameTime>({frameTime})  // Track time for positioning
                                // .add<DebugRenderBounds>()
                                .add<ScissorContainer>(canvas)
                                .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f, true});

                                filmstripData.frames.push_back(frame);
                                filmstripData.total_frames_added++;  // Track for scroll sync
                                filmstripData.elapsed_time = 0.0f;   // Reset immediately for seamless transition

                                // Update mel spec scroll tracking for Uniform mode sync
                                if (filmstripData.mode == FilmstripMode::Uniform) {
                                    auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
                                    if (sysAudioRenderer.is_valid()) {
                                        const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
                                        if (melSpec) {
                                            filmstripData.last_capture_scroll_commands = melSpec->totalScrollCommands;
                                        }
                                    }
                                }

                                // Keep frame_limit + 1 frames for smooth scrolling (extra frame offscreen)
                                if (filmstripData.frames.size() > (size_t)(filmstripData.frame_limit + 1))
                                {
                                    // Delete NVG image when removing old frame
                                    ImageRenderable* oldImg = filmstripData.frames[0].try_get_mut<ImageRenderable>();
                                    if (oldImg && oldImg->imageHandle > 0) {
                                        nvgDeleteImage(graphics.vg, oldImg->imageHandle);
                                    }
                                    filmstripData.frames[0].destruct();
                                    filmstripData.frames.erase(filmstripData.frames.begin());
                                }
                            }
                        });

                        std::cout << "[VNC CAPTURE] Created frame texture directly (handle: " << imgHandle << ")" << std::endl;
                    } else {
                        std::cerr << "[VNC CAPTURE ERROR] Failed to create NVG image from pixel data" << std::endl;
                    }
                }
            } else {
                LOG_ERROR(LogCategory::VNC_CLIENT, "Surface or pixels is null");
            }

            // Submit frame to DINO embedder (async, non-blocking)
            if (g_dinoEmbedder.isLoaded()) {
                // TODO: Consider inclusion of embeddings for entity binding clustering at a different rate than uniform captures...
                // and register it for entity binding
                bool writeEmbedding = shouldCapture;
                g_dinoEmbedder.submitFrame(job.pixelData.data(), job.width, job.height, job.pitch, writeEmbedding);
                // Results are printed by the worker thread
            }
        });

    // VNC cleanup system - handles thread lifecycle and resource cleanup
    auto vncCleanupSystem = world->system<VNCClientHandle>()
        .kind(flecs::OnUpdate)
        .each([](flecs::entity e, VNCClientHandle& handle) {
            VNCClient& vnc = *handle;
            // Handle connection state transitions
            switch (vnc.connectionState.load()) {
                case VNCClient::CONNECTING:
                    // Connection in progress, handled by network thread
                    break;

                case VNCClient::CONNECTED:
                    // Normal operation
                    break;

                case VNCClient::ERROR:
                case VNCClient::DISCONNECTING:
                    // Stop thread and cleanup if it's still running
                    if (vnc.threadRunning) {
                        LOG_INFO(LogCategory::VNC_CLIENT, "Stopping VNC thread for {}", vnc.toString());
                        vnc.threadShouldStop = true;
                        vnc.inputQueueCV.notify_one();

                        // Wait for thread to finish
                        if (vnc.messageThread.joinable()) {
                            vnc.messageThread.join();
                        }

                        vnc.connectionState = VNCClient::DISCONNECTED;
                    }
                    break;

                case VNCClient::DISCONNECTED:
                    // Ready for reconnection or final cleanup
                    // If reference count is 0, entity should be destroyed
                    if (vnc.reference_count <= 0) {
                        LOG_INFO(LogCategory::VNC_CLIENT, "Cleaning up VNC client {}", vnc.toString());

                        // Cleanup OpenGL resources (MUST be on main thread)
                        if (vnc.pbo) {
                            glDeleteBuffers(1, &vnc.pbo);
                            vnc.pbo = 0;
                        }
                        if (vnc.vncTexture) {
                            glDeleteTextures(1, &vnc.vncTexture);
                            vnc.vncTexture = 0;
                        }

                        // Cleanup VNC client
                        if (vnc.client) {
                            rfbClientCleanup(vnc.client);
                            vnc.client = nullptr;
                        }

                        // Cleanup SDL surface
                        if (vnc.surface) {
                            SDL_FreeSurface(vnc.surface);
                            vnc.surface = nullptr;
                        }

                        // Entity can now be safely removed
                        e.destruct();
                    }
                    break;
            }
        });

    // X11 window outline rendering system - draws X11 window outlines on top of VNC texture
    auto x11OutlineRenderSystem = world->system<X11Container, Position, ImageRenderable>()
        .term_at(1).second<Local>()
        .kind(flecs::PostUpdate)
        .immediate()
        .with<IsStreamingFrom>(flecs::Wildcard)
        .each([&](flecs::entity e, X11Container& container, Position& localPos, ImageRenderable& img) {
            flecs::entity vnc_entity = e.target<IsStreamingFrom>();
            auto* handle = vnc_entity.try_get<VNCClientHandle>();
            if (!handle || !*handle) return;
            VNCClient& vnc = **handle;
            if (!vnc.connected || !vnc.client) return;

            // TODO: Query to determine what VNC Streams that there are that are visible in Editor VNCStream type

            // TODO: There are no longer quadrants, this is an outdated and no longer acceptable
            // model of how multiple VNC containers are identified
            // if (container.last_update_timestamp.empty()) {2000
            //     std::cout << "[X11 Render] Container " << vnc << " has no data" << std::endl;
            //     return;
            // }

            // Calculate scale factor from VNC coordinates to shephleetess screen coordinates
            float scale_x = img.width / (float)vnc.width;
            float scale_y = img.height / (float)vnc.height;

            Position worldPos = e.get<Position, World>();
            RenderQueue& queue = world->ensure<RenderQueue>();

            // Query all visible windows in this container
            auto windows = world->query_builder<X11WindowInfo, X11WindowBounds, X11WindowVisibility>()
                .with<X11VisibleWindow>()
                .with(flecs::ChildOf, e)
                .build();

            static int debugFrameCount = 0;
            if (debugFrameCount++ % 60 == 0) {  // Log once per second at 60fps
                int window_count = 0;
                windows.each([&](flecs::entity, X11WindowInfo&, X11WindowBounds&, X11WindowVisibility&) { window_count++; });
                LOG_DEBUG(LogCategory::X11_OUTLINE, "VNC {} has {} windows", vnc.toString(), window_count);
            }

            // Render each visible window outline
            windows.each([&](flecs::entity win_e,
                            X11WindowInfo& info,
                            X11WindowBounds& bounds,
                            X11WindowVisibility& vis) {
                // Map X11 window coordinates to shephleetess screen coordinates
                float win_x = worldPos.x + (bounds.x * scale_x);
                float win_y = worldPos.y + (bounds.y * scale_y);
                float win_w = bounds.width * scale_x;
                float win_h = bounds.height * scale_y;

                // Bright red outline color
                uint32_t outline_color = 0xe85d20ff;

                float outline_thickness = 1.0f;  // Increased thickness for visibility

                // Draw window outline (top, bottom, left, right) at very high Z-index
                queue.addRectCommand(
                    {win_x, win_y},
                    {win_w, outline_thickness, true, outline_color},
                    2000  // Very high Z-index to ensure visibility above everything
                );
                queue.addRectCommand(
                    {win_x, win_y + win_h - outline_thickness},
                    {win_w, outline_thickness, true, outline_color},
                    2000
                );
                queue.addRectCommand(
                    {win_x, win_y},
                    {outline_thickness, win_h, true, outline_color},
                    2000
                );
                queue.addRectCommand(
                    {win_x + win_w - outline_thickness, win_y},
                    {outline_thickness, win_h, true, outline_color},
                    2000
                );

                // Optionally, draw window name if visibility is high enough
                // if (vis.visibility_percent > 50.0f && !info.name.empty()) {
                //     queue.addTextCommand(
                //         {win_x + 4.0f, win_y + 16.0f},  // Small offset from top-left corner
                //         {info.name, "sans", 12.0f, 0xFFFFFFFF, NVG_ALIGN_LEFT | NVG_ALIGN_TOP},
                //         2001  // Z-index above outline
                //     );
                // }
            });
        });

    int fontHandle = nvgCreateFont(vg, "ATARISTOCRAT", "../assets/ATARISTOCRAT.ttf");
    int charisSILFontHandle = nvgCreateFont(vg, "CharisSIL", "../assets/CharisSIL-Regular.ttf");
    int jetBriansMonoFontHandle = nvgCreateFont(vg, "JetBrainsMono", "../assets/JetBrainsMono-Regular.ttf");

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        int winWidth, winHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        float devicePixelRatio = (float)fbWidth / (float)winWidth;

        // Resize 3D rendering resources if window size changed
        resize3DRendering(graphics, winWidth, winHeight);

        // Tilt animation disabled - using orthographic projection
        // graphics.tiltAngle += 0.01f;

        // PHASE 1: Render UI to framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, graphics.fbo);
        glViewport(0, 0, graphics.uiWidth, graphics.uiHeight);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // The Droid panel's 3D passes run inside world->progress() below and
        // bind framebuffers of their own; this is where they hand control back,
        // so nanovg's flush at nvgEndFrame still lands in the UI framebuffer.
        panel3d::set_ui_target(graphics.fbo, graphics.uiWidth, graphics.uiHeight);

        glfwStateEntity.set<Window>({window, winWidth, winHeight});

        nvgBeginFrame(vg, graphics.uiWidth, graphics.uiHeight, 1.0f);

        world->defer_begin();
        glfwPollEvents();
        world->defer_end();
        world->progress();

        FrameMark;
        nvgEndFrame(vg);

        // PHASE 2: Render 3D plane with UI texture to screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbWidth, fbHeight);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glUseProgram(graphics.shaderProgram);

        // Set up matrices
        float model[16], view[16], projection[16];

        // Model matrix - identity (no tilt)
        mat4Identity(model);

        // View matrix (camera positioned so plane fills viewport)
        // With 45° FOV, distance = halfHeight / tan(fov/2) = 1.0 / tan(22.5°) ≈ 2.414
        mat4LookAt(view, 0.0f, 0.0f, 2.414f,  // Eye position (calculated to fill screen)
                   0.0f, 0.0f, 0.0f,           // Look at center
                   0.0f, 1.0f, 0.0f);          // Up vector

        // Perspective projection matrix
        float aspect = (float)fbWidth / (float)fbHeight;
        mat4Perspective(projection, 0.785398f, aspect, 0.1f, 100.0f);  // 45 degrees FOV

        // Set uniforms
        GLint modelLoc = glGetUniformLocation(graphics.shaderProgram, "model");
        GLint viewLoc = glGetUniformLocation(graphics.shaderProgram, "view");
        GLint projLoc = glGetUniformLocation(graphics.shaderProgram, "projection");

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model);
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);

        // Bind UI texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, graphics.fboTexture);
        glUniform1i(glGetUniformLocation(graphics.shaderProgram, "uiTexture"), 0);

        // Get glow uniforms
        GLint glowPassLoc = glGetUniformLocation(graphics.shaderProgram, "glowPass");
        GLint glowExpandLoc = glGetUniformLocation(graphics.shaderProgram, "glowExpand");
        GLint chromaLoc = glGetUniformLocation(graphics.shaderProgram, "chromaStrength");

        // Calculate chromatic aberration (fades out after central triangle forms)
        float chromaAmount = 0.0f;
        float chromaDuration = 2.2f;  // Fade out over 2.2 seconds
        if (graphics.decelerationTime < chromaDuration) {
            float t = graphics.decelerationTime / chromaDuration;
            chromaAmount = 0.3f * (1.0f - t) * (1.0f - t);  // Quadratic fadeout, max 0.3
        }
        glUniform1f(chromaLoc, chromaAmount);

        // Draw either plane or grid with debris
        if (graphics.useGridMode) {
            // Disable backface culling for tetrahedrons (we want to see all faces)
            glDisable(GL_CULL_FACE);

            // Draw noise tetrahedrons (grey debris flying past)
            if (!graphics.noiseParticles.empty()) {
                glUniform1i(glowPassLoc, 0);  // Normal pass for noise
                // Set default values for attributes not in noise vertex format
                glVertexAttrib3f(2, 0.33f, 0.33f, 0.34f);  // Barycentric (center, no edge glow)
                glVertexAttrib1f(3, 0.0f);  // Glow = 0
                glVertexAttrib2f(4, 0.0f, 0.0f);  // Centroid offset = 0
                glBindTexture(GL_TEXTURE_2D, graphics.greyTexture);
                glBindVertexArray(graphics.noiseVAO);
                glDrawElements(GL_TRIANGLES, graphics.noiseVertexCount, GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            }

            // PASS 1: Draw normal textured triangles
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUniform1i(glowPassLoc, 0);  // Normal pass

            glBindTexture(GL_TEXTURE_2D, graphics.fboTexture);
            glBindVertexArray(graphics.gridVAO);
            glDrawElements(GL_TRIANGLES, graphics.gridVertexCount, GL_UNSIGNED_INT, 0);
            glDisable(GL_BLEND);

            // PASS 2: Draw outer glow on top (expanded triangles with glow color)
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // Additive blending for glow
            glDepthMask(GL_FALSE);  // Don't write to depth buffer for glow
            glDisable(GL_DEPTH_TEST);  // Render on top of everything

            glUniform1i(glowPassLoc, 1);  // Glow pass
            glUniform1f(glowExpandLoc, 0.008f);  // Small expansion for subtle glow

            glDrawElements(GL_TRIANGLES, graphics.gridVertexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
            glEnable(GL_CULL_FACE);
        } else {
            // Draw single plane (normal texture mode)
            glUniform1i(glowPassLoc, 0);  // Normal pass
            glUniform1f(glowExpandLoc, 0.0f);  // No expansion
            glUniform1f(chromaLoc, 0.0f);  // No chromatic aberration
            glBindTexture(GL_TEXTURE_2D, graphics.fboTexture);
            glBindVertexArray(graphics.planeVAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        glDisable(GL_DEPTH_TEST);

        glfwSwapBuffers(window);
    }

    // Cleanup 3D rendering resources
    cleanup3DRendering(graphics);
    panel3d::shutdown(*world);

    nvgDeleteGL2(vg);

    libssh2_exit();

    glfwTerminate();
    return 0;
}
