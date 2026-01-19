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

using json = nlohmann::json;

struct TextSize { float w, h; };

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
    int quadrant;
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

/*
These luminous phenomena still manifest themselves
from time to time, as when a new idea opening up possibilities
strikes me, but they are no longer exciting, being of relatively
small intensity. When I close my eyes I invariably observe first,
a background of very dark and uniform blue, not unlike the
sky on a clear but starless night. In a few seconds this field
becomes animated with innumerable scintillating flakes of
green, arranged in several layers and advancing towards me.
Then there appears, to the right, a beautiful pattern of two
systems of parallel and closely spaced lines, at right angles to
one another, in all sorts of colors with yellow-green and gold
predominating. Immediately thereafter the lines grow brighter and the whole is thickly sprinkled with dots of twinkling
light. This picture moves slowly across the field of vision and
in about ten seconds vanishes to the left, leaving behind a
ground of rather unpleasant and inert grey which quickly
gives way to a billowy sea of clouds, seemingly trying to mould
themselves in living shapes. It is curious that I cannot project a
form into this grey until the second phase is reached. Every
time, before falling asleep, images of persons or objects flit
before my view. When I see them I know that I am about to lose
consciousness. If they are absent and refuse to come it means a
sleepless night.
*/

// ECS Components

// struct Position {
//     float x, y;
// };

typedef Vector2 Position;

struct Edge
{
    Position p0;// LibVNC
#include <rfb/rfbclient.h>
    Position p1;
};

struct Local {};
struct World {};

struct Velocity {
    float dx, dy;
};

struct UIElementSize
{
    float width, height;
};

struct DebugRenderBounds {};
struct UIElementBounds {
    float xmin, ymin, xmax, ymax;
};

enum class ServerStatus
{
    Offline,
    Loading,
    Ready,
};

struct ServerScript
{
    std::string name;
    std::string conda_env;
    std::string launcher_path;
};

struct RenderStatus {
    bool visible;
    // float transparency = 1.0f;
};

// Unified layout box - used for both horizontal and vertical layouts
struct LayoutBox
{
  enum Direction { Horizontal, Vertical };
  Direction dir = Horizontal;
  float padding = 0.0f;
  float move_dir = 1.0f; // 1 = right/down, -1 = left/up
};

struct ColumnFitConstraint
{
    int count;
};

struct FitChildren 
{
    float scale_factor;
};

struct FlowLayoutBox
{
  float x_progress;
  float y_progress;
  float padding = 0.0f;
  float line_height = 0.0f;
  float line_spacing = 0.0f;
};

struct RenderGradient
{
    uint32_t start;
    uint32_t end;
};

struct RectRenderable {
    float width, height;
    bool stroke;
    uint32_t color;
};

// Forward declaration
struct RenderCommand; 

// Example, diamond
struct CustomRenderable
{
    float width, height;
    bool stroke;
    uint32_t color;
    uint32_t gradient_start = 0;
    uint32_t gradient_end = 0;

    std::function<void(NVGcontext*, const RenderCommand*, const CustomRenderable&)> render_function;
};

struct RoundedRectRenderable {
    float width, height, radius;
    bool stroke;
    uint32_t color;
};

struct LineRenderable {
    float x1, y1;
    float x2, y2;
    
    float thickness;
    uint32_t color;
};

struct QuadraticBezierRenderable {
    float x1, y1;
    float cx, cy;
    float x2, y2;
    
    float thickness;
    uint32_t color;
};

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
    float radius = 80.0f;

    float centerX = 100.0f;
    float centerY = 100.0f;
    int segments = 24;
    float thickness = 2.0f;

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

struct DiurnalHour
{
    size_t segment;
};

struct DynamicTextWrap 
{
    float pad;
};

struct DynamicTextWrapContainer {};

struct TextRenderable {
    std::string text;
    std::string fontFace;
    float fontSize;
    uint32_t color;
    float scaleY = 1.0f;
    float wrapWidth = 0.0f;  // 0 = no wrapping, >0 = wrap at this width
};

struct ImageCreator
{
    std::string path;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    NVGcolor tint = nvgRGBA(255, 255, 255, 255);
};

struct ImageRenderable
{
    int imageHandle;
    float scaleX, scaleY;

    float width, height;
    NVGcolor tint = nvgRGBA(255, 255, 255, 255);

    // Texture offset for smooth scrolling (shifts UV sampling, not element position)
    float texOffsetX = 0.0f;
    float texOffsetY = 0.0f;
};

struct ZIndex {
    int layer;
};

struct Window {
    GLFWwindow* handle;
    int width, height;
};

struct CursorState
{
    double x, y;
};

struct AddTagOnLeftClick{};
struct ShowEditorPanels {};
struct SetPanelEditorType {};
struct SelectServer {};

struct LeftClickEvent {};
struct Dragging {};
struct DynamicPartition {};
struct DynamicMerge {};
struct LeftReleaseEvent {};
struct RightClick {};
struct RightRelease {};

struct AddTagOnHoverEnter {};
struct HoverEnterEvent {};

struct AddTagOnHoverExit {};
struct HoverExitEvent {};

struct ServerHUDOverlay {};
struct ServerDescription
{
    ecs_entity_t selected;
};
struct ShowServerHUDOverlay {};
struct HideServerHUDOverlay {};

struct HighlightBFOInheritanceHierarchy {};
struct ResetBFOSprites {};

struct CloseEditorSelector {};
struct SetMenuHighlightColor {};
struct SetMenuStandardColor {};

struct ChatMessage {
    std::string author;
    std::string text;
};

struct ChatState {
    std::vector<ChatMessage> messages;
    std::string draft;
    bool input_focused;
};

struct ChatMessageView {
    int index;
};

struct FocusChatInput {};
struct SendChatMessage {};

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

// Known entities from previous interpretations (for context/binding)
struct KnownEntity {
    std::string id;
    std::string label;
    std::string color;  // Hex color string
    int display_number;  // The MNIST digit number for this entity
};

// Token types for annotated sentence
enum class TokenType {
    PlainText,       // Regular word
    Entity,          // {{text, n}} - entity binding
    Relationship     // {{text, R:src:tgt}} - relationship with source/target digits
};

// Token in annotated sentence
struct SentenceToken {
    std::string text;
    int binding_digit;  // -1 for unassigned, 0-9 for MNIST digit (entities)
    int source_digit;   // -1 for wildcard, 0-9 for MNIST digit (relationships)
    int target_digit;   // -1 for wildcard, 0-9 for MNIST digit (relationships)
    int reified_digit;  // -1 for not reified, 0-9 for reified relationship entity
    TokenType type;

    bool is_binding() const { return type != TokenType::PlainText; }
    // For selection: relationships have 3 selectable parts (source, rel, target)
    // Reified indicator is inside badge but not separately selectable
    int selection_width() const {
        return type == TokenType::Relationship ? 3 : 1;
    }
};

// Cache for entity colors (binding_digit -> color)
std::map<int, uint32_t> entity_color_cache;

// Get color via Unix socket to persistent Python server (fast after first call)
uint32_t get_entity_color(int binding_digit, const std::string& entity_text) {
    // Check cache first
    auto it = entity_color_cache.find(binding_digit);
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
        entity_color_cache[binding_digit] = 0x4d9be6FF;
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
            entity_color_cache[binding_digit] = 0x4d9be6FF;
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
        entity_color_cache[binding_digit] = 0x4d9be6FF;
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

    entity_color_cache[binding_digit] = color;
    return color;
}

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
                    int digit = -1;
                    int source_digit = -1;
                    int target_digit = -1;
                    int reified_digit = -1;

                    if (!spec.empty()) {
                        char prefix = spec[0];
                        if (prefix == 'R' || prefix == 'r') {
                            type = TokenType::Relationship;
                            // Check for reified format: R (src:tgt)reified
                            size_t paren_open = spec.find('(');
                            size_t paren_close = spec.find(')');
                            if (paren_open != std::string::npos && paren_close != std::string::npos && paren_close > paren_open) {
                                // Reified format: R (src:tgt)reified
                                std::string inner = spec.substr(paren_open + 1, paren_close - paren_open - 1);
                                std::string reified_str = spec.substr(paren_close + 1);
                                size_t colon = inner.find(':');
                                if (colon != std::string::npos) {
                                    std::string src_str = inner.substr(0, colon);
                                    std::string tgt_str = inner.substr(colon + 1);
                                    if (src_str != "*") try { source_digit = std::stoi(src_str); } catch (...) {}
                                    if (tgt_str != "*") try { target_digit = std::stoi(tgt_str); } catch (...) {}
                                }
                                if (!reified_str.empty()) try { reified_digit = std::stoi(reified_str); } catch (...) {}
                            } else {
                                // Non-reified format: R src:tgt
                                size_t space = spec.find(' ');
                                std::string rest = (space != std::string::npos) ? spec.substr(space + 1) : spec.substr(1);
                                // Trim whitespace
                                while (!rest.empty() && std::isspace(rest.front())) rest.erase(0, 1);
                                size_t colon = rest.find(':');
                                if (colon != std::string::npos) {
                                    std::string src_str = rest.substr(0, colon);
                                    std::string tgt_str = rest.substr(colon + 1);
                                    if (src_str != "*") try { source_digit = std::stoi(src_str); } catch (...) {}
                                    if (tgt_str != "*") try { target_digit = std::stoi(tgt_str); } catch (...) {}
                                }
                            }
                        } else {
                            try { digit = std::stoi(spec); } catch (...) {}
                        }
                    }
                    tokens.push_back({text, digit, source_digit, target_digit, reified_digit, type});
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
            tokens.push_back({sentence.substr(word_start, i - word_start), -1, -1, -1, -1, TokenType::PlainText});
        }
    }
    return tokens;
}

// Convert tokens back to template string
std::string tokens_to_template(const std::vector<SentenceToken>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) result += " ";
        if (tokens[i].is_binding()) {
            std::string spec;
            switch (tokens[i].type) {
                case TokenType::Relationship: {
                    std::string src = tokens[i].source_digit >= 0 ? std::to_string(tokens[i].source_digit) : "*";
                    std::string tgt = tokens[i].target_digit >= 0 ? std::to_string(tokens[i].target_digit) : "*";
                    if (tokens[i].reified_digit >= 0) {
                        // Reified format: R (src:tgt)reified
                        spec = "R (" + src + ":" + tgt + ")" + std::to_string(tokens[i].reified_digit);
                    } else {
                        // Non-reified format: R src:tgt
                        spec = "R " + src + ":" + tgt;
                    }
                    break;
                }
                default:
                    spec = tokens[i].binding_digit >= 0 ? std::to_string(tokens[i].binding_digit) : "*";
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
std::vector<KnownEntity> known_entities;
int next_entity_number = 1;  // Global counter for entity display numbers

// Previous sentences for context (last N sentences)
std::mutex previous_sentences_mutex;
std::vector<std::string> previous_sentences;
const int MAX_PREVIOUS_SENTENCES = 10;  // Keep last N sentences for context

// Particle animation for grid triangles - moving with velocity to collide at target
struct TriangleParticle {
    float targetX, targetY, targetZ;  // Final grid position (collision point)
    float localX, localY, localZ;     // Rotated local offset from tetrahedron centroid
    float vx, vy, vz;  // Velocity
    float collisionTime;  // When this particle reaches its target
    float u, v;  // UV coordinates (don't animate)
    float elapsedTime;  // Current time
    float hitTime;  // Time when particle first hit/locked (for glow effect)
    float baryX, baryY, baryZ;  // Barycentric coordinates for edge detection
    int vertexIndex;  // Which vertex in the buffer
    bool locked;  // Has reached target
    bool isCentral;  // Part of central triangle (for impact glow intensity)
    float pulseScale;  // Random scale variation for pulse (0.6 - 1.4)
    float pulseRotation;  // Random rotation for pulse asymmetry
};

// Noise tetrahedron that flies past without joining grid
struct NoiseTetrahedron {
    float x, y, z;      // Current position (centroid)
    float vz;           // Velocity towards camera
    float scale;        // Size multiplier
    // Random rotation (axis-angle)
    float axisX, axisY, axisZ;
    float rotAngle;
};

struct TimeEventRowChannel 
{   
    int scaleForMinimumCount;
};

struct CopyChildHeight {};

struct Graphics {
    NVGcontext* vg;

    // 3D rendering resources
    GLuint fbo;
    GLuint fboTexture;
    GLuint fboDepthRenderBuffer;
    GLuint planeVAO;
    GLuint planeVBO;
    GLuint planeEBO;
    GLuint gridVAO;
    GLuint gridVBO;
    GLuint gridEBO;
    GLuint shaderProgram;
    GLuint greyTexture;  // 1x1 grey texture for noise tetrahedrons
    float tiltAngle;
    int uiWidth;
    int uiHeight;
    bool useGridMode;
    float gridModeTransitionTimer;  // Timer for delayed transition to plane mode
    bool allParticlesLocked;        // Track if all particles are locked
    int gridVertexCount;

    // Particle system data
    std::vector<TriangleParticle> particles;
    std::vector<float> gridVertices;  // Store vertices for dynamic updates

    // Noise tetrahedrons (fly past, don't join grid)
    GLuint noiseVAO;
    GLuint noiseVBO;
    std::vector<NoiseTetrahedron> noiseParticles;
    std::vector<float> noiseVertices;
    int noiseVertexCount;

    // FTL deceleration state
    float decelerationTime;      // Time since start of deceleration
    float decelerationDuration;  // Total deceleration period
};

enum class EditorType
{
    Void,
    PeachCore,
    ImaginaryInterlocutor,
    VNCStream,
    Healthbar,
    // Respawn,
    // Genome,
    Embodiment,
    Vision,
    Hearing,
    Memory,
    Bookshelf,
    Episodic,
    BFO,
    SceneGraph,
    // SystemNavigator,

    // Bookshelf,
    // MelSpectrogram,
    // VNCStream,
    // CameraVideoFeed,
    // RobotActuators,
    // VirtualHumanoid,
    // EpisodicMemoryTimeline,
    // Reification,
    // ReadingRepresentation,
    // ProgramSynthesis,
    // Servers,
    // Healthbar,
    // Inventory,
    // Scheduling,
    // Chat,
    // ComputeProvisioning,
    // Backup
};

struct ParentClass {};
struct BFOSprite {};

// Node representing the editor area...
struct EditorNodeArea
{
    float width, height;
};

struct EditorShiftRegion
{
    UIElementBounds bounds;
    int cursor_type; // ex, GLFW_CROSSHAIR_CURSOR
    flecs::entity split_target;
};

struct EditorModifyPartitionRegion
{
    UIElementBounds bounds;
    flecs::entity split_target;
};

struct EditorRoot 
{
    std::vector<EditorShiftRegion> shift_regions;
    std::vector<EditorModifyPartitionRegion> modify_partition_regions;
};

struct EditorVisual {};
struct EditorHeader {};
struct EditorOutline {};
struct EditorCanvas {};
struct EditorLeaf {};

struct UpperNode {};
struct LowerNode {};

struct LeftNode {};
struct RightNode {};

struct Align
{
    float self_horizontal;
    float self_vertical;
    float horizontal;
    float vertical;
};

// Expand Rect or RoundedRect to UIElement bounds of parent
// with some padding
struct Expand
{
    bool x_enabled;
    // TODO: Use a padding primitive rather than placing these on expand...
    float pad_left, pad_right;
    float x_percent; // 0.0 to 1.0

    bool y_enabled;
    float pad_top, pad_bottom;
    float y_percent;

    // If true (ImageRenderable only), do not upscale beyond native image size
    // (after applying ImageRenderable.scaleX/scaleY).
    bool cap_to_intrinsic = false;
};

// This is a special system tag to reduce UIElementBounds for LayoutBox
struct ContractBoundsPostLayout {};

struct UIContainer
{
    int pad_horizontal;
    int pad_vertical;
};

// Post expand layer to 'fit within editor panel bounds'
struct Constrain
{
    bool fit_x; // Scale x to fit within bounds (maintain ratio)
    bool fit_y; // Scale y to fit within bounds (maintain ratio)
};

// What the actual fuck is this?
struct ProportionalConstraint {
    float max_width;
    float max_height;
};

// This needs to be refactored to be a direct enum relationship in flecs once you grow up and become a competent person
struct EditorLeafData
{
    EditorType editor_type;
};

struct FilmstripChannel{};
// Relationship tag to link mel spec display to its source renderer entity
struct MelSpecSource{};
// Determine how to chunk and choose which frames to display

enum class FilmstripMode {
    Uniform,     // Frames added at fixed time intervals, adjacent layout
    Stegosaurus  // Frames added on DINO spikes, time-based positioning with overlaps
};

// Component to track when a filmstrip frame was captured (for time-based positioning)
struct FilmstripFrameTime {
    double capture_time;  // glfwGetTime() when captured
};

struct FilmstripData
{
    // TODO: This should probably be a dynamic value based on container width...
    int frame_limit;
    // Should they be stored as entities or paths?
    std::vector<flecs::entity> frames;
    // Scroll tracking for realtime left-to-right scroll over 24 seconds
    float scroll_offset = 0.0f;  // Current scroll progress (0.0 to 1.0 of one frame width)
    float elapsed_time = 0.0f;   // Time elapsed in current scroll cycle
    size_t total_frames_added = 0;  // Counter incremented each time a frame is pushed
    size_t last_seen_frame_count = 0;  // To detect when total_frames_added changes
    static constexpr float SCROLL_DURATION = 24.0f;  // Seconds for full container to scroll

    // Mode selection
    FilmstripMode mode = FilmstripMode::Uniform;

    // Stegosaurus mode parameters
    float spike_threshold = 0.15f;   // Minimum cosDiff to trigger a spike (0.0-1.0)
    float last_dino_value = 0.0f;    // Last DINO cosDiff value seen
    float spike_cooldown = 1.0f;     // Minimum seconds between spike captures
    float time_since_spike = 0.0f;   // Time since last spike capture
    bool pending_capture = false;    // Flag to signal capture should happen
    double pending_spike_time = 0.0; // Time when the spike was detected (for accurate frame positioning)
};

// Line chart channel for streaming data visualization (e.g., DINO cosine similarity)
struct LineChartData
{
    std::vector<float> values;  // Circular buffer of values (0.0 to 1.0 normalized)
    size_t write_pos = 0;       // Current write position in circular buffer
    size_t capacity = 0;        // Max number of data points
    float min_value = 0.0f;     // Min value for normalization
    float max_value = 1.0f;     // Max value for normalization
    uint32_t fill_color = 0xFFFFFF40;  // RGBA fill color (semi-transparent white)
    uint32_t line_color = 0xFFFFFFFF;  // RGBA line color (solid white)
    float sample_interval = 0.0f;  // Seconds between samples (0 = every frame)
    float time_since_sample = 0.0f;  // Time accumulator for sampling
    float scroll_offset = 0.0f;  // Smooth scroll offset (0.0 to 1.0 of one sample width)
    static constexpr float WINDOW_DURATION = 24.0f;  // Total time window in seconds (matches filmstrip)

    void push(float value) {
        if (capacity == 0) return;
        if (values.size() < capacity) {
            values.push_back(value);
        } else {
            values[write_pos] = value;
        }
        write_pos = (write_pos + 1) % capacity;
        scroll_offset = 0.0f;  // Reset scroll when new sample is added
    }

    // Get value at index (0 = oldest, size-1 = newest)
    float get(size_t index) const {
        if (values.empty() || index >= values.size()) return 0.0f;
        if (values.size() < capacity) {
            return values[index];
        }
        return values[(write_pos + index) % capacity];
    }

    size_t size() const { return values.size(); }
};

// Tag for LineChart channel type
struct LineChartChannel{};

enum class PanelSplitType
{
    Horizontal,
    Vertical
};

struct DragContext {
    bool active;
    flecs::entity target;
    PanelSplitType dim;
    float startPercent;
};

struct PanelSplit 
{
    // use percent instead of pixel count for proportional expansion when
    // window size changes
    float percent; // 0 to 100
    PanelSplitType dim;
};

enum class RenderType {
    Rectangle,
    RoundedRectangle,
    Text,
    Image,
    Line,
    QuadraticBezier,
    CustomRenderable,
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

// ============================================================================
// SFTP File Transfer
// ============================================================================

#include "sftp_client.h"
#include <libssh2.h>
#include <libssh2_sftp.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <unistd.h>

// SFTP connection function - establishes SSH and SFTP session
bool sftp_connect(SFTPClient* sftp) {
    LOG_INFO(LogCategory::SFTP, "Connecting to {}:{} (ptr: {})", sftp->host, sftp->port, (void*)sftp);

    sftp->conn_state = SFTPClient::CONNECTING;

    // 1. Create socket and connect to host:port
    LOG_INFO(LogCategory::SFTP, "Creating socket...");
    sftp->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sftp->sock < 0) {
        sftp->conn_state = SFTPClient::ERROR;
        LOG_ERROR(LogCategory::SFTP, "Failed to create socket");
        return false;
    }
    LOG_INFO(LogCategory::SFTP, "Socket created: {}", sftp->sock);

    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_port = htons(sftp->port);
    sin.sin_addr.s_addr = inet_addr(sftp->host.c_str());

    LOG_INFO(LogCategory::SFTP, "Connecting socket to {}:{}...", sftp->host, sftp->port);
    if (connect(sftp->sock, (struct sockaddr*)&sin, sizeof(struct sockaddr_in)) != 0) {
        sftp->conn_state = SFTPClient::ERROR;
        LOG_ERROR(LogCategory::SFTP, "Failed to connect to {}:{} - {}", sftp->host, sftp->port, strerror(errno));
        close(sftp->sock);
        sftp->sock = -1;
        return false;
    }
    LOG_INFO(LogCategory::SFTP, "Socket connected successfully");

    // 2. Initialize libssh2 session
    LOG_INFO(LogCategory::SFTP, "Initializing SSH session...");
    sftp->session = libssh2_session_init();
    if (!sftp->session) {
        sftp->conn_state = SFTPClient::ERROR;
        LOG_ERROR(LogCategory::SFTP, "Failed to initialize SSH session");
        close(sftp->sock);
        sftp->sock = -1;
        return false;
    }
    LOG_INFO(LogCategory::SFTP, "SSH session initialized");

    // Set blocking mode
    LOG_INFO(LogCategory::SFTP, "Setting blocking mode...");
    libssh2_session_set_blocking(sftp->session, 1);

    // 3. Perform SSH handshake
    LOG_INFO(LogCategory::SFTP, "Starting SSH handshake...");
    int rc = libssh2_session_handshake(sftp->session, sftp->sock);
    if (rc) {
        sftp->conn_state = SFTPClient::ERROR;
        LOG_ERROR(LogCategory::SFTP, "SSH handshake failed (error {})", rc);
        libssh2_session_free(sftp->session);
        sftp->session = nullptr;
        close(sftp->sock);
        sftp->sock = -1;
        return false;
    }
    LOG_INFO(LogCategory::SFTP, "SSH handshake completed");

    const char * fingerprint = libssh2_hostkey_hash(sftp->session, LIBSSH2_HOSTKEY_HASH_SHA1);

    fprintf(stderr, "Fingerprint: ");
    for(int i = 0; i < 20; i++) 
    {
        fprintf(stderr, "%02X ", (unsigned char)fingerprint[i]);
    }
    fprintf(stderr, "\n");

    // 4. Authenticate using password
    LOG_INFO(LogCategory::SFTP, "Authenticating user: {}", sftp->username);
    rc = libssh2_userauth_password(sftp->session,
        sftp->username.c_str(),
        sftp->password.c_str());

    if (rc) {
        sftp->conn_state = SFTPClient::ERROR;
        LOG_ERROR(LogCategory::SFTP, "SSH password authentication failed (user: {}, error: {})", sftp->username, rc);
        libssh2_session_disconnect(sftp->session, "Auth failed");
        libssh2_session_free(sftp->session);
        sftp->session = nullptr;
        close(sftp->sock);
        sftp->sock = -1;
        return false;
    }
    LOG_INFO(LogCategory::SFTP, "Authentication successful");

    // 5. Initialize SFTP subsystem
    sftp->sftp_session = libssh2_sftp_init(sftp->session);
    if (!sftp->sftp_session) {
        sftp->conn_state = SFTPClient::ERROR;
        LOG_ERROR(LogCategory::SFTP, "Failed to initialize SFTP session");
        libssh2_session_disconnect(sftp->session, "SFTP init failed");
        libssh2_session_free(sftp->session);
        sftp->session = nullptr;
        close(sftp->sock);
        sftp->sock = -1;
        return false;
    }

    sftp->conn_state = SFTPClient::CONNECTED;
    LOG_INFO(LogCategory::SFTP, "SFTP connected successfully to {}@{}", sftp->username, sftp->host);

    return true;
}

// SFTP worker thread - processes file transfer queue
void sftp_worker_thread(SFTPClient* sftp) {
    LOG_INFO(LogCategory::SFTP, "SFTP thread started for {} (ptr: {})", sftp->host, (void*)sftp);

    sftp->thread_running = true;

    while (!sftp->thread_should_stop) {
        // 1. Wait for transfer request (with timeout)
        FileTransferRequest request;
        {
            std::unique_lock<std::mutex> lock(sftp->queue_mutex);
            sftp->queue_cv.wait_for(lock, std::chrono::milliseconds(100),
                [sftp]() { return !sftp->transfer_queue.empty() || sftp->thread_should_stop; });

            if (sftp->transfer_queue.empty()) continue;
            request = sftp->transfer_queue.front();
            sftp->transfer_queue.pop_front();
        }

        // 2. Ensure connection (lazy connect)
        if (sftp->conn_state != SFTPClient::CONNECTED) {
            if (!sftp_connect(sftp)) {
                // Update progress with error
                std::lock_guard<std::mutex> plock(sftp->progress_mutex);
                sftp->current_progress.state = FileTransferProgress::FAILED;
                sftp->current_progress.error_message = "Connection failed (see logs)";
                continue;
            }
        }

        LOG_INFO(LogCategory::SFTP, "Transferring: {} -> {}", request.local_path, request.remote_path);

        // 3. Open local file
        FILE* local_file = fopen(request.local_path.c_str(), "rb");
        if (!local_file) {
            std::string error = "Cannot open local file: " + request.local_path;
            LOG_ERROR(LogCategory::SFTP, "{}", error);
            std::lock_guard<std::mutex> lock(sftp->progress_mutex);
            sftp->current_progress.state = FileTransferProgress::FAILED;
            sftp->current_progress.error_message = error;
            continue;
        }

        // 4. Open remote file via SFTP
        LIBSSH2_SFTP_HANDLE* sftp_handle = libssh2_sftp_open(
            sftp->sftp_session, request.remote_path.c_str(),
            LIBSSH2_FXF_WRITE | LIBSSH2_FXF_CREAT | LIBSSH2_FXF_TRUNC,
            LIBSSH2_SFTP_S_IRUSR | LIBSSH2_SFTP_S_IWUSR | LIBSSH2_SFTP_S_IRGRP | LIBSSH2_SFTP_S_IROTH);

        if (!sftp_handle) {
            unsigned long sftp_err = libssh2_sftp_last_error(sftp->sftp_session);
            std::string error = "Cannot open remote file (SFTP error " + std::to_string(sftp_err) + "): " + request.remote_path;
            LOG_ERROR(LogCategory::SFTP, "{}", error);
            fclose(local_file);
            std::lock_guard<std::mutex> lock(sftp->progress_mutex);
            sftp->current_progress.state = FileTransferProgress::FAILED;
            sftp->current_progress.error_message = error;
            continue;
        }

        // 5. Transfer file in chunks, updating progress
        {
            std::lock_guard<std::mutex> lock(sftp->progress_mutex);
            sftp->current_progress.state = FileTransferProgress::TRANSFERRING;
            sftp->current_progress.bytes_transferred = 0;
        }

        char buffer[32768];
        size_t total_written = 0;
        bool transfer_failed = false;

        while (!feof(local_file) && !sftp->thread_should_stop) {
            size_t nread = fread(buffer, 1, sizeof(buffer), local_file);
            if (nread <= 0) break;

            // Write to SFTP
            char* ptr = buffer;
            size_t remaining = nread;
            while (remaining > 0) {
                ssize_t nwritten = libssh2_sftp_write(sftp_handle, ptr, remaining);
                if (nwritten < 0) {
                    std::string error = "SFTP write failed (error " + std::to_string(nwritten) + ")";
                    LOG_ERROR(LogCategory::SFTP, "{}", error);
                    std::lock_guard<std::mutex> lock(sftp->progress_mutex);
                    sftp->current_progress.state = FileTransferProgress::FAILED;
                    sftp->current_progress.error_message = error;
                    transfer_failed = true;
                    break;
                }
                ptr += nwritten;
                remaining -= nwritten;
                total_written += nwritten;

                // Update progress (thread-safe)
                {
                    std::lock_guard<std::mutex> lock(sftp->progress_mutex);
                    sftp->current_progress.bytes_transferred = total_written;
                    sftp->current_progress.progress_percent =
                        (float)total_written / request.file_size * 100.0f;
                }
            }

            if (transfer_failed) break;
        }

        // 6. Close handles
        libssh2_sftp_close(sftp_handle);
        fclose(local_file);

        // 7. Mark completed or failed
        if (!transfer_failed && !sftp->thread_should_stop) {
            std::lock_guard<std::mutex> lock(sftp->progress_mutex);
            sftp->current_progress.state = FileTransferProgress::COMPLETED;
            sftp->current_progress.completion_time = std::chrono::steady_clock::now();
            sftp->current_progress.progress_percent = 100.0f;
            LOG_INFO(LogCategory::SFTP, "Transfer completed: {} ({} bytes)",
                request.remote_path, total_written);
        }
    }

    // Cleanup connection
    if (sftp->sftp_session) {
        libssh2_sftp_shutdown(sftp->sftp_session);
        sftp->sftp_session = nullptr;
    }
    if (sftp->session) {
        libssh2_session_disconnect(sftp->session, "Shutdown");
        libssh2_session_free(sftp->session);
        sftp->session = nullptr;
    }
    if (sftp->sock >= 0) {
        close(sftp->sock);
        sftp->sock = -1;
    }

    sftp->thread_running = false;
    LOG_INFO(LogCategory::SFTP, "SFTP thread stopped for {}", sftp->host);
}

// End SFTP File Transfer

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

    bool operator<(const RenderCommand& other) const {
        return zIndex < other.zIndex;
    }
};


void draw_double_arrow(NVGcontext* vg, const RenderCommand* cmd, const CustomRenderable& data)
{
    float x = cmd->pos.x;
    float y = cmd->pos.y;

    float leftArrowStartX = x + data.height/2.0f;
    float rightArrowStartX = x + data.width-data.height/2.0f;
    float midY = y + data.height / 2.0f;

    nvgBeginPath(vg);

    nvgMoveTo(vg, leftArrowStartX, y);
    nvgLineTo(vg, x, midY);
    nvgLineTo(vg, leftArrowStartX, y+data.height);
    nvgLineTo(vg, rightArrowStartX, y+data.height);
    nvgLineTo(vg, x+data.width, midY);
    nvgLineTo(vg, rightArrowStartX, y);

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

// Forward declaration for vector-based version
flecs::entity create_badge_impl(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule, bool is_double_arrow,
                           const std::vector<std::string>& prefix_ids,
                           const std::vector<uint32_t>& prefix_tints,
                           const std::vector<std::string>& postfix_ids,
                           const std::vector<uint32_t>& postfix_tints);

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
                           const std::vector<uint32_t>& postfix_tints) {

    // --- 1. Color Logic (matching comp_gen.py) ---
    uint32_t dark = base_color;
    uint32_t very_dark = scale_color(base_color, 0.2f);
    uint32_t light = scale_color(base_color, 1.3f); // Clamped to 255 in scale_color
    uint32_t white = 0xFFFFFFFF;

    // Outline: Light variation with 50% alpha (128)
    uint32_t outline_color = (light & 0xFFFFFF00) | 0x80;

    // --- 2. Dimensions & Shape ---
    float corner_radius = 4.0f;
    float badge_height = 25.0f; // Base height

    if (is_capsule) {
        corner_radius = badge_height / 2.0f;
    }

    // --- 3. Create Entities ---

    flecs::entity badge = flecs::entity::null();

    float xPad = 6.0f;

    if (is_double_arrow)
    {
        xPad += 25.0f/2;
        badge = world->entity()
            .is_a(UIElement)
            .child_of(parent)
            .set<CustomRenderable>({100.0f, 25.0f, true, outline_color, 0, 0, draw_double_arrow})
            .set<RenderGradient>({dark, very_dark}) // Vertical gradient
            .set<UIContainer>({xPad, 0})
            .set<ZIndex>({20});

    } else
    {
        badge = world->entity()
            .is_a(UIElement)
            .child_of(parent)
            .set<RoundedRectRenderable>({100.0f, badge_height, corner_radius, false, 0x000000FF})
            .set<RenderGradient>({dark, very_dark}) // Vertical gradient
            .set<UIContainer>({xPad, 0})
            .set<ZIndex>({20});

        // Outline Overlay
        world->entity()
            .is_a(UIElement)
            .child_of(badge)
            .set<Expand>({true, 0, 0, 1.0f, true, 0, 0, 1.0f})
            .set<RoundedRectRenderable>({100.0f, badge_height, corner_radius, true, outline_color})
            .set<ZIndex>({22});
    }

    // Helper lambda to create MNIST digit with tint
    // Helper lambda to create MNIST digit or wildcard image
    auto create_slot_image = [&](flecs::entity parent_entity, const std::string& symbol, uint32_t tint) {
        uint32_t tint_color = scale_color(tint, 1.3f);
        unsigned char r = (tint_color >> 24) & 0xFF;
        unsigned char g = (tint_color >> 16) & 0xFF;
        unsigned char b = (tint_color >> 8) & 0xFF;
        unsigned char a = (tint_color) & 0xFF;

        std::string image_path;
        if (symbol == "*") {
            // Wildcard - use wildcard.png
            image_path = "wildcard.png";
        } else {
            // Standard MNIST digit
            image_path = "mnist/set_0/" + symbol + ".png";
        }

        world->entity()
            .is_a(UIElement)
            .child_of(parent_entity)
            .set<ImageCreator>({image_path, 0.9f, 0.9f, nvgRGBA(r, g, b, a)})
            .set<ZIndex>({25});
    };

    // Helper lambda to create text element
    auto create_text_element = [&](flecs::entity parent_entity, const char* txt, uint32_t color_val) {
        world->entity()
            .is_a(UIElement)
            .child_of(parent_entity)
            .set<Position, Local>({0.0f, 6.0f})
            .set<TextRenderable>({txt, "Inter", 16.0f, color_val})
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
            create_slot_image(parent_entity, ids[i], tint);
        }

        if (is_set) {
            create_text_element(parent_entity, "} ", white);
        }
    };

    // Create content container if we have prefix or postfix
    flecs::entity badge_text_parent = badge;
    if (!prefix_ids.empty() || !postfix_ids.empty())
    {
        auto badge_content = world->entity()
            .is_a(UIElement)
            .set<LayoutBox>({LayoutBox::Horizontal, 0.0f})
            .set<Position, Local>({xPad, 0.0f})
            .add(flecs::OrderedChildren)
            .child_of(badge);

        badge_text_parent = badge_content;
        badge.set<UIContainer>({xPad, 0});
    }

    // Render prefix IDs (sources)
    render_id_set(badge_text_parent, prefix_ids, prefix_tints);

    // Text with Gradient
    world->entity()
        .is_a(UIElement)
        .child_of(badge_text_parent)
        .set<Position, Local>({xPad, 6.0f})
        .set<TextRenderable>({text, "Inter", 16.0f, white, 1.2f})
        .set<RenderGradient>({white, light})
        .set<ZIndex>({25});

    // Render postfix IDs (targets)
    render_id_set(badge_text_parent, postfix_ids, postfix_tints);

    return badge;
}

// Recreate UI entities from the template string for a WordAnnotationSelector
// Call this from key handlers instead of creating entities in a per-frame system
void recreate_annotation_entities(WordAnnotationSelector& selector) {
    if (!selector.parent_entity.is_valid()) return;

    // Store old entities to delete after creating new ones (prevents flicker)
    std::vector<flecs::entity> old_entities = std::move(selector.ui_entities);
    selector.ui_entities.clear();
    selector.selection_entities.clear();

    // Parse template and create new entities
    auto tokens = parse_sentence_template(selector.sentence_template);
    auto UIElement = world->lookup("UIElement");

    // Calculate token_count as sum of selection widths
    selector.token_count = 0;
    for (const auto& token : tokens) {
        selector.token_count += token.selection_width();
    }

    for (size_t ti = 0; ti < tokens.size(); ti++) {
        const auto& token = tokens[ti];

        if (token.type == TokenType::Relationship) {
            // Create relationship badge with source/target images as children
            // Note: ImageCreator observer prepends "../assets/" so paths are relative to that

            // Look up source and target entity colors first for outline averaging
            uint32_t src_color = 0xc72783FF; // default pink
            uint32_t tgt_color = 0xc72783FF;
            if (token.source_digit >= 0) {
                auto it = entity_color_cache.find(token.source_digit);
                if (it != entity_color_cache.end()) {
                    src_color = it->second;
                }
            }
            if (token.target_digit >= 0) {
                auto it = entity_color_cache.find(token.target_digit);
                if (it != entity_color_cache.end()) {
                    tgt_color = it->second;
                }
            }

            // Average the RGB components for outline color
            uint8_t avg_r = (((src_color >> 24) & 0xFF) + ((tgt_color >> 24) & 0xFF)) / 2;
            uint8_t avg_g = (((src_color >> 16) & 0xFF) + ((tgt_color >> 16) & 0xFF)) / 2;
            uint8_t avg_b = (((src_color >> 8) & 0xFF) + ((tgt_color >> 8) & 0xFF)) / 2;
            uint32_t avg_color = (avg_r << 24) | (avg_g << 16) | (avg_b << 8) | 0xFF;

            // 1. Create the double arrow badge container
            uint32_t base_color = avg_color;
            uint32_t dark = base_color;
            uint32_t very_dark = scale_color(base_color, 0.2f);
            uint32_t light = scale_color(base_color, 1.3f);
            uint32_t outline_color = (light & 0xFFFFFF00) | 0x80;
            float xPad = 6.0f + 25.0f/2;

            flecs::entity badge = world->entity()
                .is_a(UIElement)
                .child_of(selector.parent_entity)
                .set<CustomRenderable>({100.0f, 25.0f, true, outline_color, 0, 0, draw_double_arrow})
                .set<RenderGradient>({dark, very_dark})
                .set<UIContainer>({xPad, 0})
                .set<ZIndex>({20});

            // 2. Create content container with horizontal layout
            flecs::entity badge_content = world->entity()
                .is_a(UIElement)
                .set<LayoutBox>({LayoutBox::Horizontal, 0.0f})
                .set<Position, Local>({xPad, 0.0f})
                .add(flecs::OrderedChildren)
                .child_of(badge);

            // 3. Source MNIST/wildcard image (child of badge_content)
            std::string src_path = token.source_digit >= 0
                ? "mnist/set_0/" + std::to_string(token.source_digit) + ".png"
                : "wildcard.png";
            // Tint with entity color if bound (reuse src_color from earlier lookup)
            NVGcolor src_tint = nvgRGBA((src_color >> 24) & 0xFF, (src_color >> 16) & 0xFF, (src_color >> 8) & 0xFF, 255);
            flecs::entity source_ent = world->entity()
                .is_a(UIElement)
                .child_of(badge_content)
                .set<ImageCreator>({src_path, 1.0f, 1.0f, src_tint})
                .set<ZIndex>({25});

            // 4. Relationship text (with optional reified indicator above)
            flecs::entity text_parent = badge_content;
            if (token.reified_digit >= 0) {
                // Create vertical container for reified indicator + text
                flecs::entity text_column = world->entity()
                    .is_a(UIElement)
                    .set<LayoutBox>({LayoutBox::Vertical, 0.0f})
                    .add(flecs::OrderedChildren)
                    .child_of(badge_content);

                // Reified indicator (3d_node + digit) above text
                flecs::entity reified_row = world->entity()
                    .is_a(UIElement)
                    .set<LayoutBox>({LayoutBox::Horizontal, 0.0f})
                    .add(flecs::OrderedChildren)
                    .child_of(text_column);

                // Tint reified with entity color if it exists
                NVGcolor reified_tint = nvgRGBA(255, 255, 255, 255);
                if (token.reified_digit >= 0) {
                    auto it = entity_color_cache.find(token.reified_digit);
                    if (it != entity_color_cache.end()) {
                        uint32_t c = it->second;
                        reified_tint = nvgRGBA((c >> 24) & 0xFF, (c >> 16) & 0xFF, (c >> 8) & 0xFF, 255);
                    }
                }

                world->entity()
                    .is_a(UIElement)
                    .child_of(reified_row)
                    .set<ImageCreator>({"3d_node.png", 0.6f, 0.6f, reified_tint})
                    .set<ZIndex>({25});

                std::string reified_path = "mnist/set_0/" + std::to_string(token.reified_digit) + ".png";
                world->entity()
                    .is_a(UIElement)
                    .child_of(reified_row)
                    .set<ImageCreator>({reified_path, 0.6f, 0.6f, reified_tint})
                    .set<ZIndex>({25});

                text_parent = text_column;
            }

            flecs::entity text_ent = world->entity()
                .is_a(UIElement)
                .child_of(text_parent)
                .set<Position, Local>({0.0f, token.reified_digit >= 0 ? 0.0f : 6.0f})
                .set<TextRenderable>({token.text.c_str(), "Inter", 16.0f, 0xFFFFFFFF, 1.2f})
                .set<RenderGradient>({0xFFFFFFFF, light})
                .set<ZIndex>({25});

            // 5. Target MNIST/wildcard image (child of badge_content)
            std::string tgt_path = token.target_digit >= 0
                ? "mnist/set_0/" + std::to_string(token.target_digit) + ".png"
                : "wildcard.png";
            // Tint with entity color if bound (reuse tgt_color from earlier lookup)
            NVGcolor tgt_tint = nvgRGBA((tgt_color >> 24) & 0xFF, (tgt_color >> 16) & 0xFF, (tgt_color >> 8) & 0xFF, 255);
            flecs::entity target_ent = world->entity()
                .is_a(UIElement)
                .child_of(badge_content)
                .set<ImageCreator>({tgt_path, 1.0f, 1.0f, tgt_tint})
                .set<ZIndex>({25});

            // Only badge needs to be in ui_entities (children deleted with parent)
            selector.ui_entities.push_back(badge);
            // Selection entities track 3 parts: source, badge, target
            // Reified indicator is inside badge but not separately selectable
            selector.selection_entities.push_back(source_ent);
            selector.selection_entities.push_back(badge);
            selector.selection_entities.push_back(target_ent);

        } else if (token.type == TokenType::Entity) {
            // Entity binding: normal badge with semantic color
            std::string digit_str = token.binding_digit >= 0 ? std::to_string(token.binding_digit) : "";
            uint32_t entity_color = token.binding_digit >= 0
                ? get_entity_color(token.binding_digit, token.text)
                : 0x4488FFFF;
            flecs::entity badge = create_badge(selector.parent_entity, UIElement,
                token.text.c_str(), entity_color,
                false, false,
                digit_str, "",
                0, entity_color);
            selector.ui_entities.push_back(badge);
            selector.selection_entities.push_back(badge);
        } else {
            // Plain text entity
            flecs::entity text_ent = world->entity()
                .is_a(UIElement)
                .child_of(selector.parent_entity)
                .set<TextRenderable>({token.text.c_str(), "Inter", 16.0f, 0x777777FF})
                .set<ZIndex>({17});
            selector.ui_entities.push_back(text_ent);
            selector.selection_entities.push_back(text_ent);
        }
    }

    // Delete old entities after new ones are created (prevents flicker)
    for (auto& ent : old_entities) {
        if (ent.is_valid()) {
            ent.destruct();
        }
    }

    selector.dirty = false;
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

    // TODO: Editor type entities...

    // if (leaf.has<EditorLeafData>() && leaf.get<EditorLeafData>().editor_type == EditorType::Bookshelf)
    // {
    // For bookshelf, we should create the horizontal layout with all the books

}

// TODO: load this from config file
std::vector<std::string> editor_types = 
{
    "Void",
    // "ECS Graph", // Entity component relationship
    "Peach Core",
    "Interlocutor", // Queue or Stream
    "VNC Stream",
    "Healthbar",
    // "Gynoid",
    "Embodiment",
    "Vision",
    "Hearing",
    "Memory",
    "Bookshelf",
    "Episodic",
    "BFO",
    "Scene Graph",
};

struct VNCData
{
    flecs::entity vnc_stream;
    VNCClientHandle client;  // Stable handle that survives ECS moves
};

VNCData get_vnc_source(const std::string& host, int port) {
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

struct ScissorContainer {};
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

// Factory function to populate editor content, whether the panel is initialized or changed
void create_editor_content(flecs::entity leaf, EditorType editor_type, flecs::entity UIElement)
{
    std::cout << "Change panel type to " << editor_types[(int)editor_type] << std::endl;
    if (editor_type == EditorType::PeachCore)
    {

        auto server_hud = world->entity("ServerHUD")
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 0.0f})
        .add<FitChildren>()
        .set<Expand>({true, 4.0f, 4.0f, 1.0f, true, 4.0f, 4.0f, 1.0f})
        .add(flecs::OrderedChildren)
        .child_of(leaf.target<EditorCanvas>());

        auto panel_overlay = world->entity()
        .is_a(UIElement)
        .set<ZIndex>({15})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x000000DD})
        .add<ServerDescription>()
        .child_of(leaf.target<EditorCanvas>());

        std::vector<std::string> server_icons = {"aeri_memory", "claude", "flecs", "x11", "parakeet", "chatterbox", "doctr", "huggingface", "dino2", "alpaca", "modal"}; // , "autodistill", "yolo", 
        // std::vector<std::string> server_icons = {"peach_core"};

        for (const auto& icon : server_icons)
        {
            // .set<ServerScript>({"chatterbox", "chatterbox", "../chatter_server"})
            // .add(ServerStatus::Offline)
            // .add<AddTagOnLeftClick, SelectServer>(); 
            flecs::entity server_icon = world->entity()
            .is_a(UIElement)
            .add<ServerHUDOverlay>(panel_overlay)
            .add<AddTagOnHoverEnter, ShowServerHUDOverlay>()
            .add<AddTagOnHoverExit, HideServerHUDOverlay>()
            .child_of(server_hud)
            .set<Constrain>({true, true}) 
            .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f, true})
            .set<ImageCreator>({"../assets/server_hud/" + icon + ".png", 1.0f, 1.0f})
            .set<ZIndex>({10});
            // TODO: Server dot should only exist if the server is active...
            world->entity()
            .is_a(UIElement)
            .child_of(server_icon)
            .set<ImageCreator>({"../assets/server_dot.png", 1.0f, 1.0f})
            .set<Align>({-0.5f, -0.5f, 0.5f, 0.9f})
            .set<ZIndex>({12})
            .set<Expand>({false, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 1.0f, true});
        }
        
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
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        create_badge(badges, UIElement, "Healthbar", 0x61c300ff);
    }
    else if (editor_type == EditorType::Embodiment)
    {
        auto grey_bkg = world->entity()
        .is_a(UIElement)
        .set<ZIndex>({9})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x868485ff})
        .child_of(leaf.target<EditorCanvas>());

        auto profile = world->entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<ImageCreator>({"../assets/heonae_profile.png", 1.0f, 1.0f})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 1.0f})
        .set<Align>({-0.5f, -0.5f, 0.5f, 0.5f})
        .set<Constrain>({true, true})
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
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        
        create_badge(badges, UIElement, "Heonae", 0xc72783ff);
        // create_badge(badges, UIElement, "Kahlo", 0x782910ff);
        create_badge(badges, UIElement, "Virtual", 0xe575eeff);
        create_badge(badges, UIElement, "Physical", 0x619393ff);
        
    }
    else if (editor_type == EditorType::ImaginaryInterlocutor)
    {
        auto canvas = leaf.target<EditorCanvas>();

        auto chat_root = world->entity()
            .is_a(UIElement)
            .child_of(canvas)
            .set<ZIndex>({5});

        // auto input_text_bar = world->entity()
        //     .is_a(UIElement)
        //     .child_of(chat_root)
        //     .set<RoundedRectRenderable>({100.0f, 32.0f, 2.0f, false, 0x444444FF})
        //     .set<Expand>({true, 0.0f, 4.0f, 1.0f, false, 0, 0, 0})
        //     .set<ZIndex>({10});

        auto messages_panel = world->entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .set<RoundedRectRenderable>({100.0f, 100.0f, 4.0f, false, 0x050505FF})
            // TODO: Expand to fill remaining space in VerticalLayout...
            // .set<Expand>({true, 8.0f, 8.0f, 1.0f, true, 8.0f, 36.0f, })
            .set<ZIndex>({11});

        auto input_panel = world->entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .set<RoundedRectRenderable>({100.0f, 36.0f, 2.0f, true, 0x555555FF})
            .add<AddTagOnLeftClick, FocusChatInput>()
            .set<ZIndex>({11});

            // TODO: We need to scale the input bkg to the text/content size...
        auto input_bkg = world->entity()
            .is_a(UIElement)
            .child_of(input_panel)
            .set<RoundedRectRenderable>({10.0f, 10.0f, 2.0f, false, 0x222327FF})
            .set<Expand>({true, 0, 0, 1, true, 0, 0, 1})
            .set<ZIndex>({10});

        auto input_text = world->entity()
            .is_a(UIElement)
            // .add<DebugRenderBounds>()
            .child_of(input_panel)
            .set<Position, Local>({8.0f, 8.0f})
            .set<TextRenderable>({"", "Inter", 16.0f, 0xFFFFFFFF})
            .set<ZIndex>({12});

        auto message_list = world->entity()
            .is_a(UIElement)
            .child_of(messages_panel)
            .add(flecs::OrderedChildren)
            .set<Position, Local>({12.0f, 16.0f})
            .set<LayoutBox>({LayoutBox::Vertical, 4.0f, 1.0f})
            .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 0.0f});

        auto msg_container = world->entity()
        .is_a(UIElement)
        .set<UIContainer>({4, 4})
        .child_of(message_list);

        auto meta_input = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .add(flecs::OrderedChildren)
        // .add<DebugRenderBounds>()
        .child_of(message_list);

        auto black_bkg = world->entity()
        .is_a(UIElement)
        .set<ZIndex>({9})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x000000FF})
        .child_of(leaf.target<EditorCanvas>());
        
        create_badge(meta_input, UIElement, "Wesley", 0xf5a652ff, false, false, "0");

        create_badge(meta_input, UIElement, "types", 0xa34d1aff, false, true);



        leaf.set<ChatPanel>({messages_panel, input_panel, input_text, message_list});
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
            "cover_james.jpg", 
            "cover_cognitive_theory.jpg", 
            "cover_soar.jpg", 
            "cover_readings_in_kr.jpg"
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
        .set<Position, Local>({48.0f, 0.0f})
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
        auto badges = world->entity()
        .is_a(UIElement)
        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());

        create_badge(badges, UIElement, "Docker", 0x1d60e6ff);
        create_badge(badges, UIElement, "plasma-productivity-1", 0x9740f6ff, true);
        create_badge(badges, UIElement, "192.168.1.104", 0xa7a7a7ff, true);
        create_badge(badges, UIElement, "5901", 0xf64242ff, true);
        create_badge(badges, UIElement, "Kubuntu", 0xe2521fff);
        create_badge(badges, UIElement, "22.04", 0xe2521fff, true);

        const char* vnc_host = getenv("VNC_SERVER_HOST");
        if (!vnc_host) {
            vnc_host = "localhost";
            vnc_host = "192.168.1.104";
        }
        std::cout << "[VNC] VNC server host: " << vnc_host << " (ports 5901-5904)" << std::endl;
        
        int port = 5901;
        std::string host_string = std::string(vnc_host) + ":" + std::to_string(port);

        VNCData data = get_vnc_source(vnc_host, port);
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
        .set<Position, Local>({48.0f, 0.0f})
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
        // Capacity: ~240 samples for 24 seconds at ~10 samples/second (matching DINO inference rate)
        {
            LineChartData chartData;
            chartData.capacity = 240;
            chartData.min_value = 0.0f;
            chartData.max_value = 1.0f;
            chartData.fill_color = 0xFFFFFF30;  // Semi-transparent white fill
            chartData.line_color = 0xFFFFFFFF;  // Solid white line
            chartData.sample_interval = 0.1f;   // Sample every 100ms

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
            world->entity()
                .is_a(UIElement)
                .child_of(channels)
                .set<RectRenderable>({10.0f, 24.0f, false, i % 2 == 0 ? 0x222327FF : 0x121212FF })
                .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
                .add<ScissorContainer>(leaf.target<EditorCanvas>())
                .add<TimeEventRowChannel>()
                .set<ZIndex>({12});
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

    void addRoundedRectCommand(const Position& pos, const RoundedRectRenderable& renderable, int zIndex, bool useGradient = false, RenderGradient renderGradient = {0, 0}) {
        commands.push_back({pos, renderable, RenderType::RoundedRectangle, zIndex, 0, useGradient, renderGradient});
    }

    void addTextCommand(const Position& pos, const TextRenderable& renderable, int zIndex, bool useGradient = false, RenderGradient renderGradient = {0, 0}) {
        commands.push_back({pos, renderable, RenderType::Text, zIndex, 0, useGradient, renderGradient});
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
        std::sort(commands.begin(), commands.end());
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
    if (!glfw_state.is_valid()) {
        LOG_WARN(LogCategory::SFTP, "File drop ignored - no GLFWState entity");
        return;
    }

    const CursorState* cursorState = glfw_state.try_get<CursorState>();
    if (!cursorState) {
        LOG_WARN(LogCategory::SFTP, "File drop ignored - no cursor state");
        return;
    }

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

    if (!target_vnc_entity.is_valid()) {
        LOG_INFO(LogCategory::SFTP, "File drop ignored - no VNC panel under mouse");
        return;
    }

    // 3. Get/create SFTPClient component
    SFTPClient& sftp = target_vnc_entity.ensure<SFTPClient>();

    // 4. Initialize worker thread if first time
    if (!sftp.thread_running) {
        const VNCClientHandle* handle = target_vnc_entity.try_get<VNCClientHandle>();
        if (!handle || !*handle) {
            LOG_ERROR(LogCategory::SFTP, "VNC entity has no VNCClientHandle component");
            return;
        }
        const VNCClient* vnc = handle->get();

        sftp.host = vnc->host;
        sftp.port = 23;
        sftp.username = "grok";
        sftp.password = "GrokValentine42!";
        sftp.thread_should_stop = false;

        LOG_INFO(LogCategory::SFTP, "Starting SFTP worker thread for {} (SFTPClient at {})", sftp.host, (void*)&sftp);
        sftp.worker_thread = std::thread(sftp_worker_thread, &sftp);

        LOG_INFO(LogCategory::SFTP, "Started SFTP worker thread for {}", sftp.host);
    }

    // 5. Queue file transfers
    for (int i = 0; i < count; i++) {
        std::string local_path = paths[i];
        std::string filename = local_path.substr(local_path.find_last_of("/\\") + 1);
        std::string remote_path = "/home/grok/Downloads/" + filename;

        // Get file size
        struct stat st;
        if (stat(local_path.c_str(), &st) != 0) {
            LOG_ERROR(LogCategory::SFTP, "Cannot stat file: {}", local_path);
            continue;
        }

        FileTransferRequest request;
        request.local_path = local_path;
        request.remote_path = remote_path;
        request.file_size = st.st_size;

        {
            std::lock_guard<std::mutex> lock(sftp.queue_mutex);
            sftp.transfer_queue.push_back(request);

            // Initialize progress for first file
            if (sftp.transfer_queue.size() == 1) {
                std::lock_guard<std::mutex> plock(sftp.progress_mutex);
                sftp.current_progress.filename = filename;
                sftp.current_progress.total_bytes = st.st_size;
                sftp.current_progress.bytes_transferred = 0;
                sftp.current_progress.progress_percent = 0.0f;
                sftp.current_progress.state = FileTransferProgress::IDLE;
            }
        }
        sftp.queue_cv.notify_one();

        LOG_INFO(LogCategory::SFTP, "Queued file transfer: {} -> {} ({} bytes)",
                 local_path, remote_path, st.st_size);
    }

    // 6. Add tag for rendering system
    target_vnc_entity.add<HasSFTPTransfer>();
}

// Track mouse button state for VNC
static int g_vncButtonMask = 0;

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
            interactive_elements.each([&](flecs::entity ui_element, AddTagOnLeftClick, UIElementBounds& bounds) {
                if (point_in_bounds(cursor_state->x, cursor_state->y, bounds))
                {
                    world->event<LeftClickEvent>()
                    .id<UIElementBounds>()
                    .entity(ui_element)
                    .emit();
                } 
            });
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

static void char_callback(GLFWwindow* window, unsigned int codepoint)
{
    // Multiple chat query... only one active
    ChatState* chat = world->try_get_mut<ChatState>();
    if (chat && chat->input_focused)
    {
        if (codepoint >= 32 && codepoint < 127)
        {
            chat->draft.push_back(static_cast<char>(codepoint));
        }
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

void apply_editor_config(const EditorConfigNode& config, flecs::entity node, flecs::entity UIElement);

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

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
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
                        recreate_annotation_entities(selector);
                        // Save newly created entities back
                        new_msg.ui_entities = selector.ui_entities;
                        new_msg.selection_entities = selector.selection_entities;
                        new_msg.token_count = selector.token_count;
                    }
                });
                return;
            }
        }

        // Number keys (0-9) - assign digit to selected token/part
        if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
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
                        int width = tokens[ti].selection_width();
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
                        int width = tokens[ti].selection_width();
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
                            new_sel_start += tokens[i].selection_width();
                            new_tokens.push_back(tokens[i]);
                        }
                        new_tokens.push_back({combined_text, digit, -1, -1, -1, TokenType::Entity});
                        for (int i = end_tok + 1; i < (int)tokens.size(); i++) {
                            new_tokens.push_back(tokens[i]);
                        }

                        selector.sentence_template = tokens_to_template(new_tokens);
                        selector.dirty = true;
                        selector.start_index = new_sel_start;
                        selector.end_index = new_sel_start;
                        recreate_annotation_entities(selector);
                    } else if (tokens[token_idx].type == TokenType::Relationship) {
                        // Assign digit to relationship part
                        // Both reified and non-reified: 0=source, 1=badge, 2=target
                        // Reified has additional: 3=reified (but reified digit set via R key)
                        if (sub_part == 0) {
                            tokens[token_idx].source_digit = digit;
                        } else if (sub_part == 2) {
                            tokens[token_idx].target_digit = digit;
                        }
                        // badge part (1) and reified part (3) don't accept digit assignment
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        recreate_annotation_entities(selector);
                    } else if (tokens[token_idx].type == TokenType::Entity) {
                        // Update entity digit
                        tokens[token_idx].binding_digit = digit;
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        recreate_annotation_entities(selector);
                    }

                    handled = true;
                }
            });
            if (handled) return;
        }

        // W key - set wildcard for relationship source/target
        if (key == GLFW_KEY_W)
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
                        int width = tokens[ti].selection_width();
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            sub_part = selector.start_index - expanded_pos;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    if (tokens[token_idx].type == TokenType::Relationship) {
                        // Set wildcard (-1) for source or target
                        // Both reified and non-reified: 0=source, 1=badge, 2=target
                        if (sub_part == 0) {
                            tokens[token_idx].source_digit = -1;
                        } else if (sub_part == 2) {
                            tokens[token_idx].target_digit = -1;
                        }
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        recreate_annotation_entities(selector);
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
                if (selector.active && !selector.sentence_template.empty()) {
                    auto tokens = parse_sentence_template(selector.sentence_template);
                    if (tokens.empty()) return;

                    // Find token at selection
                    int expanded_pos = 0;
                    int token_idx = -1;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = tokens[ti].selection_width();
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
                        tokens[token_idx].binding_digit = -1;
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        recreate_annotation_entities(selector);
                        handled = true;
                        return;
                    }

                    // If relationship, bind source to closest entity before, target to closest entity after
                    if (tokens[token_idx].type == TokenType::Relationship) {
                        int source_digit = -1;
                        int target_digit = -1;

                        // Find closest entity with lower index
                        for (int i = token_idx - 1; i >= 0; i--) {
                            if (tokens[i].type == TokenType::Entity && tokens[i].binding_digit >= 0) {
                                source_digit = tokens[i].binding_digit;
                                break;
                            }
                        }

                        // Find closest entity with higher index
                        for (int i = token_idx + 1; i < (int)tokens.size(); i++) {
                            if (tokens[i].type == TokenType::Entity && tokens[i].binding_digit >= 0) {
                                target_digit = tokens[i].binding_digit;
                                break;
                            }
                        }

                        tokens[token_idx].source_digit = source_digit;
                        tokens[token_idx].target_digit = target_digit;
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        recreate_annotation_entities(selector);
                        handled = true;
                        return;
                    }

                    // Collect all used digits
                    std::set<int> used_digits;
                    for (const auto& tok : tokens) {
                        if (tok.type == TokenType::Entity && tok.binding_digit >= 0) {
                            used_digits.insert(tok.binding_digit);
                        }
                        if (tok.type == TokenType::Relationship) {
                            if (tok.source_digit >= 0) used_digits.insert(tok.source_digit);
                            if (tok.target_digit >= 0) used_digits.insert(tok.target_digit);
                        }
                    }

                    // Find first available digit (0-9)
                    int available_digit = -1;
                    for (int d = 0; d <= 9; d++) {
                        if (used_digits.find(d) == used_digits.end()) {
                            available_digit = d;
                            break;
                        }
                    }
                    if (available_digit < 0) return; // All digits used

                    int end_token_idx = -1;
                    expanded_pos = 0;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = tokens[ti].selection_width();
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
                        new_sel_start += tokens[i].selection_width();
                        new_tokens.push_back(tokens[i]);
                    }
                    new_tokens.push_back({combined_text, available_digit, -1, -1, -1, TokenType::Entity});
                    for (int i = end_tok + 1; i < (int)tokens.size(); i++) {
                        new_tokens.push_back(tokens[i]);
                    }

                    selector.sentence_template = tokens_to_template(new_tokens);
                    selector.dirty = true;
                    selector.start_index = new_sel_start;
                    selector.end_index = new_sel_start;
                    recreate_annotation_entities(selector);
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
                        int width = tokens[ti].selection_width();
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }
                    if (token_idx < 0) return;

                    if (tokens[token_idx].type == TokenType::Entity ||
                        tokens[token_idx].type == TokenType::Relationship) {
                        tokens[token_idx].type = TokenType::PlainText;
                        tokens[token_idx].binding_digit = -1;
                        tokens[token_idx].source_digit = -1;
                        tokens[token_idx].target_digit = -1;
                        tokens[token_idx].reified_digit = -1;
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        recreate_annotation_entities(selector);
                        handled = true;
                    }
                }
            });
            if (handled) return;
        }

        // R key - create relationship or reify existing relationship
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
                        int width = tokens[ti].selection_width();
                        if (selector.start_index >= expanded_pos && selector.start_index < expanded_pos + width) {
                            token_idx = (int)ti;
                            break;
                        }
                        expanded_pos += width;
                    }

                    // If on existing relationship, toggle reification
                    if (token_idx >= 0 && tokens[token_idx].type == TokenType::Relationship) {
                        if (tokens[token_idx].reified_digit >= 0) {
                            // Already reified - remove reification
                            tokens[token_idx].reified_digit = -1;
                        } else {
                            // Reify: find first available digit
                            std::set<int> used_digits;
                            for (const auto& tok : tokens) {
                                if (tok.type == TokenType::Entity && tok.binding_digit >= 0) {
                                    used_digits.insert(tok.binding_digit);
                                }
                                if (tok.type == TokenType::Relationship) {
                                    if (tok.source_digit >= 0) used_digits.insert(tok.source_digit);
                                    if (tok.target_digit >= 0) used_digits.insert(tok.target_digit);
                                    if (tok.reified_digit >= 0) used_digits.insert(tok.reified_digit);
                                }
                            }
                            int available_digit = -1;
                            for (int d = 0; d <= 9; d++) {
                                if (used_digits.find(d) == used_digits.end()) {
                                    available_digit = d;
                                    break;
                                }
                            }
                            if (available_digit >= 0) {
                                tokens[token_idx].reified_digit = available_digit;
                            }
                        }
                        selector.sentence_template = tokens_to_template(tokens);
                        selector.dirty = true;
                        recreate_annotation_entities(selector);
                        handled = true;
                        return;
                    }

                    // Calculate expanded selection range (accounting for relationship width)
                    expanded_pos = 0;
                    int token_start = -1, token_end = -1;
                    for (size_t ti = 0; ti < tokens.size(); ti++) {
                        int width = tokens[ti].selection_width();
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
                        new_sel_start += tokens[i].selection_width();
                        new_tokens.push_back(tokens[i]);
                    }

                    // Add single relationship token with wildcard source/target
                    new_tokens.push_back({combined_text, -1, -1, -1, -1, TokenType::Relationship});

                    for (int i = token_end + 1; i < (int)tokens.size(); i++) {
                        new_tokens.push_back(tokens[i]);
                    }

                    // Update template and recreate UI entities
                    selector.sentence_template = tokens_to_template(new_tokens);
                    selector.dirty = true;
                    // Select source + relationship (first 2 of 3 parts)
                    selector.start_index = new_sel_start;
                    selector.end_index = new_sel_start + 1;
                    recreate_annotation_entities(selector);

                    handled = true;
                }
            });
            if (handled) return;
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

                    auto messageBox = world->entity()
                    .is_a(UIElement)
                    .child_of(chat_panel.message_list)
                    .set<LayoutBox>({LayoutBox::Horizontal});

                    // Create the background bubble
                    auto example_message_bkg = world->entity()
                        .is_a(UIElement)
                        .child_of(messageBox) // Attached to the ChatPanel found via query
                        .set<RoundedRectRenderable>({100.0f, 16.0f, 2.0f, false, 0x121212FF})
                        // .set<Expand>({true, 4.0f, 4.0f, 1.0f, false, 0, 0, 0})
                        .set<UIContainer>({8, 6})
                        // .add<DebugRenderBounds>()
                        .set<ZIndex>({15});

                        auto message_content = world->entity()
                        .is_a(UIElement)
                        .set<Position, Local>({8, 8})
                        // .child_of(example_message_bkg)
                        .child_of(example_message_bkg)
                        // .add<DebugRenderBounds>()
                        .set<LayoutBox>({LayoutBox::Vertical});

                    // Create the text content using the actual draft
                    auto example_message_text = world->entity()
                        .is_a(UIElement)
                        .child_of(message_content)
                        .set<TextRenderable>({chat->draft.c_str(), "Inter", 16.0f, 0xFFFFFFFF})
                        .add<DynamicTextWrapContainer>(chat_panel.messages_panel)
                        .set<DynamicTextWrap>({48.0f})
                        .set<ZIndex>({17});
                    // TODO: Set a wraparound width...

                    // I put it outside the message since it is a meta annotation
                    // This might only need to be visible during certain 'entity binding' interface modes...
                    auto messageBfoSprite = world->entity()
                    .is_a(UIElement)
                    .child_of(messageBox)
                    .set<ZIndex>({20})
                    .set<ImageCreator>({"../assets/bfo/generically_dependent_continuant.png", 1.0f, 1.0f});

                    // Run interpretation async - badges will be created when it completes
                    auto pending = std::make_shared<PendingInterpretation>();
                    pending->draft = chat->draft;
                    pending->message_list = chat_panel.message_list;

                    {
                        std::lock_guard<std::mutex> lock(pending_interpretations_mutex);
                        pending_interpretations.push_back(pending);
                    }

                    // Capture known entities for context
                    std::string context_json;
                    {
                        std::lock_guard<std::mutex> lock(known_entities_mutex);
                        json entities_array = json::array();
                        // TODO: Consider a simplified in-place representation
                        for (const auto& entity : known_entities) {
                            entities_array.push_back({
                                {"id", entity.id},
                                {"label", entity.label},
                                {"color", entity.color},
                                {"display_number", entity.display_number}
                            });
                        }
                        context_json = entities_array.dump();
                    }

                    // Capture previous sentences for context
                    std::string sentences_json;
                    {
                        std::lock_guard<std::mutex> lock(previous_sentences_mutex);
                        json sentences_array = json::array();
                        for (const auto& sentence : previous_sentences) {
                            sentences_array.push_back(sentence);
                        }
                        sentences_json = sentences_array.dump();
                    }

                    // Add current sentence to previous sentences
                    {
                        std::lock_guard<std::mutex> lock(previous_sentences_mutex);
                        previous_sentences.push_back(pending->draft);
                        // Keep only the last N sentences
                        while (previous_sentences.size() > MAX_PREVIOUS_SENTENCES) {
                            previous_sentences.erase(previous_sentences.begin());
                        }
                    }

                    std::thread([pending, context_json, sentences_json]() {
                        // Escape the draft for shell
                        std::string escaped_draft;
                        for (char c : pending->draft) {
                            if (c == '\'' || c == '\\' || c == '"' || c == '$' || c == '`') {
                                escaped_draft += '\\';
                            }
                            escaped_draft += c;
                        }

                        // Escape the context JSON for shell
                        std::string escaped_context;
                        for (char c : context_json) {
                            if (c == '\'' || c == '\\' || c == '"' || c == '$' || c == '`') {
                                escaped_context += '\\';
                            }
                            escaped_context += c;
                        }

                        // Escape the sentences JSON for shell
                        std::string escaped_sentences;
                        for (char c : sentences_json) {
                            if (c == '\'' || c == '\\' || c == '"' || c == '$' || c == '`') {
                                escaped_sentences += '\\';
                            }
                            escaped_sentences += c;
                        }
                        
                        bool interpret_with_llm = false;
                        if (interpret_with_llm)
                        {
                            std::string cmd = "python3 ../scripts/interpretation.py \"" + escaped_draft + "\" \"" + escaped_context + "\" \"" + escaped_sentences + "\" 2>&1";
                            FILE* pipe = popen(cmd.c_str(), "r");
                            if (pipe) {
                                char buffer[4096];
                                std::string result;
                                while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                                    result += buffer;
                                }
                                pclose(pipe);
                                pending->result = result;
                            }
                        }
                        pending->completed.store(true);
                    }).detach();
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

// Shader sources for 3D plane rendering
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aBary;
layout (location = 3) in float aGlow;
layout (location = 4) in vec2 aCentroidOffset;

out vec2 TexCoord;
out vec3 Bary;
out float Glow;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int glowPass;  // 0 = normal, 1 = outer glow pass
uniform float glowExpand;  // How much to expand for glow (e.g., 0.015)

void main()
{
    vec3 pos = aPos;

    // In glow pass, expand vertices outward from centroid
    // Glow encoding: 0-2 = normal glow, 10+ = central pulse (10 = start, 12 = fully expanded)
    if (glowPass == 1 && aGlow > 0.0) {
        vec2 expandDir = normalize(aCentroidOffset);
        float expandAmount;

        if (aGlow >= 10.0) {
            // Central pulse: decode scale and progress
            // Format: glow = 10.0 + (scale-0.6)*5.0 + progress*0.5
            float encoded = aGlow - 10.0;
            float scaleEnc = floor(encoded / 0.5) * 0.5;  // Quantized scale portion
            float pulseProgress = clamp((encoded - scaleEnc) / 0.5, 0.0, 1.0);
            float pulseScale = scaleEnc / 5.0 + 0.6;  // Recover 0.6-1.4 range

            // Ease out for smooth deceleration as it expands
            float easedProgress = 1.0 - (1.0 - pulseProgress) * (1.0 - pulseProgress);

            // Add rotational asymmetry using centroid offset angle
            float angle = atan(aCentroidOffset.y, aCentroidOffset.x);
            float wobble = 1.0 + 0.3 * sin(angle * 3.0 + pulseScale * 10.0);  // Asymmetric shape

            expandAmount = glowExpand * pulseScale * 2.0 * easedProgress * 20.0 * wobble;
        } else {
            expandAmount = glowExpand * aGlow;
        }
        pos.xy += expandDir * expandAmount;
    }

    gl_Position = projection * view * model * vec4(pos, 1.0);
    TexCoord = aTexCoord;
    Bary = aBary;
    Glow = aGlow;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Bary;
in float Glow;

uniform sampler2D uiTexture;
uniform int glowPass;  // 0 = normal, 1 = outer glow pass
uniform float chromaStrength;  // Chromatic aberration intensity

void main()
{
    if (glowPass == 1) {
        // Outer glow pass: soft diffuse glow
        if (Glow <= 0.0) discard;

        // Distance from center using barycentric (0.33 at center, 0 at edges)
        float minBary = min(min(Bary.x, Bary.y), Bary.z);

        // Check if this is a central pulse (glow >= 10.0)
        if (Glow >= 10.0) {
            // Central pulse with rounded corners - decode scale and progress
            float encoded = Glow - 10.0;
            float scaleEnc = floor(encoded / 0.5) * 0.5;
            float pulseProgress = clamp((encoded - scaleEnc) / 0.5, 0.0, 1.0);
            float pulseScale = scaleEnc / 5.0 + 0.6;

            // Create rounded corners by using a smooth distance from edges
            // Transform barycentric to a rounded shape - vary by scale
            float cornerRadius = 0.1 + pulseScale * 0.1;  // Bigger pulses = rounder corners
            float smoothEdge = smoothstep(0.0, cornerRadius, minBary);

            // Radial falloff from center for soft edge blur
            float centerDist = 1.0 - minBary * 3.0;  // 0 at center, 1 at edges
            float edgeSoftness = 0.3;
            float radialFalloff = 1.0 - smoothstep(1.0 - edgeSoftness, 1.0, centerDist);

            // Combine rounded corners with radial falloff
            float shapeMask = smoothEdge * radialFalloff;

            // Intensity fades as pulse expands outward - vary by scale
            float fadeIntensity = 1.0 - pulseProgress * (0.5 + pulseScale * 0.3);
            // Add a bright leading edge that travels outward - vary ring width by scale
            float ringWidth = 0.15 + pulseScale * 0.15;
            float ringPos = pulseProgress;
            float normalizedDist = centerDist;
            float ringIntensity = exp(-pow((normalizedDist - ringPos) / ringWidth, 2.0) * 2.0);

            float glowIntensity = (fadeIntensity * 0.15 + ringIntensity * 0.3) * shapeMask;
            glowIntensity *= 0.5;  // Subtle enough to see character beneath

            // Warm shield pulse color (slightly cyan-shifted for energy feel)
            vec3 glowColor = mix(vec3(1.0, 0.95, 0.7), vec3(0.7, 0.95, 1.0), pulseProgress * 0.3);

            FragColor = vec4(glowColor * glowIntensity, glowIntensity * 0.4);
        } else {
            // Normal glow for outer triangles
            // Soft radial falloff - bright at outer edge, fading inward
            float edgeness = 1.0 - minBary * 3.0;  // 1 at edges, 0 at center

            // Very gentle cubic falloff for soft blur
            float t = clamp(edgeness, 0.0, 1.0);
            float glowIntensity = t * t * (3.0 - 2.0 * t);  // Smooth hermite
            glowIntensity *= Glow * 0.35;  // Subtle intensity

            // Soft warm yellow glow
            vec3 glowColor = vec3(1.0, 0.92, 0.6);

            FragColor = vec4(glowColor * glowIntensity, glowIntensity * 0.6);
        }
    } else {
        // Normal pass: render textured triangle
        // Use transparency for UVs outside the 0-1 range (edge triangles)
        if (TexCoord.x < 0.0 || TexCoord.x > 1.0 || TexCoord.y < 0.0 || TexCoord.y > 1.0) {
            FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        } else {
            // Chromatic aberration: offset RGB channels radially from center
            vec2 center = vec2(0.5, 0.5);
            vec2 dir = TexCoord - center;
            float dist = length(dir);
            vec2 offset = dir * chromaStrength * dist;  // Stronger at edges

            // Sample each channel at slightly different positions
            float r = texture(uiTexture, TexCoord + offset).r;
            float g = texture(uiTexture, TexCoord).g;
            float b = texture(uiTexture, TexCoord - offset).b;
            float a = texture(uiTexture, TexCoord).a;

            FragColor = vec4(r, g, b, a);
        }
    }
}
)";

// Compile shader helper
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
    }

    return shader;
}

// Helper function to check if a point is inside a triangle using barycentric coordinates
bool pointInTriangle(float px, float py,
                     float ax, float ay, float bx, float by, float cx, float cy) {
    float v0x = cx - ax, v0y = cy - ay;
    float v1x = bx - ax, v1y = by - ay;
    float v2x = px - ax, v2y = py - ay;

    float dot00 = v0x * v0x + v0y * v0y;
    float dot01 = v0x * v1x + v0y * v1y;
    float dot02 = v0x * v2x + v0y * v2y;
    float dot11 = v1x * v1x + v1y * v1y;
    float dot12 = v1x * v2x + v1y * v2y;

    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

// Generate flat triangle grid pattern (alternating up/down triangles)
// Each triangle has 3 vertices and 1 face
void generateTriangularGrid(std::vector<float>& vertices, std::vector<unsigned int>& indices,
                           float width, float height, int subdivisionsX, int subdivisionsY) {
    vertices.clear();
    indices.clear();

    int vertexIndex = 0;

    // Calculate triangle dimensions for triangular tiling
    float triWidth = width / subdivisionsX;
    float triHeight = triWidth * 0.866025f; // sqrt(3)/2 for equilateral triangles

    // Giant triangle boundary (upward pointing - flat bottom edge, point at top)
    float triSize = std::min(width, height) * 0.4f;
    // Snap bottom Y to nearest row boundary for clean flat edge alignment
    float giantBottomY = -triSize * 0.4f;
    giantBottomY = floor(giantBottomY / triHeight) * triHeight;  // Snap to grid row
    float giantTopY = giantBottomY + triSize * 0.866025f;  // Point at top
    float halfBase = (giantTopY - giantBottomY) / 0.866025f * 0.5f;  // Half base width

    float giantTriAx = -halfBase, giantTriAy = giantBottomY;   // Bottom left (flat edge)
    float giantTriBx =  halfBase, giantTriBy = giantBottomY;   // Bottom right (flat edge)
    float giantTriCx =  0.0f,     giantTriCy = giantTopY;      // Top center (point)

    // Adjust number of rows based on height
    int numRows = (int)(height / triHeight) + 1;

    // Generate triangle tiling pattern
    for (int row = 0; row < numRows; row++) {
        // Calculate Y position for this row
        float rowY = (row * triHeight) - height * 0.5f;

        // Determine if this is an even or odd row (for offset)
        bool isEvenRow = (row % 2 == 0);

        // Number of triangles in this row (add extra for edge coverage)
        int numTrisInRow = subdivisionsX * 2 + 2;

        for (int col = -1; col < numTrisInRow; col++) {
            // Determine if this is an upward or downward pointing triangle
            bool isUpward = (col % 2 == 0);

            // Calculate base X position (start one column earlier to fill left gap)
            float baseX = (col * triWidth * 0.5f) - width * 0.5f;
            if (!isEvenRow) {
                baseX -= triWidth * 0.25f; // Offset odd rows left to fill gap
            }

            float x0, x1, x2, y0, y1, y2;

            if (isUpward) {
                // Upward pointing triangle (△)
                x0 = baseX;                    // left
                x1 = baseX + triWidth;         // right
                x2 = baseX + triWidth * 0.5f;  // top (center)

                y0 = rowY;
                y1 = rowY;
                y2 = rowY + triHeight;
            } else {
                // Downward pointing triangle (▽)
                x0 = baseX;                    // left
                x1 = baseX + triWidth;         // right
                x2 = baseX + triWidth * 0.5f;  // bottom (center)

                y0 = rowY + triHeight;
                y1 = rowY + triHeight;
                y2 = rowY;
            }

            // Calculate UV coordinates
            float u0 = (x0 + width * 0.5f) / width;
            float u1 = (x1 + width * 0.5f) / width;
            float u2 = (x2 + width * 0.5f) / width;

            float v0 = (y0 + height * 0.5f) / height;
            float v1 = (y1 + height * 0.5f) / height;
            float v2 = (y2 + height * 0.5f) / height;

            // Calculate centroid for offset computation
            float centX = (x0 + x1 + x2) / 3.0f;
            float centY = (y0 + y1 + y2) / 3.0f;

            // Vertex 0 (barycentric: 1,0,0)
            vertices.push_back(x0);
            vertices.push_back(y0);
            vertices.push_back(0.0f);  // Flat on Z=0
            vertices.push_back(u0);
            vertices.push_back(v0);
            vertices.push_back(1.0f);  // baryX
            vertices.push_back(0.0f);  // baryY
            vertices.push_back(0.0f);  // baryZ
            vertices.push_back(0.0f);  // glow
            vertices.push_back(x0 - centX);  // offsetX (direction from centroid)
            vertices.push_back(y0 - centY);  // offsetY

            // Vertex 1 (barycentric: 0,1,0)
            vertices.push_back(x1);
            vertices.push_back(y1);
            vertices.push_back(0.0f);
            vertices.push_back(u1);
            vertices.push_back(v1);
            vertices.push_back(0.0f);  // baryX
            vertices.push_back(1.0f);  // baryY
            vertices.push_back(0.0f);  // baryZ
            vertices.push_back(0.0f);  // glow
            vertices.push_back(x1 - centX);  // offsetX
            vertices.push_back(y1 - centY);  // offsetY

            // Vertex 2 (barycentric: 0,0,1)
            vertices.push_back(x2);
            vertices.push_back(y2);
            vertices.push_back(0.0f);
            vertices.push_back(u2);
            vertices.push_back(v2);
            vertices.push_back(0.0f);  // baryX
            vertices.push_back(0.0f);  // baryY
            vertices.push_back(1.0f);  // baryZ
            vertices.push_back(0.0f);  // glow
            vertices.push_back(x2 - centX);  // offsetX
            vertices.push_back(y2 - centY);  // offsetY

            // Single triangle face
            indices.push_back(vertexIndex);
            indices.push_back(vertexIndex + 1);
            indices.push_back(vertexIndex + 2);

            vertexIndex += 3; // 3 vertices per triangle
        }
    }
}

// Apply rotation to a point around origin (Rodrigues' rotation formula simplified for unit axis)
void rotatePoint(float& x, float& y, float& z, float axisX, float axisY, float axisZ, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    float dot = x * axisX + y * axisY + z * axisZ;
    float crossX = axisY * z - axisZ * y;
    float crossY = axisZ * x - axisX * z;
    float crossZ = axisX * y - axisY * x;

    float newX = x * c + crossX * s + axisX * dot * (1 - c);
    float newY = y * c + crossY * s + axisY * dot * (1 - c);
    float newZ = z * c + crossZ * s + axisZ * dot * (1 - c);

    x = newX;
    y = newY;
    z = newZ;
}

// Initialize particle animation - FTL deceleration into Thornfield
// Screen tetrahedrons move with velocities, calculated to collide at their target positions
void initializeParticles(Graphics& graphics, const std::vector<float>& targetVertices,
                         float width, float height, float duration = 3.0f) {
    // Re-seed each time for different pattern on every spawn
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    static std::mt19937 gen(seed);
    gen.seed(seed);

    // Velocity distribution - coming from far away (negative Z) towards camera/grid (positive Z)
    std::uniform_real_distribution<float> vxDist(-0.3f, 0.3f);   // Small lateral drift
    std::uniform_real_distribution<float> vyDist(-0.3f, 0.3f);   // Small vertical drift
    std::uniform_real_distribution<float> vzDist(1.5f, 4.0f);    // Moving towards camera (+Z direction)
    std::uniform_real_distribution<float> rotAngleDist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> collisionTimeDist(0.2f, 1.0f);  // Central triangle timing (fast)
    std::uniform_real_distribution<float> outerCollisionTimeDist(1.2f, 2.5f);  // Outer grid delayed

    // Calculate giant triangle bounds (same as in generateTriangularGrid)
    // Upward pointing - flat bottom edge, point at top
    float triWidth = width / 160;  // Match subdivisions (80x2)
    float triHeight = triWidth * 0.866025f;
    float triSize = std::min(width, height) * 0.4f;
    float bottomY = -triSize * 0.4f;
    bottomY = floor(bottomY / triHeight) * triHeight;
    float topY = bottomY + triSize * 0.866025f;
    float halfBase = (topY - bottomY) / 0.866025f * 0.5f;
    float giantTriAx = -halfBase, giantTriAy = bottomY;   // Bottom left (flat edge)
    float giantTriBx =  halfBase, giantTriBy = bottomY;   // Bottom right (flat edge)
    float giantTriCx =  0.0f,     giantTriCy = topY;      // Top center (point)

    graphics.particles.clear();
    graphics.gridVertices = targetVertices;

    // Each vertex has 11 floats: x, y, z, u, v, baryX, baryY, baryZ, glow, offsetX, offsetY
    int numVertices = targetVertices.size() / 11;
    // 3 vertices per triangle
    int numTriangles = numVertices / 3;

    for (int t = 0; t < numTriangles; t++) {
        // Calculate triangle's centroid and find min Y vertex
        float centroidX = 0, centroidY = 0, centroidZ = 0;
        float minVertY = 1e10f;
        for (int v = 0; v < 3; v++) {
            int i = t * 3 + v;
            float vy = targetVertices[i * 11 + 1];
            centroidX += targetVertices[i * 11 + 0];
            centroidY += vy;
            centroidZ += targetVertices[i * 11 + 2];
            minVertY = std::min(minVertY, vy);
        }
        centroidX /= 3.0f;
        centroidY /= 3.0f;
        centroidZ /= 3.0f;

        // Check if this triangle is inside the central giant triangle
        bool isInCentralTriangle = pointInTriangle(centroidX, centroidY,
                                                    giantTriAx, giantTriAy,
                                                    giantTriBx, giantTriBy,
                                                    giantTriCx, giantTriCy);

        // Exclude downward-pointing triangles at the bottom edge (they have points, not flat edges)
        // Downward triangles have centroid above their minimum Y vertex
        bool isDownwardPointing = (centroidY > minVertY + triHeight * 0.2f);
        bool isAtBottomEdge = (minVertY < bottomY + triHeight * 0.5f);
        if (isInCentralTriangle && isDownwardPointing && isAtBottomEdge) {
            isInCentralTriangle = false;  // Exclude from central triangle
        }

        // Calculate collision time based on position
        float collisionTime;
        float vx, vy, vz;

        if (isInCentralTriangle) {
            // Central triangle loads first with random timing
            collisionTime = collisionTimeDist(gen);
            // Random velocity for central triangles
            vx = vxDist(gen);
            vy = vyDist(gen);
            vz = vzDist(gen);
        } else {
            // Outer triangles (torus/thorns) stay stable until central triangle forms
            // Then "wither away" - triangles near center extract first like decay/fire

            float distFromCenter = sqrt(centroidX * centroidX + centroidY * centroidY);
            float maxDist = sqrt(width * width + height * height) * 0.5f;  // Half diagonal
            float normalizedDist = std::min(distFromCenter / maxDist, 1.0f);

            // Withering timing: wait for central triangle to form (1.5s), then decay
            // Closer to center = extract sooner (inverse of before)
            // Narrow time window (0.8s) for gradual peeling effect
            float witherStart = 1.5f;  // Start after central triangle is mostly formed
            float witherDuration = 0.8f;  // Narrow window for decay effect

            // Invert: closer triangles (low normalizedDist) wither first
            float witherOrder = 1.0f - normalizedDist;

            std::uniform_real_distribution<float> jitterDist(-0.05f, 0.05f);
            float baseTime = witherStart + witherOrder * witherDuration;
            collisionTime = baseTime + jitterDist(gen);

            // Calculate spawn position on thorny stem torus around central triangle
            // Triangles cluster at discrete thorn positions to form visible spikes

            // Torus parameters
            float majorRadius = width * 0.7f;   // Main ring radius
            float minorRadius = height * 0.1f;  // Stem tube thickness

            // Number of thorns around the torus
            int numThorns = 16;

            // Decide if this triangle is part of stem or a thorn
            std::uniform_real_distribution<float> partDist(0.0f, 1.0f);
            bool isStem = partDist(gen) < 0.25f;  // 25% form the stem, 75% form thorns

            float torusX, torusY, torusZ;

            if (isStem) {
                // Stem triangles - distributed along the torus surface
                std::uniform_real_distribution<float> uDist(0.0f, 2.0f * M_PI);
                std::uniform_real_distribution<float> vDist(0.0f, 2.0f * M_PI);
                float u = uDist(gen);
                float v = vDist(gen);

                torusX = (majorRadius + minorRadius * cos(v)) * cos(u);
                torusY = (majorRadius + minorRadius * cos(v)) * sin(u);
                torusZ = minorRadius * sin(v);
            } else {
                // Thorn triangles - cluster at discrete thorn positions
                std::uniform_real_distribution<float> thornIndexDist(0.0f, (float)numThorns);
                int thornIndex = (int)thornIndexDist(gen);

                // Various thorn sizes - some small, some large
                std::uniform_real_distribution<float> thornSizeDist(0.1f, 0.35f);
                float thornLength = thornSizeDist(gen) * height;

                // Thorn profile: 0 = spikey isosceles, 1 = fat equilateral
                std::uniform_real_distribution<float> profileDist(0.0f, 1.0f);
                float thornProfile = profileDist(gen);

                // Position around main ring for this thorn
                float u = (thornIndex / (float)numThorns) * 2.0f * M_PI;

                // Each thorn has a fixed outward direction (v angle)
                std::uniform_real_distribution<float> vVariation(-0.2f, 0.2f);
                float v = (thornIndex % 6) * (M_PI / 3.0f) + vVariation(gen);  // 6 directions

                // Base position on torus surface
                float baseX = (majorRadius + minorRadius * cos(v)) * cos(u);
                float baseY = (majorRadius + minorRadius * cos(v)) * sin(u);
                float baseZ = minorRadius * sin(v);

                // Thorn direction (outward from tube surface)
                float thornDirX = cos(v) * cos(u);
                float thornDirY = cos(v) * sin(u);
                float thornDirZ = sin(v);

                // Position along the thorn (0 = base, 1 = tip)
                std::uniform_real_distribution<float> alongThorn(0.0f, 1.0f);
                float tPos = alongThorn(gen);

                // Thorn profile affects spread vs length ratio
                // Spikey (profile=0): narrow spread, elongated
                // Equilateral (profile=1): wide spread, shorter effective length
                float baseSpread = 0.03f + thornProfile * 0.12f;  // 0.03 to 0.15
                float lengthScale = 1.0f - thornProfile * 0.4f;   // 1.0 to 0.6
                thornLength *= lengthScale;

                // Thorn tapers - spread decreases toward tip (more dramatic for spikey)
                float taperPower = 1.0f + (1.0f - thornProfile) * 1.5f;  // 1.0 to 2.5
                float spread = pow(1.0f - tPos, taperPower) * baseSpread * height;
                std::uniform_real_distribution<float> spreadDist(-1.0f, 1.0f);

                // Calculate perpendicular directions for spread
                float perpX1 = -sin(u);
                float perpY1 = cos(u);
                float perpZ1 = 0.0f;
                float perpX2 = thornDirY * perpZ1 - thornDirZ * perpY1;
                float perpY2 = thornDirZ * perpX1 - thornDirX * perpZ1;
                float perpZ2 = thornDirX * perpY1 - thornDirY * perpX1;

                float spreadOffset1 = spreadDist(gen) * spread;
                float spreadOffset2 = spreadDist(gen) * spread;

                torusX = baseX + thornDirX * tPos * thornLength + perpX1 * spreadOffset1 + perpX2 * spreadOffset2;
                torusY = baseY + thornDirY * tPos * thornLength + perpY1 * spreadOffset1 + perpY2 * spreadOffset2;
                torusZ = baseZ + thornDirZ * tPos * thornLength + perpZ1 * spreadOffset1 + perpZ2 * spreadOffset2;
            }

            // Tilt the torus around the X-axis (planetary ring angle)
            float tiltAngle = -0.45f;  // About 25 degrees
            float cosT = cos(tiltAngle);
            float sinT = sin(tiltAngle);
            float arcX = torusX;
            float arcY = torusY * cosT - torusZ * sinT;
            float arcZ = torusY * sinT + torusZ * cosT;

            // Offset to position the ring
            arcY += height * 0.15f;

            // Z position - push spawn further back
            float spawnZ = arcZ - 4.0f;

            // Calculate velocity to travel from arc spawn to target in collision time
            vx = (centroidX - arcX) / collisionTime;
            vy = (centroidY - arcY) / collisionTime;
            vz = (0.0f - spawnZ) / collisionTime;  // Target Z is 0
        }

        // Random rotation axis (normalized) for initial orientation
        float axisX = axisDist(gen);
        float axisY = axisDist(gen);
        float axisZ = axisDist(gen);
        float axisLen = sqrt(axisX*axisX + axisY*axisY + axisZ*axisZ);
        if (axisLen > 0.001f) {
            axisX /= axisLen;
            axisY /= axisLen;
            axisZ /= axisLen;
        } else {
            axisX = 0; axisY = 0; axisZ = 1;
        }
        float rotAngle = rotAngleDist(gen);

        // Apply to all 3 vertices of this triangle
        for (int v = 0; v < 3; v++) {
            int i = t * 3 + v;
            TriangleParticle p;

            // Target position from grid (where it will collide)
            p.targetX = targetVertices[i * 11 + 0];
            p.targetY = targetVertices[i * 11 + 1];
            p.targetZ = targetVertices[i * 11 + 2];

            // Local offset from centroid (maintains triangle shape)
            float localX = p.targetX - centroidX;
            float localY = p.targetY - centroidY;
            float localZ = p.targetZ - centroidZ;

            // Apply random rotation to local offset (tumbling debris orientation)
            rotatePoint(localX, localY, localZ, axisX, axisY, axisZ, rotAngle);

            // Store rotated local offset (for maintaining shape during flight)
            p.localX = localX;
            p.localY = localY;
            p.localZ = localZ;

            // Velocity (same for all vertices of this triangle)
            p.vx = vx;
            p.vy = vy;
            p.vz = vz;

            // Collision time
            p.collisionTime = collisionTime;

            // UVs don't animate
            p.u = targetVertices[i * 11 + 3];
            p.v = targetVertices[i * 11 + 4];

            // Barycentric coordinates (from grid generation)
            p.baryX = targetVertices[i * 11 + 5];
            p.baryY = targetVertices[i * 11 + 6];
            p.baryZ = targetVertices[i * 11 + 7];

            p.elapsedTime = 0.0f;
            p.hitTime = -1.0f;  // Not hit yet
            p.vertexIndex = i;
            p.locked = false;
            p.isCentral = isInCentralTriangle;

            // Pulse variation based on debris velocity magnitude and rotation
            float velocityMag = sqrt(vx*vx + vy*vy + vz*vz);
            p.pulseScale = 0.6f + (velocityMag / 4.0f) * 0.8f;  // Faster debris = bigger pulse
            p.pulseScale = std::min(p.pulseScale, 1.4f);
            p.pulseRotation = rotAngle;  // Use debris rotation for pulse asymmetry

            graphics.particles.push_back(p);
        }
    }
}

// Update particle positions based on velocity trajectories toward collision points
void updateParticles(Graphics& graphics, float deltaTime) {
    for (auto& p : graphics.particles) {
        p.elapsedTime += deltaTime;

        float x, y, z;
        float glow = 0.0f;

        if (p.locked || p.elapsedTime >= p.collisionTime) {
            // Track first hit time for glow effect
            if (!p.locked) {
                p.hitTime = p.elapsedTime;
            }
            p.locked = true;

            // Time since impact
            float timeSinceHit = p.elapsedTime - p.hitTime;

            // Impact overshoot effect - deflects backward then bounces back
            // Central triangles have stronger overshoot (debris impact on shield)
            float overshootAmount = p.isCentral ? 0.425f : 0.112f;
            float overshootDuration = p.isCentral ? 1.0f : 0.48f;
            float overshootZ = 0.0f;

            if (timeSinceHit < overshootDuration) {
                // Damped spring oscillation for bounce-back effect
                // z(t) = A * sin(ωt) * e^(-bt) where ω gives ~1.5 oscillations
                float t = timeSinceHit / overshootDuration;
                float omega = 4.5f * M_PI;  // ~1.5 oscillations
                float damping = 4.0f;
                overshootZ = overshootAmount * sin(omega * t) * exp(-damping * t);
            }

            // Position with overshoot (negative Z = pushed back toward viewer)
            x = p.targetX;
            y = p.targetY;
            z = p.targetZ - overshootZ;

            // Compute glow based on triangle type
            if (p.isCentral) {
                // Central pulse: encode scale and progress in glow value
                // Format: glow = 10.0 + (scale-0.6)*5.0 + progress*0.5
                // Scale range 0.6-1.4 -> 0-4, progress 0-1 -> 0-0.5
                // Total range: 10.0 to 14.5
                float pulseDuration = 0.8f;
                if (timeSinceHit < pulseDuration) {
                    float pulseProgress = timeSinceHit / pulseDuration;
                    float scaleEnc = (p.pulseScale - 0.6f) * 5.0f;  // 0-4
                    glow = 10.0f + scaleEnc + pulseProgress * 0.5f;
                } else {
                    glow = 0.0f;  // Pulse complete
                }
            } else {
                // Normal glow for outer triangles
                // float glowDuration = 0.8f;
                // if (timeSinceHit < glowDuration) {
                //     glow = exp(-3.0f * timeSinceHit / glowDuration);
                // }
                glow = 0.0f;
                // float pulseDuration = 0.1f;
                // if (timeSinceHit < pulseDuration) {
                //     float pulseProgress = timeSinceHit / pulseDuration;
                //     float scaleEnc = (p.pulseScale - 0.6f) * 1.0f;  // 0-4
                //     glow = 30.0f + scaleEnc + pulseProgress * 0.5f;
                // } else {
                //     glow = 0.0f;  // Pulse complete
                // }
                float glowDuration = 0.5f;
                glow = exp(-5.0f * timeSinceHit / glowDuration);

            }
        } else {
            // Flying towards collision point along velocity trajectory
            float timeToCollision = p.collisionTime - p.elapsedTime;
            float t = p.elapsedTime / p.collisionTime;  // 0 at start, 1 at collision

            // Centroid follows trajectory: starts far, arrives at target centroid
            float centroidX = p.targetX - p.vx * timeToCollision;
            float centroidY = p.targetY - p.vy * timeToCollision;
            float centroidZ = p.targetZ - p.vz * timeToCollision;

            // Rotated local offset blends out as we approach collision (tumbling -> aligned)
            float localBlend = 1.0f - t;
            x = centroidX + p.localX * localBlend;
            y = centroidY + p.localY * localBlend;
            z = centroidZ + p.localZ * localBlend;
        }

        // Update in grid vertices buffer (11 floats per vertex: x,y,z, u,v, baryX,baryY,baryZ, glow, offsetX,offsetY)
        int offset = p.vertexIndex * 11;
        graphics.gridVertices[offset + 0] = x;
        graphics.gridVertices[offset + 1] = y;
        graphics.gridVertices[offset + 2] = z;
        // UV (3,4) and barycentric (5,6,7) stay the same
        graphics.gridVertices[offset + 8] = glow;
    }
}

// Upload updated vertices to GPU
void uploadParticleVertices(Graphics& graphics) {
    glBindBuffer(GL_ARRAY_BUFFER, graphics.gridVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, graphics.gridVertices.size() * sizeof(float), graphics.gridVertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Initialize noise tetrahedrons - grey debris all around (we've just hit the debris field)
void initializeNoiseTetrahedrons(Graphics& graphics, int count = 300) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    // Spread debris all around in 3D space
    std::uniform_real_distribution<float> xDist(-8.0f, 8.0f);
    std::uniform_real_distribution<float> yDist(-6.0f, 6.0f);
    std::uniform_real_distribution<float> zDistFar(-15.0f, -3.0f);   // Far debris field
    std::uniform_real_distribution<float> zDistNear(-3.0f, 2.0f);    // Near-camera debris (passes through early)
    std::uniform_real_distribution<float> velocityDist(4.0f, 10.0f);  // FTL speeds
    std::uniform_real_distribution<float> scaleDist(0.015f, 0.06f);
    std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * M_PI);

    // Initialize FTL deceleration state
    graphics.decelerationTime = 0.0f;
    graphics.decelerationDuration = 5.0f;  // 5 seconds to decelerate

    graphics.noiseParticles.clear();
    graphics.noiseParticles.reserve(count);

    // Spawn 40% of particles near the camera so they pass through early
    int nearCount = count * 4 / 10;

    for (int i = 0; i < count; i++) {
        NoiseTetrahedron n;
        // Distribute throughout 3D space around the viewer
        n.x = xDist(gen);
        n.y = yDist(gen);
        // Near particles pass through camera area early, before screen forms
        n.z = (i < nearCount) ? zDistNear(gen) : zDistFar(gen);
        n.vz = velocityDist(gen);
        n.scale = scaleDist(gen);

        // Random rotation axis (normalized)
        n.axisX = axisDist(gen);
        n.axisY = axisDist(gen);
        n.axisZ = axisDist(gen);
        float axisLen = sqrt(n.axisX*n.axisX + n.axisY*n.axisY + n.axisZ*n.axisZ);
        if (axisLen > 0.001f) {
            n.axisX /= axisLen;
            n.axisY /= axisLen;
            n.axisZ /= axisLen;
        } else {
            n.axisX = 0; n.axisY = 0; n.axisZ = 1;
        }
        n.rotAngle = angleDist(gen);

        graphics.noiseParticles.push_back(n);
    }

    // Pre-allocate vertex buffer (4 verts * 5 floats per tetrahedron)
    graphics.noiseVertices.resize(count * 4 * 5);
    graphics.noiseVertexCount = count * 12; // 4 faces * 3 indices
}

// Respawn a noise tetrahedron at far distance
void respawnNoiseTetrahedron(NoiseTetrahedron& n) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(-8.0f, 8.0f);
    std::uniform_real_distribution<float> yDist(-6.0f, 6.0f);
    std::uniform_real_distribution<float> velocityDist(4.0f, 10.0f);
    std::uniform_real_distribution<float> scaleDist(0.015f, 0.06f);
    std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * M_PI);

    n.x = xDist(gen);
    n.y = yDist(gen);
    n.z = -20.0f;  // Respawn far away
    n.vz = velocityDist(gen);
    n.scale = scaleDist(gen);

    // New random rotation
    n.axisX = axisDist(gen);
    n.axisY = axisDist(gen);
    n.axisZ = axisDist(gen);
    float axisLen = sqrt(n.axisX*n.axisX + n.axisY*n.axisY + n.axisZ*n.axisZ);
    if (axisLen > 0.001f) {
        n.axisX /= axisLen;
        n.axisY /= axisLen;
        n.axisZ /= axisLen;
    } else {
        n.axisX = 0; n.axisY = 0; n.axisZ = 1;
    }
    n.rotAngle = angleDist(gen);
}

// Update noise tetrahedrons - move towards camera with FTL deceleration
void updateNoiseTetrahedrons(Graphics& graphics, float deltaTime) {
    // Update deceleration time
    graphics.decelerationTime += deltaTime;

    // Calculate deceleration factor: starts at 1.0 (full speed), decays towards 0.1 (crawl)
    float t = std::min(graphics.decelerationTime / graphics.decelerationDuration, 1.0f);
    // Exponential decay for deceleration feel
    float speedMultiplier = 0.1f + 0.9f * exp(-3.0f * t);

    for (auto& n : graphics.noiseParticles) {
        // Apply decelerated velocity
        n.z += n.vz * speedMultiplier * deltaTime;

        // Respawn if past camera (but only if still decelerating fast enough)
        if (n.z > 5.0f) {
            if (speedMultiplier > 0.15f) {
                respawnNoiseTetrahedron(n);
            } else {
                // At near-stop, just keep them drifting slowly past
                n.z = 5.1f; // Park them just past camera
            }
        }
    }
}

// Generate vertices for noise tetrahedrons
void generateNoiseVertices(Graphics& graphics) {
    int idx = 0;
    for (const auto& n : graphics.noiseParticles) {
        float s = n.scale;
        float h = s * 0.8f;  // Apex height

        // Base triangle vertices (local, centered at origin)
        float lx0 = -s,    ly0 = -s * 0.577f, lz0 = 0;
        float lx1 = s,     ly1 = -s * 0.577f, lz1 = 0;
        float lx2 = 0,     ly2 = s * 1.155f,  lz2 = 0;
        float lx3 = 0,     ly3 = 0,           lz3 = h;  // Apex

        // Apply rotation to each local vertex
        rotatePoint(lx0, ly0, lz0, n.axisX, n.axisY, n.axisZ, n.rotAngle);
        rotatePoint(lx1, ly1, lz1, n.axisX, n.axisY, n.axisZ, n.rotAngle);
        rotatePoint(lx2, ly2, lz2, n.axisX, n.axisY, n.axisZ, n.rotAngle);
        rotatePoint(lx3, ly3, lz3, n.axisX, n.axisY, n.axisZ, n.rotAngle);

        // Translate to world position
        float x0 = n.x + lx0, y0 = n.y + ly0, z0 = n.z + lz0;
        float x1 = n.x + lx1, y1 = n.y + ly1, z1 = n.z + lz1;
        float x2 = n.x + lx2, y2 = n.y + ly2, z2 = n.z + lz2;
        float x3 = n.x + lx3, y3 = n.y + ly3, z3 = n.z + lz3;

        // Grey UV
        float u = 0.5f, v = 0.5f;

        // Vertex 0
        graphics.noiseVertices[idx++] = x0;
        graphics.noiseVertices[idx++] = y0;
        graphics.noiseVertices[idx++] = z0;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
        // Vertex 1
        graphics.noiseVertices[idx++] = x1;
        graphics.noiseVertices[idx++] = y1;
        graphics.noiseVertices[idx++] = z1;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
        // Vertex 2
        graphics.noiseVertices[idx++] = x2;
        graphics.noiseVertices[idx++] = y2;
        graphics.noiseVertices[idx++] = z2;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
        // Vertex 3 (apex)
        graphics.noiseVertices[idx++] = x3;
        graphics.noiseVertices[idx++] = y3;
        graphics.noiseVertices[idx++] = z3;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
    }
}

// Upload noise vertices to GPU
void uploadNoiseVertices(Graphics& graphics) {
    glBindBuffer(GL_ARRAY_BUFFER, graphics.noiseVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, graphics.noiseVertices.size() * sizeof(float), graphics.noiseVertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Initialize 3D rendering resources
void initialize3DRendering(Graphics& graphics, int width, int height) {
    graphics.uiWidth = width;
    graphics.uiHeight = height;
    graphics.tiltAngle = 0.0f;

    // Create framebuffer for UI rendering
    glGenFramebuffers(1, &graphics.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, graphics.fbo);

    // Create texture to render UI to
    glGenTextures(1, &graphics.fboTexture);
    glBindTexture(GL_TEXTURE_2D, graphics.fboTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, graphics.fboTexture, 0);

    // Create depth and stencil renderbuffer
    glGenRenderbuffers(1, &graphics.fboDepthRenderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, graphics.fboDepthRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, graphics.fboDepthRenderBuffer);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer is not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Create shader program
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    graphics.shaderProgram = glCreateProgram();
    glAttachShader(graphics.shaderProgram, vertexShader);
    glAttachShader(graphics.shaderProgram, fragmentShader);
    glLinkProgram(graphics.shaderProgram);

    GLint success;
    glGetProgramiv(graphics.shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(graphics.shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create plane geometry
    float aspectRatio = (float)width / (float)height;
    float planeWidth = 2.0f * aspectRatio;
    float planeHeight = 2.0f;

    float vertices[] = {
        // positions                              // tex     // bary (unused)   // glow  // centroid offset
        -planeWidth/2, -planeHeight/2, 0.0f,   0.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom left
         planeWidth/2, -planeHeight/2, 0.0f,   1.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom right
         planeWidth/2,  planeHeight/2, 0.0f,   1.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // top right
        -planeWidth/2,  planeHeight/2, 0.0f,   0.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f   // top left
    };

    unsigned int indices[] = {
        0, 1, 2,
        0, 2, 3
    };

    glGenVertexArrays(1, &graphics.planeVAO);
    glGenBuffers(1, &graphics.planeVBO);
    glGenBuffers(1, &graphics.planeEBO);

    glBindVertexArray(graphics.planeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphics.planeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Barycentric coords attribute (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Glow factor attribute (location 3)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);

    // Centroid offset attribute (location 4)
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(9 * sizeof(float)));
    glEnableVertexAttribArray(4);

    glBindVertexArray(0);

    // Create triangular grid geometry (20x20 subdivisions for smooth tessellation)
    std::vector<float> gridVertices;
    std::vector<unsigned int> gridIndices;
    generateTriangularGrid(gridVertices, gridIndices, planeWidth, planeHeight, 80, 80);

    graphics.gridVertexCount = gridIndices.size();
    graphics.useGridMode = true; // Start with grid mode to show particle animation
    graphics.gridModeTransitionTimer = 0.0f;
    graphics.allParticlesLocked = false;

    glGenVertexArrays(1, &graphics.gridVAO);
    glGenBuffers(1, &graphics.gridVBO);
    glGenBuffers(1, &graphics.gridEBO);

    glBindVertexArray(graphics.gridVAO);

    // Initialize particles with random positions that will animate to grid
    initializeParticles(graphics, gridVertices, planeWidth, planeHeight, 3.0f);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.gridVBO);
    // Use DYNAMIC_DRAW since we'll update vertices each frame during animation
    glBufferData(GL_ARRAY_BUFFER, graphics.gridVertices.size() * sizeof(float), graphics.gridVertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphics.gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndices.size() * sizeof(unsigned int), gridIndices.data(), GL_STATIC_DRAW);

    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Barycentric coords attribute (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Glow factor attribute (location 3)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);

    // Centroid offset attribute (location 4)
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(9 * sizeof(float)));
    glEnableVertexAttribArray(4);

    glBindVertexArray(0);

    // Create grey texture for noise tetrahedrons
    unsigned char greyPixel[4] = {100, 100, 100, 255};  // Dark grey RGBA
    glGenTextures(1, &graphics.greyTexture);
    glBindTexture(GL_TEXTURE_2D, graphics.greyTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, greyPixel);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Initialize noise tetrahedrons
    initializeNoiseTetrahedrons(graphics, 300);

    // Generate indices for noise tetrahedrons (4 faces * 3 verts each)
    std::vector<unsigned int> noiseIndices;
    for (int t = 0; t < (int)graphics.noiseParticles.size(); t++) {
        unsigned int base = t * 4;
        // Face 0: base (v0, v1, v2)
        noiseIndices.push_back(base + 0);
        noiseIndices.push_back(base + 1);
        noiseIndices.push_back(base + 2);
        // Face 1: side (v0, v1, apex)
        noiseIndices.push_back(base + 0);
        noiseIndices.push_back(base + 1);
        noiseIndices.push_back(base + 3);
        // Face 2: side (v1, v2, apex)
        noiseIndices.push_back(base + 1);
        noiseIndices.push_back(base + 2);
        noiseIndices.push_back(base + 3);
        // Face 3: side (v2, v0, apex)
        noiseIndices.push_back(base + 2);
        noiseIndices.push_back(base + 0);
        noiseIndices.push_back(base + 3);
    }
    graphics.noiseVertexCount = noiseIndices.size();

    glGenVertexArrays(1, &graphics.noiseVAO);
    glGenBuffers(1, &graphics.noiseVBO);
    GLuint noiseEBO;
    glGenBuffers(1, &noiseEBO);

    glBindVertexArray(graphics.noiseVAO);

    // Generate initial vertices
    generateNoiseVertices(graphics);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.noiseVBO);
    glBufferData(GL_ARRAY_BUFFER, graphics.noiseVertices.size() * sizeof(float), graphics.noiseVertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, noiseEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, noiseIndices.size() * sizeof(unsigned int), noiseIndices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// Helper function to create a 4x4 identity matrix
void mat4Identity(float* mat) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
}

// Helper function to create a perspective projection matrix
void mat4Perspective(float* mat, float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    mat4Identity(mat);
    mat[0] = 1.0f / (aspect * tanHalfFov);
    mat[5] = 1.0f / tanHalfFov;
    mat[10] = -(far + near) / (far - near);
    mat[11] = -1.0f;
    mat[14] = -(2.0f * far * near) / (far - near);
    mat[15] = 0.0f;
}

// Helper function to create an orthographic projection matrix
void mat4Ortho(float* mat, float left, float right, float bottom, float top, float near, float far) {
    mat4Identity(mat);
    mat[0] = 2.0f / (right - left);
    mat[5] = 2.0f / (top - bottom);
    mat[10] = -2.0f / (far - near);
    mat[12] = -(right + left) / (right - left);
    mat[13] = -(top + bottom) / (top - bottom);
    mat[14] = -(far + near) / (far - near);
}

// Helper function to create a look-at view matrix
void mat4LookAt(float* mat, float eyeX, float eyeY, float eyeZ,
                float centerX, float centerY, float centerZ,
                float upX, float upY, float upZ) {
    // Calculate forward vector
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;
    float fLen = sqrt(fx*fx + fy*fy + fz*fz);
    fx /= fLen; fy /= fLen; fz /= fLen;

    // Calculate right vector
    float rx = fy * upZ - fz * upY;
    float ry = fz * upX - fx * upZ;
    float rz = fx * upY - fy * upX;
    float rLen = sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rLen; ry /= rLen; rz /= rLen;

    // Calculate up vector
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    mat4Identity(mat);
    mat[0] = rx; mat[4] = ry; mat[8] = rz;
    mat[1] = ux; mat[5] = uy; mat[9] = uz;
    mat[2] = -fx; mat[6] = -fy; mat[10] = -fz;
    mat[12] = -(rx * eyeX + ry * eyeY + rz * eyeZ);
    mat[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
    mat[14] = (fx * eyeX + fy * eyeY + fz * eyeZ);
}

// Helper function to create a rotation matrix around Y axis
void mat4RotateY(float* mat, float angle) {
    mat4Identity(mat);
    float c = cos(angle);
    float s = sin(angle);
    mat[0] = c;
    mat[2] = s;
    mat[8] = -s;
    mat[10] = c;
}

// Resize framebuffer and plane when window size changes
void resize3DRendering(Graphics& graphics, int width, int height) {
    if (width == graphics.uiWidth && height == graphics.uiHeight) {
        return; // No change, skip resize
    }

    graphics.uiWidth = width;
    graphics.uiHeight = height;

    // Resize framebuffer texture
    glBindTexture(GL_TEXTURE_2D, graphics.fboTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Resize depth/stencil renderbuffer
    glBindRenderbuffer(GL_RENDERBUFFER, graphics.fboDepthRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Update plane geometry to match aspect ratio
    float aspectRatio = (float)width / (float)height;
    float planeWidth = 2.0f * aspectRatio;
    float planeHeight = 2.0f;

    float vertices[] = {
        // positions                              // tex     // bary (unused)   // glow  // centroid offset
        -planeWidth/2, -planeHeight/2, 0.0f,   0.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom left
         planeWidth/2, -planeHeight/2, 0.0f,   1.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom right
         planeWidth/2,  planeHeight/2, 0.0f,   1.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // top right
        -planeWidth/2,  planeHeight/2, 0.0f,   0.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f   // top left
    };

    glBindBuffer(GL_ARRAY_BUFFER, graphics.planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Update grid geometry to match aspect ratio
    std::vector<float> gridVertices;
    std::vector<unsigned int> gridIndices;
    generateTriangularGrid(gridVertices, gridIndices, planeWidth, planeHeight, 80, 80);

    graphics.gridVertexCount = gridIndices.size();

    // Reinitialize particles for new grid dimensions
    initializeParticles(graphics, gridVertices, planeWidth, planeHeight, 4.0f);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.gridVBO);
    glBufferData(GL_ARRAY_BUFFER, graphics.gridVertices.size() * sizeof(float), graphics.gridVertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphics.gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndices.size() * sizeof(unsigned int), gridIndices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Cleanup 3D resources
void cleanup3DRendering(Graphics& graphics) {
    glDeleteFramebuffers(1, &graphics.fbo);
    glDeleteTextures(1, &graphics.fboTexture);
    glDeleteRenderbuffers(1, &graphics.fboDepthRenderBuffer);
    glDeleteVertexArrays(1, &graphics.planeVAO);
    glDeleteBuffers(1, &graphics.planeVBO);
    glDeleteBuffers(1, &graphics.planeEBO);
    glDeleteVertexArrays(1, &graphics.gridVAO);
    glDeleteBuffers(1, &graphics.gridVBO);
    glDeleteBuffers(1, &graphics.gridEBO);
    glDeleteProgram(graphics.shaderProgram);
}

int main(int, char *[]) {

    ecs_os_set_api_defaults();
    ecs_os_api_t os_api = ecs_os_get_api();
    os_api.perf_trace_push_ = trace_push;
    os_api.perf_trace_pop_ = trace_pop;
    ecs_os_set_api(&os_api);

    flecs::world world_instance;
    world = &world_instance;

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

    query_server::initialize(world, &spatial_manager);

    // TODO: query_server::register_spatial_handler
    // Can this be moved elsewhere?

    // Start socket server
    int port = 8000;
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
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Thornfield", monitor, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
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

    world->component<DiurnalHour>();

    world->component<ChatMessage>();
    world->component<ChatMessageView>();
    world->component<ChatState>().add(flecs::Singleton);
    world->component<ChatPanel>();
    world->component<FocusChatInput>();
    world->component<SendChatMessage>();
    world->set<ChatState>({std::vector<ChatMessage>{}, "", false});

    // SFTP components
    world->component<SFTPClient>();
    world->component<FileTransferProgress>();
    world->component<FileTransferRequest>();
    world->component<HasSFTPTransfer>();

    world->component<DragContext>().add(flecs::Singleton);
    world->set<DragContext>({false, flecs::entity::null(), PanelSplitType::Horizontal, 0.0f});

    world->observer<ImageCreator, Graphics>()
    .event(flecs::OnSet)
    .each([&](flecs::entity e, ImageCreator& img, Graphics& graphics)
    {
        int imgHandle = nvgCreateImage(graphics.vg, ("../assets/" + img.path).c_str(), 0);

        if (imgHandle == -1) {
            std::cerr << "Failed to load " << img.path << std::endl;
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
    .term_at(1).second<FocusChatInput>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds&, AddTagOnLeftClick)
    {
        ChatState& chat = world->ensure<ChatState>();
        chat.input_focused = true;
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

        auto editor_type_selector = world->entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        // .add<DebugRenderBounds>()
        .set<Position, Local>({-1.0f, 19.0f});

        auto editor_type_selector_square_corner = world->entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<RectRenderable>({16.0f, 16.0f, false, 0x282828FF})
        .set<ZIndex>({30});

        auto editor_type_selector_bkg = world->entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<RoundedRectRenderable>({196.0f, 256.0f, 4.0f, false, 0x282828FF})
        .set<Expand>({false, 0, 0, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({30});

        auto editor_type_list = world->entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<LayoutBox>({LayoutBox::Vertical, 2.0f})
        // .add<DebugRenderBounds>()
        .set<Position, Local>({4.0f, 4.0f});

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
            .set<RoundedRectRenderable>({196.0f-12.0f, 20.0f, 2.0f, false, 0x383838FF})
            .set<Position, Local>({2.0f, 0.0f})
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
            .set<TextRenderable>({editor_type_name.c_str(), "ATARISTOCRAT", 16.0f, 0xFFFFFFFF})
            .set<Position, Local>({4.0f, 2.0f})
            .set<ZIndex>({40});
            
            editor_type_index++;
        }

    });

    // Create text entities with different z-indices
    // auto text1 = world->entity("Text1")
    //     .is_a(UIElement)
    //     .set<Position, Local>({400.0f, 100.0f})
    //     .set<TextRenderable>({"Behind boxes", "ATARISTOCRAT", 24.0f, 0xFFFFFFFF, NVG_ALIGN_CENTER})
    //     .set<ZIndex>({0});

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
        .is_a(UIElement)
        .set<ImageCreator>({"../assets/ecs_header.png", 1.0f, 1.0f})
        .set<ZIndex>({5});

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
            auto canvas = leaf.target<EditorCanvas>();
            const RectRenderable* canvas_rect = canvas.try_get<RectRenderable>();
            if (!canvas_rect) return;

            float canvas_w = canvas_rect->width;
            float canvas_h = canvas_rect->height;
            const float pad = 8.0f;
            const float input_h = 72.0f;  // Taller input to accommodate multi-line text

            auto& messages_rect = panel.messages_panel.ensure<RoundedRectRenderable>();
            messages_rect.width = canvas_w - pad * 2.0f;
            messages_rect.height = canvas_h - input_h - pad * 3.0f;
            panel.messages_panel.ensure<Position, Local>() = {pad, pad};

            auto& input_rect = panel.input_panel.ensure<RoundedRectRenderable>();
            input_rect.width = canvas_w - pad * 2.0f;
            input_rect.height = input_h;
            panel.input_panel.ensure<Position, Local>() = {pad, canvas_h - input_h - pad};

            ChatState& chat = world->ensure<ChatState>();
            std::string caret = chat.input_focused ? "|" : "";
            if (auto* input_tr = panel.input_text.try_get_mut<TextRenderable>())
            {
                input_tr->text = chat.draft + caret;
                input_tr->wrapWidth = input_rect.width - 16.0f;  // Wrap within input panel with padding
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

                if (!expand) {
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

                if (!expand) {
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

    // world->system<ImageRenderable, ProportionalConstraint, Graphics>()
    // {

    // }

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

    auto debugRenderBounds = world->system<RenderQueue, UIElementBounds, DebugRenderBounds>()
    .term_at(0).src(renderQueueEntity)
    .each([](flecs::entity e, RenderQueue& render_queue, UIElementBounds& bounds, DebugRenderBounds)
    {
        RectRenderable debug_bound {bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin, true, 0xFFFF00FF};
        render_queue.addRectCommand({bounds.xmin, bounds.ymin}, debug_bound, 100);
    });

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
            queue.addRoundedRectCommand(pos, renderable, zIndex.layer, rg, rg ? *rg : RenderGradient{0, 0});
        });


    auto rectQueueSystem = world->system<Position, RectRenderable, ZIndex, RenderStatus>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
        .each([&](flecs::entity e, Position& pos, RectRenderable& renderable, ZIndex& zIndex, RenderStatus& status) {
            if (status.visible)
            {
                RenderQueue& queue = world->ensure<RenderQueue>();
                // flecs::entity scissorEntity = flecs::entity::null();
                if (e.has<ScissorContainer>(flecs::Wildcard))
                {
                    flecs::entity scissorEntity = e.target<ScissorContainer>();
                    queue.commands.push_back({pos, renderable, RenderType::Rectangle, zIndex.layer, scissorEntity});
                    // TODO: Pushback target UIElementBounds as the scissorRegion of RenderCommand
                } else
                {
                    queue.commands.push_back({pos, renderable, RenderType::Rectangle, zIndex.layer, 0});
                }
            }
        });

    world->system<TextRenderable, DynamicTextWrap>()
    .kind(flecs::PreUpdate)
    .with<DynamicTextWrapContainer>(flecs::Wildcard)
    .each([&](flecs::entity e, TextRenderable& text, DynamicTextWrap& data)
    {
        UIElementSize& uiElementSize = e.target<DynamicTextWrapContainer>().ensure<UIElementSize>();
        text.wrapWidth = uiElementSize.width - data.pad;
        std::cout << "Set text wrap to " << text.wrapWidth;
    });

    auto textQueueSystem = world->system<Position, TextRenderable, ZIndex, RenderGradient*>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .term_at(3).optional()
    .each([&](flecs::entity e, Position& pos, TextRenderable& renderable, ZIndex& zIndex, RenderGradient* rg) {
        RenderQueue& queue = world->ensure<RenderQueue>();
        queue.addTextCommand(pos, renderable, zIndex.layer, rg, rg ? *rg : RenderGradient{0, 0});
    });

    auto imageQueueSystem = world->system<Position, ImageRenderable, ZIndex, RenderStatus>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, ImageRenderable& renderable, ZIndex& zIndex, RenderStatus& status) {
        RenderQueue& queue = world->ensure<RenderQueue>();
        if (status.visible)
        {
            // queue.addImageCommand(pos, renderable, zIndex.layer);
                if (e.has<ScissorContainer>(flecs::Wildcard))
                {
                    flecs::entity scissorEntity = e.target<ScissorContainer>();
                    queue.commands.push_back({pos, renderable, RenderType::Image, zIndex.layer, scissorEntity});
                } else
                {
                    queue.commands.push_back({pos, renderable, RenderType::Image, zIndex.layer, 0});
                }
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
        if (e.has<ScissorContainer>(flecs::Wildcard))
        {
            flecs::entity scissorEntity = e.target<ScissorContainer>();
            queue.commands.push_back({pos, renderable, RenderType::CustomRenderable, zIndex.layer, scissorEntity});
            // TODO: Pushback target UIElementBounds as the scissorRegion of RenderCommand
        } else
        {
            queue.commands.push_back({pos, renderable, RenderType::CustomRenderable, zIndex.layer, 0});
        }
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

        // DEBUG BOUNDS
        // for (EditorShiftRegion& shift_region : editor_root->shift_regions)
        // {
        //     RenderQueue& queue = world->ensure<RenderQueue>();
        //     RectRenderable debug_rect;
        //     debug_rect.width = shift_region.bounds.xmax - shift_region.bounds.xmin;
        //     debug_rect.height = shift_region.bounds.ymax - shift_region.bounds.ymin;
        //     debug_rect.color = 0xFF00FFFF;
        //     debug_rect.stroke = true;
        //     queue.addRectCommand({shift_region.bounds.xmin, shift_region.bounds.ymin}, debug_rect, 100);
        // }

        // std::cout << cursor_state->x << ", " << cursor_state->y << std::endl;

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
            // TODO: Apply scissor regions to relevant entity
            for (const auto& cmd : queue.commands) {
                if (ecs_is_valid(*world, cmd.scissorEntity) && ecs_is_alive(*world, cmd.scissorEntity))
                {
                    UIElementBounds* scissorBounds = ecs_ensure(*world, cmd.scissorEntity, UIElementBounds);
                    nvgScissor(graphics.vg, scissorBounds->xmin, scissorBounds->ymin, scissorBounds->xmax - scissorBounds->xmin, scissorBounds->ymax - scissorBounds->ymin);
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

            // Base position: parent left edge (no scroll offset - data flows naturally via circular buffer)
            // Right edge is always "now", newest data point at parent_bounds->xmax
            float base_x = parent_bounds->xmin;
            float y_bottom = bounds.ymax;

            size_t num_points = chart.size();

            // Calculate point spacing - chart extends one frame past each edge (left and right)
            // Using capacity-3 makes capacity points span width + 2*point_spacing
            float point_spacing = width / (float)(chart.capacity > 3 ? chart.capacity - 3 : 1);

            // If we have data, draw it
            if (num_points >= 1) {
                // Build polygon path: start at bottom of first point, trace line, close at bottom
                nvgBeginPath(graphics.vg);

                // Calculate x offset:
                // - Start one frame to the left of visible area (- point_spacing)
                // - Adjust for partially filled buffer
                // - Apply smooth scroll offset
                float x_offset = (float)(chart.capacity - num_points) * point_spacing - chart.scroll_offset * point_spacing - point_spacing;

                // Start at bottom of the first data point's x position
                float first_data_x = base_x + x_offset;
                nvgMoveTo(graphics.vg, first_data_x, y_bottom);

                // Draw line through all data points (from oldest to newest)
                for (size_t i = 0; i < num_points; i++) {
                    float value = chart.get(i);
                    // Normalize value to 0-1 range
                    float normalized = (value - chart.min_value) / (chart.max_value - chart.min_value);
                    normalized = std::max(0.0f, std::min(1.0f, normalized));

                    float x = base_x + x_offset + i * point_spacing;
                    float y = y_bottom - normalized * height;

                    nvgLineTo(graphics.vg, x, y);
                }

                // Close polygon: line to bottom at last point, then back to start
                float last_data_x = base_x + x_offset + (num_points - 1) * point_spacing;
                nvgLineTo(graphics.vg, last_data_x, y_bottom);
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
                    float value = chart.get(i);
                    float normalized = (value - chart.min_value) / (chart.max_value - chart.min_value);
                    normalized = std::max(0.0f, std::min(1.0f, normalized));

                    float x = base_x + x_offset + i * point_spacing;
                    float y = y_bottom - normalized * height;

                    if (i == 0) {
                        nvgMoveTo(graphics.vg, x, y);
                    } else {
                        nvgLineTo(graphics.vg, x, y);
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

    // SFTP progress indicator rendering system
    auto sftpProgressRenderSystem = world->system<Position, ImageRenderable>()
        .with<IsStreamingFrom>(flecs::Wildcard)
        .term_at(0).second<World>()
        .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, Position& pos, ImageRenderable& img) {
            flecs::entity vnc_entity = e.target<IsStreamingFrom>();

            // Check if this VNC has an active SFTP transfer
            if (!vnc_entity.has<HasSFTPTransfer>()) return;

            const SFTPClient* sftp = vnc_entity.try_get<SFTPClient>();
            if (!sftp) return;

            // Read progress (thread-safe)
            FileTransferProgress progress;
            {
                std::lock_guard<std::mutex> lock(sftp->progress_mutex);
                progress = sftp->current_progress;
            }

            // Skip if idle or hide after 2 seconds of completion
            if (progress.state == FileTransferProgress::IDLE) return;

            if (progress.state == FileTransferProgress::COMPLETED) {
                auto elapsed = std::chrono::steady_clock::now() - progress.completion_time;
                if (elapsed > std::chrono::seconds(2)) {
                    vnc_entity.remove<HasSFTPTransfer>();
                    return;
                }
            }

            // Calculate position (bottom-right of VNC panel)
            float indicator_width = 300.0f;
            float indicator_height = 60.0f;
            float indicator_x = pos.x + img.width - indicator_width - 20.0f;
            float indicator_y = pos.y + img.height - indicator_height - 20.0f;

            // Add to render queue
            RenderQueue& queue = world->ensure<RenderQueue>();

            // Background
            queue.addRoundedRectCommand(
                {indicator_x, indicator_y},
                {indicator_width, indicator_height, 8.0f, false, 0x000000AA},
                1500  // High Z-index
            );

            // Progress bar background
            float bar_x = indicator_x + 10.0f;
            float bar_y = indicator_y + 35.0f;
            float bar_width = indicator_width - 20.0f;
            float bar_height = 15.0f;

            queue.addRoundedRectCommand(
                {bar_x, bar_y},
                {bar_width, bar_height, 4.0f, false, 0x333333FF},
                1501
            );

            // Progress bar fill
            if (progress.progress_percent > 0) {
                float fill_width = bar_width * (progress.progress_percent / 100.0f);
                uint32_t fill_color = progress.state == FileTransferProgress::FAILED ?
                    0xFF0000FF : 0x00AA00FF;  // Red for error, green for success

                queue.addRoundedRectCommand(
                    {bar_x, bar_y},
                    {fill_width, bar_height, 4.0f, false, fill_color},
                    1502
                );
            }

            // Filename text
            queue.addTextCommand(
                {indicator_x + 10.0f, indicator_y + 15.0f},
                {progress.filename, "sans-bold", 14.0f, 0xFFFFFFFF, 1.0f},
                1503
            );

            // Progress percentage text
            char percent_str[32];
            snprintf(percent_str, sizeof(percent_str), "%.1f%%", progress.progress_percent);

            queue.addTextCommand(
                {bar_x + bar_width - 50.0f, bar_y + 12.0f},
                {std::string(percent_str), "sans", 12.0f, 0xFFFFFFFF, 1.0f},
                1503
            );

            // Error message if failed
            if (progress.state == FileTransferProgress::FAILED && !progress.error_message.empty()) {
                queue.addTextCommand(
                    {indicator_x + 10.0f, indicator_y + 45.0f},
                    {progress.error_message, "sans", 11.0f, 0xFF0000FF, 1.0f},
                    1503
                );
            }
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
        float melSpecOffset = 0.0f;
        auto sysAudioRenderer = world->lookup("SystemAudioRenderer");
        if (sysAudioRenderer.is_valid()) {
            const MelSpecRender* melSpec = sysAudioRenderer.try_get<MelSpecRender>();
            if (melSpec && melSpec->fillProgress >= 1.0f) {
                melSpecOffset = melSpec->renderOffset;
            }
        }

        if (data.mode == FilmstripMode::Uniform) {
            // UNIFORM MODE: Position each child frame directly based on index and scroll offset
            // Right-aligned: frames start from the right side, newest frame at rightmost
            int frame_index = 0;
            int num_frames = 0;
            e.children([&](flecs::entity) { num_frames++; });

            e.children([&](flecs::entity child)
            {
                // Right-align: offset so last frame is at position frame_limit * frame_width (rightmost)
                // base_offset puts first frame at the right position when we have fewer than max frames
                int base_offset = data.frame_limit + 1 - num_frames;
                // Apply mel spec offset (shift right when mel spec is behind to match its timing)
                float x_pos = ((base_offset + frame_index) * frame_width) - (data.scroll_offset * frame_width) + (melSpecOffset * container_width);

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
                frame_index++;
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

        // Update smooth scroll offset every frame (0 to 1 over sample_interval)
        if (chart.sample_interval > 0) {
            chart.scroll_offset = std::min(1.0f, chart.time_since_sample / chart.sample_interval);
        }

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
            chart.push(similarity);  // This resets scroll_offset to 0
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
            if (surface && surface->pixels) {
                // Submit vision processing job to background thread instead of blocking
                VisionProcessingJob job;
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

                // Submit frame to DINO embedder (async, non-blocking)
                if (g_dinoEmbedder.isLoaded()) {
                    g_dinoEmbedder.submitFrame(job.pixelData.data(), job.width, job.height, job.pitch);
                    // Results are printed by the worker thread
                }

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
                bool shouldCapture = false;
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
                                // Uniform mode: capture every 3 seconds
                                shouldCapture = (currentTime - vnc.lastCaptureTime >= 3.0);
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
                                // capture time for Uniform mode
                                double frameTime = (filmstripData.mode == FilmstripMode::Stegosaurus)
                                    ? filmstripData.pending_spike_time
                                    : currentTime;

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
    int interFontHandle = nvgCreateFont(vg, "Inter", "../assets/CharisSIL-Regular.ttf");

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

        glfwStateEntity.set<Window>({window, winWidth, winHeight});

        nvgBeginFrame(vg, graphics.uiWidth, graphics.uiHeight, 1.0f);

        world->defer_begin();
        glfwPollEvents();
        world->defer_end();
        world->progress();

        // Process completed interpretations and create badges
        {
            std::lock_guard<std::mutex> lock(pending_interpretations_mutex);
            auto it = pending_interpretations.begin();
            while (it != pending_interpretations.end()) {
                if ((*it)->completed.load()) {
                    auto& pending = *it;
                    auto UIElement = world->lookup("UIElement");

                    // Create "Heonae understands" header
                    auto meta_response = world->entity()
                        .is_a(UIElement)
                        .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
                        .add(flecs::OrderedChildren)
                        .child_of(pending->message_list);

                    // TODO: Reification underline or sphere indicator
                    // create_badge(meta_response, UIElement, "Heonae", 0xc72783ff, false, false, "1");
                    // create_badge(meta_response, UIElement, "understands", 0xc72783ff, false, true);

                    // Split draft into words for interleaving
                    std::vector<std::string> words;
                    std::istringstream iss(pending->draft);
                    std::string word;
                    while (iss >> word) {
                        words.push_back(word);
                    }

                    // If no LLM interpretation, just create text entities without badges
                    if (pending->result.empty()) {
                        auto meta_response_data = world->entity()
                            .is_a(UIElement)
                            // .set<FlowLayoutBox>({0.0f, 0.0f, 2.0f, 0.0f, 2.0f})
                            // .set<LayoutBox>({LayoutBox::Horizontal, 0.0f})
                            .set<FlowLayoutBox>({0.0f, 0.0f, 2.0f, 0.0f, 2.0f})
                            // .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
                            .add(flecs::OrderedChildren)
                            .child_of(pending->message_list);

                        // Register message in global list
                        g_annotatable_messages.push_back({pending->draft, meta_response_data});
                        int msg_idx = (int)g_annotatable_messages.size() - 1;

                        // Create or update the single global annotation selector
                        static flecs::entity g_annotation_entity = flecs::entity::null();
                        if (!g_annotation_entity.is_valid() || !g_annotation_entity.is_alive()) {
                            g_annotation_entity = world->entity()
                                .set<WordAnnotationSelector>({
                                    pending->draft,           // sentence_template
                                    {},                       // ui_entities
                                    {},                       // selection_entities
                                    meta_response_data,       // parent_entity
                                    0, 0,                     // start_index, end_index
                                    0,                        // token_count
                                    false,                    // active
                                    true,                     // dirty
                                    0x4488FFAA                // highlight_color
                                })
                                .set<Position, World>({0, 0})
                                .set<RectRenderable>({0, 0, false, 0x4488FFAA})
                                .set<RenderStatus>({true})
                                .set<ZIndex>({15});
                        } else {
                            // Save current message state before switching
                            WordAnnotationSelector& selector = g_annotation_entity.ensure<WordAnnotationSelector>();
                            if (g_current_message_idx >= 0 && g_current_message_idx < (int)g_annotatable_messages.size()) {
                                auto& old_msg = g_annotatable_messages[g_current_message_idx];
                                old_msg.sentence_template = selector.sentence_template;
                                old_msg.ui_entities = selector.ui_entities;
                                old_msg.selection_entities = selector.selection_entities;
                                old_msg.token_count = selector.token_count;
                            }
                            // Point selector to new message (don't destroy old entities!)
                            selector.ui_entities.clear();
                            selector.selection_entities.clear();
                            selector.sentence_template = pending->draft;
                            selector.parent_entity = meta_response_data;
                            selector.start_index = 0;
                            selector.end_index = 0;
                            selector.dirty = true;
                        }

                        g_current_message_idx = msg_idx;

                        // Create UI entities immediately
                        WordAnnotationSelector& selector = g_annotation_entity.ensure<WordAnnotationSelector>();
                        recreate_annotation_entities(selector);
                    } else {
                    // Parse JSON and create dynamic badges
                    try {
                        json result = json::parse(pending->result);

                        // Binding type for relationship slots
                        enum class SlotBindingType {
                            Standard,  // Single bound entity
                            Set,       // Multiple bound entities
                            Wildcard   // Unbound/intensional slot
                        };

                        // Build a map of word indices to node/edge info
                        struct Annotation {
                            std::string label;
                            uint32_t color;
                            bool is_relationship;
                            bool is_binding;  // True if this binds to an existing entity
                            // For relationships: vectors of source/target IDs and colors
                            std::vector<std::string> prefix_ids;   // Source node numbers (for relationships) or empty
                            std::vector<uint32_t> prefix_tints;    // Source node colors
                            std::vector<std::string> postfix_ids;  // Target node numbers (for relationships) or node's own number
                            std::vector<uint32_t> postfix_tints;   // Target node colors or node's own color
                            SlotBindingType prefix_type;   // Type of source binding
                            SlotBindingType postfix_type;  // Type of target binding
                            int end_idx;  // End index of this annotation span
                        };
                        std::map<int, Annotation> start_annotations;

                        // Helper to determine slot binding type
                        auto get_slot_type = [](const std::vector<std::string>& ids) -> SlotBindingType {
                            if (ids.empty()) return SlotBindingType::Standard;
                            if (ids.size() == 1 && ids[0] == "*") return SlotBindingType::Wildcard;
                            if (ids.size() > 1) return SlotBindingType::Set;
                            return SlotBindingType::Standard;
                        };

                        // Parse hex color string to uint32_t with alpha
                        auto parse_hex_color = [](const std::string& hex) -> uint32_t {
                            std::string clean_hex = hex;
                            // Remove leading '#' if present
                            if (!clean_hex.empty() && clean_hex[0] == '#') {
                                clean_hex = clean_hex.substr(1);
                            }
                            // Parse as RGB, add full alpha
                            uint32_t rgb = std::stoul(clean_hex, nullptr, 16);
                            return (rgb << 8) | 0xff;
                        };

                        // Fallback color generator based on ID hash
                        auto hash_color = [](const std::string& id) -> uint32_t {
                            std::hash<std::string> hasher;
                            size_t h = hasher(id);
                            uint8_t r = 80 + (h % 150);
                            uint8_t g = 80 + ((h >> 8) % 150);
                            uint8_t b = 80 + ((h >> 16) % 150);
                            return (r << 24) | (g << 16) | (b << 8) | 0xff;
                        };

                        std::map<std::string, uint32_t> node_colors;
                        std::map<std::string, std::string> node_numbers;  // Map node ID to its display number
                        std::vector<KnownEntity> new_entities;  // New entities to add after processing

                        if (result.contains("nodes")) {
                            for (auto& node : result["nodes"]) {
                                std::string id = node["id"].get<std::string>();
                                std::string label = node["label"].get<std::string>();
                                int start_idx = node["start_index"].get<int>();
                                int end_idx = node["end_index"].get<int>();
                                bool is_new = node.value("is_new", true);
                                std::string binds_to = node.value("binds_to", "");

                                uint32_t color;
                                std::string display_num;
                                bool is_binding = false;
                                std::string color_hex;

                                if (!is_new && !binds_to.empty()) {
                                    // This is a binding to an existing entity
                                    is_binding = true;
                                    std::lock_guard<std::mutex> lock(known_entities_mutex);
                                    for (const auto& known : known_entities) {
                                        if (known.id == binds_to) {
                                            color = parse_hex_color(known.color);
                                            display_num = std::to_string(known.display_number);
                                            color_hex = known.color;
                                            break;
                                        }
                                    }
                                    // Fallback if binding target not found
                                    if (display_num.empty()) {
                                        color = hash_color(id);
                                        display_num = "?";
                                    }
                                } else {
                                    // This is a new entity
                                    if (node.contains("color")) {
                                        try {
                                            color_hex = node["color"].get<std::string>();
                                            color = parse_hex_color(color_hex);
                                        } catch (...) {
                                            color = hash_color(id);
                                            // Convert hash color back to hex for storage
                                            std::stringstream ss;
                                            ss << std::hex << ((color >> 8) & 0xFFFFFF);
                                            color_hex = ss.str();
                                        }
                                    } else {
                                        color = hash_color(id);
                                        std::stringstream ss;
                                        ss << std::hex << ((color >> 8) & 0xFFFFFF);
                                        color_hex = ss.str();
                                    }

                                    // Assign new display number
                                    int assigned_num;
                                    {
                                        std::lock_guard<std::mutex> lock(known_entities_mutex);
                                        assigned_num = next_entity_number++;
                                    }
                                    display_num = std::to_string(assigned_num);

                                    // Queue this entity to be added to known entities
                                    new_entities.push_back({id, label, color_hex, assigned_num});
                                }

                                node_colors[id] = color;
                                node_numbers[id] = display_num;

                                // For nodes: no prefix, single postfix with the node's number
                                start_annotations[start_idx] = {label, color, false, is_binding, {}, {}, {display_num}, {color}, SlotBindingType::Standard, SlotBindingType::Standard, end_idx};
                            }
                        }

                        // Add new entities to the global known entities list
                        {
                            std::lock_guard<std::mutex> lock(known_entities_mutex);
                            for (const auto& entity : new_entities) {
                                known_entities.push_back(entity);
                            }
                        }

                        // Process edges (relationships)
                        if (result.contains("edges")) {
                            for (auto& edge : result["edges"]) {
                                bool in_situ = edge.value("in-situ", false);
                                if (in_situ && edge.contains("start_index")) {
                                    std::string rel = edge["relationship"].get<std::string>();
                                    int start_idx = edge["start_index"].get<int>();
                                    int end_idx = edge["end_index"].get<int>();

                                    // Use semantic color from API if available
                                    uint32_t color;
                                    if (edge.contains("color")) {
                                        try {
                                            color = parse_hex_color(edge["color"].get<std::string>());
                                        } catch (...) {
                                            color = 0xad734bff;
                                        }
                                    } else {
                                        color = 0xad734bff;
                                    }

                                    // Wildcard color - muted purple/gray for unbound slots
                                    const uint32_t WILDCARD_COLOR = 0x8866aaff;

                                    // Collect source IDs and colors (now an array)
                                    std::vector<std::string> source_nums;
                                    std::vector<uint32_t> source_colors;
                                    if (edge.contains("sources")) {
                                        for (const auto& src : edge["sources"]) {
                                            std::string source = src.get<std::string>();
                                            if (source == "*") {
                                                // Wildcard - unbound slot
                                                source_nums.push_back("*");
                                                source_colors.push_back(WILDCARD_COLOR);
                                            } else {
                                                source_nums.push_back(node_numbers.count(source) ? node_numbers[source] : "?");
                                                source_colors.push_back(node_colors.count(source) ? node_colors[source] : 0x888888ff);
                                            }
                                        }
                                    }

                                    // Collect target IDs and colors (now an array)
                                    std::vector<std::string> target_nums;
                                    std::vector<uint32_t> target_colors;
                                    if (edge.contains("targets")) {
                                        for (const auto& tgt : edge["targets"]) {
                                            std::string target = tgt.get<std::string>();
                                            if (target == "*") {
                                                // Wildcard - unbound slot
                                                target_nums.push_back("*");
                                                target_colors.push_back(WILDCARD_COLOR);
                                            } else {
                                                target_nums.push_back(node_numbers.count(target) ? node_numbers[target] : "?");
                                                target_colors.push_back(node_colors.count(target) ? node_colors[target] : 0x888888ff);
                                            }
                                        }
                                    }

                                    // Determine binding types
                                    SlotBindingType src_type = get_slot_type(source_nums);
                                    SlotBindingType tgt_type = get_slot_type(target_nums);

                                    start_annotations[start_idx] = {rel, color, true, false, source_nums, source_colors, target_nums, target_colors, src_type, tgt_type, end_idx};
                                }
                            }
                        }

                        // Create flow layout for interleaved text and badges
                        auto meta_response_data = world->entity()
                            .is_a(UIElement)
                            .set<FlowLayoutBox>({0.0f, 0.0f, 2.0f, 0.0f, 2.0f})
                            .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
                            .add(flecs::OrderedChildren)
                            .child_of(pending->message_list);
                        
                        // We're currently disabling the model to test annotation
                        for (size_t i = 0; i < words.size(); i++) 
                        {
                            flecs::entity text_annotator = world->entity()
                                .is_a(UIElement)
                                .set<UIContainer>({4, 4})
                                .set<RoundedRectRenderable>({0.0f, 0.0f, 2.0f, true, 0xFFFFFFFF})
                                .set<ZIndex>({20})
                                .child_of(meta_response_data);
                                
                            flecs::entity text_seq = world->entity()
                                .is_a(UIElement)
                                // .child_of(meta_response_data)
                                .child_of(text_annotator)
                                .set<TextRenderable>({words[i].c_str(), "Inter", 16.0f, 0x777777FF})
                                .set<ZIndex>({17});
                        }

                        // Interleave text with badges - badges replace their word spans
                        // std::string current_text;
                        // for (size_t i = 0; i < words.size(); ) {
                        //     // Check if an annotation starts at this position
                        //     if (start_annotations.count(i)) {
                        //         // Flush any accumulated text
                        //         if (!current_text.empty()) {

                        //             flecs::entity text_annotator = world->entity()
                        //                 .is_a(UIElement)
                        //                 .set<UIContainer>({4, 4})
                        //                 .set<RoundedRectRenderable>({0.0f, 0.0f, 2.0f, true, 0xFFFFFFFF})
                        //                 .set<ZIndex>({20})
                        //                 .child_of(meta_response_data);
                                        
                        //             flecs::entity text_seq = world->entity()
                        //                 .is_a(UIElement)
                        //                 // .child_of(meta_response_data)
                        //                 .child_of(text_annotator)
                        //                 .set<TextRenderable>({current_text.c_str(), "Inter", 16.0f, 0x777777FF})
                        //                 .set<ZIndex>({17});
                                    
                                        
                                    
                        //             current_text.clear();
                        //         }

                        //         auto& ann = start_annotations[i];
                        //         create_badge(meta_response_data, UIElement, ann.label.c_str(),
                        //                    ann.color, false, ann.is_relationship,
                        //                    ann.prefix_ids, ann.prefix_tints,
                        //                    ann.postfix_ids, ann.postfix_tints);

                        //         // Skip all words covered by this annotation (start_idx to end_idx inclusive)
                        //         i = ann.end_idx + 1;
                        //     } else {
                        //         // Add the word to accumulated text
                        //         if (!current_text.empty()) current_text += " ";
                        //         current_text += words[i];
                        //         i++;
                        //     }
                        // }

                        // Flush remaining text
                        // if (!current_text.empty()) {
                        //     world->entity()
                        //         .is_a(UIElement)
                        //         .child_of(meta_response_data)
                        //         .set<TextRenderable>({current_text.c_str(), "Inter", 16.0f, 0xFFFFFFFF})
                        //         .set<ZIndex>({17});
                        // }

                    } catch (const json::exception& e) {
                        std::cerr << "[Interpretation] JSON parse error: " << e.what() << std::endl;
                        std::cerr << "[Interpretation] Raw result: " << pending->result << std::endl;
                    }
                    } // end else (has result)

                    it = pending_interpretations.erase(it);
                } else {
                    ++it;
                }
            }
        }

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

    nvgDeleteGL2(vg);

    // Cleanup SFTP threads before shutdown
    auto sftp_cleanup = world->query<SFTPClient>();
    sftp_cleanup.each([](flecs::entity e, SFTPClient& sftp) {
        if (sftp.thread_running) {
            sftp.thread_should_stop = true;
            sftp.queue_cv.notify_all();
            if (sftp.worker_thread.joinable()) {
                sftp.worker_thread.join();
            }
        }
    });

    libssh2_exit();

    glfwTerminate();
    return 0;
}
