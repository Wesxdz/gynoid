#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
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
#include <raymath.h>

#include <ctime>
#include <chrono>

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

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <stack>

#include <libssh2.h>

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

struct HorizontalLayoutBox
{
  float x_progress;
  float padding = 0.0f;
  float move_dir = 1; // Right
};

struct FitChildren {};

struct VerticalLayoutBox 
{
  float y_progress;
  float padding;
  float move_dir = 1; // Down
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
    float scaleX, scaleY = 1.0;
};

struct ImageRenderable
{
    int imageHandle;
    float scaleX, scaleY;

    float width, height;
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

// Language Game chat components
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

struct Graphics {
    NVGcontext* vg;
};

enum class EditorType
{
    Void,
    PeachCore,
    VNCStream,
    Healthbar,
    // Respawn,
    // Genome,
    Embodiment,
    LanguageGame,
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

struct ProportionalConstraint {
    float max_width;
    float max_height;
};

// This needs to be refactored to be a direct enum relationship in flecs once you grow up and become a competent person
struct EditorLeafData
{
    EditorType editor_type;
};

struct SpaceframeChannel{};

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

void calculate_recursive_bounds(flecs::entity parent, UIElementBounds& total_bounds, bool& first) {
    // Iterate over all children of this entity
    parent.children([&](flecs::entity child) {
        if (child.has<UIElementBounds>() && !child.has<Expand>()) {
            const auto* child_bounds = child.try_get<UIElementBounds>();
            
            if (first) {
                total_bounds = *child_bounds;
                first = false;
            } else {
                if (child_bounds->xmin < total_bounds.xmin) total_bounds.xmin = child_bounds->xmin;
                if (child_bounds->ymin < total_bounds.ymin) total_bounds.ymin = child_bounds->ymin;
                if (child_bounds->xmax > total_bounds.xmax) total_bounds.xmax = child_bounds->xmax;
                if (child_bounds->ymax > total_bounds.ymax) total_bounds.ymax = child_bounds->ymax;
            }
        }
        
        // Recurse deeper if the child itself has children
        calculate_recursive_bounds(child, total_bounds, first);
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
void vnc_message_thread(VNCClient* vnc) {
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

flecs::entity create_badge(flecs::entity parent, flecs::entity UIElement, 
                           const char* text, uint32_t base_color, 
                           bool is_capsule = false, bool is_double_arrow = false, int bind_to_entity = -1) {
    
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

    int xPad = 6.0f;

    if (is_double_arrow)
    {
        xPad += 25.0f/2;
        badge = world->entity()
            .is_a(UIElement)
            .child_of(parent)
            .set<CustomRenderable>({100.0f, 25.0f, true, outline_color, draw_double_arrow})
            .set<RenderGradient>({dark, very_dark}) // Vertical gradient
            .set<UIContainer>({xPad, 3})
            .set<ZIndex>({20});

    } else
    {        
        badge = world->entity()
            .is_a(UIElement)
            .child_of(parent)
            .set<RoundedRectRenderable>({100.0f, badge_height, corner_radius, false, 0x000000FF})
            .set<RenderGradient>({dark, very_dark}) // Vertical gradient
            .set<UIContainer>({xPad, 3})
            .set<ZIndex>({20});
    
        // Outline Overlay
        world->entity()
            .is_a(UIElement)
            .child_of(badge)
            .set<Expand>({true, 0, 0, 1.0f, true, 0, 0, 1.0f})
            .set<RoundedRectRenderable>({100.0f, badge_height, corner_radius, true, outline_color})
            .set<ZIndex>({22});
    }

    flecs::entity badge_text_parent = badge;
    if (bind_to_entity >= 0)
    {
        auto badge_content = world->entity()
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 0.0f})
        .set<Position, Local>({xPad, 0.0f}) // reduce Y spacing for MNIST
        // .add<DebugRenderBounds>()
        .child_of(badge);
        
        badge_text_parent = badge_content;
    }

    // Text with Gradient
    world->entity()
        .is_a(UIElement)
        .child_of(badge_text_parent)
        .set<Position, Local>({xPad, 6.0f})
        .set<TextRenderable>({text, "Inter", 16.0f, white, 1.2f})
        .set<RenderGradient>({white, light})   // Apply gradient to text
        .set<ZIndex>({25});

    if (bind_to_entity >= 0)
    {
        world->entity()
        .is_a(UIElement)
        .child_of(badge_text_parent)
        .set<ImageCreator>({"../assets/mnist/set_0/" + std::to_string(bind_to_entity) + ".png", 0.9f, 0.9f})
        .set<ZIndex>({25});

        badge.set<UIContainer>({xPad, 0});
    }
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
    "VNC Stream",
    "Healthbar",
    // "Gynoid",
    "Embodiment",
    "Language Game", // Queue or Stream
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
    VNCClient* client;  // Pointer to avoid copying non-copyable VNCClient
};

VNCData get_vnc_source(const std::string& host, int port) {
    std::string addr = host + ":" + std::to_string(port);

    // 1. Check if we already have this connection
    auto existing = world->lookup(addr.c_str());
    if (existing && existing.has<VNCClient>()) {
        existing.ensure<VNCClient>().reference_count++;
        return {existing, &existing.ensure<VNCClient>()};
    }

    // 2. Create VNCClient with async connection setup
    VNCClient client_data;
    client_data.host = host;
    client_data.port = port;
    client_data.reference_count = 1;
    client_data.surfaceMutex = std::make_shared<std::mutex>();
    client_data.dirtyRectQueue = std::make_shared<DirtyRectQueue>();
    client_data.connectionState = VNCClient::CONNECTING;

    // Create OpenGL texture (will be resized when connected)
    glGenTextures(1, &client_data.vncTexture);
    glBindTexture(GL_TEXTURE_2D, client_data.vncTexture);
    // Allocate empty texture initially (will resize when connected)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Create PBO for async texture upload
    glGenBuffers(1, &client_data.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, client_data.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 1920 * 1080 * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    client_data.nvgHandle = nvglCreateImageFromHandleGL2(world->try_get<Graphics>()->vg,
        client_data.vncTexture, 1920, 1080, 0);

    // Create entity BEFORE starting thread (move client_data)
    flecs::entity created = world->entity(addr.c_str()).set<VNCClient>(std::move(client_data));

    // Start network thread AFTER entity is created
    VNCClient& vnc_ref = created.ensure<VNCClient>();
    vnc_ref.threadShouldStop = false;
    vnc_ref.messageThread = std::thread(vnc_message_thread, &vnc_ref);

    return {created, &vnc_ref};
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
        .set<HorizontalLayoutBox>({0.0f, 0.0f})
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

        std::vector<std::string> server_icons = {"aeri_memory", "flecs", "x11", "parakeet", "chatterbox", "doctr", "huggingface", "dino2", "autodistill", "yolo", "alpaca", "modal"};
        // std::vector<std::string> server_icons = {"peach_core"};

        for (const auto& icon : server_icons)
        {
            flecs::entity server_icon;

            if (icon == "chatterbox")
            { 
                server_icon = world->entity()
                .is_a(UIElement)
                .add<ServerHUDOverlay>(panel_overlay)
                .add<AddTagOnHoverEnter, ShowServerHUDOverlay>()
                .add<AddTagOnHoverExit, HideServerHUDOverlay>()
                .child_of(server_hud)
                .set<ImageCreator>({"../assets/server_hud/" + icon + ".png", 1.0f, 1.0f})
                .set<ZIndex>({10})
                .set<ServerScript>({"chatterbox", "chatterbox", "../chatter_server"})
                .set<Constrain>({true, true}) 
                .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f, true})
                .add(ServerStatus::Offline)

                .add<AddTagOnLeftClick, SelectServer>(); 
            } else
            {
                server_icon = world->entity()
                .is_a(UIElement)
                .add<ServerHUDOverlay>(panel_overlay)
                .add<AddTagOnHoverEnter, ShowServerHUDOverlay>()
                .add<AddTagOnHoverExit, HideServerHUDOverlay>()
                .child_of(server_hud)
                .set<Constrain>({true, true}) 
                .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f, true})
                .set<ImageCreator>({"../assets/server_hud/" + icon + ".png", 1.0f, 1.0f})
                .set<ZIndex>({10});
            }
            // TODO: Server dot should only exist if the server is active...
            world->entity()
            .is_a(UIElement)
            .child_of(server_icon)
            .set<ImageCreator>({"../assets/server_dot.png", 1.0f, 1.0f})
            .set<Align>({-0.5f, -0.5f, 0.5f, 0.9f})
            .set<ZIndex>({12});
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
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        create_badge(badges, UIElement, "Healthbar", 0x61c300ff);
    }
    else if (editor_type == EditorType::Embodiment)
    {
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
        .child_of(profile);

        auto badges = world->entity()
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        
        create_badge(badges, UIElement, "Heonae", 0xc72783ff);
        // create_badge(badges, UIElement, "Kahlo", 0x782910ff);
        create_badge(badges, UIElement, "Virtual", 0xe575eeff);
        create_badge(badges, UIElement, "Physical", 0x619393ff);
        
    }
    else if (editor_type == EditorType::LanguageGame)
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
            .set<ZIndex>({10});

        auto input_panel = world->entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .set<RoundedRectRenderable>({100.0f, 36.0f, 2.0f, true, 0x555555FF})
            .add<AddTagOnLeftClick, FocusChatInput>()
            .set<ZIndex>({10});

            // TODO: We need to scale the input bkg to the text/content size...
        auto input_bkg = world->entity()
            .is_a(UIElement)
            .child_of(input_panel)
            .set<RoundedRectRenderable>({10.0f, 10.0f, 2.0f, false, 0x222327FF})
            .set<Expand>({true, 0, 0, 1, true, 0, 0, 1})
            .set<ZIndex>({9});

        auto input_text = world->entity()
            .is_a(UIElement)
            // .add<DebugRenderBounds>()
            .child_of(input_panel)
            .set<Position, Local>({8.0f, 8.0f})
            .set<TextRenderable>({"", "Inter", 16.0f, 0xFFFFFFFF})
            .set<ZIndex>({12});

        auto message_list = world->entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .add(flecs::OrderedChildren)
            .set<Position, Local>({12.0f, 16.0f})
            .set<VerticalLayoutBox>({0.0f, 4.0f, 1.0f});

        auto msg_container = world->entity()
        .is_a(UIElement)
        .set<UIContainer>({4, 4})
        .child_of(message_list);

        auto meta_input = world->entity()
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .add(flecs::OrderedChildren)
        // .add<DebugRenderBounds>()
        .child_of(message_list);

        auto black_bkg = world->entity()
        .is_a(UIElement)
        .set<ZIndex>({8})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<RectRenderable>({0.0f, 0.0f, false, 0x000000FF})
        .child_of(leaf.target<EditorCanvas>());
        
        create_badge(meta_input, UIElement, "Wesley", 0xf5a652ff, false, false, 0);

        create_badge(meta_input, UIElement, "types", 0xa34d1aff, false, true);



        leaf.set<ChatPanel>({messages_panel, input_panel, input_text, message_list});
    }
    else if (editor_type == EditorType::Bookshelf)
    {
        auto bookshelf_layer = world->entity()
            .is_a(UIElement)
            .child_of(leaf.target<EditorCanvas>())
            .set<HorizontalLayoutBox>({0.0f, 8.0f})
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
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());

        create_badge(badges, UIElement, "System", 0x1361b0ff, false);
        create_badge(badges, UIElement, "Microphone", 0xc43131ff, false);

        // Create vertical layout for mel spectrograms
        auto hearing_layer = world->entity()
            .is_a(UIElement)
            .child_of(leaf.target<EditorCanvas>())
            .set<VerticalLayoutBox>({0.0f, 4.0f})
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
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
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
        // Modulus rows

        // TODO: Fractal granularity navigation
        auto channels = world->entity()
        .is_a(UIElement)
        .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        .add<VerticalLayoutBox>()
        .add(flecs::OrderedChildren)
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .child_of(leaf.target<EditorCanvas>());

        auto channels_2 = world->entity()
        .is_a(UIElement)
        .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        .add<VerticalLayoutBox>()
        .add(flecs::OrderedChildren)
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .child_of(leaf.target<EditorCanvas>());

        auto channels_3 = world->entity()
        .is_a(UIElement)
        .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
        .add<VerticalLayoutBox>()
        .add(flecs::OrderedChildren)
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .child_of(leaf.target<EditorCanvas>());

        auto frameChannel = world->entity()
        .is_a(UIElement)
        .add<FitChildren>()
        .set<Expand>({true, 0, 0, 1, true, 0, 0, 0.5})
        .add<HorizontalLayoutBox>()
        .add(flecs::OrderedChildren)
        .add<ScissorContainer>(leaf.target<EditorCanvas>())
        .child_of(leaf.target<EditorCanvas>());

        leaf.target<EditorCanvas>().add<SpaceframeChannel>(frameChannel);

        // TODO: Implement scissors/vertical scrollbar

        for (size_t i = 0; i < 6; i++)
        {
            world->entity()
                .is_a(UIElement)
                .child_of(channels)
                .set<RectRenderable>({10.0f, 24.0f, false, i % 2 == 0 ? 0x222327FF : 0x121212FF })
                .set<Expand>({true, 0, 0, 1, false, 0, 0, 0})
                .add<ScissorContainer>(leaf.target<EditorCanvas>())
                .set<ZIndex>({20});
        }
            
        for (size_t i = 0; i < 2; i++)
        {
            world->entity()
            .is_a(UIElement)
            .set<Align>({0.0f, 0.0f, 0.8f, 0.0f})
            .set<CustomRenderable>({24*3, 24*3, false, i % 2 == 1 ? 0x222327FF : 0x121212FF, draw_diamond})
            .set<ZIndex>({25})
            .add<ScissorContainer>(leaf.target<EditorCanvas>())
            .child_of(channels_2);
            // .child_of(leaf.target<EditorCanvas>());

            world->entity()
            .is_a(UIElement)
            .child_of(channels_3)
            .set<Align>({0.0f, 0.0f, 0.8f, 0.0f})
            // .add<DebugRenderBounds>()
            .set<RectRenderable>({10.0f, 24.0f*3, false, i % 2 == 1 ? 0x222327FF : 0x121212FF })
            .add<ScissorContainer>(leaf.target<EditorCanvas>())
            .set<Expand>({true, 24*1.5f, 0, 0.2f, false, 0, 0, 0})
            .set<ZIndex>({24});
        }
    } else if (editor_type == EditorType::BFO)
    {
        auto bfo_editor = world->entity()
        .is_a(UIElement)
        // .add<HorizontalLayoutBox>()
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
    } else if (editor_type == EditorType::Void)
    {
        auto test = world->entity()
        .is_a(UIElement)
        .set<CustomRenderable>({100.0f, 25.0f, true, 0xFFFFFFFF, draw_double_arrow}) // Support custom arrow badge
        .set<ZIndex>({30})
        .child_of(leaf.target<EditorCanvas>());
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
        auto editor_icon_bkg_square = world->entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<Position, Local>({4.0f, 12.0f})
        .set<TextRenderable>({editor_types[(int)editor_type].c_str(), "Inter", 16.0f, 0xFFFFFFFF})
        .set<ZIndex>({1000});
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
        const VNCClient* vnc = vnc_entity.try_get<VNCClient>();

        if (vnc && vnc->eventPassthroughEnabled && vnc->connected) {
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
        const VNCClient* vnc = target_vnc_entity.try_get<VNCClient>();
        if (!vnc) {
            LOG_ERROR(LogCategory::SFTP, "VNC entity has no VNCClient component");
            return;
        }

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
    auto query = world->query<VNCClient, Position, ImageRenderable>();
    query.each([&](flecs::entity e, VNCClient& vnc, Position& pos, ImageRenderable& img) {
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
                    .add<HorizontalLayoutBox>();

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
                        .add<VerticalLayoutBox>();

                    // Create the text content using the actual draft
                    auto example_message_text = world->entity()
                        .is_a(UIElement)
                        .child_of(message_content)
                        .set<TextRenderable>({chat->draft.c_str(), "Inter", 16.0f, 0xFFFFFFFF})
                        .set<ZIndex>({17});

                    // I put it outside the message since it is a meta annotation
                    // This might only need to be visible during certain 'entity binding' interface modes...
                    auto messageBfoSprite = world->entity()
                    .is_a(UIElement)
                    .child_of(messageBox)
                    .set<ZIndex>({20})
                    .set<ImageCreator>({"../assets/bfo/generically_dependent_continuant.png", 1.0f, 1.0f});

                    auto meta_response = world->entity()
                    .is_a(UIElement)
                    .set<HorizontalLayoutBox>({0.0f, 2.0f})
                    .add(flecs::OrderedChildren)
                    // .add<DebugRenderBounds>()
                    .child_of(chat_panel.message_list);

                    create_badge(meta_response, UIElement, "Heonae", 0xc72783ff, false, false, 1);

                    create_badge(meta_response, UIElement, "understands", 0xc72783ff, false, true);

                });

            chat->draft.clear();
        }
    }
    } 
    }
    // TODO: Check for focused VNC Stream...
    // Use static cached query to avoid creating new query on every key press
    static auto vnc_query = world->query<VNCClient>();
    vnc_query.each([&](flecs::entity e, VNCClient& vnc) {
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

    // Initialize libssh2
    int rc = libssh2_init(0);
    if (rc) {
        std::cerr << "libssh2 initialization failed (" << rc << ")" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1200, 800, "Thornfield", NULL, NULL);
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

    glViewport(0, 0, 800, 600);
    NVGcontext* vg = nvgCreateGL2(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
    if (vg == NULL) {
        std::cerr << "Failed to initialize NanoVG" << std::endl;
        glfwTerminate();
        return -1;
    }

    world->component<Position>()
    .member<float>("x")
    .member<float>("y");
    world->component<Velocity>();
    world->component<RectRenderable>();
    world->component<CustomRenderable>();
    world->component<TextRenderable>();
    world->component<ImageCreator>();
    world->component<ImageRenderable>();
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

    world->component<HorizontalLayoutBox>();
    world->component<VerticalLayoutBox>();
    
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
        e.set<ImageRenderable>({imgHandle, img.scaleX, img.scaleY, 0.0f, 0.0f});
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
        .set<VerticalLayoutBox>({0.0f, 2.0f})
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

    // Hierarchical positioning system - computes world positions from local positions
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

    auto boundsCalculationSystem = world->system<Position, UIElementBounds, UIElementSize>()
        .term_at(0).second<World>()
        .kind(flecs::OnLoad) 
        .each([&](flecs::entity e, Position& worldPos, UIElementBounds& bounds, UIElementSize& size) {
            ZoneScoped;
            // Reset bounds to invalid state at start of each frame
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

            msg_views.each([&](flecs::entity, ChatMessageView& view, TextRenderable& tr, Position& pos)
            {
                int msg_index = start + view.index;
                if (msg_index < total)
                {
                    const auto& msg = chat.messages[msg_index];
                    tr.text = msg.author + ": " + msg.text;
                    tr.wrapWidth = messages_rect.width - 12.0f;  // Wrap messages within panel
                    pos.x = 6.0f;
                    pos.y = 6.0f + view.index * 18.0f;
                }
                else
                {
                    tr.text.clear();
                }
            });
        });

    world->system<const UIElementBounds, HorizontalLayoutBox>()
    .kind(flecs::PreUpdate)
    .term_at(0).parent() 
    .with<FitChildren>()
    .each([](flecs::entity e, const UIElementBounds& container_bounds, HorizontalLayoutBox& box) {
        float container_w = container_bounds.xmax - container_bounds.xmin;
        float container_h = container_bounds.ymax - container_bounds.ymin;
        
        float total_intrinsic_w = 0;
        int child_count = 0;

        // Pass 1: Sum up current natural widths
        e.children([&](flecs::entity child) {
            const ImageRenderable* img = child.try_get<ImageRenderable>();
            if (img && img->width > 0) {
                total_intrinsic_w += img->width;
                child_count++;
            }
        });

        if (child_count == 0 || total_intrinsic_w <= 0) return;

        // Pass 2: Calculate Scale Factors
        float total_padding = box.padding * (child_count - 1);
        float available_w = container_w - total_padding;
        
        // This is our ideal horizontal scale factor
        float scale_factor = available_w / total_intrinsic_w;

        // Pass 3: Distribute Constraints with Y-Limit
        e.children([&](flecs::entity child) {
            const ImageRenderable* img = child.try_get<ImageRenderable>();
            if (img) {
                float aspect = img->width / img->height;
                float target_w = img->width * scale_factor;
                float target_h = target_w / aspect;

                // --- THE Y-OVERFLOW FIX ---
                // If the target width makes the book too tall, 
                // clamp based on container height instead.
                if (target_h > container_h) {
                    target_h = container_h;
                    target_w = target_h * aspect;
                }

                child.set<ProportionalConstraint>({ target_w, target_h });
            }
        });
    });

    world->system<HorizontalLayoutBox, UIElementSize>("ResetHProgress")
        .kind(flecs::PostLoad)
        .each([](flecs::entity e, HorizontalLayoutBox& box, UIElementSize& container_size)
        {
            box.x_progress = 0.0f;
            float max_height = 0.0f; 

            e.children([&](flecs::entity child)
            {
                Position& pos = child.ensure<Position, Local>();
                pos.x = box.x_progress;
                
                const UIElementSize* child_size = child.try_get<UIElementSize>();
                
                if (child_size) {
                    box.x_progress += child_size->width + box.padding;
                    if (child_size->height > max_height) {
                        max_height = child_size->height;
                    }
                }
            });

            // FIX: Check if we have an Expand component
            const Expand* expand = e.try_get<Expand>();

            // Only auto-resize WIDTH if not expanding in X
            if (!expand || !expand->x_enabled) {
                container_size.width = box.x_progress; 
            }
            
            // Only auto-resize HEIGHT if not expanding in Y
            // This allows the bookshelf to keep its parent's height
            if (!expand || !expand->y_enabled) {
                container_size.height = max_height;
            }
        });

    world->system<VerticalLayoutBox, UIElementSize>("ResetVProgress")
    .kind(flecs::PostLoad)
    .each([](flecs::entity e, VerticalLayoutBox& box, UIElementSize& container_size) {
        float current_y = 0.0f;
        float max_width = 0.0f;

        e.children([&](flecs::entity child) {
            const UIElementSize* child_size = child.try_get<UIElementSize>();
            if (!child_size) return;

            Position& pos = child.ensure<Position, Local>();
            
            if (box.move_dir == -1.0f) {
                // Stack upwards by moving the child into negative space 
                // rather than moving the parent into positive space.
                current_y -= (child_size->height + box.padding);
                pos.y = current_y;
            } else {
                pos.y = current_y;
                current_y += child_size->height + box.padding;
            }

            max_width = std::max(max_width, child_size->width);
        });

        // Update container size without shifting the container's own Position
        const Expand* expand = e.try_get<Expand>();
        if (!expand || !expand->y_enabled) container_size.height = std::abs(current_y);
        if (!expand || !expand->x_enabled) container_size.width = max_width;

        // Immediately recalculate bounds for dimensions VerticalLayoutBox controls
        // so Expand systems see the updated size this frame (prevents flickering)
        if (e.has<UIElementBounds>()) {
            UIElementBounds& bounds = e.ensure<UIElementBounds>();
            const Position& world_pos = e.get<Position, World>();
            if (!expand || !expand->y_enabled) {
                bounds.ymax = world_pos.y + container_size.height;
            }
            if (!expand || !expand->x_enabled) {
                bounds.xmax = world_pos.x + container_size.width;
            }
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
                    parent_bounds->xmin = std::min(parent_bounds->xmin, bounds.xmin);
                    parent_bounds->xmax = std::max(parent_bounds->xmax, bounds.xmax);
                    parent_bounds->ymin = std::min(parent_bounds->ymin, bounds.ymin);
                    parent_bounds->ymax = std::max(parent_bounds->ymax, bounds.ymax);
                }
                }
        });

    world->system<UIElementBounds, UIContainer, RoundedRectRenderable, UIElementSize>()
    .kind(flecs::PreFrame)
    .each([&](flecs::entity e, UIElementBounds& bounds, UIContainer& container, RoundedRectRenderable& renderable, UIElementSize& size)
    {
        UIElementBounds children_aabb = {0, 0, 0, 0};
        bool first = true;

        // 1. Traverse children to find the collective bounding box
        calculate_recursive_bounds(e, children_aabb, first);

        // 2. If children were found, update the parent's bounds
        if (!first) {
            bounds = children_aabb;
        }

        // 3. Apply padding and update render/size components
        renderable.width = (bounds.xmax - bounds.xmin) + (container.pad_horizontal * 2);
        renderable.height = (bounds.ymax - bounds.ymin) + (container.pad_vertical * 2);
        
        size.width = renderable.width;
        size.height = renderable.height;
    });

    world->system<UIElementBounds, UIContainer, CustomRenderable, UIElementSize>()
    .kind(flecs::PreFrame)
    .each([&](flecs::entity e, UIElementBounds& bounds, UIContainer& container, CustomRenderable& renderable, UIElementSize& size)
    {
        UIElementBounds children_aabb = {0, 0, 0, 0};
        bool first = true;
        calculate_recursive_bounds(e, children_aabb, first);
        if (!first) {
            bounds = children_aabb;
        }
        renderable.width = (bounds.xmax - bounds.xmin) + (container.pad_horizontal * 2);
        renderable.height = (bounds.ymax - bounds.ymin) + (container.pad_vertical * 2);
        
        size.width = renderable.width;
        size.height = renderable.height;
    });


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
                    parent_bounds->xmin = std::min(parent_bounds->xmin, bounds.xmin);
                    parent_bounds->xmax = std::max(parent_bounds->xmax, bounds.xmax);
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
        if (!e.parent().has<HorizontalLayoutBox>())
        {
            pos.x = align.horizontal * (parent_bounds->xmax - parent_bounds->xmin) + ui_size.width * align.self_horizontal + (expand ? expand->pad_left : 0);
        }
        if (!e.parent().has<VerticalLayoutBox>())
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

world->system<UIElementBounds*, ImageRenderable, Expand, Constrain*, Graphics>()
    .term_at(0).parent()
    .term_at(3).optional()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, ImageRenderable& sprite, Expand& expand, Constrain* constrain, Graphics& graphics) {        
        if (!bounds) return;

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
            float avail_w = (bounds->xmax - bounds->xmin) - (expand.pad_left + expand.pad_right);
            float avail_h = (bounds->ymax - bounds->ymin) - (expand.pad_top + expand.pad_bottom);

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
            queue.addImageCommand(pos, renderable, zIndex.layer);
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
                            NVGpaint imgPaint = nvgImagePattern(graphics.vg, cmd.pos.x, cmd.pos.y,
                                                              image.width, image.height, 0.0f,
                                                              image.imageHandle, 1.0); // image.alpha
                            nvgBeginPath(graphics.vg);
                            nvgRect(graphics.vg, cmd.pos.x, cmd.pos.y, image.width, image.height);
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

    auto vncInitSystem = world->system<VNCClient>()
        .kind(flecs::PreUpdate)
        .each([](flecs::iter& it, size_t i, VNCClient& vnc) {
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
            VNCClient& vnc = e.target<IsStreamingFrom>().ensure<VNCClient>();
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
            VNCClient& vnc = e.target<IsStreamingFrom>().ensure<VNCClient>();
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

    auto vncTextureUpdateSystem = world->system<VNCClient>()
        .kind(flecs::OnUpdate)
        .each([&](flecs::entity e, VNCClient& vnc) {
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

                // Periodic PNG capture every 3 seconds
                double currentTime = glfwGetTime();
                if (currentTime - vnc.lastCaptureTime >= 3.0) {
                    // Prepare RGBA data buffer for fpng (fpng expects RGBA format)
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

                    // Generate filename
                    std::string frame_filename = "frame_" + std::to_string(vnc.framesCaptured) + ".png";

                    // Save PNG using fpng
                    // TODO: Save at smaller resolution? (Probably)
                    if (fpng::fpng_encode_image_to_file(frame_filename.c_str(), rgbaData.data(),
                                                         surface->w, surface->h, 4, 0)) {
                        std::cout << "[VNC CAPTURE] Saved " << frame_filename << std::endl;
                        vnc.framesCaptured++;
                        vnc.lastCaptureTime = currentTime;

                        // TODO: Now, we should load this as a 'Spaceframe' in the Episodic panel

                        // TODO: Find Episodic memory panel...
                        flecs::query q_editor_panels = world->query_builder<EditorLeafData>()
                        .build();

                        q_editor_panels.each([&](flecs::entity leaf, EditorLeafData& leaf_data) 
                        {
                            if (leaf_data.editor_type == EditorType::Episodic)
                            {
                                // TODO: Check for rows that are designed for framespace render
                                auto messageBfoSprite = world->entity()
                                .is_a(UIElement)
                                // .set<ProportionalConstraint>({800.0f, 200.0f})
                                // hmmm....
                                .set<ImageCreator>({"../build/" + frame_filename, 1.0f, 1.0f})
                                .set<ZIndex>({30})
                                 .set<Constrain>({true, true})
                                 .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f, true})
                                .child_of(leaf.target<EditorCanvas>().target<SpaceframeChannel>());
                            }
                        });

                    } else {
                        std::cerr << "[VNC CAPTURE ERROR] Failed to save " << frame_filename << std::endl;
                    }
                }
            } else {
                LOG_ERROR(LogCategory::VNC_CLIENT, "Surface or pixels is null");
            }
        });

    // VNC cleanup system - handles thread lifecycle and resource cleanup
    auto vncCleanupSystem = world->system<VNCClient>()
        .kind(flecs::OnUpdate)
        .each([](flecs::entity e, VNCClient& vnc) {
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
            VNCClient& vnc = vnc_entity.ensure<VNCClient>();
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

        glViewport(0, 0, fbWidth, fbHeight);
        // glClearColor(22.0f/255.0f, 22.0f/255.0f, 22.0f/255.0f, 0.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        glfwStateEntity.set<Window>({window, winWidth, winHeight});

        nvgBeginFrame(vg, winWidth, winHeight, devicePixelRatio);
        
        world->defer_begin();
        glfwPollEvents();
        world->defer_end();
        world->progress();
        FrameMark;
        nvgEndFrame(vg);

        glfwSwapBuffers(window);
    }

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
