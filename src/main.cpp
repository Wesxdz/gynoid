#include <iostream>
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

static const char* VNC_SURFACE_TAG = "vnc_surface";
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

struct VNCClient {
    rfbClient* client = nullptr;
    SDL_Surface* surface = nullptr;
    bool connected = false;
    std::string host;
    int port;
    int width = 0;
    int height = 0;
    int quadrant = 0;  // Which quadrant this client belongs to (0-3)
};

struct VNCUpdateRect {
    int x, y, w, h;
};

struct VNCTexture {
    GLuint texture = 0;
    int nvgHandle = -1;
    int width = 0;
    int height = 0;
    bool needsUpdate = false;
    std::vector<VNCUpdateRect> dirtyRects;  // Track which regions need updating
};

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
};

struct HorizontalLayoutBox
{
  float x_progress;
  float padding = 0.0f;
};

struct VerticalLayoutBox 
{
  float y_progress;
  float padding;
};


struct RectRenderable {
    float width, height;
    bool stroke;
    uint32_t color;
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
    int alignment;
};

struct ImageCreator
{
    std::string path;
    float scaleX, scaleY;
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
    Embodiment,
    LanguageGame,
    Vision,
    Hearing,
    Memory,
    Bookshelf,
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

// Post expand layer to 'fit within editor panel bounds'
struct Constrain
{
    bool fit_x; // Scale x to fit within bounds (maintain ratio)
    bool fit_y; // Scale y to fit within bounds (maintain ratio)
};

struct EditorLeafData
{
    EditorType editor_type;
};

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
};

flecs::world world;

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
    auto query = world.query<VNCClient, VNCTexture>();
    int updateCount = 0;
    query.each([&](flecs::entity e, VNCClient& vnc, VNCTexture& tex) {
        if (vnc.client == client) {
            // Add the dirty rectangle to the update queue
            tex.dirtyRects.push_back({x, y, w, h});
            tex.needsUpdate = true;
            updateCount++;
            LOG_TRACE(LogCategory::VNC_CLIENT, "Added dirty rect total rects: {}", tex.dirtyRects.size());
        }
    });
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

// End VNC Stream


void create_editor(flecs::entity leaf, EditorNodeArea& node_area, flecs::world world, flecs::entity UIElement)
{
    leaf.set<EditorLeafData>({EditorType::Void});

    auto editor_visual = world.entity()
        .is_a(UIElement)
        .set<Position, Local>({1.0f, 1.0f})
        .set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, false, 0x010222})
        .child_of(leaf);

    leaf.add<EditorVisual>(editor_visual);

    auto editor_outline = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, true, 0x111222FF})
        .set<ZIndex>({5});

    leaf.add<EditorOutline>(editor_outline);

    auto editor_header = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        // .set<RectRenderable>({node_area.width-2-8.0f, 27.0f, true, 0x00000000})
        .set<Expand>({true, 4.0f, 4.0f, 1.0f, false, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({8});

    leaf.add<EditorHeader>(editor_header);

    // Add 'expand to parent UIElement bounds with padding'
    auto editor_canvas = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({4.0f, 23.0f})
        // .add<DebugRenderBounds>()
        .set<RectRenderable>({node_area.width-2-8.0f, node_area.height-2-23.0f, false, 0x121212FF})
        .set<Expand>({true, 4.0f, 4.0f, 1.0f, true, 27.0f, 0.0f, 1.0f})
        .set<ZIndex>({8});

    leaf.add<EditorCanvas>(editor_canvas);

    auto editor_header_bkg = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({0.0f, 0.0f})
        .set<RoundedRectRenderable>({0.0f, 22.0f, 4.0f, false, 0x282828FF})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0, 0, 0})
        .set<ZIndex>({2});

    auto editor_icon_bkg = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({8.0f, 2.0f})
        .set<RoundedRectRenderable>({32.0f, 20.0f, 4.0f, false, 0x282828FF})
        .add<EditorCanvas>(editor_canvas)
        .add<EditorLeaf>(leaf)
        .add<AddTagOnLeftClick, ShowEditorPanels>()
        .set<ZIndex>({4});

    auto editor_icon = world.entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({2.0f, 0.0f})
        .set<ImageCreator>({"../assets/embodiment.png", 1.0f, 1.0f})
        .set<ZIndex>({12});

    auto editor_dropdown = world.entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({22.0f, 8.0f})
        .set<ImageCreator>({"../assets/arrow_down.png", 1.0f, 1.0f})
        .set<ZIndex>({12});
        
    world.entity()
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
};


// Factory function to populate editor content, whether the panel is initialized or changed
void create_editor_content(flecs::entity leaf, EditorType editor_type, flecs::entity UIElement)
{
    std::cout << "Change panel type to " << editor_types[(int)editor_type] << std::endl;
    if (editor_type == EditorType::PeachCore)
    {

        auto server_hud = world.entity("ServerHUD")
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .add(flecs::OrderedChildren)
        .child_of(leaf.target<EditorCanvas>());

        std::vector<std::string> server_icons = {"peach_core", "aeri_memory", "flecs", "x11", "parakeet", "chatterbox", "doctr", "huggingface", "dino2", "alpaca", "modal"};
        // std::vector<std::string> server_icons = {"peach_core"};

        for (const auto& icon : server_icons)
        {
            flecs::entity server_icon;

            if (icon == "chatterbox")
            { 
                server_icon = world.entity()
                .is_a(UIElement)
                .child_of(server_hud)
                .set<ImageCreator>({"../assets/server_hud/" + icon + ".png", 1.0f, 1.0f})
                .set<ZIndex>({10})
                server_icon.set<ServerScript>({"chatterbox", "chatterbox", "../chatter_server"})
                .add(ServerStatus::Offline)
                .add<AddTagOnLeftClick, SelectServer>(); 
            } else
            {
                server_icon = world.entity()
                .is_a(UIElement)
                .child_of(server_hud)
                .set<ImageCreator>({"../assets/server_hud/" + icon + ".png", 1.0f, 1.0f})
                .set<ZIndex>({10});
            }
            // TODO: Server dot should only exist if the server is active...
            world.entity()
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
            world.entity()
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
        world.entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<LineRenderable>({0.0f, 0.0f, 100.0f, 0.0f, 2.0f, 0x00FF00FF})
        .set<Align>({0.0f, 0.0f, 0.0f, 0.5f})
        .set<Expand>({true, 0, 0, 1.0, false, 0, 0, 0})
        .set<ZIndex>({10});

        auto threshold_line = world.entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<RectRenderable>({0.0f, 0.0f, false, 0x00FF0055})
        .set<RenderStatus>({})
        .set<Align>({0.0f, 0.0f, 0.0f, 0.5f})
        .set<Expand>({true, 0, 0, 1.0, true, 0, 0, 0.5})
        .set<ZIndex>({10});

        auto endowment_monthly_income = world.entity()
        .is_a(UIElement)
        .child_of(threshold_line)
        .set<Position, Local>({4.0f, -4.0f})
        .set<TextRenderable>({"0.5%", "ATARISTOCRAT", 16.0f, 0xFFFFFFFF})
        .set<ZIndex>({30});
        
        auto threshold_amount = world.entity()
        .is_a(UIElement)
        .child_of(threshold_line)
        .set<Position, Local>({4.0f, 12.0f})
        .set<TextRenderable>({"$1200", "ATARISTOCRAT", 16.0f, 0x00FF00FF})
        .set<ZIndex>({30});

        auto badges = world.entity()
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/healthbar_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});
    }
    else if (editor_type == EditorType::Embodiment)
    {
        world.entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<ImageCreator>({"../assets/heonae_work.png", 1.0f, 1.0f})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 1.0f})
        .set<Align>({-0.5f, -0.5f, 0.5f, 0.5f})
        .set<Constrain>({true, true})
        .set<ZIndex>({10});

        auto badges = world.entity()
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/heonae_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/virtual_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/physical_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});
        
    }
    else if (editor_type == EditorType::LanguageGame)
    {
        auto canvas = leaf.target<EditorCanvas>();

        auto chat_root = world.entity()
            .is_a(UIElement)
            .child_of(canvas)
            .set<ZIndex>({5});

        // auto input_text_bar = world.entity()
        //     .is_a(UIElement)
        //     .child_of(chat_root)
        //     .set<RoundedRectRenderable>({100.0f, 32.0f, 2.0f, false, 0x444444FF})
        //     .set<Expand>({true, 0.0f, 4.0f, 1.0f, false, 0, 0, 0})
        //     .set<ZIndex>({10});

        auto messages_panel = world.entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .set<RoundedRectRenderable>({100.0f, 100.0f, 4.0f, false, 0x121212FF})
            // TODO: Expand to fill remaining space in VerticalLayout...
            // .set<Expand>({true, 8.0f, 8.0f, 1.0f, true, 8.0f, 36.0f, })
            .set<ZIndex>({10});

        auto input_panel = world.entity()
            .is_a(UIElement)
            .child_of(chat_root)
            .set<RoundedRectRenderable>({100.0f, 36.0f, 2.0f, true, 0x555555FF})
            .add<AddTagOnLeftClick, FocusChatInput>()
            .set<ZIndex>({10});

        auto input_bkg = world.entity()
            .is_a(UIElement)
            .child_of(input_panel)
            .set<RoundedRectRenderable>({10.0f, 10.0f, 2.0f, false, 0x222327FF})
            .set<Expand>({true, 0, 0, 1, true, 0, 0, 1})
            .set<ZIndex>({9});

        auto input_text = world.entity()
            .is_a(UIElement)
            .child_of(input_panel)
            .set<Position, Local>({8.0f, 8.0f})
            .set<TextRenderable>({"", "Inter", 14.0f, 0xFFFFFFFF, NVG_ALIGN_TOP | NVG_ALIGN_LEFT})
            .set<ZIndex>({12});

        const int kMaxMessages = 30;
        for (int i = 0; i < kMaxMessages; ++i)
        {
            world.entity()
                .is_a(UIElement)
                .child_of(messages_panel)
                .set<ChatMessageView>({i})
                .set<Position, Local>({6.0f, 6.0f + i * 18.0f})
                .set<TextRenderable>({"", "Inter", 14.0f, 0xDDDDDDFF, NVG_ALIGN_TOP | NVG_ALIGN_LEFT})
                .set<ZIndex>({7});
        }

        leaf.set<ChatPanel>({messages_panel, input_panel, input_text});
    }
    else if (editor_type == EditorType::Bookshelf)
    {
        auto bookshelf_layer = world.entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<HorizontalLayoutBox>({0.0f, 8.0f})
        .set<Expand>({true, 0.0f, 0.0f, 1.0f, true, 0.0f, 0.0f, 1.0f});
        
        // TODO: Load from folder target
        world.entity()
        .is_a(UIElement)
        .child_of(bookshelf_layer)
        .set<ImageCreator>({"../assets/cover_james.jpg", 1.0f, 1.0f})
        .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({10});
    
        world.entity()
        .is_a(UIElement)
        .child_of(bookshelf_layer)
        .set<ImageCreator>({"../assets/cover_cognitive_theory.jpg", 1.0f, 1.0f})
        .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({10});
    
        world.entity()
        .is_a(UIElement)
        .child_of(bookshelf_layer)
        .set<ImageCreator>({"../assets/cover_soar.jpg", 1.0f, 1.0f})
        .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({10});
    
        world.entity()
        .is_a(UIElement)
        .child_of(bookshelf_layer)
        .set<ImageCreator>({"../assets/cover_readings_in_kr.jpg", 1.0f, 1.0f})
        .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({10});

        world.entity()
        .is_a(UIElement)
        .child_of(bookshelf_layer)
        .set<ImageCreator>({"../assets/cuct.png", 1.0f, 1.0f})
        .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({10});
    } 
    else if (editor_type == EditorType::Hearing)
    {
        std::cout << "Creating Hearing editor..." << std::endl;

        auto badges = world.entity()
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/system_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/microphone_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        // Create vertical layout for mel spectrograms
        auto hearing_layer = world.entity()
            .is_a(UIElement)
            .child_of(leaf.target<EditorCanvas>())
            .set<VerticalLayoutBox>({0.0f, 4.0f})
            .set<Align>({-0.5f, -0.5f, 0.5f, 0.5f});
        // Look up the mel spec renderer entities
        auto micRenderer = world.lookup("MelSpecRenderer");
        auto sysAudioRenderer = world.lookup("SystemAudioRenderer");

        std::cout << "MicRenderer exists: " << (micRenderer ? "yes" : "no") << std::endl;
        std::cout << "SysAudioRenderer exists: " << (sysAudioRenderer ? "yes" : "no") << std::endl;

        if (micRenderer && micRenderer.has<MelSpecRender>())
        {
            auto melSpec = micRenderer.get<MelSpecRender>();
            std::cout << "Mic texture handle: " << melSpec.nvgTextureHandle
                      << " size: " << melSpec.width << "x" << melSpec.height << std::endl;

            // Create microphone mel spec display
            world.entity()
                .is_a(UIElement)
                .child_of(hearing_layer)
                .set<ImageRenderable>({melSpec.nvgTextureHandle, 1.0f, 1.0f, (float)melSpec.width, (float)melSpec.height})
                // .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 0.5f, false})
                // .set<Constrain>({true, true})
                .set<ZIndex>({10});

            // Add label for microphone
            // world.entity()
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
            world.entity()
                .is_a(UIElement)
                .child_of(hearing_layer)
                .set<ImageRenderable>({melSpec.nvgTextureHandle, 1.0f, 1.0f, (float)melSpec.width, (float)melSpec.height})
                // .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0.0f, 0.0f, 0.5f, false})
                // .set<Constrain>({true, true})
                .set<ZIndex>({10});

            // Add label for system audio
            // world.entity()
            //     .is_a(UIElement)
            //     .child_of(hearing_layer)
            //     .set<Position, Local>({8.0f, 0.0f})
            //     .set<TextRenderable>({"System Audio", "ATARISTOCRAT", 12.0f, 0xAAAAAAFF})
            //     .set<ZIndex>({20});
        }
    }
    else if (editor_type == EditorType::VNCStream)
    {
        auto badges = world.entity()
        .is_a(UIElement)
        .set<HorizontalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({48.0f, 0.0f})
        .child_of(leaf.target<EditorHeader>());
        
        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/docker_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/kubuntu_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/192.168.1.104_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        world.entity()
        .is_a(UIElement)
        .child_of(badges)
        .set<ImageCreator>({"../assets/5901_badge.png", 1.0f, 1.0f})
        .set<ZIndex>({20});

        const char* vnc_host = getenv("VNC_SERVER_HOST");
        if (!vnc_host) {
            vnc_host = "localhost";
            vnc_host = "192.168.1.104";
        }
        std::cout << "[VNC] VNC server host: " << vnc_host << " (ports 5901-5904)" << std::endl;
        
        int port = 5901;
        std::string host_string = std::string(vnc_host) + ":" + std::to_string(port);

        rfbClient* vncClient = connectToTurboVNC(vnc_host, port);
        if (vncClient) {
            std::cout << "[VNC] VNC client connected successfully" << std::endl;
            SDL_Surface* surface = (SDL_Surface*)rfbClientGetClientData(vncClient, (void*)VNC_SURFACE_TAG);
            
            // Create OpenGL texture for VNC framebuffer
            GLuint vncTexture;
            glGenTextures(1, &vncTexture);
            glBindTexture(GL_TEXTURE_2D, vncTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, vncClient->width, vncClient->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            
            // Create NanoVG image from OpenGL texture
            int nvgVNCHandle = nvglCreateImageFromHandleGL2(world.try_get<Graphics>()->vg, vncTexture, vncClient->width, vncClient->height, 0);
            
            flecs::entity vnc_entity = world.entity()
            .is_a(UIElement)
            .set<VNCTexture>({vncTexture, nvgVNCHandle, vncClient->width, vncClient->height, false})
            .set<ImageRenderable>({nvgVNCHandle, 1.0f, 1.0f, vncClient->width, vncClient->height})
            .set<VNCClient>({
                vncClient,
                surface,
                true,
                vnc_host,
                port,
                vncClient->width,
                vncClient->height,
                0
            })
            .set<ZIndex>({9})
            .set<Expand>({true, 0.0f, 0.0f, 1.0f, false, 0, 0, 0})
            .set<Constrain>({true, true})
            .set<Align>({-0.5f, -0.5f, 0.5f, 0.5f})
            // .add<DebugRenderBounds>()
            .child_of(leaf.target<EditorCanvas>());
        }


    } else
    {
        auto editor_icon_bkg_square = world.entity()
        .is_a(UIElement)
        .child_of(leaf.target<EditorCanvas>())
        .set<Position, Local>({4.0f, 12.0f})
        .set<TextRenderable>({editor_types[(int)editor_type].c_str(), "ATARISTOCRAT", 16.0f, 0xFFFFFFFF})
        .set<ZIndex>({30});
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
    create_editor_content(leaf, editor_type, UIElement);
}

void merge_editor(flecs::entity non_leaf, flecs::world world, flecs::entity UIElement)
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
    create_editor(non_leaf, intermediate_area, world, UIElement);
}

void split_editor(PanelSplit split, flecs::entity leaf, flecs::world world, flecs::entity UIElement)
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
        auto left_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(left_node_area)
            .add(flecs::OrderedChildren);
        create_editor(left_editor_leaf, left_node_area, world, UIElement);
        leaf.add<LeftNode>(left_editor_leaf);

        EditorNodeArea right_node_area = {node_area->width*(1.0f-split.percent), node_area->height};
        auto right_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({node_area->width*split.percent, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(right_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(right_editor_leaf, right_node_area, world, UIElement);
        leaf.add<RightNode>(right_editor_leaf);
    }
    if (split.dim == PanelSplitType::Vertical)
    {   
        EditorNodeArea upper_node_area = {node_area->width, node_area->height*split.percent};
        auto upper_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(upper_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(upper_editor_leaf, upper_node_area, world, UIElement);
        leaf.add<UpperNode>(upper_editor_leaf);
        
        EditorNodeArea lower_node_area = {node_area->width, node_area->height*(1-split.percent)};
        auto lower_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, node_area->height*split.percent})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(lower_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(lower_editor_leaf, lower_node_area, world, UIElement);
        leaf.add<LowerNode>(lower_editor_leaf);
    }
}

struct RenderCommand {
    Position pos;
    std::variant<RoundedRectRenderable, RectRenderable, TextRenderable, ImageRenderable, LineRenderable, QuadraticBezierRenderable> renderData;
    RenderType type;
    int zIndex;

    bool operator<(const RenderCommand& other) const {
        return zIndex < other.zIndex;
    }
};

struct RenderQueue {
    std::vector<RenderCommand> commands;

    void clear() {
        commands.clear();
    }

    void addRectCommand(const Position& pos, const RectRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Rectangle, zIndex});
    }

    void addRoundedRectCommand(const Position& pos, const RoundedRectRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::RoundedRectangle, zIndex});
    }

    void addTextCommand(const Position& pos, const TextRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Text, zIndex});
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
    Window& window_comp = world.lookup("GLFWState").ensure<Window>();
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
    CursorState& cursor_state = world.lookup("GLFWState").ensure<CursorState>();
    // TODO: Query for hoverable UIElement 
    flecs::query hoverable_elements = world.query_builder<AddTagOnHoverEnter, AddTagOnHoverExit, UIElementBounds>()
    .term_at(0).second(flecs::Wildcard).optional()
    .term_at(1).second(flecs::Wildcard).optional()
    .build();

    hoverable_elements.each([&](flecs::entity ui_element, AddTagOnHoverEnter, AddTagOnHoverExit, UIElementBounds& bounds) {
        bool in_bounds_prior = point_in_bounds(cursor_state.x, cursor_state.y, bounds);
        bool in_bounds_post = point_in_bounds(xpos, ypos, bounds);
        
        if (!in_bounds_prior && in_bounds_post && ui_element.has<AddTagOnHoverEnter>(flecs::Wildcard))
        {   
            world.event<HoverEnterEvent>()
            .id<UIElementBounds>()
            .entity(ui_element)
            .enqueue();
        } else if (in_bounds_prior && !in_bounds_post && ui_element.has<AddTagOnHoverExit>(flecs::Wildcard))
        {
            // TODO: Store hover state...
            world.event<HoverExitEvent>()
            .id<UIElementBounds>()
            .entity(ui_element)
            .enqueue();  
        }
    });

    cursor_state.x = xpos;
    cursor_state.y = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    flecs::entity glfw_state = world.lookup("GLFWState");
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            flecs::query interactive_elements = world.query_builder<AddTagOnLeftClick, UIElementBounds>()
            .term_at(0).second(flecs::Wildcard)
            .build();
            // TODO: Eventually this should use a more efficient partition bound check as the first layer
            const CursorState* cursor_state = world.lookup("GLFWState").try_get<CursorState>();
            interactive_elements.each([&](flecs::entity ui_element, AddTagOnLeftClick, UIElementBounds& bounds) {
                if (point_in_bounds(cursor_state->x, cursor_state->y, bounds))
                {
                    world.event<LeftClickEvent>()
                    .id<UIElementBounds>()
                    .entity(ui_element)
                    .emit();
                } 
            });
            world.event<LeftClickEvent>()
            .id<CursorState>()
            .entity(glfw_state)
            .emit();
        } else if (action == GLFW_RELEASE)
        {
            world.event<LeftReleaseEvent>()
            .id<CursorState>()
            .entity(glfw_state)
            .emit();
        }
    }
}

static void char_callback(GLFWwindow* window, unsigned int codepoint)
{
    ChatState* chat = world.try_get_mut<ChatState>();
    if (!chat || !chat->input_focused) return;
    if (codepoint >= 32 && codepoint < 127)
    {
        chat->draft.push_back(static_cast<char>(codepoint));
    }
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    ChatState* chat = world.try_get_mut<ChatState>();
    if (!chat || !chat->input_focused) return;

    if (key == GLFW_KEY_BACKSPACE)
    {
        if (!chat->draft.empty()) chat->draft.pop_back();
    }
    else if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER)
    {
        if (!chat->draft.empty())
        {
            chat->messages.push_back({"You", chat->draft});
            chat->draft.clear();
        }
    }
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

int main(int, char *[]) {
    glfwSetErrorCallback(error_callback);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "ECS Graphstar", NULL, NULL);
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

    world.component<Position>();
    world.component<Velocity>();
    world.component<RectRenderable>();
    world.component<TextRenderable>();
    world.component<ImageCreator>();
    world.component<ImageRenderable>();
    world.component<ZIndex>();
    world.component<Window>();
    world.component<CursorState>();
    world.component<Graphics>().add(flecs::Singleton);
    world.component<RenderQueue>();
    world.component<UIElementBounds>();
    world.component<UIElementSize>();

    world.component<EditorNodeArea>();
    world.component<PanelSplit>();

    world.component<Align>();
    world.component<Expand>();
    world.component<Constrain>();

    world.component<HorizontalLayoutBox>();
    world.component<VerticalLayoutBox>();
    
    world.component<DiurnalHour>();

    world.component<ChatMessage>();
    world.component<ChatMessageView>();
    world.component<ChatState>().add(flecs::Singleton);
    world.component<ChatPanel>();
    world.component<FocusChatInput>();
    world.component<SendChatMessage>();
    world.set<ChatState>({std::vector<ChatMessage>{}, "", false});

    world.component<DragContext>().add(flecs::Singleton);
    world.set<DragContext>({false, flecs::entity::null(), PanelSplitType::Horizontal, 0.0f});

    world.observer<ImageCreator, Graphics>()
    .event(flecs::OnSet)
    .each([&](flecs::entity e, ImageCreator& img, Graphics& graphics)
    {
        int imgHandle = nvgCreateImage(graphics.vg, ("../assets/" + img.path).c_str(), 0);

        if (imgHandle == -1) {
            std::cerr << "Failed to load " << img.path << std::endl;
        }
        e.set<ImageRenderable>({imgHandle, img.scaleX, img.scaleY, 0.0f, 0.0f});
    });

    world.observer<ImageRenderable, Graphics>()
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

    auto glfwStateEntity = world.entity("GLFWState")
        .set<Window>({window, 800, 600})
        .set<CursorState>({cursorXPos, cursorYPos});

    auto graphicsEntity = world.entity("Graphics")
        .set<Graphics>({vg});

    auto renderQueueEntity = world.entity("RenderQueue")
        .set<RenderQueue>({});

    // Initialize debug logger module (MUST BE FIRST!)
    DebugLogModule(world);

    // Initialize mel spectrogram rendering module (must be after Graphics entity is created)
    MelSpecRenderModule(world);

    auto UIElement = world.prefab("UIElement")
        .set<Position, Local>({0.0f, 0.0f})
        .set<Position, World>({0.0f, 0.0f})
        .set<UIElementBounds>({0, 0, 0, 0})
        .set<UIElementSize>({0.0f, 0.0f})
        .set<RenderStatus>({true})
        .set<ZIndex>({0});

    // TODO: Text search field
    // auto FieldEntry = world.prefab("FieldEntry")


    world.observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<CloseEditorSelector>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        std::cout << "Hover exit editor selector region" << std::endl;
        e.destruct();
    });

    world.observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<CloseEditorSelector>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        std::cout << "Select a particular panel..." << std::endl;
        e.destruct();
    });

    world.observer<UIElementBounds, AddTagOnHoverEnter>()
    .term_at(1).second<SetMenuHighlightColor>()
    .event<HoverEnterEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverEnter)
    {
        RoundedRectRenderable& bkg = e.ensure<RoundedRectRenderable>();
        bkg.color = 0x585858FF;
    });

    world.observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<SetMenuStandardColor>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        RoundedRectRenderable& bkg = e.ensure<RoundedRectRenderable>();
        bkg.color = 0x383838FF;
    });

    world.observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<SetPanelEditorType>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        replace_editor_content(e.target<EditorLeaf>(), e.get_constant<EditorType>(), UIElement);
    });

    world.observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<FocusChatInput>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds&, AddTagOnLeftClick)
    {
        ChatState& chat = world.ensure<ChatState>();
        chat.input_focused = true;
    });

    world.observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<SendChatMessage>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds&, AddTagOnLeftClick)
    {
        ChatState& chat = world.ensure<ChatState>();
        if (!chat.draft.empty())
        {
            chat.messages.push_back({"You", chat.draft});
            chat.draft.clear();
        }
    });

    world.observer<UIElementBounds, AddTagOnLeftClick>()
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

    world.observer<UIElementBounds, AddTagOnLeftClick>()
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

        auto editor_hover_region = world.entity()
        .is_a(UIElement)
        .child_of(e)
        // .add<DebugRenderBounds>()
        .add<AddTagOnHoverExit, CloseEditorSelector>()
        .add<AddTagOnLeftClick, CloseEditorSelector>();

        auto editor_icon_bkg_square = world.entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .set<Position, Local>({-1.0f, 10.0f})
        .set<RectRenderable>({32.0f, 12.0f, false, 0x282828FF})
        .set<ZIndex>({7});

        auto editor_type_selector = world.entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        // .add<DebugRenderBounds>()
        .set<Position, Local>({-1.0f, 19.0f});

        auto editor_type_selector_square_corner = world.entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<RectRenderable>({16.0f, 16.0f, false, 0x282828FF})
        .set<ZIndex>({30});

        auto editor_type_selector_bkg = world.entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<RoundedRectRenderable>({196.0f, 256.0f, 4.0f, false, 0x282828FF})
        .set<Expand>({false, 0, 0, 1.0f, true, 0.0f, 0.0f, 1.0f})
        .set<ZIndex>({30});

        auto editor_type_list = world.entity()
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
            auto edtior_type_btn = world.entity()
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


            world.entity()
            .is_a(UIElement)
            .child_of(edtior_type_btn)
            .set<TextRenderable>({editor_type_name.c_str(), "ATARISTOCRAT", 16.0f, 0xFFFFFFFF, NVG_ALIGN_TOP | NVG_ALIGN_LEFT})
            .set<Position, Local>({4.0f, 2.0f})
            .set<ZIndex>({40});
            
            editor_type_index++;
        }

    });

    // Create text entities with different z-indices
    // auto text1 = world.entity("Text1")
    //     .is_a(UIElement)
    //     .set<Position, Local>({400.0f, 100.0f})
    //     .set<TextRenderable>({"Behind boxes", "ATARISTOCRAT", 24.0f, 0xFFFFFFFF, NVG_ALIGN_CENTER})
    //     .set<ZIndex>({0});

    auto movementSystem = world.system<Position, Velocity>()
    .term_at(0).second<Local>()
        .each([](flecs::iter& it, size_t i, Position& pos, Velocity& vel) {
            float deltaTime = it.delta_system_time();

            pos.x += vel.dx * deltaTime;
            pos.y += vel.dy * deltaTime;
        });

    // Hierarchical positioning system - computes world positions from local positions
    auto hierarchicalQuery = world.query_builder<const Position, const Position*, Position>()
        .term_at(0).second<Local>()      // Local position
        .term_at(1).second<World>()      // Parent world position
        .term_at(2).second<World>()      // This entity's world position
        .term_at(1).parent().cascade()   // Get parent position in breadth-first order
        .build();

    auto hierarchicalSystem = world.system()
        .kind(flecs::OnLoad)  // Run after layout systems to compute world positions
        .each([&]() {
            // std::cout << "Update hierarchy" << std::endl;
            hierarchicalQuery.each([](const Position& local, const Position* parentWorld, Position& world) {
                world.x = local.x;
                world.y = local.y;
                if (parentWorld) {
                    world.x += parentWorld->x;
                    world.y += parentWorld->y;
                }
            });
        });

    auto boundsCalculationSystem = world.system<Position, UIElementBounds, UIElementSize>()
        .term_at(0).second<World>()
        .kind(flecs::OnLoad) 
        .each([&](flecs::entity e, Position& worldPos, UIElementBounds& bounds, UIElementSize& size) {
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

    auto editor_root = world.entity("editor_root")
        .set<Position, Local>({0.0f, 28.0f})
        .set<Position, World>({0.0f, 0.0f})
        .set<EditorNodeArea>({800.0f, 600.0f - 28.0f}) // TOOD: Observer to update root to window width/height updates
        .add<EditorRoot>()
        .add(flecs::OrderedChildren);

    auto editor_header = world.entity()
        .is_a(UIElement)
        .set<ImageCreator>({"../assets/ecs_header.png", 1.0f, 1.0f})
        .set<ZIndex>({5});

    // create_editor(editor_root, world, UIElement);
    split_editor({0.5, PanelSplitType::Horizontal}, editor_root, world, UIElement);
    auto right_node = editor_root.target<RightNode>();
    split_editor({0.35, PanelSplitType::Vertical}, right_node, world, UIElement);
    auto left_node = editor_root.target<LeftNode>();
    split_editor({0.25, PanelSplitType::Vertical}, left_node, world, UIElement);

    float diurnal_pos = 0.0f;
    world.system<DiurnalHour, QuadraticBezierRenderable>()
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

    world.system<Position, EditorNodeArea, PanelSplit, CursorState>()
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

    auto propagateEditorRoot = world.system<Window, EditorNodeArea, EditorRoot>("EditorPropagate")
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
            flecs::entity editor_root = world.lookup("editor_root");

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
                    outline.set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, true, 0x5f5f5fFF});
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

    auto sizeCalculationSystem = world.system<UIElementSize>()
        .kind(flecs::PreFrame)
        .each([&](flecs::entity e, UIElementSize& size) {

            if (e.has<RectRenderable>()) {
                auto rect = e.get<RectRenderable>();
                size.width = rect.width;
                size.height = rect.height;
            }
            else if (e.has<RoundedRectRenderable>()) {
                auto rect = e.get<RoundedRectRenderable>();
                size.width = rect.width;
                size.height = rect.height;
                //std::cout << "Setting size to" << size.width << std::endl;
            } else if (e.has<ImageRenderable>()) {
                auto img = e.get<ImageRenderable>();
                size.width = img.width;
                size.height = img.height;
            } else if (e.has<TextRenderable>()) {
                auto text = e.get<TextRenderable>();
                // Approximate text bounds
                float approxWidth = text.text.length() * text.fontSize * 0.6f;
                float approxHeight = text.fontSize;
                size.width = approxWidth;
                size.height = approxHeight;
            }
        });

    // Update Language Game chat UI each frame
    world.system<ChatPanel, EditorNodeArea>()
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
            const float input_h = 36.0f;

            auto& messages_rect = panel.messages_panel.ensure<RoundedRectRenderable>();
            messages_rect.width = canvas_w - pad * 2.0f;
            messages_rect.height = canvas_h - input_h - pad * 3.0f;
            panel.messages_panel.ensure<Position, Local>() = {pad, pad};

            auto& input_rect = panel.input_panel.ensure<RoundedRectRenderable>();
            input_rect.width = canvas_w - pad * 2.0f;
            input_rect.height = input_h;
            panel.input_panel.ensure<Position, Local>() = {pad, canvas_h - input_h - pad};

            ChatState& chat = world.ensure<ChatState>();
            std::string caret = chat.input_focused ? "|" : "";
            if (auto* input_tr = panel.input_text.try_get_mut<TextRenderable>())
            {
                input_tr->text = chat.draft + caret;
            }

            const int kMaxMessages = 30;
            int total = (int)chat.messages.size();
            int start = std::max(0, total - kMaxMessages);

            flecs::query msg_views = world.query_builder<ChatMessageView, TextRenderable, Position>()
                .term_at(0).src(panel.messages_panel)
                .build();

            msg_views.each([&](flecs::entity, ChatMessageView& view, TextRenderable& tr, Position& pos)
            {
                int msg_index = start + view.index;
                if (msg_index < total)
                {
                    const auto& msg = chat.messages[msg_index];
                    tr.text = msg.author + ": " + msg.text;
                    pos.x = 6.0f;
                    pos.y = 6.0f + view.index * 18.0f;
                }
                else
                {
                    tr.text.clear();
                }
            });
        });

    world.system<HorizontalLayoutBox, UIElementSize>("ResetHProgress")
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

    world.system<VerticalLayoutBox, UIElementSize>("ResetVProgress")
    .kind(flecs::PostLoad)
    .each([](flecs::entity e, VerticalLayoutBox& box, UIElementSize& container_size)
    {
        box.y_progress = 0.0f;
        float max_width = 0.0f;

        e.children([&](flecs::entity child)
        {
            Position& pos = child.ensure<Position, Local>();
            pos.y = box.y_progress;
            
            const UIElementSize* child_size = child.try_get<UIElementSize>();
            
            if (child_size) {
                box.y_progress += child_size->height + box.padding;
                if (child_size->width > max_width) {
                    max_width = child_size->width;
                }
            }
        });

        const Expand* expand = e.try_get<Expand>();

        // Only auto-resize HEIGHT if not expanding in Y
        if (!expand || !expand->y_enabled) {
            container_size.height = box.y_progress;
        }

        // Only auto-resize WIDTH if not expanding in X
        if (!expand || !expand->x_enabled) {
            container_size.width = max_width;
        }
    });

    auto cursorEvents = world.observer<CursorState, EditorRoot>()
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
                    split_editor({0.05, PanelSplitType::Horizontal}, partition_region.split_target, world, UIElement);
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

    world.system<Position, EditorNodeArea, PanelSplit*, CursorState>()
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
                    merge_editor(e, world, UIElement);
                    // TODO: Temporary merger to allow reversal?
                    merge_editor(e.parent(), world, UIElement);
                }
            } else
            {
                if (e.has<DynamicMerge>())
                {
                    e.remove<DynamicMerge>();
                    split_editor({0.05, PanelSplitType::Horizontal}, e, world, UIElement);
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

    auto query_dragging = world.query_builder<PanelSplit>()
    .with<Dragging>()
    .build();

    world.observer<CursorState, EditorRoot>()
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

    auto bubbleUpBoundsQuery = world.query_builder<UIElementBounds, UIElementBounds*, RenderStatus*>()
        .term_at(1).parent().up()  // Parent UIElementBounds
        .term_at(2).optional()          // Optional RenderStatus
        .build();

    auto bubbleUpBoundsSystem = world.system<UIElementBounds, UIElementBounds*, UIElementSize, RenderStatus*>()
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

    auto bubbleUpBoundsSecondarySystem = world.system<UIElementBounds, UIElementBounds*, RenderStatus*>()
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

    world.system<Position, UIElementBounds*, UIElementSize, UIElementBounds, Align>()
    .term_at(0).second<Local>()
    .term_at(1).parent()
    .kind(flecs::PreFrame)
    .each([&](flecs::entity e, Position& pos, UIElementBounds* parent_bounds, UIElementSize& ui_size, UIElementBounds& bounds, Align& align) 
    {
        pos.x = align.horizontal * (parent_bounds->xmax - parent_bounds->xmin) + ui_size.width * align.self_horizontal;
        pos.y = align.vertical * (parent_bounds->ymax - parent_bounds->ymin) + ui_size.height * align.self_vertical;
    });

    world.system<UIElementBounds*, RectRenderable, Expand>()
    .term_at(0).parent()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, RectRenderable& rect, Expand& expand) {
        // std::cout << bounds->xmin << std::endl;
        // std::cout << bounds->xmax << std::endl;
        
        if (expand.x_enabled)
        {
            rect.width = (bounds->xmax - bounds->xmin - (expand.pad_left + expand.pad_right))*expand.x_percent;
        }
        if (expand.y_enabled)
        {
            rect.height = (bounds->ymax - bounds->ymin - (expand.pad_top + expand.pad_bottom))*expand.y_percent;
        }
    });

    world.system<UIElementBounds*, RoundedRectRenderable, Expand>()
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

    world.system<UIElementBounds*, LineRenderable, Expand>()
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

    world.system<UIElementBounds*, ImageRenderable, Expand, Constrain*, Graphics>()
    .term_at(0).parent()
    .term_at(3).optional()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, ImageRenderable& sprite, Expand& expand, Constrain* constrain, Graphics& graphics) {        
        if (bounds)
        {
            int img_width, img_height;
            nvgImageSize(graphics.vg, sprite.imageHandle, &img_width, &img_height);
            
            if (img_width == 0 || img_height == 0) return;

            float aspect = (float)img_width / (float)img_height;
            float bounds_w = bounds->xmax - bounds->xmin;
            float bounds_h = bounds->ymax - bounds->ymin;
            
            // Calculate available space after padding
            float avail_w = bounds_w - (expand.pad_left + expand.pad_right);
            float avail_h = bounds_h - (expand.pad_top + expand.pad_bottom);

            float desired_w = sprite.width;
            float desired_h = sprite.height;

            if (expand.x_enabled)
            {
                // Start by trying to fill the width
                float target_w = avail_w;

                // If we also need to fit Y, check if our target width causes a height overflow
                if (constrain && constrain->fit_y)
                {
                    float height_from_width = target_w / aspect;
                    
                    // If the resulting height is too big, constrain by height instead
                    if (height_from_width > avail_h)
                    {

                        target_w = avail_h * aspect;
                    }
                }
                
                desired_w = target_w * expand.x_percent;
                
                // If Y is not enabled, calculate height based on the aspect ratio of the final width
                if (!expand.y_enabled)
                {
                    desired_h = desired_w / aspect;
                }
            }
            
            if (expand.y_enabled)
            {
                // Logic for Y-driven expansion (e.g. for vertical lists or sidebars)
                float target_h = avail_h;

                if (!expand.x_enabled) 
                {
                    if (constrain && constrain->fit_x)
                    {
                         float width_from_height = target_h * aspect;
                         if (width_from_height > avail_w)
                         {
                             target_h = avail_w / aspect;
                         }
                    }
                    desired_h = target_h * expand.y_percent;
                    desired_w = desired_h * aspect;
                }
                else 
                {
                    // If BOTH X and Y are enabled (and we have constraints), 
                    // we perform a centered "Aspect Fit" inside the box.
                    if (constrain && (constrain->fit_x || constrain->fit_y))
                    {
                        float scale_x = avail_w / img_width;
                        float scale_y = avail_h / img_height;
                        float scale = std::min(scale_x, scale_y);

                        desired_w = img_width * scale * expand.x_percent;
                        desired_h = img_height * scale * expand.y_percent;
                    }
                    else
                    {
                        // No constraints = stretch to fill
                        desired_h = avail_h * expand.y_percent;
                        // sprite.width was already set in the x_enabled block
                    }
                }
            }

            if (expand.cap_to_intrinsic) {
                float max_w = img_width * sprite.scaleX;
                float max_h = img_height * sprite.scaleY;
                if (max_w > 0.0f && max_h > 0.0f) {
                    float cap_scale = 1.0f;
                    if (desired_w > max_w) cap_scale = std::min(cap_scale, max_w / desired_w);
                    if (desired_h > max_h) cap_scale = std::min(cap_scale, max_h / desired_h);
                    if (cap_scale < 1.0f) {
                        desired_w *= cap_scale;
                        desired_h *= cap_scale;
                    }
                }
            }

            sprite.width = desired_w;
            sprite.height = desired_h;
        }
    });

    auto debugRenderBounds = world.system<RenderQueue, UIElementBounds, DebugRenderBounds>()
    .term_at(0).src(renderQueueEntity)
    .each([](flecs::entity e, RenderQueue& render_queue, UIElementBounds& bounds, DebugRenderBounds) 
    {
        RectRenderable debug_bound {bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin, true, 0xFFFF00FF};
        render_queue.addRectCommand({bounds.xmin, bounds.ymin}, debug_bound, 100);
    });

    auto roundedRectQueueSystem = world.system<Position, RoundedRectRenderable, ZIndex>()
    .term_at(0).second<World>()
    .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, Position& pos, RoundedRectRenderable& renderable, ZIndex& zIndex) {
            RenderQueue& queue = world.ensure<RenderQueue>();
            queue.addRoundedRectCommand(pos, renderable, zIndex.layer);
        });


    auto rectQueueSystem = world.system<Position, RectRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
        .each([&](flecs::entity e, Position& pos, RectRenderable& renderable, ZIndex& zIndex) {
            RenderQueue& queue = world.ensure<RenderQueue>();
            queue.addRectCommand(pos, renderable, zIndex.layer);
        });

    auto textQueueSystem = world.system<Position, TextRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, TextRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world.ensure<RenderQueue>();
        queue.addTextCommand(pos, renderable, zIndex.layer);
    });

    auto imageQueueSystem = world.system<Position, ImageRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, ImageRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world.ensure<RenderQueue>();
        queue.addImageCommand(pos, renderable, zIndex.layer);
    });

    auto lineQueueSystem = world.system<Position, LineRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, LineRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world.ensure<RenderQueue>();
        queue.addLineCommand(pos, renderable, zIndex.layer);
    });

    auto quadraticBezierQueueSystem = world.system<Position, QuadraticBezierRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, QuadraticBezierRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world.ensure<RenderQueue>();
        queue.addQuadraticBezierCommand(pos, renderable, zIndex.layer);
    });

    world.system<Position, EditorNodeArea, EditorLeafData, EditorRoot>()
    .term_at(0).second<World>()
    .term_at(3).src(editor_root)
    .run([](flecs::iter& it)
    {
        auto editor_root = world.lookup("editor_root").try_get_mut<EditorRoot>();
        editor_root->modify_partition_regions.clear();
        while (it.next()) {
            it.each();
        }
    }, [](flecs::entity e, Position& world_pos, EditorNodeArea& node_area, EditorLeafData& leaf_data, EditorRoot& editor_root) 
    {
        editor_root.modify_partition_regions.push_back({{world_pos.x, world_pos.y, world_pos.x + 8.0f, world_pos.y + 24.0f}, e});
    });

    int scale_region_dist = 8;
    
    world.system<Window, CursorState, EditorNodeArea, PanelSplit, Position, EditorRoot>()
    .term_at(0).src(glfwStateEntity)
    .term_at(1).src(glfwStateEntity)
    .term_at(4).second<World>()
    .term_at(5).src(editor_root)
    .run([scale_region_dist](flecs::iter& it)
    {
        auto editor_root = world.lookup("editor_root").try_get_mut<EditorRoot>();
        editor_root->shift_regions.clear();
        while (it.next()) {
            it.each();
        }
        
        auto window = it.field<Window>(0);
        auto cursor_state = it.field<CursorState>(1);
        

        // DEBUG BOUNDS
        // for (EditorShiftRegion& shift_region : editor_root->shift_regions)
        // {
        //     RenderQueue& queue = world.ensure<RenderQueue>();
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

    auto renderExecutionSystem = world.system<RenderQueue, Graphics>()
        .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, RenderQueue& queue, Graphics& graphics) {
            queue.sort();

            for (const auto& cmd : queue.commands) {
                switch (cmd.type) {
                    case RenderType::RoundedRectangle: {
                        const auto& rect = std::get<RoundedRectRenderable>(cmd.renderData);
                        nvgBeginPath(graphics.vg);
                        nvgRoundedRect(graphics.vg, cmd.pos.x, cmd.pos.y, rect.width, rect.height, rect.radius);

                        uint8_t r = (rect.color >> 24) & 0xFF;
                        uint8_t g = (rect.color >> 16) & 0xFF;
                        uint8_t b = (rect.color >> 8) & 0xFF;

                        if (rect.stroke)
                        {
                            nvgStrokeWidth(graphics.vg, 1.0f);
                            nvgStrokeColor(graphics.vg, nvgRGB(r, g, b));
                            nvgStroke(graphics.vg);
                        } else
                        {
                            nvgFillColor(graphics.vg, nvgRGB(r, g, b));
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
                        nvgFontSize(graphics.vg, text.fontSize);
                        nvgFontFace(graphics.vg, text.fontFace.c_str());
                        nvgTextAlign(graphics.vg, text.alignment);

                        uint8_t r = (text.color >> 24) & 0xFF;
                        uint8_t g = (text.color >> 16) & 0xFF;
                        uint8_t b = (text.color >> 8) & 0xFF;

                        nvgFillColor(graphics.vg, nvgRGB(r, g, b));
                        nvgText(graphics.vg, cmd.pos.x, cmd.pos.y, text.text.c_str(), nullptr);
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
            }

            queue.clear();
        });

    auto vncInitSystem = world.system<VNCClient>()
        .kind(flecs::PreUpdate)
        .each([](flecs::iter& it, size_t i, VNCClient& vnc) {
            static bool initialized[1] = {false};
            if (!initialized[vnc.quadrant] && vnc.connected && vnc.client) {
                std::cout << "[VNC INIT] Requesting full framebuffer update for quadrant " << vnc.quadrant << "..." << std::endl;
                SendFramebufferUpdateRequest(vnc.client, 0, 0, vnc.client->width, vnc.client->height, FALSE);
                initialized[vnc.quadrant] = true;
                std::cout << "[VNC INIT] Initial update request sent for quadrant " << vnc.quadrant << " " << vnc.client->width << "x" << vnc.client->height << std::endl;
            }
        });

    // Process VNC messages to trigger update callbacks
    auto vncMessageProcessingSystem = world.system<VNCClient>()
        .kind(flecs::OnUpdate)
        .each([](flecs::entity e, VNCClient& vnc) {
            if (!vnc.connected || !vnc.client) return;

            // Process pending VNC messages (non-blocking)
            // WaitForMessage with timeout of 0 makes it non-blocking
            int result = WaitForMessage(vnc.client, 0);
            if (result > 0) {
                // Message available, process it
                if (!HandleRFBServerMessage(vnc.client)) {
                    LOG_ERROR(LogCategory::VNC_CLIENT, "Failed to handle VNC message for quadrant {}", vnc.quadrant);
                }
            } else if (result < 0) {
                LOG_ERROR(LogCategory::VNC_CLIENT, "VNC connection error for quadrant {}", vnc.quadrant);
                vnc.connected = false;
            }
            // result == 0 means no messages, which is fine
        });

    auto vncTextureUpdateSystem = world.system<VNCClient, VNCTexture>()
        .kind(flecs::OnUpdate)
        .each([](flecs::entity e, VNCClient& vnc, VNCTexture& tex) {
            if (!vnc.connected || !vnc.client) {
                static bool warned = false;
                LOG_DEBUG(LogCategory::VNC_CLIENT, "Client not connected or null");
                return;
            }
            
            if (!tex.needsUpdate) return;
            std::cout << "VNC Texture Update System" << std::endl;
            
            LOG_TRACE(LogCategory::VNC_CLIENT, "Updating OpenGL texture {} for quadrant {}", tex.texture, vnc.quadrant);

            SDL_Surface* surface = (SDL_Surface*)rfbClientGetClientData(vnc.client, (void*)VNC_SURFACE_TAG);
            if (surface && surface->pixels) {
                // Submit vision processing job to background thread instead of blocking
                VisionProcessingJob job;
                job.quadrant = vnc.quadrant;
                job.paletteFile = "../assets/palettes/resurrect-64.hex";
                job.outputPath = "/tmp/vision_quad_" + std::to_string(vnc.quadrant) + ".png";
                job.width = surface->w;
                job.height = surface->h;
                job.pitch = surface->pitch;

                // Copy pixel data to avoid race conditions with VNC updates
                size_t dataSize = surface->pitch * surface->h;
                job.pixelData.resize(dataSize);
                memcpy(job.pixelData.data(), surface->pixels, dataSize);

                g_visionQueue.submit(job);
                LOG_TRACE(LogCategory::VNC_CLIENT, "Submitted vision processing job for quadrant {}", vnc.quadrant);


                LOG_TRACE(LogCategory::VNC_CLIENT, "Processing {} dirty rectangles", tex.dirtyRects.size());
                LOG_TRACE(LogCategory::VNC_CLIENT, "Surface info: {}x{}, format: {}", surface->w, surface->h, SDL_GetPixelFormatName(surface->format->format));

                // Update OpenGL texture from SDL surface pixels
                glBindTexture(GL_TEXTURE_2D, tex.texture);

                // Set pixel unpack alignment to handle pitch correctly
                int bytesPerPixel = surface->format->BytesPerPixel;
                int expectedPitch = surface->w * bytesPerPixel;

                // CRITICAL: When updating partial rects, GL_UNPACK_ROW_LENGTH tells OpenGL
                // how many pixels are in a row of the SOURCE data (the surface)
                // This must be set to surface->pitch / bytesPerPixel regardless of whether
                // we're updating the full surface or a partial rect
                int rowLengthPixels = surface->pitch / bytesPerPixel;

                LOG_TRACE(LogCategory::VNC_CLIENT, "Surface pitch: {}, bytes per pixel: {}", surface->pitch, surface->format->BytesPerPixel);

                // Always set row length for partial updates
                glPixelStorei(GL_UNPACK_ROW_LENGTH, rowLengthPixels);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

                // Determine the correct pixel format based on SDL surface format
                GLenum format = GL_BGRA;
                if (surface->format->Rmask == 0xFF) {
                    format = GL_RGBA;
                }

                LOG_TRACE(LogCategory::VNC_CLIENT, "Using OpenGL format: {}", (format == GL_BGRA ? "BGRA" : "RGBA"));

                // Process each dirty rectangle
                for (const auto& rectIn : tex.dirtyRects) {
                    // Clamp rect to surface/texture bounds to be safe
                    int rx = std::max(0, rectIn.x);
                    int ry = std::max(0, rectIn.y);
                    int rw = std::min(rectIn.w, surface->w - rx);
                    int rh = std::min(rectIn.h, surface->h - ry);
                    if (rw <= 0 || rh <= 0) continue;

                    // Pointer to top-left of the dirty region
                    uint8_t* regionStart = // NOTE: Changed to non-const for modification
                        static_cast<uint8_t*>(surface->pixels) + ry * surface->pitch + rx * 4;

                    // --- CRITICAL ADDITION: Force full alpha for the dirty rectangle ---
                    // This assumes a 32-bit format (bpp=4) where the alpha channel is the
                    // last byte of the 4-byte pixel (e.g., RGBA or BGRA).
                    // If the format is different, the offset (bpp - 1) needs adjustment.
                    for (int y = 0; y < rh; ++y) {
                        uint8_t* rowStart = regionStart + y * surface->pitch;
                        for (int x = 0; x < rw; ++x) {
                            // Set the alpha component (last byte) to 0xFF (fully opaque)
                            rowStart[x * 4 + (4 - 1)] = 0xFF;
                        }
                    }
                    // ------------------------------------------------------------------

                    // Critical change: we keep GL_UNPACK_ROW_LENGTH set so GL steps by `pitch` each row.
                    glTexSubImage2D(GL_TEXTURE_2D, 0, rx, ry, rw, rh, format, GL_UNSIGNED_BYTE, regionStart);
                }

                // Reset to defaults
                glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

                LOG_TRACE(LogCategory::VNC_CLIENT, "All rects updated successfully");

                // Save screenshot every second
                struct timeval current_time;
                gettimeofday(&current_time, NULL);
                long elapsed_ms = (current_time.tv_sec - g_last_screenshot_time.tv_sec) * 1000 +
                                  (current_time.tv_usec - g_last_screenshot_time.tv_usec) / 1000;

                // if (elapsed_ms >= 1000) {
                //     save_vnc_screenshot(surface, vnc.quadrant);
                //     g_last_screenshot_time = current_time;
                // }

                // Clear dirty rects and mark as updated
                tex.dirtyRects.clear();
                tex.needsUpdate = false;

                std::cout << "Updated tVNC texture " << std::endl;
            } else {
                LOG_ERROR(LogCategory::VNC_CLIENT, "Surface or pixels is null");
            }
        });

    int fontHandle = nvgCreateFont(vg, "ATARISTOCRAT", "../assets/ATARISTOCRAT.ttf");
    int interFontHandle = nvgCreateFont(vg, "Inter", "../assets/OpenSans-Regular.ttf");

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
        
        world.defer_begin();
        glfwPollEvents();
        world.defer_end();
        world.progress();

        nvgEndFrame(vg);

        glfwSwapBuffers(window);
    }

    nvgDeleteGL2(vg);
    glfwTerminate();
    return 0;
}
