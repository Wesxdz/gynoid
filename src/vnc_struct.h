#include <iostream>       // Required for std::ostream
#include <string>         // Required for std::string and std::to_string
#include <vector>         // Required for std::vector (dirtyRects)
#include <memory>         // Required for std::shared_ptr
#include <mutex>          // Required for std::mutex
#include <thread>         // Required for std::thread
#include <atomic>         // Required for std::atomic
#include <condition_variable> // Required for std::condition_variable
#include <deque>          // Required for std::deque (input event queue)

// --- Graphics and Windowing ---
#include <glad/glad.h>    // Required for GLuint
#include <SDL2/SDL.h>     // Required for SDL_Surface

// --- LibVNC ---
#include <rfb/rfbclient.h> // Required for rfbClient*

struct VNCUpdateRect {
    int x, y, w, h;
};

// Lock-free dirty rectangle queue for network thread → main thread communication
struct DirtyRectQueue {
    std::vector<VNCUpdateRect> pending;
    std::mutex mutex;

    void push(const VNCUpdateRect& rect) {
        std::lock_guard<std::mutex> lock(mutex);
        pending.push_back(rect);
    }

    std::vector<VNCUpdateRect> swap() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<VNCUpdateRect> result;
        std::swap(result, pending);
        return result;
    }
};

// Input event for main thread → network thread communication
struct InputEvent {
    enum Type { POINTER, KEY } type;
    union {
        struct { int x; int y; int buttonMask; } pointer;
        struct { uint32_t keysym; int down; } key;
    } data;
};

struct VNCClient {
    rfbClient* client = nullptr;
    SDL_Surface* surface = nullptr;
    GLuint vncTexture;
    int nvgHandle = -1;
    bool connected = false;
    bool initialized = false;
    std::string host;
    int port;
    int width = 0;
    int height = 0;
    int reference_count = 0;
    std::shared_ptr<std::mutex> surfaceMutex;
    std::atomic<bool> needsUpdate{false};
    std::vector<VNCUpdateRect> dirtyRects;
    bool eventPassthroughEnabled;

    // Threading infrastructure
    std::thread messageThread;                      // Dedicated network I/O thread
    std::atomic<bool> threadRunning{false};         // Thread lifecycle flag
    std::atomic<bool> threadShouldStop{false};      // Shutdown signal

    // Input event queue (main thread → network thread)
    std::deque<InputEvent> inputQueue;
    std::mutex inputQueueMutex;
    std::condition_variable inputQueueCV;

    // Dirty rectangle queue (network thread → main thread)
    std::shared_ptr<DirtyRectQueue> dirtyRectQueue;

    // PBO for async texture upload
    GLuint pbo = 0;                                 // Pixel Buffer Object

    // Connection state machine
    enum ConnectionState {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        DISCONNECTING,
        ERROR
    };
    std::atomic<ConnectionState> connectionState{DISCONNECTED};
    std::string errorMessage;
    std::mutex errorMutex;

    // Delete copy constructor and copy assignment (non-copyable due to thread/mutex/atomic)
    VNCClient(const VNCClient&) = delete;
    VNCClient& operator=(const VNCClient&) = delete;

    // Default constructor
    VNCClient() = default;

    // Move constructor
    VNCClient(VNCClient&& other) noexcept
        : client(other.client), surface(other.surface), vncTexture(other.vncTexture),
          nvgHandle(other.nvgHandle), connected(other.connected), initialized(other.initialized),
          host(std::move(other.host)), port(other.port), width(other.width), height(other.height),
          reference_count(other.reference_count), surfaceMutex(std::move(other.surfaceMutex)),
          needsUpdate(other.needsUpdate.load()), dirtyRects(std::move(other.dirtyRects)),
          eventPassthroughEnabled(other.eventPassthroughEnabled),
          messageThread(std::move(other.messageThread)),
          threadRunning(other.threadRunning.load()),
          threadShouldStop(other.threadShouldStop.load()),
          inputQueue(std::move(other.inputQueue)),
          inputQueueMutex(), inputQueueCV(),
          dirtyRectQueue(std::move(other.dirtyRectQueue)),
          pbo(other.pbo),
          connectionState(other.connectionState.load()),
          errorMessage(std::move(other.errorMessage)),
          errorMutex()
    {
        other.client = nullptr;
        other.surface = nullptr;
        other.pbo = 0;
    }

    // Move assignment
    VNCClient& operator=(VNCClient&& other) noexcept {
        if (this != &other) {
            client = other.client;
            surface = other.surface;
            vncTexture = other.vncTexture;
            nvgHandle = other.nvgHandle;
            connected = other.connected;
            initialized = other.initialized;
            host = std::move(other.host);
            port = other.port;
            width = other.width;
            height = other.height;
            reference_count = other.reference_count;
            surfaceMutex = std::move(other.surfaceMutex);
            needsUpdate.store(other.needsUpdate.load());
            dirtyRects = std::move(other.dirtyRects);
            eventPassthroughEnabled = other.eventPassthroughEnabled;
            messageThread = std::move(other.messageThread);
            threadRunning.store(other.threadRunning.load());
            threadShouldStop.store(other.threadShouldStop.load());
            inputQueue = std::move(other.inputQueue);
            dirtyRectQueue = std::move(other.dirtyRectQueue);
            pbo = other.pbo;
            connectionState.store(other.connectionState.load());
            errorMessage = std::move(other.errorMessage);

            other.client = nullptr;
            other.surface = nullptr;
            other.pbo = 0;
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const VNCClient& vnc) {
        return os << vnc.host << ":" << vnc.port;
    }

    std::string toString() const {
        return host + ":" + std::to_string(port);
    }
};

struct IsStreamingFrom {};
struct ActiveIndicator {};