#include <iostream>       // Required for std::ostream
#include <string>         // Required for std::string and std::to_string
#include <vector>         // Required for std::vector (dirtyRects)
#include <memory>         // Required for std::shared_ptr
#include <mutex>          // Required for std::mutex

// --- Graphics and Windowing ---
#include <glad/glad.h>    // Required for GLuint
#include <SDL2/SDL.h>     // Required for SDL_Surface

// --- LibVNC ---
#include <rfb/rfbclient.h> // Required for rfbClient*

struct VNCUpdateRect {
    int x, y, w, h;
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
    bool needsUpdate = false;
    std::vector<VNCUpdateRect> dirtyRects;

    friend std::ostream& operator<<(std::ostream& os, const VNCClient& vnc) {
        return os << vnc.host << ":" << vnc.port;
    }
    
    std::string toString() const {
        return host + ":" + std::to_string(port);
    }
};

struct IsStreamingFrom {};