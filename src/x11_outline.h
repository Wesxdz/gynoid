#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <flecs.h>
#include <iostream>

// ECS Components for X11 windows

// Separate component for window bounds (clipped bounds only)
struct X11WindowBounds {
    int x, y, width, height;
};

// Window metadata component (no bounds embedded)
struct X11WindowInfo {
    std::string id;
    std::string name;
    int z_order;
};

// Window visibility metrics component
struct X11WindowVisibility {
    int visible_pixels;
    int onscreen_pixels;
    float visibility_percent;
};

// Container component for X11 quadrants
struct X11Container {
    std::string container_id;
    int quadrant;  // 0-3
    X11WindowBounds screen;
    X11WindowBounds work_area;
    bool has_work_area;
    std::string last_update_timestamp;
};

// Tag to mark X11 window entities as visible
struct X11VisibleWindow {};

// Tag to mark X11 window entities as special windows
struct X11SpecialWindow {};

// Temporary structure for parsing JSON (not an ECS component)
struct X11OutlineData {
    std::string container_id;
    std::string timestamp;
    int quadrant;
    struct WindowData {
        std::string id;
        std::string name;
        X11WindowBounds bounds;  // clipped bounds
        int z_order;
        int visible_pixels;
        int onscreen_pixels;
        float visibility_percent;
    };
    std::vector<WindowData> visible_windows;
    std::vector<WindowData> special_windows;
    X11WindowBounds screen;
    X11WindowBounds work_area;
    bool has_work_area;
};

// Thread-safe queue for X11 outline updates
struct X11OutlineQueue {
    std::vector<X11OutlineData> pending;
    std::mutex mutex;

    void push(const X11OutlineData& data) {
        std::lock_guard<std::mutex> lock(mutex);
        pending.push_back(data);
    }

    std::vector<X11OutlineData> pop_all() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<X11OutlineData> result = std::move(pending);
        pending.clear();
        return result;
    }
};

// Socket server component
struct X11OutlineServer {
    std::thread server_thread;
    int port;
    bool running;
    X11OutlineQueue* queue;  // Pointer to shared queue

    X11OutlineServer() : port(5299), running(false), queue(nullptr) {}
};

// Parse container ID to quadrant (plasma-vnc-1 -> 0, plasma-vnc-2 -> 1, etc.)
inline int container_id_to_quadrant(const std::string& container_id) {
    // Check for full container name format
    if (container_id.find("plasma-vnc-1") != std::string::npos) return 0;
    if (container_id.find("plasma-vnc-2") != std::string::npos) return 1;
    if (container_id.find("plasma-vnc-3") != std::string::npos) return 2;
    if (container_id.find("plasma-vnc-4") != std::string::npos) return 3;

    // Check for simple numeric format (1 -> 0, 2 -> 1, etc.)
    if (container_id == "1") return 0;
    if (container_id == "2") return 1;
    if (container_id == "3") return 2;
    if (container_id == "4") return 3;

    return -1;  // Unknown container
}

// Module export
void X11OutlineModule(flecs::world& ecs);
