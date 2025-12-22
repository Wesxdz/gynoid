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
    X11WindowBounds screen;
    X11WindowBounds work_area;
    bool has_work_area;
    std::string last_update_timestamp;
};

// Tag to mark X11 window entities as visible
struct X11VisibleWindow {};

// Tag to mark X11 window entities as special windows
struct X11SpecialWindow {};

struct X11WindowData {
    std::string id;
    std::string name;
    X11WindowBounds bounds;
    int z_order;
    int visible_pixels;
    int onscreen_pixels;
    float visibility_percent;
};

// Temporary structure for parsing JSON (not an ECS component)
struct X11OutlineData {
    std::string container_id;
    std::string timestamp;
    std::vector<X11WindowData> visible_windows;
    std::vector<X11WindowData> special_windows;
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

// Module export
void X11OutlineModule(flecs::world& ecs);
