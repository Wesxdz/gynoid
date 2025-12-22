#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <flecs.h>

// Blender region bounds (subpanels within an area)
struct BlenderRegionBounds {
    std::string type;  // e.g., "HEADER", "WINDOW", "TOOLS", "NAVIGATION_BAR"
    int x, y, width, height;
};

// Blender panel bounds (relative to Blender window)
struct BlenderPanelBounds {
    std::string type;  // e.g., "VIEW_3D", "PROPERTIES", "OUTLINER"
    int x, y, width, height;
    std::vector<BlenderRegionBounds> regions;
};

// Blender panel data from a specific container/quadrant
struct BlenderPanelData {
    std::string container_id;
    std::string timestamp;
    int quadrant;  // 0-3, mapped from container_id
    std::vector<BlenderPanelBounds> panels;
};

// Component to store Blender panel data per quadrant
struct BlenderQuadrantPanels {
    BlenderPanelData data;
    mutable std::mutex mutex;
    bool has_data = false;

    void update(const BlenderPanelData& new_data) {
        std::lock_guard<std::mutex> lock(mutex);
        data = new_data;
        has_data = true;
    }

    BlenderPanelData get() const {
        std::lock_guard<std::mutex> lock(mutex);
        return data;
    }
};

// Socket server component
struct BlenderPanelServer {
    std::thread server_thread;
    int port;
    bool running;

    BlenderPanelServer() : port(9998), running(false) {}
};

// Parse container ID to quadrant (same as X11 outline system)
inline int blender_container_id_to_quadrant(const std::string& container_id) {
    if (container_id.find("plasma-vnc-1") != std::string::npos) return 0;
    if (container_id.find("plasma-vnc-2") != std::string::npos) return 1;
    if (container_id.find("plasma-vnc-3") != std::string::npos) return 2;
    if (container_id.find("plasma-vnc-4") != std::string::npos) return 3;

    if (container_id == "1") return 0;
    if (container_id == "2") return 1;
    if (container_id == "3") return 2;
    if (container_id == "4") return 3;

    return -1;
}

// Module export
void BlenderPanelModule(flecs::world& ecs);
