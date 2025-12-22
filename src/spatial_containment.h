#pragma once

#include <flecs.h>
#include <string>
#include <vector>
#include <algorithm>

// =============================================================================
// CORE SPATIAL COMPONENTS
// =============================================================================

// Universal spatial bounds component for all spatial entities
struct SpatialBounds {
    int x, y;           // Position (top-left corner)
    int width, height;  // Dimensions

    SpatialBounds() : x(0), y(0), width(0), height(0) {}
    SpatialBounds(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}

    // Check if a point is contained within these bounds
    bool contains_point(int px, int py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }

    // Check if these bounds overlap with another
    bool overlaps(const SpatialBounds& other) const {
        return !(x + width <= other.x || other.x + other.width <= x ||
                 y + height <= other.y || other.y + other.height <= y);
    }

    // Check if these bounds fully contain another bounds
    bool contains(const SpatialBounds& other) const {
        return other.x >= x && other.y >= y &&
               (other.x + other.width) <= (x + width) &&
               (other.y + other.height) <= (y + height);
    }

    // Calculate area
    int area() const {
        return width * height;
    }

    // Calculate intersection area with another bounds
    int intersection_area(const SpatialBounds& other) const {
        if (!overlaps(other)) return 0;

        int x1 = std::max(x, other.x);
        int y1 = std::max(y, other.y);
        int x2 = std::min(x + width, other.x + other.width);
        int y2 = std::min(y + height, other.y + other.height);

        return (x2 - x1) * (y2 - y1);
    }
};

// =============================================================================
// OCR COMPONENTS
// =============================================================================

// OCR-specific bounds (using xmin/ymin/xmax/ymax convention)
struct OCRBounds {
    int xmin, ymin, xmax, ymax;

    OCRBounds() : xmin(0), ymin(0), xmax(0), ymax(0) {}
    OCRBounds(int x1, int y1, int x2, int y2) : xmin(x1), ymin(y1), xmax(x2), ymax(y2) {}

    int width() const { return xmax - xmin; }
    int height() const { return ymax - ymin; }
    int area() const { return width() * height(); }

    // Convert to SpatialBounds
    SpatialBounds to_spatial() const {
        return SpatialBounds(xmin, ymin, width(), height());
    }

    // Create from SpatialBounds
    static OCRBounds from_spatial(const SpatialBounds& bounds) {
        return OCRBounds(bounds.x, bounds.y, bounds.x + bounds.width, bounds.y + bounds.height);
    }

    bool contains_point(int px, int py) const {
        return px >= xmin && px < xmax && py >= ymin && py < ymax;
    }
};

// OCR detected text
struct OCRText {
    std::string text;

    OCRText() = default;
    OCRText(const std::string& t) : text(t) {}
};

// OCR confidence score
struct OCRConfidence {
    float value;

    OCRConfidence() : value(1.0f) {}
    OCRConfidence(float v) : value(v) {}
};

// OCR detection timestamp
struct OCRTimestamp {
    double creation_time;

    OCRTimestamp() : creation_time(0.0) {}
    OCRTimestamp(double t) : creation_time(t) {}
};

// =============================================================================
// BLENDER PANEL COMPONENTS
// =============================================================================

// Blender panel type enumeration
enum class PanelType {
    VIEW_3D,
    PROPERTIES,
    OUTLINER,
    TIMELINE,
    GRAPH_EDITOR,
    SHADER_EDITOR,
    COMPOSITOR,
    TEXT_EDITOR,
    CONSOLE,
    INFO,
    PREFERENCES,
    FILE_BROWSER,
    ASSET_BROWSER,
    SPREADSHEET,
    OTHER
};

// Convert string to PanelType
inline PanelType parse_panel_type(const std::string& type_str) {
    if (type_str == "VIEW_3D") return PanelType::VIEW_3D;
    if (type_str == "PROPERTIES") return PanelType::PROPERTIES;
    if (type_str == "OUTLINER") return PanelType::OUTLINER;
    if (type_str == "TIMELINE" || type_str == "DOPESHEET_EDITOR") return PanelType::TIMELINE;
    if (type_str == "GRAPH_EDITOR") return PanelType::GRAPH_EDITOR;
    if (type_str == "SHADER_EDITOR") return PanelType::SHADER_EDITOR;
    if (type_str == "NODE_EDITOR") return PanelType::COMPOSITOR;
    if (type_str == "TEXT_EDITOR") return PanelType::TEXT_EDITOR;
    if (type_str == "CONSOLE") return PanelType::CONSOLE;
    if (type_str == "INFO") return PanelType::INFO;
    if (type_str == "PREFERENCES") return PanelType::PREFERENCES;
    if (type_str == "FILE_BROWSER") return PanelType::FILE_BROWSER;
    if (type_str == "ASSETS") return PanelType::ASSET_BROWSER;
    if (type_str == "SPREADSHEET") return PanelType::SPREADSHEET;
    return PanelType::OTHER;
}

// Convert PanelType to string
inline const char* panel_type_to_string(PanelType type) {
    switch (type) {
        case PanelType::VIEW_3D: return "VIEW_3D";
        case PanelType::PROPERTIES: return "PROPERTIES";
        case PanelType::OUTLINER: return "OUTLINER";
        case PanelType::TIMELINE: return "TIMELINE";
        case PanelType::GRAPH_EDITOR: return "GRAPH_EDITOR";
        case PanelType::SHADER_EDITOR: return "SHADER_EDITOR";
        case PanelType::COMPOSITOR: return "COMPOSITOR";
        case PanelType::TEXT_EDITOR: return "TEXT_EDITOR";
        case PanelType::CONSOLE: return "CONSOLE";
        case PanelType::INFO: return "INFO";
        case PanelType::PREFERENCES: return "PREFERENCES";
        case PanelType::FILE_BROWSER: return "FILE_BROWSER";
        case PanelType::ASSET_BROWSER: return "ASSET_BROWSER";
        case PanelType::SPREADSHEET: return "SPREADSHEET";
        default: return "OTHER";
    }
}

// Blender panel component
struct BlenderPanel {
    PanelType type;

    BlenderPanel() : type(PanelType::OTHER) {}
    BlenderPanel(PanelType t) : type(t) {}
};

// Blender region type enumeration
enum class RegionType {
    HEADER,
    WINDOW,
    TOOLS,
    TOOL_PROPS,
    NAVIGATION_BAR,
    EXECUTE,
    FOOTER,
    CHANNELS,
    PREVIEW,
    HUD,
    UI,
    OTHER
};

// Convert string to RegionType
inline RegionType parse_region_type(const std::string& type_str) {
    if (type_str == "HEADER") return RegionType::HEADER;
    if (type_str == "WINDOW") return RegionType::WINDOW;
    if (type_str == "TOOLS") return RegionType::TOOLS;
    if (type_str == "TOOL_PROPS") return RegionType::TOOL_PROPS;
    if (type_str == "NAVIGATION_BAR") return RegionType::NAVIGATION_BAR;
    if (type_str == "EXECUTE") return RegionType::EXECUTE;
    if (type_str == "FOOTER") return RegionType::FOOTER;
    if (type_str == "CHANNELS") return RegionType::CHANNELS;
    if (type_str == "PREVIEW") return RegionType::PREVIEW;
    if (type_str == "HUD") return RegionType::HUD;
    if (type_str == "UI") return RegionType::UI;
    return RegionType::OTHER;
}

// Convert RegionType to string
inline const char* region_type_to_string(RegionType type) {
    switch (type) {
        case RegionType::HEADER: return "HEADER";
        case RegionType::WINDOW: return "WINDOW";
        case RegionType::TOOLS: return "TOOLS";
        case RegionType::TOOL_PROPS: return "TOOL_PROPS";
        case RegionType::NAVIGATION_BAR: return "NAVIGATION_BAR";
        case RegionType::EXECUTE: return "EXECUTE";
        case RegionType::FOOTER: return "FOOTER";
        case RegionType::CHANNELS: return "CHANNELS";
        case RegionType::PREVIEW: return "PREVIEW";
        case RegionType::HUD: return "HUD";
        case RegionType::UI: return "UI";
        default: return "OTHER";
    }
}

// Blender region component
struct BlenderRegion {
    RegionType type;

    BlenderRegion() : type(RegionType::OTHER) {}
    BlenderRegion(RegionType t) : type(t) {}
};

// =============================================================================
// QUADRANT COMPONENTS
// =============================================================================

// Quadrant identifier component (0-3)
struct Quadrant {
    int id;  // 0-3

    Quadrant() : id(0) {}
    Quadrant(int i) : id(i) {}
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Helper to walk up the entity hierarchy to find an ancestor with a specific component
template<typename T>
flecs::entity find_ancestor_with(flecs::entity e) {
    auto current = e;
    while (current.is_alive()) {
        if (current.has<T>()) {
            return current;
        }
        current = current.parent();
    }
    return flecs::entity::null();
}

// Helper to check if an entity is contained within another entity (via ChildOf hierarchy)
inline bool is_contained_in(flecs::entity child, flecs::entity parent) {
    auto current = child;
    while (current.is_alive()) {
        if (current == parent) {
            return true;
        }
        current = current.parent();
    }
    return false;
}

// Helper to get all children of an entity
inline std::vector<flecs::entity> get_children(flecs::entity parent) {
    std::vector<flecs::entity> children;
    parent.children([&](flecs::entity child) {
        children.push_back(child);
    });
    return children;
}

// Helper to get all descendants of an entity (recursive)
inline void get_descendants_recursive(flecs::entity parent, std::vector<flecs::entity>& descendants) {
    parent.children([&](flecs::entity child) {
        descendants.push_back(child);
        get_descendants_recursive(child, descendants);
    });
}

inline std::vector<flecs::entity> get_all_descendants(flecs::entity parent) {
    std::vector<flecs::entity> descendants;
    get_descendants_recursive(parent, descendants);
    return descendants;
}

// =============================================================================
// MODULE EXPORT
// =============================================================================

void SpatialContainmentModule(flecs::world& ecs);
