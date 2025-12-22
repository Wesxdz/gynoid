#pragma once

#include "spatial_containment.h"
#include "ocr_stream.h"
#include "blender_panel.h"
#include <flecs.h>
#include <vector>
#include <string>

// =============================================================================
// OCR ENTITY CREATION AND MANAGEMENT
// =============================================================================

// Create an OCR bound entity with proper containment relationships
// Returns the created entity
inline flecs::entity create_ocr_entity(
    flecs::world& world,
    const OCRWord& word,
    int quadrant,
    flecs::entity parent = flecs::entity::null())
{
    // Generate unique name for the OCR entity
    std::string entity_name = "OCR_Q" + std::to_string(quadrant) +
                              "_" + word.text +
                              "_" + std::to_string(word.xmin) +
                              "_" + std::to_string(word.ymin);

    // Create the entity
    auto ocr_entity = world.entity(entity_name.c_str())
        .set<OCRBounds>({word.xmin, word.ymin, word.xmax, word.ymax})
        .set<OCRText>({word.text})
        .set<OCRConfidence>({word.confidence})
        .set<OCRTimestamp>({word.creationTime})
        .set<Quadrant>({quadrant});

    // Set parent relationship if provided
    if (parent.is_alive()) {
        ocr_entity.child_of(parent);
    }

    return ocr_entity;
}

// Update or create OCR entities for a quadrant
// Returns vector of created/updated entities
inline std::vector<flecs::entity> update_ocr_entities_for_quadrant(
    flecs::world& world,
    const OCRResult& ocr_result,
    flecs::entity quadrant_entity)
{
    std::vector<flecs::entity> entities;

    for (const auto& word : ocr_result.words) {
        auto ocr_entity = create_ocr_entity(world, word, ocr_result.quadrant, quadrant_entity);
        entities.push_back(ocr_entity);
    }

    return entities;
}

// Find the best parent entity for an OCR bound based on spatial containment
// Checks Blender regions first, then panels, then X11 windows
inline flecs::entity find_spatial_parent_for_ocr(
    flecs::world& world,
    const OCRBounds& ocr_bounds,
    int quadrant)
{
    SpatialBounds ocr_spatial = ocr_bounds.to_spatial();
    flecs::entity best_parent = flecs::entity::null();
    int best_area = INT_MAX;  // Prefer smallest containing parent

    // First, try to find a containing Blender region
    world.each([&](flecs::entity e, BlenderRegion& region, SpatialBounds& bounds, Quadrant& q) {
        if (q.id == quadrant && bounds.contains(ocr_spatial)) {
            int area = bounds.area();
            if (area < best_area) {
                best_area = area;
                best_parent = e;
            }
        }
    });

    if (best_parent.is_alive()) {
        return best_parent;
    }

    // Next, try to find a containing Blender panel
    world.each([&](flecs::entity e, BlenderPanel& panel, SpatialBounds& bounds, Quadrant& q) {
        if (q.id == quadrant && bounds.contains(ocr_spatial)) {
            int area = bounds.area();
            if (area < best_area) {
                best_area = area;
                best_parent = e;
            }
        }
    });

    if (best_parent.is_alive()) {
        return best_parent;
    }

    // Finally, try to find a containing X11 window
    world.each([&](flecs::entity e, const X11WindowBounds& bounds, Quadrant& q) {
        if (q.id == quadrant) {
            SpatialBounds window_bounds(bounds.x, bounds.y, bounds.width, bounds.height);
            if (window_bounds.contains(ocr_spatial)) {
                int area = window_bounds.area();
                if (area < best_area) {
                    best_area = area;
                    best_parent = e;
                }
            }
        }
    });

    return best_parent;  // May be null if no containing parent found
}

// Create OCR entities with automatic spatial parent detection
inline std::vector<flecs::entity> create_ocr_entities_with_spatial_parents(
    flecs::world& world,
    const OCRResult& ocr_result,
    int quadrant)
{
    std::vector<flecs::entity> entities;

    for (const auto& word : ocr_result.words) {
        OCRBounds bounds(word.xmin, word.ymin, word.xmax, word.ymax);
        auto parent = find_spatial_parent_for_ocr(world, bounds, quadrant);

        auto ocr_entity = create_ocr_entity(world, word, quadrant, parent);
        entities.push_back(ocr_entity);
    }

    return entities;
}

// =============================================================================
// QUERY HELPERS
// =============================================================================

// Find all OCR entities in a specific region type (e.g., HEADER)
inline std::vector<flecs::entity> find_ocr_in_region_type(
    flecs::world& world,
    RegionType region_type,
    int quadrant = -1)  // -1 means all quadrants
{
    std::vector<flecs::entity> results;

    world.each([&](flecs::entity e, OCRText& text, OCRBounds& bounds, Quadrant& q) {
        if (quadrant >= 0 && q.id != quadrant) return;

        // Walk up hierarchy to find region
        auto parent = e.parent();
        while (parent.is_alive()) {
            if (parent.has<BlenderRegion>()) {
                const auto* region = parent.try_get<BlenderRegion>();
                if (region && region->type == region_type) {
                    results.push_back(e);
                    break;
                }
            }
            parent = parent.parent();
        }
    });

    return results;
}

// Find all OCR entities in a specific panel type (e.g., VIEW_3D)
inline std::vector<flecs::entity> find_ocr_in_panel_type(
    flecs::world& world,
    PanelType panel_type,
    int quadrant = -1)
{
    std::vector<flecs::entity> results;

    world.each([&](flecs::entity e, OCRText& text, OCRBounds& bounds, Quadrant& q) {
        if (quadrant >= 0 && q.id != quadrant) return;

        // Walk up hierarchy to find panel
        auto parent = e.parent();
        while (parent.is_alive()) {
            if (parent.has<BlenderPanel>()) {
                const auto* panel = parent.try_get<BlenderPanel>();
                if (panel && panel->type == panel_type) {
                    results.push_back(e);
                    break;
                }
            }
            parent = parent.parent();
        }
    });

    return results;
}

// Find all OCR text containing a specific string
inline std::vector<flecs::entity> find_ocr_with_text(
    flecs::world& world,
    const std::string& search_text,
    bool case_sensitive = false)
{
    std::vector<flecs::entity> results;

    world.each([&](flecs::entity e, OCRText& text) {
        std::string haystack = text.text;
        std::string needle = search_text;

        if (!case_sensitive) {
            std::transform(haystack.begin(), haystack.end(), haystack.begin(), ::tolower);
            std::transform(needle.begin(), needle.end(), needle.begin(), ::tolower);
        }

        if (haystack.find(needle) != std::string::npos) {
            results.push_back(e);
        }
    });

    return results;
}

// Find all OCR entities within pixel bounds
inline std::vector<flecs::entity> find_ocr_in_bounds(
    flecs::world& world,
    int x, int y, int width, int height,
    int quadrant = -1)
{
    std::vector<flecs::entity> results;
    SpatialBounds search_bounds(x, y, width, height);

    world.each([&](flecs::entity e, OCRBounds& bounds, Quadrant& q) {
        if (quadrant >= 0 && q.id != quadrant) return;

        SpatialBounds ocr_spatial = bounds.to_spatial();
        if (search_bounds.contains(ocr_spatial)) {
            results.push_back(e);
        }
    });

    return results;
}

// Get OCR text for all entities, optionally filtered by region/panel
struct OCRTextInfo {
    std::string text;
    OCRBounds bounds;
    int quadrant;
    RegionType region_type;
    PanelType panel_type;
    bool has_region;
    bool has_panel;
};

inline std::vector<OCRTextInfo> get_all_ocr_with_context(flecs::world& world) {
    std::vector<OCRTextInfo> results;

    world.each([&](flecs::entity e, OCRText& text, OCRBounds& bounds, Quadrant& q) {
        OCRTextInfo info;
        info.text = text.text;
        info.bounds = bounds;
        info.quadrant = q.id;
        info.has_region = false;
        info.has_panel = false;

        // Walk up hierarchy to find region and panel
        auto parent = e.parent();
        while (parent.is_alive()) {
            if (!info.has_region && parent.has<BlenderRegion>()) {
                const auto* region = parent.try_get<BlenderRegion>();
                if (region) {
                    info.region_type = region->type;
                    info.has_region = true;
                }
            }
            if (!info.has_panel && parent.has<BlenderPanel>()) {
                const auto* panel = parent.try_get<BlenderPanel>();
                if (panel) {
                    info.panel_type = panel->type;
                    info.has_panel = true;
                }
            }
            parent = parent.parent();
        }

        results.push_back(info);
    });

    return results;
}

// =============================================================================
// CLEANUP HELPERS
// =============================================================================

// Remove all OCR entities for a specific quadrant
inline void clear_ocr_entities_for_quadrant(flecs::world& world, int quadrant) {
    std::vector<flecs::entity> to_delete;

    world.each([&](flecs::entity e, OCRText&, Quadrant& q) {
        if (q.id == quadrant) {
            to_delete.push_back(e);
        }
    });

    for (auto e : to_delete) {
        e.destruct();
    }
}

// Remove stale OCR entities (older than max_age seconds)
inline void remove_stale_ocr_entities(flecs::world& world, double current_time, double max_age) {
    std::vector<flecs::entity> to_delete;

    world.each([&](flecs::entity e, OCRTimestamp& timestamp) {
        if (current_time - timestamp.creation_time > max_age) {
            to_delete.push_back(e);
        }
    });

    for (auto e : to_delete) {
        e.destruct();
    }
}
