#pragma once

#include "spatial_containment.h"
#include "blender_panel.h"
#include <flecs.h>
#include <vector>
#include <string>

// =============================================================================
// BLENDER PANEL AND REGION ENTITY CREATION
// =============================================================================

// Create a Blender panel entity
inline flecs::entity create_blender_panel_entity(
    flecs::world& world,
    const BlenderPanelBounds& panel_data,
    int quadrant,
    flecs::entity parent = flecs::entity::null())
{
    PanelType panel_type = parse_panel_type(panel_data.type);

    std::string entity_name = "BlenderPanel_Q" + std::to_string(quadrant) +
                              "_" + panel_data.type +
                              "_" + std::to_string(panel_data.x) +
                              "_" + std::to_string(panel_data.y);

    auto panel_entity = world.entity(entity_name.c_str())
        .set<BlenderPanel>({panel_type})
        .set<SpatialBounds>({panel_data.x, panel_data.y, panel_data.width, panel_data.height})
        .set<Quadrant>({quadrant});

    if (parent.is_alive()) {
        panel_entity.child_of(parent);
    }

    return panel_entity;
}

// Create a Blender region entity (child of panel)
inline flecs::entity create_blender_region_entity(
    flecs::world& world,
    const BlenderRegionBounds& region_data,
    int quadrant,
    flecs::entity panel_entity)
{
    RegionType region_type = parse_region_type(region_data.type);

    std::string entity_name = "BlenderRegion_Q" + std::to_string(quadrant) +
                              "_" + region_data.type +
                              "_" + std::to_string(region_data.x) +
                              "_" + std::to_string(region_data.y);

    auto region_entity = world.entity(entity_name.c_str())
        .set<BlenderRegion>({region_type})
        .set<SpatialBounds>({region_data.x, region_data.y, region_data.width, region_data.height})
        .set<Quadrant>({quadrant})
        .child_of(panel_entity);

    return region_entity;
}

// Create panel and region entities from BlenderPanelData
inline std::vector<flecs::entity> create_blender_hierarchy_for_quadrant(
    flecs::world& world,
    const BlenderPanelData& panel_data,
    flecs::entity quadrant_entity = flecs::entity::null())
{
    std::vector<flecs::entity> entities;

    for (const auto& panel : panel_data.panels) {
        // Create panel entity
        auto panel_entity = create_blender_panel_entity(
            world, panel, panel_data.quadrant, quadrant_entity);
        entities.push_back(panel_entity);

        // Create region entities as children of panel
        for (const auto& region : panel.regions) {
            auto region_entity = create_blender_region_entity(
                world, region, panel_data.quadrant, panel_entity);
            entities.push_back(region_entity);
        }
    }

    return entities;
}

// =============================================================================
// QUERY HELPERS
// =============================================================================

// Find a specific panel by type and quadrant
inline flecs::entity find_panel_by_type(
    flecs::world& world,
    PanelType panel_type,
    int quadrant = -1)
{
    flecs::entity result = flecs::entity::null();

    world.each([&](flecs::entity e, BlenderPanel& panel, Quadrant& q) {
        if (panel.type == panel_type && (quadrant < 0 || q.id == quadrant)) {
            result = e;
            return;  // Stop iteration
        }
    });

    return result;
}

// Find all regions of a specific type
inline std::vector<flecs::entity> find_regions_by_type(
    flecs::world& world,
    RegionType region_type,
    int quadrant = -1)
{
    std::vector<flecs::entity> results;

    world.each([&](flecs::entity e, BlenderRegion& region, Quadrant& q) {
        if (region.type == region_type && (quadrant < 0 || q.id == quadrant)) {
            results.push_back(e);
        }
    });

    return results;
}

// Find all panels in a quadrant
inline std::vector<flecs::entity> find_all_panels_in_quadrant(
    flecs::world& world,
    int quadrant)
{
    std::vector<flecs::entity> results;

    world.each([&](flecs::entity e, BlenderPanel&, Quadrant& q) {
        if (q.id == quadrant) {
            results.push_back(e);
        }
    });

    return results;
}

// Get all regions within a specific panel
inline std::vector<flecs::entity> get_regions_in_panel(flecs::entity panel_entity) {
    std::vector<flecs::entity> regions;

    panel_entity.children([&](flecs::entity child) {
        if (child.has<BlenderRegion>()) {
            regions.push_back(child);
        }
    });

    return regions;
}

// =============================================================================
// CLEANUP HELPERS
// =============================================================================

// Remove all Blender panel and region entities for a quadrant
inline void clear_blender_entities_for_quadrant(flecs::world& world, int quadrant) {
    std::vector<flecs::entity> to_delete;

    // Find all panels (regions will be deleted cascadingly via ChildOf)
    world.each([&](flecs::entity e, BlenderPanel&, Quadrant& q) {
        if (q.id == quadrant) {
            to_delete.push_back(e);
        }
    });

    // Also find standalone regions (if any)
    world.each([&](flecs::entity e, BlenderRegion&, Quadrant& q) {
        if (q.id == quadrant) {
            to_delete.push_back(e);
        }
    });

    for (auto e : to_delete) {
        e.destruct();  // This will cascade delete children
    }
}

// =============================================================================
// UPDATE HELPERS
// =============================================================================

// Update or create Blender entities from latest panel data
// Returns list of created/updated entities
inline std::vector<flecs::entity> update_blender_entities(
    flecs::world& world,
    const BlenderQuadrantPanels& panel_component,
    int quadrant)
{
    // Clear old entities for this quadrant
    clear_blender_entities_for_quadrant(world, quadrant);

    // Get panel data
    BlenderPanelData panel_data = panel_component.get();

    // Create new hierarchy
    return create_blender_hierarchy_for_quadrant(world, panel_data);
}
