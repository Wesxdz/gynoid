#include "spatial_containment.h"
#include "debug_log.h"
#include <cstddef>

void SpatialContainmentModule(flecs::world& ecs) {
    // Register enum types first
    ecs.component<PanelType>()
        .constant("VIEW_3D", PanelType::VIEW_3D)
        .constant("PROPERTIES", PanelType::PROPERTIES)
        .constant("OUTLINER", PanelType::OUTLINER)
        .constant("TIMELINE", PanelType::TIMELINE)
        .constant("GRAPH_EDITOR", PanelType::GRAPH_EDITOR)
        .constant("SHADER_EDITOR", PanelType::SHADER_EDITOR)
        .constant("COMPOSITOR", PanelType::COMPOSITOR)
        .constant("TEXT_EDITOR", PanelType::TEXT_EDITOR)
        .constant("CONSOLE", PanelType::CONSOLE)
        .constant("INFO", PanelType::INFO)
        .constant("PREFERENCES", PanelType::PREFERENCES)
        .constant("FILE_BROWSER", PanelType::FILE_BROWSER)
        .constant("ASSET_BROWSER", PanelType::ASSET_BROWSER)
        .constant("SPREADSHEET", PanelType::SPREADSHEET)
        .constant("OTHER", PanelType::OTHER);

    ecs.component<RegionType>()
        .constant("HEADER", RegionType::HEADER)
        .constant("WINDOW", RegionType::WINDOW)
        .constant("TOOLS", RegionType::TOOLS)
        .constant("TOOL_PROPS", RegionType::TOOL_PROPS)
        .constant("NAVIGATION_BAR", RegionType::NAVIGATION_BAR)
        .constant("EXECUTE", RegionType::EXECUTE)
        .constant("FOOTER", RegionType::FOOTER)
        .constant("CHANNELS", RegionType::CHANNELS)
        .constant("PREVIEW", RegionType::PREVIEW)
        .constant("HUD", RegionType::HUD)
        .constant("UI", RegionType::UI)
        .constant("OTHER", RegionType::OTHER);

    // Register all components with reflection metadata for JSON serialization
    ecs.component<SpatialBounds>()
        .member<int>("x")
        .member<int>("y")
        .member<int>("width")
        .member<int>("height");

    ecs.component<OCRBounds>()
        .member<int>("xmin")
        .member<int>("ymin")
        .member<int>("xmax")
        .member<int>("ymax");

    ecs.component<OCRText>()
        .member<std::string>("text");

    ecs.component<OCRConfidence>()
        .member<float>("value");

    ecs.component<OCRTimestamp>()
        .member<double>("creation_time");

    ecs.component<BlenderPanel>()
        .member<PanelType>("type");

    ecs.component<BlenderRegion>()
        .member<RegionType>("type");

    ecs.component<Quadrant>()
        .member<int>("id");

    LOG_INFO(LogCategory::SYSTEM, "SpatialContainmentModule initialized with reflection metadata");
}
