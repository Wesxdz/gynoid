#pragma once

// The Holodeck panel as a flecs module -- the template extraction for pulling
// panels out of main.cpp. Importing it registers the panel's sync system and
// contributes its PanelDef to the registry; main.cpp keeps only the import.
//
// Shared components the module queries (Screen3DViewport, Dragging,
// UIElementBounds, ...) must be registered at root scope before the import,
// or the module scope would fork them into distinct component ids from the
// monolith's. main.cpp does this explicitly at the import site.

#include <flecs.h>

namespace panels {

struct Holodeck
{
    Holodeck(flecs::world& world);
};

} // namespace panels
