#pragma once

// The Droid panel as a flecs module: the render3d chassis viewport, its
// layout/pointer sync, and the cursor-riding part tooltip. Importing it
// registers the systems and contributes the PanelDef; panel3d::init (the GL
// side) stays in main.cpp with the other graphics-ready setup.
//
// Import ordering matters twice: after the shared components are registered
// at root scope (see the import site in main.cpp), and after
// boundsCalculationSystem -- the sync system runs in the same OnLoad phase
// and must read this frame's bounds, and flecs runs same-phase systems in
// declaration order.

#include <flecs.h>

namespace panels {

struct Droid
{
    Droid(flecs::world& world);
};

} // namespace panels
