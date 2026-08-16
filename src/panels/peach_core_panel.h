#pragma once

// The Peach Core panel as a flecs module: one icon per supporting server, its
// status light, and click-to-start/stop.
//
// The panel holds no server data of its own. Every icon carries a ServerOf
// reference to a registry entity from src/server_registry.h, which is where the
// process, its state, and its config entry live -- so a panel can be closed and
// reopened without disturbing anything that is running.
//
// Import after servers::module (the registry entities must exist to be drawn)
// and after the shared components are registered at root scope.

#include <flecs.h>

namespace panels {

struct PeachCore
{
    PeachCore(flecs::world& world);
};

} // namespace panels
