#pragma once

// The VarRegistry panel: the solver's scene graph for the task in context,
// browsed as an indented tree. Jane Eyre builds the same instruction list it
// publishes during a solve; this panel asks for it on demand and draws it.

#include <flecs.h>

namespace panels {

struct VarRegistry
{
    VarRegistry(flecs::world& world);
};

} // namespace panels
