#pragma once

// The Dataset panel: every task in the set as a vertical list, the open one
// outlined, each row coloured by how the solver last did on it. Clicking a row
// opens that task in the ArcTask panel.

#include <flecs.h>

namespace panels {

struct Dataset
{
    Dataset(flecs::world& world);
};

} // namespace panels
