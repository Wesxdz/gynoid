#pragma once

// The Lattice panel: where an entity sits in the subsumption order over types,
// and how several entities' lattices line up when they form a sentence -- an
// analogy being the case where the alignment itself is the content.
//
// Registering the module contributes the PanelDef; see panels/registry.h.

#include <flecs.h>

namespace panels {

struct Lattice
{
    Lattice(flecs::world& world);
};

} // namespace panels
