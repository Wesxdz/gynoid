#pragma once

// The Type Generalization panel: a triple's seven adornments (Triple, Ego,
// Outbound, Bridge, Inbound, Alter, Predicate) and their extensions -- see
// type_generalization.png. Follows the relationship selected in the
// Interlocutor's annotator, the way the Lattice panel follows the selected
// badge.
//
// Registering the module contributes the PanelDef; see panels/registry.h.

#include <flecs.h>

namespace panels {

struct TypeGen
{
    TypeGen(flecs::world& world);
};

} // namespace panels
