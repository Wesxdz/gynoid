#pragma once

// The Solve panel: runs Jane Eyre's search on the task in context and shows
// what it is doing while it does it -- the progress it prints, the hypotheses
// it attempts, and the verdict it lands on.

#include <flecs.h>

namespace panels {

struct Solve
{
    Solve(flecs::world& world);
};

} // namespace panels
