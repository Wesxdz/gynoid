#pragma once

// The ArcTask panel module: renders an ARC-AGI-1 task's train/test IO grid
// pairs, click-selects cells, and highlights the cell sets the Jane Eyre
// solver bridge answers nlp_query with. Registered as EditorType::ArcTask.

#include <flecs.h>

namespace panels {

struct ArcTask
{
    ArcTask(flecs::world& world);
};

// Steps the open panel to another task: delta walks the sorted dataset and
// wraps at both ends. False when no ArcTask panel is open, so a key handler
// can leave the arrow to whoever wants it next.
bool arc_task_step(int delta);

} // namespace panels
