#pragma once

// The ArcTask panel module: renders an ARC-AGI-1 task's train/test IO grid
// pairs, click-selects cells, and highlights the cell sets the Jane Eyre
// solver bridge answers nlp_query with. Registered as EditorType::ArcTask.

#include <flecs.h>
#include <string>

namespace panels {

struct ArcTask
{
    ArcTask(flecs::world& world);
};

// Steps the open panel to another task: delta walks the sorted dataset and
// wraps at both ends. False when no ArcTask panel is open, so a key handler
// can leave the arrow to whoever wants it next.
bool arc_task_step(int delta);

// Opens a task by name. False when no ArcTask panel is open or the dataset has
// no such task -- the caller is usually a list that believes it does, and a
// silent miss would look like a dead click.
bool arc_task_open(const std::string& task_id);

// The task the panel currently holds, empty when none. Published so a browser
// can mark which row is the open one without keeping its own idea of it.
std::string arc_task_current();

} // namespace panels
