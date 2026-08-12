#pragma once

// The panel registry: how panel types get out of main.cpp.
//
// A panel module (a flecs module under src/panels/) constructs a PanelDef in
// its import constructor and registers it here. create_editor_content consults
// the registry before its legacy else-if chain, so panels migrate one at a
// time -- the chain shrinks, the registry grows, and nothing has to move in
// one big bang.
//
// EditorType stays the id for now because saved layouts store the enum as an
// int. The string id is the successor: once layouts read/write it, the
// index-aligned enum + editor_types name list can go.

#include <flecs.h>

// Forward-declared rather than pulling in components.h: a scoped enum is a
// complete type from its declaration, and this keeps the registry includable
// without the GL/GLFW prelude components.h silently depends on.
enum class EditorType;

struct PanelDef
{
    EditorType type;
    const char* id;     // stable string id, the future layout-file key
    const char* name;   // display name; must match editor_types for now
    void (*create_content)(flecs::entity leaf, flecs::entity UIElement);
};

namespace panels {

void register_panel(const PanelDef& def);
const PanelDef* find(EditorType type);

} // namespace panels
