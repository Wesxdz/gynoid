#include "registry.h"

#include <vector>

namespace {

// Function-local static so panel modules registering during world import never
// race a global's construction order.
std::vector<PanelDef>& defs()
{
    static std::vector<PanelDef> v;
    return v;
}

} // namespace

namespace panels {

void register_panel(const PanelDef& def)
{
    defs().push_back(def);
}

const PanelDef* find(EditorType type)
{
    for (const PanelDef& def : defs()) {
        if (def.type == type) return &def;
    }
    return nullptr;
}

} // namespace panels
