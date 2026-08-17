#include "var_registry_panel.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include "../components.h"
#include "../tradewinds.h"
#include "../server_registry.h"
#include "../capabilities/ui_kit.h"
#include "registry.h"

using namespace tradewinds;

namespace {

// The solver bridge's socket, created centrally in main.cpp and shared with the
// ArcTask panel. Reserved per request like anyone else's, so two panels asking
// at once take turns rather than colliding on the REQ socket.
constexpr const char* kClientName = "ArcSolverClient";
constexpr const char* kServerId   = "arc_solver";

// Denser than the editor's usual CharisSIL 16: this is a 600-row data tree, not
// prose or a label, and Heatshield's own VarRegistry sat at 14/22 for the same
// reason. Row height follows the text so the stripes stay proportionate.
constexpr float kFontSize    = 14.0f;
constexpr float kRowHeight   = 22.0f;
constexpr float kIndentStep  = 12.0f;
constexpr uint32_t kZebraA   = 0x00000000;   // the panel's own ground
constexpr uint32_t kZebraB   = 0xFFFFFF08;   // barely lighter, every other row
constexpr uint32_t kNameColor  = 0xD8D8D8FF;
constexpr uint32_t kValueColor = 0x8FB8D8FF;
constexpr uint32_t kHighlight  = 0x6DF0FFFF;

// One line of the solver's scene graph, exactly as jane_eyre reports it: a
// slash-separated path, and either an object (which can hold others) or a value.
struct VarNode {
    std::string path;
    std::string value;       // empty for objects
    bool is_object = false;
    bool highlighted = false;
    bool expanded = true;    // what the declarer suggested; the panel may differ
};

struct VarRegistryView {
    std::string task_id;                    // the task these nodes describe
    std::vector<VarNode> nodes;
    std::vector<std::string> collapsed;     // paths the reader has folded shut

    std::string status;
    flecs::entity list;                     // rows live here
    flecs::entity status_text;
    bool dirty = true;
    bool inflight = false;
};

// Marks a row so a click knows which node it is on.
struct VarRow { std::string path; };

flecs::world* g_world = nullptr;

flecs::entity view_entity()
{
    flecs::entity found;
    if (!g_world) return found;
    g_world->query<VarRegistryView>().each([&](flecs::entity e, VarRegistryView&) {
        if (!found) found = e;
    });
    return found;
}

std::string field_str(std::map<std::string, msgpack::object>& res, const char* key)
{
    auto it = res.find(key);
    if (it == res.end()) return {};
    try { return it->second.as<std::string>(); } catch (const std::exception&) { return {}; }
}

bool server_running()
{
    if (!g_world) return false;
    flecs::entity server = servers::find(*g_world, kServerId);
    const ServerState* st = server.is_valid() ? server.try_get<ServerState>() : nullptr;
    return st && st->status != ServerStatus::Offline;
}

// The task the conversation is currently within. The panel follows the context
// rather than owning a task of its own: it shows the abstractions of whatever
// is being talked about.
std::string context_task()
{
    if (!g_world) return {};
    std::string task;
    g_world->query<ContextInfo>().each([&](flecs::entity, ContextInfo& info) {
        if (task.empty() && info.evaluator == "arc") task = info.detail;
    });
    return task;
}

int depth_of(const std::string& path)
{
    return (int)std::count(path.begin(), path.end(), '/');
}

std::string leaf_of(const std::string& path)
{
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool is_collapsed(const VarRegistryView& view, const std::string& path)
{
    return std::find(view.collapsed.begin(), view.collapsed.end(), path)
        != view.collapsed.end();
}

// A row is drawn when every ancestor above it is open. Checked against the
// path rather than a tree, because the declarer sends a flat list and the path
// already carries the hierarchy.
bool ancestors_open(const VarRegistryView& view, const std::string& path)
{
    size_t at = path.find('/');
    while (at != std::string::npos) {
        if (is_collapsed(view, path.substr(0, at))) return false;
        at = path.find('/', at + 1);
    }
    return true;
}

void scene_graph_response(std::map<std::string, msgpack::object>& res)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    VarRegistryView& view = ve.ensure<VarRegistryView>();
    view.inflight = false;
    view.dirty = true;

    if (field_str(res, "status") != "ok") {
        const std::string message = field_str(res, "message");
        view.status = message.empty() ? "scene graph request failed" : message;
        return;
    }

    view.nodes.clear();
    view.collapsed.clear();
    try {
        auto it = res.find("nodes");
        if (it == res.end()) { view.status = "no nodes in reply"; return; }

        for (const msgpack::object& row : it->second.as<std::vector<msgpack::object>>()) {
            auto entry = row.as<std::map<std::string, msgpack::object>>();

            VarNode node;
            auto path_it = entry.find("path");
            auto type_it = entry.find("type");
            auto value_it = entry.find("value");
            auto expanded_it = entry.find("expanded");
            auto high_it = entry.find("highlighted");
            if (path_it == entry.end()) continue;

            node.path = path_it->second.as<std::string>();
            node.is_object = type_it != entry.end()
                && type_it->second.as<std::string>() == "object";
            if (value_it != entry.end())
                node.value = value_it->second.as<std::string>();
            if (expanded_it != entry.end() && expanded_it->second.type == msgpack::type::BOOLEAN)
                node.expanded = expanded_it->second.via.boolean;
            if (high_it != entry.end() && high_it->second.type == msgpack::type::BOOLEAN)
                node.highlighted = high_it->second.via.boolean;

            // The declarer's suggestion is honoured once, as the starting fold:
            // a task's polys arrive shut because there are hundreds of them.
            if (node.is_object && !node.expanded) view.collapsed.push_back(node.path);
            view.nodes.push_back(std::move(node));
        }
        view.status.clear();
    } catch (const std::exception& exc) {
        view.status = std::string("unreadable scene graph: ") + exc.what();
    }
}

void toggle_row(flecs::entity row)
{
    const VarRow* info = row.try_get<VarRow>();
    flecs::entity ve = view_entity();
    if (!info || !ve.is_valid()) return;

    VarRegistryView& view = ve.ensure<VarRegistryView>();
    auto at = std::find(view.collapsed.begin(), view.collapsed.end(), info->path);
    if (at != view.collapsed.end()) view.collapsed.erase(at);
    else view.collapsed.push_back(info->path);
    view.dirty = true;

    row.world().ensure<UIInputDispatch>().consumed = true;
}

void rebuild(VarRegistryView& view)
{
    if (!view.list.is_valid() || !view.list.is_alive()) return;
    flecs::world world = view.list.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid()) return;

    std::vector<flecs::entity> old;
    view.list.children([&](flecs::entity c) { old.push_back(c); });
    for (flecs::entity c : old) c.destruct();

    if (view.status_text.is_valid() && view.status_text.is_alive())
        view.status_text.ensure<TextRenderable>().text = view.status;

    int drawn = 0;
    for (const VarNode& node : view.nodes) {
        if (!ancestors_open(view, node.path)) continue;

        // Zebra by drawn position, not by index in the list: the stripes have to
        // alternate down the rows you can actually see, or folding a subtree
        // leaves the banding broken.
        auto row = world.entity()
            .is_a(UIElement)
            .child_of(view.list)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<VarRow>({node.path})
            .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
            .set<UISize>({YGUndefined, kRowHeight})
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                   YGJustifyFlexStart, YGAlignCenter, 6.0f})
            .set<UIPadding>({0.0f, 8.0f, 0.0f,
                             8.0f + depth_of(node.path) * kIndentStep})
            .set<RectRenderable>({0.0f, kRowHeight, false,
                                  (drawn % 2) ? kZebraB : kZebraA})
            .set<ZIndex>({12});
        if (node.is_object) row.set<CallbackOnLeftClick>({toggle_row});
        drawn++;

        if (node.is_object) {
            // The arrow is the state: pointing down when the subtree is open,
            // right when it is folded away.
            world.entity()
                .is_a(UIElement)
                .child_of(row)
                .add<UIYoga>()
                .add<UINativeImageSize>()
                .set<ImageCreator>({is_collapsed(view, node.path) ? "arrow_right.png"
                                                                  : "arrow_down.png",
                                    1.0f, 1.0f, nvgRGBA(0x9A, 0x9A, 0x9A, 0xFF)})
                .set<ZIndex>({13});
        }

        world.entity()
            .is_a(UIElement)
            .child_of(row)
            .add<UIYoga>()
            .set<TextRenderable>({leaf_of(node.path), "JetBrainsMono", kFontSize,
                                  node.highlighted ? kHighlight : kNameColor})
            .set<ZIndex>({13});

        if (!node.value.empty()) {
            world.entity()
                .is_a(UIElement)
                .child_of(row)
                .add<UIYoga>()
                .set<TextRenderable>({node.value, "JetBrainsMono", kFontSize, kValueColor})
                .set<ZIndex>({13});
        }
    }
}

void create_content(flecs::entity leaf, flecs::entity UIElement)
{
    flecs::world world = leaf.world();
    auto canvas = leaf.target<EditorCanvas>();

    world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add<UIYoga>()
    .set<UIFillParent>({})
    .set<RectRenderable>({0.0f, 0.0f, false, 0x0C0D10FF})
    .set<ZIndex>({9});

    auto column = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFillParent>({8.0f, 8.0f, 8.0f, 8.0f})
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignStretch, 4.0f})
    .set<ZIndex>({10});

    auto status_text = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add<UIYoga>()
    .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
    .set<TextRenderable>({"", "JetBrainsMono", kFontSize, 0x9A9A9AFF})
    .set<ZIndex>({12});

    // The visible window onto the graph, which is always longer than the panel.
    // It takes the leftover height and clips; the rows themselves live in a
    // container that overflows it and slides.
    auto viewport = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFlexItem>({1.0f, 1.0f, YGAlignAuto})
    .set<ZIndex>({11});

    auto list = world.entity()
    .is_a(UIElement)
    .child_of(viewport)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    // Absolute, so it is free to be taller than the viewport; PanelScrollSystem
    // writes the top edge and clamps to the overflow.
    .set<UIAbsoluteEdges>({0.0f, 0.0f, 0.0f, YGUndefined})
    .set<PanelScroll>({0.0f})
    .add<ScissorContainer>(viewport)
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignStretch, 0.0f})
    .set<ZIndex>({11});

    // Against the viewport's right edge, over the rows rather than taking width
    // from them. An indicator: it says where you are and that there is more,
    // while the wheel is what moves it.
    auto scroll_track = world.entity()
    .is_a(UIElement)
    .child_of(viewport)
    .add<UIYoga>()
    .set<UIAbsoluteEdges>({YGUndefined, 2.0f, 2.0f, 2.0f})
    .set<UISize>({4.0f, YGUndefined})
    .set<RoundedRectRenderable>({4.0f, 0.0f, 2.0f, false, 0xFFFFFF14})
    .set<ZIndex>({14});

    world.entity()
    .is_a(UIElement)
    .child_of(scroll_track)
    .add<PanelScrollbarOf>(list)
    .set<Position, Local>({0.0f, 0.0f})
    .set<RoundedRectRenderable>({4.0f, 0.0f, 2.0f, false, 0xFFFFFF4D})
    .set<ZIndex>({15});

    VarRegistryView initial;
    initial.list = list;
    initial.status_text = status_text;
    initial.status = "waiting for an ARC context";
    column.set<VarRegistryView>(initial);

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());
    create_badge(badges, UIElement, "VarRegistry", 0x8f7fd0ff);
}

} // namespace

namespace panels {

VarRegistry::VarRegistry(flecs::world& world)
{
    world.module<VarRegistry>();
    g_world = &world;

    world.component<VarRegistryView>();
    world.component<VarRow>();

    // Follows the context: the graph shown is the one belonging to the task
    // being talked about, refetched when that moves.
    world.system<VarRegistryView>("VarRegistryFollowSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, VarRegistryView& view) {
        if (view.inflight) return;

        const std::string task = context_task();
        if (task.empty()) {
            if (view.status != "waiting for an ARC context" && view.nodes.empty()) {
                view.status = "waiting for an ARC context";
                view.dirty = true;
            }
            return;
        }
        if (task == view.task_id) return;

        if (!server_running()) {
            const std::string want = "solver offline -- start ARC Solver in Peach Core";
            if (view.status != want) { view.status = want; view.dirty = true; }
            return;
        }

        flecs::entity client = g_world->lookup(kClientName);
        if (!tradewinds::try_reserve(client)) return;   // shared; next frame

        view.task_id = task;
        view.nodes.clear();
        view.inflight = true;
        view.status = "reading " + task + "...";
        view.dirty = true;
        client.set<SendMapRequest>({{{"command", "scene_graph"}, {"task_id", task}}});
        client.set<AwaitResponse>({scene_graph_response});
    });

    world.system<VarRegistryView>("VarRegistryRebuildSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, VarRegistryView& view) {
        if (!view.dirty) return;
        view.dirty = false;
        rebuild(view);
    });

    register_panel({EditorType::VarRegistry, "var_registry", "VarRegistry", create_content});
}

} // namespace panels
