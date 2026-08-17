#include "solve_panel.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <string>
#include <vector>

#include <fstream>
#include <map>

#include <msgpack.hpp>
#include <nlohmann/json.hpp>

#include "../components.h"
#include "../tradewinds.h"
#include "../server_registry.h"
#include "../capabilities/arc_palette.h"
#include "../capabilities/ui_kit.h"
#include "registry.h"

namespace {

// The solver bridge. The request goes out on the REQ socket every other ARC
// panel shares; the progress comes back on a subscription of its own, because a
// stream is not a reply and must not hold the request socket hostage while a
// search runs for thirty seconds.
constexpr const char* kClientName = "ArcSolverClient";
constexpr const char* kStreamName = "ArcSolverStream";
constexpr const char* kServerId   = "arc_solver";

constexpr float kLineHeight = 16.0f;
constexpr float kFontSize   = 13.0f;

constexpr uint32_t kProgress = 0x9A9A9AFF;
constexpr uint32_t kAttempt  = 0x6A7A8AFF;
constexpr uint32_t kResult   = 0x2ECC40FF;
constexpr uint32_t kError    = 0xFF4136FF;
constexpr uint32_t kRunning  = 0xFAB257FF;
constexpr uint32_t kIdle     = 0x3A4048FF;

struct SolveLine {
    std::string text;
    uint32_t color = kProgress;
};

// Which sprite stands for an operator. An asset, like the categories: the
// solver names its operators and the editor decides how they look, so a new
// operator gets an icon without a rebuild.
const std::map<std::string, std::string>& operator_icons()
{
    static const std::map<std::string, std::string> icons = [] {
        std::map<std::string, std::string> out;
        std::ifstream file("../assets/config/operator_icons.json");
        if (!file) return out;
        try {
            nlohmann::json parsed = nlohmann::json::parse(file, nullptr, true, true);
            for (auto it = parsed.begin(); it != parsed.end(); ++it)
                out[it.key()] = it.value().get<std::string>();
        } catch (const std::exception&) { /* names in text, then */ }
        return out;
    }();
    return icons;
}

// A filter that names an ARC colour is that colour. Reading "Rose" as text and
// then remembering which red it is wastes the one glance the readout is for.
int arc_color_index(const std::string& name)
{
    std::string lower = name;
    for (char& c : lower) c = (char)std::tolower((unsigned char)c);
    for (int i = 0; i < 10; i++)
        if (lower == arc_color_names[i]) return i;
    return -1;
}

// Where the search went. Counts by operator and by filter say what it spent
// itself on; the score histogram says how close any of it came. Together they
// answer the question a bare attempt count cannot: is this search circling the
// answer, or nowhere near it?
struct AttemptShare { std::string name; int count = 0; };

struct SolveDistribution {
    int attempts = 0;
    float best = -1.0f;                  // < 0 until something scores
    std::vector<AttemptShare> by_operator;
    std::vector<AttemptShare> by_filter;
    std::vector<AttemptShare> by_type;
    std::vector<int> scores;             // log-spaced buckets, closest first
};

struct SolveView {
    std::string task_id;                 // what is being solved, or was last
    std::vector<SolveLine> lines;
    int attempts = 0;
    bool running = false;

    SolveDistribution distribution;
    flecs::entity shape;                 // where the distribution is drawn
    bool shape_dirty = false;

    flecs::entity list;                  // carries the VirtualList
    flecs::entity status_text;
    flecs::entity run_button;
    bool header_dirty = true;
};

// Marks the run/stop control.
struct SolveButton {};

flecs::world* g_world = nullptr;

flecs::entity view_entity()
{
    flecs::entity found;
    if (!g_world) return found;
    g_world->query<SolveView>().each([&](flecs::entity e, SolveView&) {
        if (!found) found = e;
    });
    return found;
}

std::string context_task()
{
    if (!g_world) return {};
    std::string task;
    g_world->query<ContextInfo>().each([&](flecs::entity, ContextInfo& info) {
        if (task.empty() && info.evaluator == "arc") task = info.detail;
    });
    return task;
}

bool server_running()
{
    if (!g_world) return false;
    flecs::entity server = servers::find(*g_world, kServerId);
    const ServerState* st = server.is_valid() ? server.try_get<ServerState>() : nullptr;
    return st && st->status != ServerStatus::Offline;
}

// Kept bounded: a search prints thousands of lines and the interesting ones are
// the last of them. Older lines are dropped rather than held, because a
// scrollback nobody reads still costs memory for as long as the panel is open.
constexpr size_t kMaxLines = 400;

void append(SolveView& view, const std::string& text, uint32_t color)
{
    if (text.empty()) return;
    view.lines.push_back({text, color});
    if (view.lines.size() > kMaxLines)
        view.lines.erase(view.lines.begin(),
                         view.lines.begin() + (view.lines.size() - kMaxLines));

    if (view.list.is_valid() && view.list.is_alive()) {
        VirtualList& list = view.list.ensure<VirtualList>();
        list.item_count = view.lines.size();
        list.dirty = true;

        // Follows the tail: a running search is watched, not browsed, and a log
        // that stays where you left it hides the thing you are waiting for.
        const UIElementSize* viewport = list.viewport.is_valid()
            ? list.viewport.try_get<UIElementSize>() : nullptr;
        const float view_height = viewport ? viewport->height : 0.0f;
        list.scroll = std::max(0.0f, view.lines.size() * kLineHeight - view_height);
    }
}

// Progress arrives as [topic, payload] while the search runs. The topics are
// jane_eyre's own: what it printed, that it tried something, and how it ended.
void published(const std::string& topic, const std::string& payload)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    SolveView& view = ve.ensure<SolveView>();

    if (topic == "progress") {
        append(view, payload, kProgress);
    } else if (topic == "attempts") {
        try {
            msgpack::object_handle handle = msgpack::unpack(payload.data(), payload.size());
            auto summary = handle.get().as<std::map<std::string, msgpack::object>>();

            SolveDistribution& dist = view.distribution;
            if (summary.count("attempts")) dist.attempts = summary["attempts"].as<int>();
            view.attempts = dist.attempts;
            dist.best = -1.0f;
            if (summary.count("best") && summary["best"].type != msgpack::type::NIL)
                dist.best = summary["best"].as<float>();
            if (summary.count("scores"))
                dist.scores = summary["scores"].as<std::vector<int>>();

            auto shares = [&](const char* key, std::vector<AttemptShare>& into) {
                into.clear();
                if (!summary.count(key)) return;
                for (const msgpack::object& row :
                         summary[key].as<std::vector<msgpack::object>>()) {
                    auto entry = row.as<std::map<std::string, msgpack::object>>();
                    into.push_back({entry.count("name") ? entry["name"].as<std::string>() : "",
                                    entry.count("count") ? entry["count"].as<int>() : 0});
                }
            };
            shares("by_operator", dist.by_operator);
            shares("by_filter", dist.by_filter);
            shares("by_type", dist.by_type);

            view.shape_dirty = true;
            view.header_dirty = true;
        } catch (const std::exception&) { /* a malformed summary is not fatal */ }
    } else if (topic == "result") {
        append(view, payload, kResult);
        view.running = false;
        view.header_dirty = true;
    } else if (topic == "error") {
        append(view, payload, kError);
        view.running = false;
        view.header_dirty = true;
    }
    // scene_graph is msgpack for the VarRegistry panel, not text: ignored here
    // rather than printed as bytes.
}

void solve_ack(std::map<std::string, msgpack::object>& res)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    SolveView& view = ve.ensure<SolveView>();

    std::string status, message;
    try {
        auto status_it = res.find("status");
        if (status_it != res.end()) status = status_it->second.as<std::string>();
        auto message_it = res.find("message");
        if (message_it != res.end()) message = message_it->second.as<std::string>();
    } catch (const std::exception&) {}

    if (status != "ok") {
        append(view, message.empty() ? "the solver refused the task" : message, kError);
        view.running = false;
        view.header_dirty = true;
    }
    // On ok there is nothing to say: the search has started and will speak for
    // itself on the subscription.
}

void run_clicked(flecs::entity button)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid() || !g_world) return;
    SolveView& view = ve.ensure<SolveView>();
    button.world().ensure<UIInputDispatch>().consumed = true;

    if (view.running) {
        // Nothing here can stop a search: the server runs it inline and only
        // answers when it is done. Saying so beats a button that lies.
        append(view, "already running -- the solver finishes or times out", kAttempt);
        return;
    }

    const std::string task = context_task();
    if (task.empty()) {
        append(view, "no ARC task in context", kError);
        return;
    }
    if (!server_running()) {
        append(view, "solver offline -- start ARC Solver in Peach Core", kError);
        return;
    }

    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) {
        append(view, "solver busy with another request", kAttempt);
        return;
    }

    view.task_id = task;
    view.lines.clear();
    view.attempts = 0;
    view.running = true;
    view.header_dirty = true;
    append(view, "solving " + task + "...", kAttempt);

    client.set<tradewinds::SendMapRequest>({{{"command", "solve"}, {"task_id", task}}});
    client.set<tradewinds::AwaitResponse>({solve_ack});
}

void build_line(flecs::entity list, size_t index, SolveView& view)
{
    flecs::world world = list.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid() || index >= view.lines.size()) return;

    world.entity()
        .is_a(UIElement)
        .child_of(list)
        .add<UIYoga>()
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
        .set<UISize>({YGUndefined, kLineHeight})
        .set<TextRenderable>({view.lines[index].text, "JetBrainsMono", kFontSize,
                              view.lines[index].color})
        .set<ZIndex>({12});
}

// The distribution, as bars. Names and counts alone make you do the division;
// a bar length against the leader shows at a glance that one filter is eating
// half the search.
void rebuild_shape(SolveView& view)
{
    if (!view.shape.is_valid() || !view.shape.is_alive()) return;
    flecs::world world = view.shape.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid()) return;

    std::vector<flecs::entity> old;
    view.shape.children([&](flecs::entity c) { old.push_back(c); });
    for (flecs::entity c : old) c.destruct();

    const SolveDistribution& dist = view.distribution;
    if (dist.attempts <= 0) return;

    auto text = [&](flecs::entity parent, const std::string& body, uint32_t color) {
        world.entity().is_a(UIElement).child_of(parent).add<UIYoga>()
            .set<TextRenderable>({body, "JetBrainsMono", kFontSize, color})
            .set<ZIndex>({13});
    };

    // How close anything came, log-spaced and closest first. A search that
    // never leaves the right-hand buckets is not converging on anything.
    if (!dist.scores.empty()) {
        auto row = world.entity().is_a(UIElement).child_of(view.shape)
            .add(flecs::OrderedChildren).add<UIYoga>()
            .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                   YGJustifyFlexStart, YGAlignFlexEnd, 2.0f})
            .set<ZIndex>({13});
        text(row, "how close", 0x7A7A7AFF);

        int tallest = 1;
        for (int count : dist.scores) tallest = std::max(tallest, count);
        static const char* edges[] = {"0", "1", "2", "4", "8", "16", "64", "+"};
        for (size_t i = 0; i < dist.scores.size() && i < 8; i++) {
            const float height = 4.0f + 24.0f * ((float)dist.scores[i] / (float)tallest);
            auto column = world.entity().is_a(UIElement).child_of(row)
                .add(flecs::OrderedChildren).add<UIYoga>()
                .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                                       YGJustifyFlexEnd, YGAlignCenter, 1.0f})
                .set<ZIndex>({13});
            world.entity().is_a(UIElement).child_of(column).add<UIYoga>()
                .set<UISize>({14.0f, height})
                .set<RoundedRectRenderable>({14.0f, height, 1.0f, false,
                                             i < 2 ? kResult : kAttempt})
                .set<ZIndex>({14});
            text(column, edges[i], 0x6A6A6AFF);
        }
    }

    // Where the effort went. Operators first: there are few of them, so a
    // lopsided split is a fact about the search rather than about the task.
    auto section = [&](const char* title, const std::vector<AttemptShare>& shares,
                       bool as_sprites) {
        if (shares.empty()) return;
        text(view.shape, title, 0x7A7A7AFF);
        const int leader = std::max(1, shares.front().count);
        for (const AttemptShare& share : shares) {
            auto row = world.entity().is_a(UIElement).child_of(view.shape)
                .add(flecs::OrderedChildren).add<UIYoga>()
                .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
                .set<UISize>({YGUndefined, kLineHeight + 8.0f})
                .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                       YGJustifyFlexStart, YGAlignCenter, 6.0f})
                .set<ZIndex>({13});

            // The operator wears the same sprite it wears in the abduction
            // panel, so a tool is one picture everywhere in the editor.
            bool named = false;
            if (as_sprites) {
                auto icon = operator_icons().find(share.name);
                if (icon != operator_icons().end()) {
                    world.entity().is_a(UIElement).child_of(row).add<UIYoga>()
                        .add<UINativeImageSize>()
                        .set<ImageCreator>({"items/" + icon->second + ".png", 1.0f, 1.0f,
                                            nvgRGBA(0xFF, 0xFF, 0xFF, 0xFF)})
                        .set<ZIndex>({14});
                    named = true;
                }
            } else {
                const int color = arc_color_index(share.name);
                if (color >= 0) {
                    create_badge(row, UIElement, share.name.c_str(), arc_palette[color]);
                    named = true;
                }
            }

            const float width = 6.0f + 70.0f * ((float)share.count / (float)leader);
            world.entity().is_a(UIElement).child_of(row).add<UIYoga>()
                .set<UISize>({width, 6.0f})
                .set<RoundedRectRenderable>({width, 6.0f, 2.0f, false, kAttempt})
                .set<ZIndex>({14});

            // The name still travels when nothing depicts it -- an operator with
            // no icon or a filter that is not a colour would otherwise be a
            // nameless bar.
            text(row, std::to_string(share.count) + (named ? "" : "  " + share.name),
                 0xB0B0B0FF);
        }
    };
    section("operators", dist.by_operator, true);
    section("filters", dist.by_filter, false);
}

void refresh_header(SolveView& view)
{
    if (!view.header_dirty) return;
    view.header_dirty = false;

    if (view.status_text.is_valid() && view.status_text.is_alive()) {
        std::string status = view.task_id.empty() ? "nothing solved yet" : view.task_id;
        if (view.running) status += "   running";
        if (view.attempts > 0) status += "   " + std::to_string(view.attempts) + " attempts";
        if (view.distribution.best >= 0.0f)
            status += "   best " + std::to_string((int)view.distribution.best);
        view.status_text.ensure<TextRenderable>().text = status;
    }
    if (view.run_button.is_valid() && view.run_button.is_alive()) {
        if (auto* rect = view.run_button.try_get_mut<RoundedRectRenderable>())
            rect->color = view.running ? kRunning : kIdle;
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

    auto header = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
    .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignCenter, 8.0f})
    .set<ZIndex>({12});

    auto run_button = world.entity()
    .is_a(UIElement)
    .child_of(header)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .add<SolveButton>()
    .set<CallbackOnLeftClick>({run_clicked})
    .set<UIPadding>({3.0f, 10.0f, 3.0f, 10.0f})
    .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                           YGJustifyCenter, YGAlignCenter, 0.0f})
    .set<RoundedRectRenderable>({0.0f, 0.0f, 3.0f, false, kIdle})
    .set<ZIndex>({13});

    world.entity()
    .is_a(UIElement)
    .child_of(run_button)
    .add<UIYoga>()
    .set<TextRenderable>({"solve", "JetBrainsMono", kFontSize, 0xF0F0F0FF})
    .set<ZIndex>({14});

    auto status_text = world.entity()
    .is_a(UIElement)
    .child_of(header)
    .add<UIYoga>()
    .set<TextRenderable>({"nothing solved yet", "JetBrainsMono", kFontSize, 0x9A9A9AFF})
    .set<ZIndex>({13});

    auto shape = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignStretch, 2.0f})
    .set<ZIndex>({12});

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
    .set<UIAbsoluteEdges>({0.0f, 0.0f, 0.0f, YGUndefined})
    .add<ScissorContainer>(viewport)
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignStretch, 0.0f})
    .set<ZIndex>({11});

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
    .add<VirtualListScrollbarOf>(list)
    .set<Position, Local>({0.0f, 0.0f})
    .set<RoundedRectRenderable>({4.0f, 0.0f, 2.0f, false, 0xFFFFFF4D})
    .set<ZIndex>({15});

    SolveView initial;
    initial.list = list;
    initial.status_text = status_text;
    initial.run_button = run_button;
    initial.shape = shape;
    column.set<SolveView>(initial);

    VirtualList lines;
    lines.viewport = viewport;
    lines.row_pitch = kLineHeight;
    lines.pad = 0.0f;
    lines.item_count = 0;
    lines.build_row = [column](flecs::entity host, size_t index) {
        if (!column.is_valid() || !column.is_alive()) return;
        build_line(host, index, column.ensure<SolveView>());
    };
    list.set<VirtualList>(lines);

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());
    create_badge(badges, UIElement, "Solve", 0xFAB257FF);
}

} // namespace

namespace panels {

Solve::Solve(flecs::world& world)
{
    world.module<Solve>();
    g_world = &world;

    world.component<SolveView>();
    world.component<SolveButton>();

    // The subscription is bound once, to the stream client main.cpp created.
    // Attached here rather than there because the handler is this panel's: the
    // socket is shared infrastructure, what to do with a message is not.
    world.system<SolveView>("SolveSubscribeSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, SolveView&) {
        flecs::entity stream = g_world->lookup(kStreamName);
        if (!stream.is_valid() || stream.has<tradewinds::AwaitPublished>()) return;
        stream.set<tradewinds::AwaitPublished>({published});
    });

    world.system<SolveView>("SolveHeaderSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, SolveView& view) {
        refresh_header(view);
        if (!view.shape_dirty) return;
        view.shape_dirty = false;
        rebuild_shape(view);
    });

    register_panel({EditorType::Solve, "solve", "Solve", create_content});
}

} // namespace panels
