#include "dataset_panel.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../components.h"
#include "../capabilities/ui_kit.h"
#include "arc_task_panel.h"
#include "registry.h"

namespace {

// The dataset the ArcTask panel reads, and the solver's record of how it did.
// Machine-specific like the other ARC paths, and a config knob for the same
// reason.
constexpr const char* kArcDataDir =
    "/home/wesxdz/arc/editor/jane_eyre/ARC-AGI/data/training";
constexpr const char* kResultsDir =
    "/home/wesxdz/arc/editor/heatshield/save/solver_results";

constexpr float kRowHeight = 22.0f;
constexpr float kFontSize  = 14.0f;

constexpr uint32_t kZebra    = 0xFFFFFF08;
constexpr uint32_t kIdColor  = 0xC8C8C8FF;
constexpr uint32_t kSolved   = 0x2ECC40FF;
constexpr uint32_t kTimeout  = 0xFF851BFF;
constexpr uint32_t kFailed   = 0xFF4136FF;
constexpr uint32_t kUnknown  = 0x4A4A4AFF;

struct DatasetEntry {
    std::string task_id;
    std::string status;      // "solved" | "timeout" | "failed" | "" (never run)
};

struct DatasetView {
    std::vector<DatasetEntry> entries;
    std::string open_task;               // the one the ArcTask panel holds
    int solved = 0;

    flecs::entity list;                  // carries the VirtualList
    flecs::entity status_text;
    flecs::entity progress_fill;
    flecs::entity progress_track;
    bool header_dirty = true;            // the counts above the list
};

// Marks a row with the task it stands for.
struct DatasetRow { std::string task_id; };

flecs::world* g_world = nullptr;

uint32_t status_color(const std::string& status)
{
    if (status == "solved")  return kSolved;
    if (status == "timeout") return kTimeout;
    if (status == "failed")  return kFailed;
    return kUnknown;
}

// The dataset, with whatever the solver last recorded about each task. Read once
// per panel: the set does not change while the editor runs, and a result that
// lands later is a reopen away.
void load_dataset(DatasetView& view)
{
    namespace fs = std::filesystem;
    view.entries.clear();
    view.solved = 0;

    std::vector<std::string> ids;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(kArcDataDir, ec))
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            ids.push_back(entry.path().stem().string());
    std::sort(ids.begin(), ids.end());

    for (const std::string& id : ids) {
        DatasetEntry entry{id, ""};
        std::ifstream result(std::string(kResultsDir) + "/" + id + ".json");
        if (result) {
            try {
                nlohmann::json parsed;
                result >> parsed;
                entry.status = parsed.value("status", std::string{});
            } catch (const std::exception&) { /* no verdict, then */ }
        }
        if (entry.status == "solved") view.solved++;
        view.entries.push_back(std::move(entry));
    }
    view.header_dirty = true;
}

void open_row(flecs::entity row)
{
    const DatasetRow* info = row.try_get<DatasetRow>();
    if (!info) return;
    panels::arc_task_open(info->task_id);
    row.world().ensure<UIInputDispatch>().consumed = true;
}

// One row. Which rows exist, when they are built and how far the container is
// nudged for sub-row scrolling are VirtualListSystem's business -- the same
// machinery the Entities panel scrolls a million rows with, rather than a
// second windowing scheme living in this panel.
void build_row(flecs::entity list, size_t index, DatasetView& view)
{
    flecs::world world = list.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid() || index >= view.entries.size()) return;

    const DatasetEntry& entry = view.entries[index];
    const bool open = entry.task_id == view.open_task;

    auto row = world.entity()
        .is_a(UIElement)
        .child_of(list)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<DatasetRow>({entry.task_id})
        .set<CallbackOnLeftClick>({open_row})
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
        .set<UISize>({YGUndefined, kRowHeight})
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyFlexStart, YGAlignCenter, 6.0f})
        .set<UIPadding>({0.0f, 8.0f, 0.0f, 8.0f})
        .set<RectRenderable>({0.0f, kRowHeight, false,
                              (index % 2) ? kZebra : 0x00000000u})
        .set<ZIndex>({12});

    // The open task is outlined rather than tinted: it is not another status,
    // it is where you are.
    if (open) {
        world.entity()
            .is_a(UIElement)
            .child_of(row)
            .add<UIYoga>()
            .set<UIAbsoluteEdges>({0.0f, 0.0f, 0.0f, 0.0f})
            .set<RoundedRectRenderable>({0.0f, kRowHeight, 2.0f, true, 0xFFFFFFFF})
            .set<ZIndex>({15});
    }

    // A verdict is a dot: colour carries it, and a word per row would out-shout
    // the ids you are actually reading.
    world.entity()
        .is_a(UIElement)
        .child_of(row)
        .add<UIYoga>()
        .set<UISize>({6.0f, 6.0f})
        .set<RoundedRectRenderable>({6.0f, 6.0f, 3.0f, false,
                                     status_color(entry.status)})
        .set<ZIndex>({13});

    world.entity()
        .is_a(UIElement)
        .child_of(row)
        .add<UIYoga>()
        .set<TextRenderable>({entry.task_id, "JetBrainsMono", kFontSize,
                              open ? 0xFFFFFFFF : kIdColor})
        .set<ZIndex>({13});
}

// The counts above the list. Rebuilt only when they change, which is when the
// dataset is read or a task is opened.
void refresh_header(DatasetView& view)
{
    if (!view.header_dirty) return;
    view.header_dirty = false;

    const int total = (int)view.entries.size();
    if (view.status_text.is_valid() && view.status_text.is_alive())
        view.status_text.ensure<TextRenderable>().text =
            std::to_string(view.solved) + " solved of " + std::to_string(total);

    if (!view.progress_fill.is_valid() || !view.progress_fill.is_alive()) return;
    const UIElementSize* track = view.progress_track.is_valid()
        ? view.progress_track.try_get<UIElementSize>() : nullptr;
    const float width = track ? track->width : 0.0f;
    const float filled = total > 0 ? width * ((float)view.solved / (float)total) : 0.0f;
    view.progress_fill.ensure<UISize>().w = filled;
    view.progress_fill.ensure<RoundedRectRenderable>().width = filled;
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

    auto progress_track = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
    .set<UISize>({YGUndefined, 4.0f})
    .set<RoundedRectRenderable>({0.0f, 4.0f, 2.0f, false, 0xFFFFFF14})
    .set<ZIndex>({12});

    auto progress_fill = world.entity()
    .is_a(UIElement)
    .child_of(progress_track)
    .add<UIYoga>()
    .set<UIAbsoluteEdges>({0.0f, 0.0f, YGUndefined, 0.0f})
    .set<UISize>({0.0f, 4.0f})
    .set<RoundedRectRenderable>({0.0f, 4.0f, 2.0f, false, kSolved})
    .set<ZIndex>({13});

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

    DatasetView initial;
    initial.list = list;
    initial.status_text = status_text;
    initial.progress_track = progress_track;
    initial.progress_fill = progress_fill;
    load_dataset(initial);
    initial.open_task = panels::arc_task_current();
    column.set<DatasetView>(initial);

    // build_row reaches back through the panel node for the data, so the list
    // holds no copy of the dataset and a reread cannot leave the two disagreeing.
    VirtualList virtual_list;
    virtual_list.viewport = viewport;
    virtual_list.row_pitch = kRowHeight;
    virtual_list.pad = 0.0f;
    virtual_list.item_count = column.get<DatasetView>().entries.size();
    virtual_list.build_row = [column](flecs::entity rows, size_t index) {
        if (!column.is_valid() || !column.is_alive()) return;
        build_row(rows, index, column.ensure<DatasetView>());
    };
    list.set<VirtualList>(virtual_list);

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());
    create_badge(badges, UIElement, "Dataset", 0x619393ff);
}

} // namespace

namespace panels {

Dataset::Dataset(flecs::world& world)
{
    world.module<Dataset>();
    g_world = &world;

    world.component<DatasetView>();
    world.component<DatasetRow>();

    // Follows whichever task the ArcTask panel has open, so arrowing through
    // tasks over there moves the outline over here.
    world.system<DatasetView>("DatasetFollowSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, DatasetView& view) {
        const std::string open = panels::arc_task_current();
        if (open == view.open_task) return;
        view.open_task = open;

        // The window has not moved, but two rows in it now look different, so
        // the list is told to rebuild what it already has.
        if (view.list.is_valid() && view.list.is_alive())
            view.list.ensure<VirtualList>().dirty = true;
    });

    world.system<DatasetView>("DatasetHeaderSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, DatasetView& view) { refresh_header(view); });

    register_panel({EditorType::Dataset, "dataset", "Dataset", create_content});
}

} // namespace panels
