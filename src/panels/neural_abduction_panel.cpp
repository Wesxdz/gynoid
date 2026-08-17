#include "neural_abduction_panel.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../components.h"
#include "../tradewinds.h"
#include "../server_registry.h"
#include "../capabilities/arc_palette.h"
#include "../capabilities/ui_kit.h"
#include "registry.h"

namespace {

// The model's output vector, in order: line N of this file names what the Nth
// probability is about. Read from assets rather than hardcoded, because the
// network's output layer and this list are the same 47 things -- retrain with a
// different vocabulary and the file moves with it.
constexpr const char* kRepresentationIndex = "../assets/config/representation_vector.txt";

// One probability per representation, predicted per task ahead of time by
// neural_heuristic/predict_priors.py. Machine-specific, like the dataset path
// in the ArcTask panel, and a config knob for the same reason.
constexpr const char* kPriorsDir =
    "/home/wesxdz/arc/editor/heatshield/save/predicted_priors";

// Where a person's own advice for a task is kept -- which representations they
// vouched for, as against what the network guessed. The same files
// extract_advice.py learns from, so a toggle here is training signal, not a
// preference: written only on an actual click, never as a side effect of
// looking at a task.
constexpr const char* kAdviceDir = "/home/wesxdz/arc/editor/heatshield/save/advice";

constexpr float kRowHeight  = 28.0f;
constexpr float kBarWidth   = 90.0f;
constexpr float kBarHeight  = 6.0f;

// Below this, the model is saying no. Such rows stay -- absence is a prediction
// too, and hiding them would misreport a confident no as no opinion -- but they
// recede.
constexpr float kAbduced = 0.5f;

constexpr uint32_t kZebra      = 0xFFFFFF08;
constexpr uint32_t kNameOn     = 0xD8E8F0FF;
constexpr uint32_t kNameOff    = 0x6A6A6AFF;
constexpr uint32_t kBarTrack   = 0xFFFFFF14;
constexpr uint32_t kBarOn      = 0x6DF0FFFF;   // the cyan Heatshield used
constexpr uint32_t kBarOff     = 0x40506AFF;

struct Prior {
    std::string name;
    float probability = 0.0f;
    bool selected = false;      // vouched for by a person, not the network
};

// What a processor found in this task, in the one shape every processor
// reports: label/value facts, a small matrix worth drawing, and cells in the
// source grid. An item that computes nothing simply never has one.
//
// This is the alternative to a panel per item: 47 tools cannot each own a
// viewport, but each can say a few things about the task it is hovering over.
struct ProcessorFinding {
    int pair_idx = 0;
    bool is_test = false;
    bool is_output = false;
    std::vector<std::pair<std::string, std::string>> facts;
    int grid_w = 0, grid_h = 0;
    std::vector<uint8_t> grid;          // row-major, palette indices
    int cell_count = 0;
};

struct ProcessorResult {
    std::string status;                 // "" pending, "ok", or the failure
    std::vector<ProcessorFinding> findings;
};

// Marks a row so a click knows which representation it is on.
struct AbductionRow { std::string name; };

// Two ways of looking at the same 47 things, because reading and choosing are
// different jobs. The ranked list answers "what does the network think"; the
// category boxes answer "what shall I vouch for", where what matters is finding
// an item among its kin rather than its rank.
enum class ViewMode { Ranked, Categories };

struct NeuralAbductionView {
    std::string task_id;
    std::vector<Prior> priors;
    std::string status;
    ViewMode mode = ViewMode::Ranked;

    flecs::entity list;
    flecs::entity status_text;
    flecs::entity mode_row;
    bool dirty = true;
};

// Marks a view-mode chip.
struct AbductionModeChip { ViewMode mode; };

// The hover card and what it is currently about, so it is only rebuilt when the
// pointer moves to a different item rather than every frame.
struct AbductionTooltip {
    std::string showing;
    // What was known about it when the card was drawn, so a finding that
    // arrives after the hover began still gets shown.
    std::string showing_state;
};

flecs::world* g_world = nullptr;

// The solver bridge, shared with the other ARC panels: reserved per request,
// never a socket of this panel's own.
constexpr const char* kClientName = "ArcSolverClient";
constexpr const char* kServerId   = "arc_solver";

// What each processor found, keyed "task/item". Cached because a hover asks and
// hovering is cheap to repeat; an entry exists the moment a request goes out, so
// the same question is never in flight twice.
std::map<std::string, ProcessorResult> g_findings;
std::string g_finding_inflight;   // the key currently at the server, if any

// The task the conversation is within: this panel says what the network thinks
// about whatever is being talked about, so it follows the context rather than
// holding a task of its own.
std::string context_task()
{
    if (!g_world) return {};
    std::string task;
    g_world->query<ContextInfo>().each([&](flecs::entity, ContextInfo& info) {
        if (task.empty() && info.evaluator == "arc") task = info.detail;
    });
    return task;
}

const std::vector<std::string>& representation_names()
{
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out;
        std::ifstream file(kRepresentationIndex);
        std::string line;
        while (std::getline(file, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
                line.pop_back();
            if (!line.empty()) out.push_back(line);
        }
        return out;
    }();
    return names;
}

// Defined below, beside the rest of the advice handling.
void load_advice(NeuralAbductionView& view);

// Loads one task's predicted priors, strongest first. The two files are
// index-aligned by construction -- the Nth probability is the Nth
// representation -- so a length mismatch means they have drifted apart and is
// worth saying rather than quietly zipping the shorter one.
void load_priors(NeuralAbductionView& view, const std::string& task_id)
{
    view.priors.clear();
    view.task_id = task_id;
    view.dirty = true;

    const std::vector<std::string>& names = representation_names();
    if (names.empty()) {
        view.status = "no representation_vector.txt in assets/config";
        return;
    }

    std::ifstream file(std::string(kPriorsDir) + "/" + task_id + "_priors.txt");
    if (!file) {
        view.status = "no prediction for " + task_id;
        return;
    }

    std::vector<float> values;
    std::string line;
    while (std::getline(file, line)) {
        try { values.push_back(std::stof(line)); }
        catch (const std::exception&) { /* blank or malformed line */ }
    }

    if (values.size() != names.size()) {
        view.status = "priors (" + std::to_string(values.size()) + ") and names ("
                    + std::to_string(names.size()) + ") disagree";
        return;
    }

    for (size_t i = 0; i < values.size(); i++)
        view.priors.push_back({names[i], values[i]});
    std::sort(view.priors.begin(), view.priors.end(),
              [](const Prior& a, const Prior& b) { return a.probability > b.probability; });

    load_advice(view);

    int abduced = 0, vouched = 0;
    for (const Prior& prior : view.priors) {
        if (prior.probability >= kAbduced) abduced++;
        if (prior.selected) vouched++;
    }
    view.status = task_id + "  " + std::to_string(abduced) + " abduced, "
                + std::to_string(vouched) + " vouched";
}

// What an item IS, for the tooltip: the same table Heatshield carried, lifted
// into an asset. Inputs and outputs are typed the way the tools themselves are
// written ("state: matrix"), which is the part worth having on hover -- a name
// says which tool, the signature says what it would do with the task.
struct ItemDescription {
    std::string title;
    std::string description;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

const std::map<std::string, ItemDescription>& item_descriptions()
{
    static const std::map<std::string, ItemDescription> table = [] {
        std::map<std::string, ItemDescription> out;
        std::ifstream file("../assets/config/item_descriptions.json");
        if (!file) return out;
        try {
            nlohmann::json parsed = nlohmann::json::parse(file, nullptr, true, true);
            for (auto it = parsed.begin(); it != parsed.end(); ++it) {
                ItemDescription described;
                described.title = it.value().value("title", it.key());
                described.description = it.value().value("description", std::string{});
                if (it.value().contains("inputs"))
                    for (const auto& in : it.value()["inputs"])
                        described.inputs.push_back(in.get<std::string>());
                if (it.value().contains("outputs"))
                    for (const auto& out_type : it.value()["outputs"])
                        described.outputs.push_back(out_type.get<std::string>());
                out[it.key()] = std::move(described);
            }
        } catch (const std::exception&) { /* names alone, then */ }
        return out;
    }();
    return table;
}

struct ItemCategory {
    std::string name;
    std::vector<std::string> items;
};

// How the representations group when choosing among them. An asset rather than
// a table in here: the grouping is an opinion about the domain, and it should be
// editable without a rebuild.
const std::vector<ItemCategory>& item_categories()
{
    static const std::vector<ItemCategory> categories = [] {
        std::vector<ItemCategory> out;
        std::ifstream file("../assets/config/item_categories.json");
        if (!file) return out;
        try {
            nlohmann::json parsed = nlohmann::json::parse(file, nullptr, true, true);
            if (!parsed.is_array()) return out;
            for (const auto& entry : parsed) {
                ItemCategory category;
                category.name = entry.value("name", std::string{});
                if (entry.contains("items"))
                    for (const auto& item : entry["items"])
                        category.items.push_back(item.get<std::string>());
                if (!category.name.empty()) out.push_back(std::move(category));
            }
        } catch (const std::exception&) { /* no grouping; UNFILED catches all */ }
        return out;
    }();
    return categories;
}

std::string advice_path(const std::string& task_id)
{
    return std::string(kAdviceDir) + "/" + task_id + "/itemset.json";
}

// Reads which representations this task has already been advised with. The file
// is the authority on its own contents: sessions saved lists of different
// lengths and orders, so the panel matches by name and never assumes its own
// 47 are what is in there.
void load_advice(NeuralAbductionView& view)
{
    std::ifstream file(advice_path(view.task_id));
    if (!file) return;

    try {
        nlohmann::json items;
        file >> items;
        if (!items.is_array()) return;
        for (const auto& item : items) {
            if (!item.contains("name")) continue;
            const std::string name = item["name"].get<std::string>();
            const bool selected = item.value("selected", false);
            for (Prior& prior : view.priors)
                if (prior.name == name) prior.selected = selected;
        }
    } catch (const std::exception&) { /* leave everything unvouched */ }
}

// Writes the advice back, preserving whatever the file already had. Entries the
// panel does not show are carried through untouched rather than dropped: an
// older session's list is a record of what a person considered, and rewriting
// it to this build's 47 names would quietly destroy it.
void save_advice(const NeuralAbductionView& view)
{
    if (view.task_id.empty()) return;

    nlohmann::json items = nlohmann::json::array();
    std::vector<std::string> written;

    std::ifstream existing(advice_path(view.task_id));
    if (existing) {
        try {
            nlohmann::json previous;
            existing >> previous;
            if (previous.is_array()) {
                for (auto& item : previous) {
                    if (!item.contains("name")) continue;
                    const std::string name = item["name"].get<std::string>();
                    bool selected = item.value("selected", false);
                    for (const Prior& prior : view.priors)
                        if (prior.name == name) selected = prior.selected;
                    items.push_back({{"name", name}, {"selected", selected}});
                    written.push_back(name);
                }
            }
        } catch (const std::exception&) { items = nlohmann::json::array(); }
    }

    for (const Prior& prior : view.priors) {
        if (std::find(written.begin(), written.end(), prior.name) != written.end())
            continue;
        items.push_back({{"name", prior.name}, {"selected", prior.selected}});
    }

    std::error_code ec;
    std::filesystem::create_directories(
        std::string(kAdviceDir) + "/" + view.task_id, ec);

    std::ofstream out(advice_path(view.task_id), std::ios::trunc);
    if (out) out << items.dump(4);   // 4 spaces, as the existing files are kept
}

flecs::entity view_entity()
{
    flecs::entity found;
    if (!g_world) return found;
    g_world->query<NeuralAbductionView>().each([&](flecs::entity e, NeuralAbductionView&) {
        if (!found) found = e;
    });
    return found;
}

// The chips are badges, and a badge's own children take the click, so the whole
// subtree carries the handler -- clicking the word has to mean the same as
// clicking beside it.
void switch_mode(flecs::entity chip)
{
    const AbductionModeChip* info = chip.try_get<AbductionModeChip>();
    flecs::entity ve = view_entity();
    if (!info || !ve.is_valid()) return;

    NeuralAbductionView& view = ve.ensure<NeuralAbductionView>();
    if (view.mode == info->mode) return;
    view.mode = info->mode;
    view.dirty = true;

    chip.world().ensure<UIInputDispatch>().consumed = true;
}

void toggle_row(flecs::entity row)
{
    const AbductionRow* info = row.try_get<AbductionRow>();
    if (!info || !g_world) return;

    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;

    NeuralAbductionView& view = ve.ensure<NeuralAbductionView>();
    for (Prior& prior : view.priors) {
        if (prior.name != info->name) continue;
        prior.selected = !prior.selected;
        break;
    }
    save_advice(view);
    view.dirty = true;

    row.world().ensure<UIInputDispatch>().consumed = true;
}

// Confidence as opacity. A prior is a continuous claim, so drawing it as one
// says more per pixel than a bar and a number do: a ghosted icon is the model
// shrugging, a solid one is the model insisting, and the eye reads the whole
// set at once instead of a row at a time.
unsigned char prior_alpha(float probability)
{
    // Gamma-corrected, not linear. The predictions are heavily skewed -- a
    // typical task has three items above 0.5 and forty near zero -- so mapping
    // straight to opacity spends the whole range on the top few and leaves
    // everything else in one indistinguishable smear. The curve lifts the low
    // and middle so a 0.05 and a 0.25 are visibly different claims, which is
    // the point of showing all 47 at once.
    const float clamped = std::clamp(probability, 0.0f, 1.0f);
    const float curved = std::pow(clamped, 0.45f);
    return (unsigned char)(28.0f + curved * 227.0f);
}

const Prior* find_prior(const NeuralAbductionView& view, const std::string& name)
{
    for (const Prior& prior : view.priors)
        if (prior.name == name) return &prior;
    return nullptr;
}

// One item as a clickable icon: what the network thinks in its brightness, what
// you have vouched for in its outline.
void build_item(flecs::entity parent, flecs::entity UIElement,
                const NeuralAbductionView& view, const std::string& name)
{
    flecs::world world = parent.world();
    const Prior* prior = find_prior(view, name);
    const bool vouched = prior && prior->selected;
    const unsigned char alpha = prior_alpha(prior ? prior->probability : 0.0f);

    auto slot = world.entity()
        .is_a(UIElement)
        .child_of(parent)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<AbductionRow>({name})
        .set<CallbackOnLeftClick>({toggle_row})
        .set<UIPadding>({3.0f, 3.0f, 3.0f, 3.0f})
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyCenter, YGAlignCenter, 0.0f})
        .set<RoundedRectRenderable>({0.0f, 0.0f, 3.0f, true,
                                     vouched ? 0xFFFFFFFFu : 0x00000000u})
        .set<ZIndex>({13});

    world.entity()
        .is_a(UIElement)
        .child_of(slot)
        .add<UIYoga>()
        .add<UINativeImageSize>()
        .set<ImageCreator>({"items/" + name + ".png", 1.0f, 1.0f,
                            nvgRGBA(0xFF, 0xFF, 0xFF, alpha)})
        .set<ZIndex>({14});
}

// The annotation view: items among their kin. Finding "symmetry_shovel" here is
// a matter of knowing it is an abstraction tool, not of remembering how strongly
// the network happened to rank it today.
void build_categories(NeuralAbductionView& view, flecs::entity UIElement)
{
    flecs::world world = view.list.world();

    std::vector<std::string> filed;
    for (const ItemCategory& category : item_categories()) {
        auto box = world.entity()
            .is_a(UIElement)
            .child_of(view.list)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            // No grow and no shrink: a box is exactly as wide as the items and
            // title inside it, which is what lets the row pack several and wrap
            // the rest.
            .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
            .set<UIPadding>({6.0f, 8.0f, 8.0f, 8.0f})
            .set<UIMaxSize>({260.0f, YGUndefined})
            .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                                   YGJustifyFlexStart, YGAlignFlexStart, 4.0f})
            .set<ZIndex>({12});

        // The box lights when anything inside it is vouched for, so a glance
        // down the panel says where your advice is concentrated.
        bool any_vouched = false;
        for (const std::string& name : category.items) {
            const Prior* prior = find_prior(view, name);
            if (prior && prior->selected) { any_vouched = true; break; }
        }
        box.set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, true,
                                        any_vouched ? 0x6DF0FFAAu : 0xFFFFFF14u});

        world.entity()
            .is_a(UIElement)
            .child_of(box)
            .add<UIYoga>()
            .set<TextRenderable>({category.name, "JetBrainsMono", 12.0f,
                                  any_vouched ? kBarOn : 0x7A7A7AFF})
            .set<ZIndex>({13});

        auto row = world.entity()
            .is_a(UIElement)
            .child_of(box)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                                   YGJustifyFlexStart, YGAlignCenter, 4.0f})
            .set<ZIndex>({13});

        for (const std::string& name : category.items) {
            build_item(row, UIElement, view, name);
            filed.push_back(name);
        }
    }

    // Anything the model can predict but no category claims. Without this a
    // representation could be predicted and never annotatable.
    std::vector<std::string> unfiled;
    for (const Prior& prior : view.priors)
        if (std::find(filed.begin(), filed.end(), prior.name) == filed.end())
            unfiled.push_back(prior.name);
    if (unfiled.empty()) return;

    auto box = world.entity()
        .is_a(UIElement)
        .child_of(view.list)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
        .set<UIPadding>({6.0f, 8.0f, 8.0f, 8.0f})
        .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                               YGJustifyFlexStart, YGAlignFlexStart, 4.0f})
        .set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, true, 0xFFFFFF14})
        .set<ZIndex>({12});

    world.entity()
        .is_a(UIElement)
        .child_of(box)
        .add<UIYoga>()
        .set<TextRenderable>({"UNFILED", "JetBrainsMono", 12.0f, 0x7A7A7AFF})
        .set<ZIndex>({13});

    auto row = world.entity()
        .is_a(UIElement)
        .child_of(box)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                               YGJustifyFlexStart, YGAlignCenter, 4.0f})
        .set<ZIndex>({13});
    for (const std::string& name : unfiled) build_item(row, UIElement, view, name);
}

void processor_response(std::map<std::string, msgpack::object>& res)
{
    const std::string key = g_finding_inflight;
    g_finding_inflight.clear();
    if (key.empty()) return;

    ProcessorResult& result = g_findings[key];
    std::string status;
    try {
        auto status_it = res.find("status");
        if (status_it != res.end()) status = status_it->second.as<std::string>();
    } catch (const std::exception&) {}

    if (status != "ok") {
        // "computes nothing yet" is the ordinary case for most items, not an
        // error worth shouting about: it is remembered so it is asked once.
        try {
            auto message_it = res.find("message");
            result.status = message_it != res.end()
                ? message_it->second.as<std::string>() : "no reply";
        } catch (const std::exception&) { result.status = "no reply"; }
        return;
    }

    result.status = "ok";
    result.findings.clear();
    try {
        auto it = res.find("results");
        if (it == res.end()) return;
        for (const msgpack::object& row : it->second.as<std::vector<msgpack::object>>()) {
            auto entry = row.as<std::map<std::string, msgpack::object>>();
            ProcessorFinding finding;

            auto source_it = entry.find("source");
            if (source_it != entry.end()) {
                const auto source = source_it->second.as<std::vector<msgpack::object>>();
                if (source.size() >= 3) {
                    finding.pair_idx = (int)source[0].as<int64_t>();
                    finding.is_test = source[1].as<int64_t>() != 0;
                    finding.is_output = source[2].as<int64_t>() != 0;
                }
            }

            auto facts_it = entry.find("facts");
            if (facts_it != entry.end())
                for (const msgpack::object& fact :
                         facts_it->second.as<std::vector<msgpack::object>>()) {
                    auto pair = fact.as<std::map<std::string, msgpack::object>>();
                    finding.facts.push_back(
                        {pair.count("label") ? pair["label"].as<std::string>() : "",
                         pair.count("value") ? pair["value"].as<std::string>() : ""});
                }

            auto grid_it = entry.find("grid");
            if (grid_it != entry.end()) {
                const auto rows = grid_it->second.as<std::vector<msgpack::object>>();
                finding.grid_h = (int)rows.size();
                for (const msgpack::object& row_obj : rows) {
                    const auto cells = row_obj.as<std::vector<msgpack::object>>();
                    finding.grid_w = std::max(finding.grid_w, (int)cells.size());
                    for (const msgpack::object& cell : cells)
                        finding.grid.push_back((uint8_t)cell.as<int64_t>());
                }
            }

            auto cells_it = entry.find("cells");
            if (cells_it != entry.end())
                finding.cell_count =
                    (int)cells_it->second.as<std::vector<msgpack::object>>().size();

            result.findings.push_back(std::move(finding));
        }
    } catch (const std::exception& exc) {
        result.status = std::string("unreadable finding: ") + exc.what();
    }
}

// Asks what an item computed for this task, once. Most items compute nothing,
// and the reply saying so is cached like any other answer.
void request_finding(const std::string& task_id, const std::string& name)
{
    if (!g_world || task_id.empty() || name.empty()) return;
    const std::string key = task_id + "/" + name;
    if (g_findings.count(key) || !g_finding_inflight.empty()) return;

    flecs::entity server = servers::find(*g_world, kServerId);
    const ServerState* state = server.is_valid() ? server.try_get<ServerState>() : nullptr;
    if (!state || state->status == ServerStatus::Offline) return;

    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) return;

    g_findings[key].status = "";          // pending, so it is not asked again
    g_finding_inflight = key;
    client.set<tradewinds::SendMapRequest>({{{"command", "processor"},
                                             {"name", name},
                                             {"task_id", task_id}}});
    client.set<tradewinds::AwaitResponse>({processor_response});
}

// The hover card: what the network predicted, what the thing is, the signature
// it works over, and what it found in this task if it computes anything.
void build_tooltip(flecs::entity card, const NeuralAbductionView& view,
                   const std::string& name)
{
    flecs::world world = card.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid()) return;

    std::vector<flecs::entity> old;
    card.children([&](flecs::entity c) { old.push_back(c); });
    for (flecs::entity c : old) c.destruct();
    if (name.empty()) return;

    auto line = [&](const std::string& text, uint32_t color, float size) {
        world.entity()
            .is_a(UIElement)
            .child_of(card)
            .add<UIYoga>()
            .set<TextRenderable>({text, "JetBrainsMono", size, color})
            .set<ZIndex>({31});
    };

    const Prior* prior = find_prior(view, name);
    if (prior) {
        char predicted[32];
        snprintf(predicted, sizeof(predicted), "predicted %d%%",
                 (int)(prior->probability * 100.0f + 0.5f));
        line(predicted, prior->probability >= kAbduced ? kBarOn : 0x7A7A7AFF, 13.0f);
    }

    const auto& table = item_descriptions();
    auto described = table.find(name);
    if (described == table.end()) {
        // No entry is a fact about the table, not about the item: say the name
        // rather than showing an empty card.
        line(name, kNameOn, 14.0f);
        line("(no description)", 0x6A6A6AFF, 13.0f);
    } else {
        line(described->second.title, kNameOn, 14.0f);
        if (!described->second.description.empty())
            line(described->second.description, 0xB0B0B0FF, 13.0f);
        if (!described->second.inputs.empty()) {
            line("inputs", 0x7A7A7AFF, 13.0f);
            for (const std::string& in : described->second.inputs)
                line("  " + in, 0x9FD8B0FF, 13.0f);
        }
        if (!described->second.outputs.empty()) {
            line("outputs", 0x7A7A7AFF, 13.0f);
            for (const std::string& out : described->second.outputs)
                line("  " + out, 0xD8C39FFF, 13.0f);
        }
    }

    // What this processor actually found in the task being looked at. Only the
    // items that compute something ever have this, which is the point: the
    // display is in the card the pointer is already on, not in a panel each of
    // 47 items would have to earn.
    auto found = g_findings.find(view.task_id + "/" + name);
    if (found == g_findings.end() || found->second.status != "ok") return;
    if (found->second.findings.empty()) {
        line("found nothing here", 0x6A6A6AFF, 13.0f);
        return;
    }

    line("", 0x00000000, 4.0f);   // a gap: the findings are a separate claim

    // A dozen findings is a wall, and a hover is a glance: the first few say
    // what was found, and the count says there is more.
    constexpr size_t kMaxShown = 4;
    const size_t shown = std::min(kMaxShown, found->second.findings.size());
    for (size_t fi = 0; fi < shown; fi++) {
        const ProcessorFinding& finding = found->second.findings[fi];
        std::string where = (finding.is_test ? "test " : "train ")
                          + std::to_string(finding.pair_idx)
                          + (finding.is_output ? " out" : " in");
        for (const auto& [label, value] : finding.facts)
            where += "   " + label + " " + value;
        line(where, 0xB0B0B0FF, 13.0f);

        // The matrix it found, drawn rather than described: a 3x2 tile is a
        // shape, and a shape is not a sentence.
        if (finding.grid_w > 0 && finding.grid_h > 0) {
            auto rows = world.entity()
                .is_a(UIElement).child_of(card).add(flecs::OrderedChildren).add<UIYoga>()
                .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                                       YGJustifyFlexStart, YGAlignFlexStart, 1.0f})
                .set<UIPadding>({2.0f, 0.0f, 4.0f, 8.0f})
                .set<ZIndex>({31});
            for (int y = 0; y < finding.grid_h; y++) {
                auto row = world.entity()
                    .is_a(UIElement).child_of(rows).add(flecs::OrderedChildren).add<UIYoga>()
                    .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                           YGJustifyFlexStart, YGAlignCenter, 1.0f})
                    .set<ZIndex>({31});
                for (int x = 0; x < finding.grid_w; x++) {
                    const size_t at = (size_t)y * finding.grid_w + x;
                    const uint8_t value = at < finding.grid.size() ? finding.grid[at] : 0;
                    world.entity()
                        .is_a(UIElement).child_of(row).add<UIYoga>()
                        .set<UISize>({8.0f, 8.0f})
                        .set<RectRenderable>({8.0f, 8.0f, false,
                                              value < 10 ? arc_palette[value] : 0x000000FF})
                        .set<ZIndex>({32});
                }
            }
        }
    }
    if (found->second.findings.size() > kMaxShown)
        line("+" + std::to_string(found->second.findings.size() - kMaxShown)
             + " more matrices", 0x6A6A6AFF, 13.0f);
}

void rebuild(NeuralAbductionView& view)
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

    // The mode chips restyle with the mode, so they are rebuilt beside the rows.
    if (view.mode_row.is_valid() && view.mode_row.is_alive()) {
        std::vector<flecs::entity> chips;
        view.mode_row.children([&](flecs::entity c) { chips.push_back(c); });
        for (flecs::entity c : chips) c.destruct();

        const std::pair<ViewMode, const char*> modes[] = {
            {ViewMode::Ranked, "ranked"}, {ViewMode::Categories, "categories"}};
        for (const auto& [mode, label] : modes) {
            flecs::entity chip = create_badge(view.mode_row, UIElement, label,
                                              view.mode == mode ? kBarOn : 0x3A4048FF);
            chip.set<AbductionModeChip>({mode})
                .set<CallbackOnLeftClick>({switch_mode});
        }
    }

    // The two views want opposite containers: rows stack, boxes flow. Set on
    // the same list rather than kept as two, so only one of them can be the
    // thing being scrolled.
    if (view.mode == ViewMode::Categories) {
        // Left to right, wrapping like text: a category is as wide as its
        // contents, and several small ones share a line instead of each taking
        // a whole row to hold three icons.
        view.list.set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                                        YGJustifyFlexStart, YGAlignFlexStart, 6.0f});
        build_categories(view, UIElement);
        return;
    }

    // Ranked: every item at once, strongest first, flowing left to right and
    // wrapping. The sprite is the name, opacity is the probability and position
    // is the rank -- three facts per icon, in the space one labelled row used to
    // take. The exact number is a hover away, which is where a number belongs.
    view.list.set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                                    YGJustifyFlexStart, YGAlignFlexStart, 4.0f});
    for (const Prior& prior : view.priors)
        build_item(view.list, UIElement, view, prior.name);
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
    .set<TextRenderable>({"waiting for an ARC context", "JetBrainsMono", 14.0f, 0x9A9A9AFF})
    .set<ZIndex>({12});

    // Reading and choosing want different shapes, so the panel carries both and
    // says which it is in.
    auto mode_row = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
    .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignCenter, 4.0f})
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
    .set<PanelScroll>({0.0f})
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
    .add<PanelScrollbarOf>(list)
    .set<Position, Local>({0.0f, 0.0f})
    .set<RoundedRectRenderable>({4.0f, 0.0f, 2.0f, false, 0xFFFFFF4D})
    .set<ZIndex>({15});

    // An overlay filling the panel, so the card has a Yoga parent to be
    // absolutely positioned inside. Without one the card would be a Yoga ROOT
    // -- parent not a Yoga node -- and Yoga leaves a root's position to whoever
    // owns it, which is what forced the hand-written Position,Local this
    // replaces. No padding: the overlay's origin IS the panel's, so nothing has
    // to be subtracted back out.
    auto overlay = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFillParent>({})
    .set<ZIndex>({30});

    world.entity()
    .is_a(UIElement)
    .child_of(overlay)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<AbductionTooltip>({""})
    // Absolute in Yoga's own terms: out of the overlay's flow, placed by edges
    // the hover system writes. Yoga owns the position, so nothing else can be
    // fighting it for the same frame.
    .set<UIAbsoluteEdges>({0.0f, 0.0f, YGUndefined, YGUndefined})
    .set<UIMaxSize>({320.0f, YGUndefined})
    .set<UIPadding>({6.0f, 8.0f, 6.0f, 8.0f})
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignFlexStart, 2.0f})
    .set<RoundedRectRenderable>({0.0f, 0.0f, 4.0f, false, 0x00000000})
    .set<ZIndex>({30});

    NeuralAbductionView initial;
    initial.list = list;
    initial.status_text = status_text;
    initial.mode_row = mode_row;
    initial.status = "waiting for an ARC context";
    column.set<NeuralAbductionView>(initial);

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());
    // Not the cyan the bars use: that colour means "the network vouches for
    // this", and a badge wearing it reads as a claim rather than a name. A
    // muted plum instead -- distinct from the Lattice violet, the Dataset teal
    // and the ArcTask blue, and far enough from Rose and Purse that it cannot
    // be mistaken for an ARC colour.
    create_badge(badges, UIElement, "NeuralAbduction", 0xB0728FFF);
}

} // namespace

namespace panels {

NeuralAbduction::NeuralAbduction(flecs::world& world)
{
    world.module<NeuralAbduction>();
    g_world = &world;

    world.component<NeuralAbductionView>();
    world.component<AbductionRow>();
    world.component<AbductionModeChip>();
    world.component<AbductionTooltip>();

    world.system<NeuralAbductionView>("NeuralAbductionFollowSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, NeuralAbductionView& view) {
        const std::string task = context_task();
        if (task.empty() || task == view.task_id) return;
        load_priors(view, task);
    });

    // Hover: which item the pointer is over, and the card that follows it.
    // Hit-tested here rather than through the hover-tag observers because the
    // card's content depends on WHICH item, not merely that one is hovered --
    // and because a rebuilt row list would drop tags mid-hover anyway.
    world.system<AbductionTooltip, UIAbsoluteEdges>("NeuralAbductionHoverSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity card, AbductionTooltip& tip, UIAbsoluteEdges& edges) {
        flecs::world world = card.world();
        flecs::entity glfw_state = world.lookup("GLFWState");
        const CursorState* cursor =
            glfw_state.is_valid() ? glfw_state.try_get<CursorState>() : nullptr;

        flecs::entity ve = view_entity();
        if (!cursor || !ve.is_valid()) return;
        const NeuralAbductionView* view = ve.try_get<NeuralAbductionView>();
        if (!view) return;

        std::string hovered;
        world.query<const AbductionRow, const UIElementBounds>()
            .each([&](flecs::entity item, const AbductionRow& row,
                      const UIElementBounds& bounds) {
                if (!hovered.empty()) return;
                if (bounds.xmax <= bounds.xmin) return;
                if (!point_in_bounds((float)cursor->x, (float)cursor->y, bounds)) return;
                hovered = row.name;
            });

        // Asked once per task and item; the reply lands later, so the card is
        // rebuilt when what is known about the hovered item changes, not only
        // when the pointer moves to a different one.
        if (!hovered.empty()) request_finding(view->task_id, hovered);

        auto known = g_findings.find(view->task_id + "/" + hovered);
        const std::string state = known == g_findings.end() ? "" : known->second.status;
        if (hovered != tip.showing || state != tip.showing_state) {
            tip.showing = hovered;
            tip.showing_state = state;
            build_tooltip(card, *view, hovered);
        }

        // Hidden by being empty, not by being parked far away. An off-canvas
        // position is a child the parent's bounds may stretch to include, and
        // those same bounds are what this system converts through -- so parking
        // at -10000 feeds its own input and the card oscillates every frame.
        // With no children and a transparent ground there is nothing to see and
        // nothing to stretch.
        if (RoundedRectRenderable* ground = card.try_get_mut<RoundedRectRenderable>())
            ground->color = hovered.empty() ? 0x00000000u : 0x11141AF0u;

        if (hovered.empty()) {
            edges.left = 0.0f;
            edges.top = 0.0f;
            return;
        }

        // Anchored by its lower left: the card grows up and to the right of the
        // pointer, so it never covers what is below and to the right -- the rows
        // you are about to move onto. Same anchor the Droid panel's tooltip uses.
        const Window* win = world.lookup("GLFWState").try_get<Window>();
        const UIElementSize* size = card.try_get<UIElementSize>();
        const float width  = size ? size->width  : 0.0f;
        const float height = size ? size->height : 0.0f;

        float tx = (float)cursor->x;
        float ty = (float)cursor->y - height;

        if (win && width > 0.0f && height > 0.0f) {
            // Out of room above or to the right: put the card on the other side
            // of the cursor rather than letting it leave the window.
            if (tx + width > (float)win->width)  tx = (float)cursor->x - width;
            if (ty < 4.0f)                       ty = (float)cursor->y;
            tx = std::clamp(tx, 4.0f, std::max(4.0f, (float)win->width - width - 4.0f));
            ty = std::clamp(ty, 4.0f, std::max(4.0f, (float)win->height - height - 4.0f));
        }

        // The cursor is window-space and the edges are relative to the overlay,
        // so the overlay's world POSITION converts between them -- computed
        // from its parent downward, never from its children. Its BOUNDS would
        // not do: bounds stretch to contain children, this card is one, and
        // converting through a number the card helps produce is how a tooltip
        // ends up chasing itself off the edge of a panel and back.
        const Position* origin = card.parent().try_get<Position, World>();
        edges.left = tx - (origin ? origin->x : 0.0f);
        edges.top  = ty - (origin ? origin->y : 0.0f);
    });

    world.system<NeuralAbductionView>("NeuralAbductionRebuildSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, NeuralAbductionView& view) {
        if (!view.dirty) return;
        view.dirty = false;
        rebuild(view);
    });

    register_panel({EditorType::NeuralAbduction, "neural_abduction",
                    "NeuralAbduction", create_content});
}

} // namespace panels
