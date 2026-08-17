#include "arc_task_panel.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <msgpack.hpp>
#include <nlohmann/json.hpp>

#include "../components.h"
#include "../tradewinds.h"
#include "../server_registry.h"
#include "../capabilities/function_library.h"
#include "../capabilities/query_contexts.h"
#include "../capabilities/ui_kit.h"
#include "registry.h"

using namespace tradewinds;

namespace {

// The Jane Eyre solver bridge's REQ socket, created centrally in main.cpp
// like every other client -- the Peach Core owns the server roster, and this
// panel adds no socket of its own.
constexpr const char* kClientName = "ArcSolverClient";
constexpr const char* kServerId   = "arc_solver";

// The plural transducer the Lexicon teaches. Shared with that panel, so every
// send goes through tradewinds' reservation like anyone else's.
constexpr const char* kMorphologyClient   = "MorphologyClient";
constexpr const char* kMorphologyServerId = "morphology";

// The renderer and the solver share one source of truth for a task: this JSON
// tree is the same one solver_server.py resolves ./ARC-AGI/data/training
// against, and task_id is the key both sides pass around. Machine-specific
// for now; a config knob when a second machine needs it.
constexpr const char* kArcDataDir =
    "/home/wesxdz/arc/editor/jane_eyre/ARC-AGI/data/training";

// Canonical ARC palette, RRGGBBAA like every color in components.h.
constexpr uint32_t kArcPalette[10] = {
    0x555555FF, // 0 slate
    0x0074D9FF, // 1 zen
    0xFF4136FF, // 2 rose
    0x2ECC40FF, // 3 grass
    0xFFDC00FF, // 4 sun
    0xAAAAAAFF, // 5 grey
    0xF012BEFF, // 6 purse
    0xFF851BFF, // 7 tang
    0x7FDBFFFF, // 8 sky
    0x985898FF, // 9 space
};
constexpr const char* kArcColorNames[10] = {
    "slate", "zen", "rose", "grass", "sun",
    "grey", "purse", "tang", "sky", "space",
};

// Heatshield's taskset layout knobs, kept as constants: one panel, one look.
constexpr float kMinCell          =  2.0f;
constexpr float kMaxCell          = 48.0f;
constexpr float kPairGap          = 16.0f;   // input column | output column
constexpr float kRowGap           =  6.0f;   // between IO rows
constexpr float kDividerThickness =  2.0f;   // train/test divider
constexpr float kDividerPad       =  6.0f;
constexpr float kDimFactor        =  0.18f;  // unselected cells while a selection exists

struct ArcGridData {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> cells;              // row-major, values 0..9
};

struct ArcIOPairData {
    ArcGridData input;
    ArcGridData output;
};

struct ArcTaskData {
    std::string name;
    std::vector<ArcIOPairData> train;
    std::vector<ArcIOPairData> test;
};

// One cell of the loaded task: which grid (pair, train/test, input/output)
// and which square. The same shape solver_server.py's nlp_query rows carry.
struct ArcCellRef {
    int pair_idx = 0;
    bool is_test = false;
    bool is_output = false;
    int x = 0, y = 0;

    bool operator==(const ArcCellRef& o) const {
        return pair_idx == o.pair_idx && is_test == o.is_test &&
               is_output == o.is_output && x == o.x && y == o.y;
    }
};

// The state of one ArcTask panel. Lives on the panel's own content root so it
// dies with the canvas when the panel type changes.
struct ArcTaskView {
    std::string task_id;
    int task_index = -1;                     // position in dataset_ids(), -1 if unknown
    ArcTaskData task;

    // Click-selected or server-returned cells: full brightness plus a stroke,
    // everything else dimmed. One list serves both because the two gestures
    // mean the same thing -- "these are the cells being talked about".
    std::vector<ArcCellRef> selection;
    bool server_selection = false;

    std::string status;                      // "" | progress | error text
    std::string structured;                  // the query as the server parsed it

    // Resolving a comprehension's words into terms. Words that name a tag
    // outright are resolved immediately; the rest queue here for the
    // morphology transducer, which is what turns "polyominoes" into
    // "polyomino". `inductive` records that some word arrived inflected --
    // the plural is the sentence saying it means every such thing, not one.
    std::vector<std::string> pending_words;
    std::string resolving;                   // word currently at the transducer
    std::vector<std::string> resolved_terms;
    std::vector<std::string> unresolved_words;
    bool inductive = false;

    flecs::entity status_text;               // TextRenderable updated in place
    flecs::entity query_row;                 // holds the query's term badges
    std::string rendered_query;              // what query_row currently shows
    bool dirty = true;                       // status line needs a refresh
};

// Marker for the CustomRenderable element the grids draw into; its laid-out
// bounds are the draw region and the click hit-test region.
struct ArcGridSurface {};

flecs::world* g_world = nullptr;

// The solver's tag vocabulary, fetched once the server is up. This is what
// makes a spoken sentence a query rather than a remark: jane_eyre's
// nl_to_structured recognises a word only when its lowercase form names a tag
// (see query_lang.py), drops everything else, and reads not/and/or as
// operators. So Thornfield can apply the same test locally -- an utterance
// naming at least one tag is a query -- and leave the authoritative
// translation and evaluation to the server.
std::vector<std::string> g_vocab;
bool g_vocab_inflight = false;

// Defined with the rest of the context handling below; needed here because the
// vocabulary is imported scoped to it.
flecs::entity context_entity(flecs::world& world);

flecs::entity view_entity()
{
    flecs::entity found;
    if (!g_world) return found;
    g_world->query<ArcTaskView>().each([&](flecs::entity e, ArcTaskView&) {
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

// nlp_query rows mix ints and Python bools depending on query_lang internals;
// accept either rather than trusting the wire type.
int msg_int(const msgpack::object& o)
{
    switch (o.type) {
        case msgpack::type::BOOLEAN:          return o.via.boolean ? 1 : 0;
        case msgpack::type::POSITIVE_INTEGER: return (int)o.via.u64;
        case msgpack::type::NEGATIVE_INTEGER: return (int)o.via.i64;
        default:                              return 0;
    }
}

// ------------------------------------------------------------- query reading

std::string lowered(const std::string& s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

// The canonical tag an annotated word names, or empty when the word means
// nothing to the solver. Case-insensitive because annotation badges carry the
// word as it was said ("zen", "above") while the vocabulary is canonical
// ("Zen", "Above"); punctuation is stripped for the same reason.
std::string vocab_term(const std::string& word)
{
    std::string cleaned;
    for (char c : word)
        if (c != ',' && c != '.' && c != '?' && c != '!' && !std::isspace((unsigned char)c))
            cleaned += c;
    if (cleaned.empty()) return {};

    const std::string lower = lowered(cleaned);
    for (const std::string& tag : g_vocab)
        if (lowered(tag) == lower) return tag;
    return {};
}

// ---------------------------------------------------------------- task data

// Every task id in the dataset, sorted, scanned once per process: arrow
// navigation is index stepping through this list, and a load by name keeps the
// index in step with it.
const std::vector<std::string>& dataset_ids()
{
    static const std::vector<std::string> ids = [] {
        namespace fs = std::filesystem;
        std::vector<std::string> out;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(kArcDataDir, ec))
            if (entry.is_regular_file() && entry.path().extension() == ".json")
                out.push_back(entry.path().stem().string());
        std::sort(out.begin(), out.end());
        return out;
    }();
    return ids;
}

ArcGridData grid_from_json(const nlohmann::json& g)
{
    ArcGridData out;
    out.height = (int)g.size();
    out.width  = out.height ? (int)g[0].size() : 0;
    out.cells.reserve((size_t)out.width * (size_t)out.height);
    for (const auto& row : g)
        for (const auto& cell : row)
            out.cells.push_back((uint8_t)cell.get<int>());
    return out;
}

bool load_task(ArcTaskView& view, const std::string& task_id)
{
    const std::string path = std::string(kArcDataDir) + "/" + task_id + ".json";
    std::ifstream f(path);
    if (!f) {
        view.status = "no such task: " + task_id;
        view.dirty = true;
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& exc) {
        view.status = task_id + ": " + exc.what();
        view.dirty = true;
        return false;
    }

    ArcTaskData t;
    t.name = task_id;
    if (j.contains("train"))
        for (const auto& p : j["train"])
            t.train.push_back({grid_from_json(p["input"]), grid_from_json(p["output"])});
    if (j.contains("test"))
        for (const auto& p : j["test"])
            t.test.push_back({grid_from_json(p["input"]), grid_from_json(p["output"])});

    view.task = std::move(t);
    view.task_id = task_id;
    view.selection.clear();
    view.server_selection = false;
    view.structured.clear();

    // Position in the dataset, so the status line reads as a place you can
    // navigate from rather than a bare id.
    const std::vector<std::string>& ids = dataset_ids();
    const auto it = std::find(ids.begin(), ids.end(), task_id);
    view.task_index = it == ids.end() ? -1 : (int)(it - ids.begin());
    view.status = view.task_index >= 0
        ? task_id + "  " + std::to_string(view.task_index + 1) + "/" +
          std::to_string(ids.size())
        : task_id;

    view.dirty = true;
    return true;
}

// ------------------------------------------------------------------- layout

// Where every grid of the task lands within the surface. One function feeds
// both the draw pass and the click hit-test, so the math can never drift.
struct GridPlace {
    const ArcGridData* g;
    float x, y;
    int pair_idx;
    bool is_test, is_output;
};

struct TaskLayout {
    float cs = 0.0f;
    float cell_pad = 0.0f;
    std::vector<GridPlace> grids;
    float divider_y = -1.0f;                 // <0 when there is no divider
    float table_left = 0.0f;
    float table_width = 0.0f;
};

// Two-column table: every input shares a column sized to the widest input,
// outputs likewise, inputs right-aligned and outputs left-aligned so the gap
// between them stays at one x. Ported from heatshield's TaskSet renderer.
TaskLayout compute_layout(const ArcTaskData& task, float ox, float oy,
                          float avail_w, float avail_h)
{
    TaskLayout out;

    int max_input_w = 0, max_output_w = 0, sum_row_units_y = 0, num_rows = 0;
    auto tally = [&](const std::vector<ArcIOPairData>& pairs) {
        for (const auto& p : pairs) {
            max_input_w  = std::max(max_input_w,  p.input.width);
            max_output_w = std::max(max_output_w, p.output.width);
            sum_row_units_y += std::max(p.input.height, p.output.height);
            num_rows++;
        }
    };
    tally(task.train);
    tally(task.test);

    const int total_units_x = max_input_w + max_output_w;
    if (num_rows == 0 || total_units_x == 0 || sum_row_units_y == 0) return out;

    const bool has_divider = !task.train.empty() && !task.test.empty();
    const float divider_h_total =
        has_divider ? (kDividerThickness + 2.0f * kDividerPad) : 0.0f;
    const float total_gap_h =
        (num_rows > 1 ? (num_rows - 1) * kRowGap : 0.0f) + divider_h_total;

    const float fit_w = (avail_w > kPairGap)
        ? (avail_w - kPairGap) / float(total_units_x) : kMaxCell;
    const float fit_h = (avail_h > total_gap_h)
        ? (avail_h - total_gap_h) / float(sum_row_units_y) : kMaxCell;

    const float cs = std::clamp(std::min(fit_w, fit_h), kMinCell, kMaxCell);
    out.cs = cs;
    out.cell_pad = std::clamp(cs * 0.08f, 0.5f, 2.0f);

    out.table_width = total_units_x * cs + kPairGap;
    out.table_left = ox + (avail_w - out.table_width) * 0.5f;
    const float input_col_right = out.table_left + max_input_w * cs;
    const float output_col_left = input_col_right + kPairGap;

    float cur_y = oy;
    auto place_rows = [&](const std::vector<ArcIOPairData>& pairs, bool is_test) {
        for (size_t pi = 0; pi < pairs.size(); pi++) {
            const ArcIOPairData& p = pairs[pi];
            const float in_h  = p.input.height  * cs;
            const float out_h = p.output.height * cs;
            const float row_h = std::max(in_h, out_h);
            out.grids.push_back({&p.input,
                                 input_col_right - p.input.width * cs,
                                 cur_y + (row_h - in_h) * 0.5f,
                                 (int)pi, is_test, false});
            out.grids.push_back({&p.output, output_col_left,
                                 cur_y + (row_h - out_h) * 0.5f,
                                 (int)pi, is_test, true});
            cur_y += row_h + kRowGap;
        }
    };

    place_rows(task.train, false);
    if (has_divider) {
        cur_y += kDividerPad - kRowGap;
        out.divider_y = cur_y;
        cur_y += kDividerThickness + kDividerPad;
    }
    place_rows(task.test, true);
    return out;
}

// --------------------------------------------------------------------- draw

// The whole task in one CustomRenderable: per color one batched fill for
// selected cells at full brightness and one for the rest dimmed, then a
// stroked outline over the selected set. Cells are not entities -- a 30x30
// grid six times over would be the largest entity population in the editor
// for the least interesting reason.
void draw_task(NVGcontext* vg, const RenderCommand*, const CustomRenderable&)
{
    if (!g_world) return;
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    const ArcTaskView* view = ve.try_get<ArcTaskView>();
    if (!view) return;
    if (view->task.train.empty() && view->task.test.empty()) return;

    UIElementBounds bounds{0, 0, 0, 0};
    bool have_bounds = false;
    g_world->query<const ArcGridSurface, const UIElementBounds>()
        .each([&](flecs::entity, const ArcGridSurface&, const UIElementBounds& b) {
            if (!have_bounds && b.xmax > b.xmin) { bounds = b; have_bounds = true; }
        });
    if (!have_bounds) return;

    const TaskLayout lay = compute_layout(view->task, bounds.xmin, bounds.ymin,
                                          bounds.xmax - bounds.xmin,
                                          bounds.ymax - bounds.ymin);
    if (lay.grids.empty()) return;

    const bool has_sel = !view->selection.empty();
    const float cs = lay.cs;
    const float cell_draw = cs - lay.cell_pad;

    for (const GridPlace& place : lay.grids) {
        const ArcGridData& g = *place.g;

        std::vector<uint8_t> mask;
        if (has_sel) {
            mask.assign((size_t)g.width * (size_t)g.height, 0);
            for (const ArcCellRef& c : view->selection) {
                if (c.pair_idx != place.pair_idx) continue;
                if (c.is_test != place.is_test) continue;
                if (c.is_output != place.is_output) continue;
                if (c.x < 0 || c.x >= g.width || c.y < 0 || c.y >= g.height) continue;
                mask[(size_t)c.y * g.width + c.x] = 1;
            }
        }

        for (uint8_t color_idx = 0; color_idx < 10; color_idx++) {
            const uint32_t col = kArcPalette[color_idx];
            const uint8_t rf = (col >> 24) & 0xFF;
            const uint8_t gf = (col >> 16) & 0xFF;
            const uint8_t bf = (col >>  8) & 0xFF;

            auto emit_batch = [&](bool want_selected, bool dim) {
                bool any = false;
                for (int y = 0; y < g.height; y++) {
                    for (int x = 0; x < g.width; x++) {
                        if (g.cells[(size_t)y * g.width + x] != color_idx) continue;
                        if (has_sel) {
                            const bool is_sel = mask[(size_t)y * g.width + x] != 0;
                            if (is_sel != want_selected) continue;
                        } else if (!want_selected) {
                            continue;
                        }
                        if (!any) { nvgBeginPath(vg); any = true; }
                        nvgRect(vg,
                                place.x + x * cs + lay.cell_pad * 0.5f,
                                place.y + y * cs + lay.cell_pad * 0.5f,
                                cell_draw, cell_draw);
                    }
                }
                if (any) {
                    if (dim) {
                        nvgFillColor(vg, nvgRGB((uint8_t)(rf * kDimFactor),
                                                (uint8_t)(gf * kDimFactor),
                                                (uint8_t)(bf * kDimFactor)));
                    } else {
                        nvgFillColor(vg, nvgRGB(rf, gf, bf));
                    }
                    nvgFill(vg);
                }
            };

            emit_batch(true, false);
            if (has_sel) emit_batch(false, true);
        }

        // One stroked path over this grid's selected cells: the "these ones"
        // outline from the deductive_integration mock.
        if (has_sel) {
            bool any = false;
            for (int y = 0; y < g.height; y++) {
                for (int x = 0; x < g.width; x++) {
                    if (!mask[(size_t)y * g.width + x]) continue;
                    if (!any) { nvgBeginPath(vg); any = true; }
                    nvgRect(vg,
                            place.x + x * cs + lay.cell_pad * 0.5f,
                            place.y + y * cs + lay.cell_pad * 0.5f,
                            cell_draw, cell_draw);
                }
            }
            if (any) {
                nvgStrokeColor(vg, nvgRGB(255, 255, 0));
                nvgStrokeWidth(vg, 1.5f);
                nvgStroke(vg);
            }
        }
    }

    if (lay.divider_y >= 0.0f) {
        nvgBeginPath(vg);
        nvgRect(vg, lay.table_left, lay.divider_y, lay.table_width, kDividerThickness);
        nvgFillColor(vg, nvgRGB(255, 255, 255));
        nvgFill(vg);
    }
}

// -------------------------------------------------------------------- click

// Cell toggle by cursor math against the same layout the draw used. A click
// on the surface but outside every grid clears the selection.
void grid_clicked(flecs::entity surface)
{
    flecs::world world = surface.world();
    const CursorState* cursor = world.lookup("GLFWState").try_get<CursorState>();
    const UIElementBounds* b = surface.try_get<UIElementBounds>();
    flecs::entity ve = view_entity();
    if (!cursor || !b || !ve.is_valid()) return;

    ArcTaskView& view = ve.ensure<ArcTaskView>();
    if (view.task.train.empty() && view.task.test.empty()) return;

    // The grid is the whole panel body; nothing beneath it wants this click.
    world.ensure<UIInputDispatch>().consumed = true;

    const TaskLayout lay = compute_layout(view.task, b->xmin, b->ymin,
                                          b->xmax - b->xmin, b->ymax - b->ymin);
    for (const GridPlace& place : lay.grids) {
        const float lx = (float)cursor->x - place.x;
        const float ly = (float)cursor->y - place.y;
        if (lx < 0.0f || ly < 0.0f) continue;
        const int x = (int)(lx / lay.cs);
        const int y = (int)(ly / lay.cs);
        if (x >= place.g->width || y >= place.g->height) continue;

        const ArcCellRef ref{place.pair_idx, place.is_test, place.is_output, x, y};
        auto it = std::find(view.selection.begin(), view.selection.end(), ref);
        if (it != view.selection.end()) view.selection.erase(it);
        else view.selection.push_back(ref);
        view.server_selection = false;
        view.status = std::to_string(view.selection.size()) + " cells selected";
        view.dirty = true;
        return;
    }

    if (!view.selection.empty()) {
        view.selection.clear();
        view.server_selection = false;
        view.status.clear();
        view.dirty = true;
    }
}

// ------------------------------------------------------------------- solver

// tradewinds' REQ recv has no timeout, so a request sent to a dead server
// wedges the client until restart. The roster knows whether the process exists;
// refuse to send while it does not. This matters twice over for morphology:
// that client is shared with the Lexicon, so wedging it would take a panel down
// with us.
bool server_running(const char* id)
{
    if (!g_world) return false;
    flecs::entity server = servers::find(*g_world, id);
    const ServerState* st = server.is_valid() ? server.try_get<ServerState>() : nullptr;
    return st && st->status != ServerStatus::Offline;
}

bool solver_running()
{
    return server_running(kServerId);
}

// Guarded by tradewinds' immediate reservation, never by has<AwaitResponse>
// -- same reasoning as the lattice: the component guard is deferred and two
// senders could double-send on the REQ socket in one frame.
bool send_solver(std::map<std::string, std::string> payload,
                 void (*handler)(std::map<std::string, msgpack::object>&))
{
    if (!g_world) return false;
    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) return false;
    client.set<SendMapRequest>({std::move(payload)});
    client.set<AwaitResponse>({handler});
    return true;
}

// solver_server.py speaks lowercase "ok"/"error" + "message", unlike the
// entity bridge's "OK"/"OFFLINE" convention.
void nlp_query_response(std::map<std::string, msgpack::object>& res)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    ArcTaskView& view = ve.ensure<ArcTaskView>();
    view.dirty = true;

    const std::string status = field_str(res, "status");
    if (status != "ok") {
        const std::string message = field_str(res, "message");
        view.status = !message.empty() ? message
                     : status.empty() ? "no reply from solver" : status;
        return;
    }

    view.structured = field_str(res, "structured");
    const std::string kind = field_str(res, "kind");

    if (kind == "cells") {
        view.selection.clear();
        view.server_selection = true;
        try {
            auto it = res.find("cells");
            if (it != res.end()) {
                for (const msgpack::object& row : it->second.as<std::vector<msgpack::object>>()) {
                    const auto v = row.as<std::vector<msgpack::object>>();
                    if (v.size() < 5) continue;
                    view.selection.push_back({msg_int(v[0]), msg_int(v[1]) != 0,
                                              msg_int(v[2]) != 0, msg_int(v[3]),
                                              msg_int(v[4])});
                }
            }
        } catch (const std::exception& exc) {
            view.status = std::string("unreadable cells: ") + exc.what();
            return;
        }
        // "every such thing" is the plural doing semantic work: the annotation
        // said polyominoes, so the selection is the whole extension rather than
        // one instance of it.
        view.status = std::to_string(view.selection.size()) + " cells"
                    + (view.inductive ? " -- every such thing" : "");
        return;
    }

    if (kind == "colors") {
        // The query projected the cell set down to its colors; there is
        // nothing to outline, so the answer lives on the status line.
        view.selection.clear();
        view.server_selection = false;
        std::string txt = "colors:";
        try {
            auto it = res.find("values");
            if (it != res.end()) {
                for (const msgpack::object& o : it->second.as<std::vector<msgpack::object>>()) {
                    const int c = msg_int(o);
                    txt += " " + std::to_string(c);
                    if (c >= 0 && c < 10) txt += std::string(" (") + kArcColorNames[c] + ")";
                }
            }
        } catch (const std::exception&) { /* partial list is still an answer */ }
        view.status = txt;
        return;
    }

    view.status = "unknown result kind: " + kind;
}

// The solver's declared vocabulary, imported as entities in Thornfield's own
// world -- one per term, scoped by Context to the task context that declared it.
// This is what lets the annotator bind a typed word to a term rather than
// compare strings: from here on "Sun" is something that exists.
void vocabulary_response(std::map<std::string, msgpack::object>& res)
{
    g_vocab_inflight = false;

    flecs::entity ve = view_entity();
    if (!ve.is_valid() || !g_world) return;
    ArcTaskView& view = ve.ensure<ArcTaskView>();
    view.dirty = true;

    if (field_str(res, "status") != "ok") {
        const std::string message = field_str(res, "message");
        view.status = message.empty() ? "vocabulary request failed" : message;
        return;
    }

    flecs::entity context = context_entity(*g_world);

    // A fresh declaration replaces the old one: the vocabulary is task
    // dependent, so a term the new task does not know must stop being
    // recognised rather than linger.
    std::vector<flecs::entity> stale;
    g_world->query_builder<VocabTerm>().build()
        .each([&](flecs::entity term, VocabTerm&) {
            if (term.has<Context>(context)) stale.push_back(term);
        });
    for (flecs::entity term : stale) term.destruct();

    g_vocab.clear();
    try {
        auto it = res.find("terms");
        if (it == res.end()) { view.status = "no terms in reply"; return; }

        for (const msgpack::object& row : it->second.as<std::vector<msgpack::object>>()) {
            auto entry = row.as<std::map<std::string, msgpack::object>>();

            std::string text, kind;
            int index = -1;
            auto text_it = entry.find("term");
            auto kind_it = entry.find("kind");
            auto index_it = entry.find("index");
            if (text_it != entry.end()) text = text_it->second.as<std::string>();
            if (kind_it != entry.end()) kind = kind_it->second.as<std::string>();
            if (index_it != entry.end()) index = msg_int(index_it->second);
            if (text.empty()) continue;

            // The declarer's index is what makes a colour term wear the colour
            // it selects: index 4 is the same 4 the cells are painted from.
            const uint32_t color = (kind == "color" && index >= 0 && index < 10)
                ? kArcPalette[index] : 0u;

            g_world->entity()
                .child_of(context)          // namespaced: no collision with a word
                .set_name(text.c_str())
                .add<Context>(context)
                .set<VocabTerm>({text, kind, index, color});

            g_vocab.push_back(text);
        }
        view.status = std::to_string(g_vocab.size()) + " terms recognised in this task";
    } catch (const std::exception& exc) {
        view.status = std::string("unreadable vocabulary: ") + exc.what();
    }
}

// ------------------------------------------------- comprehension -> selection

// Sends whatever the comprehension turned out to name. Terms are juxtaposed,
// which query_lang reads as conjunction, so "Sun CompositePoly" selects the
// cells that are both.
void finish_resolution(flecs::entity panel, ArcTaskView& view)
{
    view.resolving.clear();

    std::string query;
    for (const std::string& term : view.resolved_terms) {
        if (!query.empty()) query += " ";
        query += term;
    }

    // What the panel could not read is worth saying plainly: these are the
    // words waiting to be taught, and nothing about them is guessed at here.
    std::string unread;
    for (const std::string& word : view.unresolved_words) {
        if (!unread.empty()) unread += ", ";
        unread += word;
    }

    if (query.empty()) {
        view.status = unread.empty()
            ? "nothing to select on in that comprehension"
            : "not understood: " + unread + " -- demonstrate it in the Lexicon";
        view.structured.clear();
        view.dirty = true;
        return;
    }

    if (!unread.empty())
        view.status = "not understood: " + unread;
    panel.set<ArcQuerySubmit>({query});
}

// One word back from the transducer. A lemma that names a tag resolves the
// word AND marks the query inductive: the plural is the sentence saying it
// means every such thing. A word the transducer cannot reach is left
// unresolved rather than guessed -- inflection is demonstrated in the Lexicon,
// never spelled out here.
void analyze_response(std::map<std::string, msgpack::object>& res)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    ArcTaskView& view = ve.ensure<ArcTaskView>();

    const std::string word = view.resolving;
    const std::string lemma = field_str(res, "lemma");
    const std::string term = lemma.empty() ? std::string() : vocab_term(lemma);

    if (!term.empty()) {
        view.resolved_terms.push_back(term);
        if (lemma != lowered(word)) view.inductive = true;
    } else {
        view.unresolved_words.push_back(word);
    }

    view.resolving.clear();
    if (view.pending_words.empty()) finish_resolution(ve, view);
}

// --------------------------------------------------------- query as syntax

// The parsed query worn as badges: one per term, colour tags in their own
// palette colour so "Zen" reads as the blue it selects, operators as plain
// glyphs between them. This is the utterance shown as the query it became --
// the panel's answer to "did you understand me", rather than a raw string.
void rebuild_query_row(ArcTaskView& view)
{
    if (!view.query_row.is_valid() || !view.query_row.is_alive()) return;
    flecs::world world = view.query_row.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid()) return;

    std::vector<flecs::entity> old;
    view.query_row.children([&](flecs::entity c) { old.push_back(c); });
    for (flecs::entity c : old) c.destruct();

    // Space-joined tags and operators, exactly as query_lang emits them.
    std::vector<std::string> terms;
    {
        std::string current;
        for (char c : view.structured) {
            if (std::isspace((unsigned char)c)) {
                if (!current.empty()) { terms.push_back(current); current.clear(); }
            } else {
                current += c;
            }
        }
        if (!current.empty()) terms.push_back(current);
    }

    for (const std::string& term : terms) {
        const bool is_operator = term == "&" || term == "|" || term == "!" ||
                                 term == "(" || term == ")" || term == "~";
        if (is_operator) {
            world.entity().is_a(UIElement).child_of(view.query_row).add<UIYoga>()
                .set<TextRenderable>({term, "CharisSIL", 16.0f, 0x8A8A8AFF})
                .set<ZIndex>({13});
            continue;
        }

        // A colour term wears the colour it selects; everything else takes the
        // panel's own blue.
        uint32_t color = 0x0074D9FF;
        const std::string lower = lowered(term);
        for (int i = 0; i < 10; i++) {
            if (lower == kArcColorNames[i]) {
                color = kArcPalette[i];
                break;
            }
        }
        create_badge(view.query_row, UIElement, term.c_str(), color);
    }

    view.rendered_query = view.structured;
}

// ------------------------------------------------------------------ context

// The ARC task as something statements can be made In. One entity for the
// panel's context rather than one per task: the task it currently holds is a
// property of the context (its detail), so a conversation scoped here follows
// the panel as it navigates rather than going stale on the id it started with.
flecs::entity context_entity(flecs::world& world)
{
    flecs::entity context = world.entity("ArcTaskContext");
    if (!context.has<ContextInfo>())
        context.set<ContextInfo>({"arc", "ArcTask", "", "a"});
    return context;
}

// Keeps the context's detail honest -- what task a conversation scoped here is
// actually talking about.
void publish_context_detail(flecs::world& world, const std::string& task_id)
{
    flecs::entity context = context_entity(world);
    ContextInfo& info = context.ensure<ContextInfo>();
    if (info.detail != task_id) info.detail = task_id;
}

// -------------------------------------------------------------------- panel

void create_content(flecs::entity leaf, flecs::entity UIElement)
{
    flecs::world world = leaf.world();
    auto canvas = leaf.target<EditorCanvas>();

    world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add<UIYoga>()
    .set<UIFillParent>({})
    // Pure black, not the editor's usual 0x121317 panel ground: the ARC
    // palette is judged against it, and slate (0x555555) is a real cell
    // colour that a tinted ground would muddy.
    .set<RectRenderable>({0.0f, 0.0f, false, 0x000000ff})
    .set<ZIndex>({9});

    // Query terms on top, grids filling the middle, status line below. No
    // input of its own: a query is something you say and then annotate in the
    // Interlocutor, and this panel reads those bindings.
    auto column = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFillParent>({12.0f, 12.0f, 12.0f, 12.0f})
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignStretch, 8.0f})
    .set<ZIndex>({10});

    // The annotation, worn as the query it became: one badge per term. Empty
    // until an annotated stop names something the solver can select on.
    auto query_row = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
    .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                           YGJustifyCenter, YGAlignCenter, 6.0f})
    .set<ZIndex>({13});

    // The grid surface: all of the task's grids drawn by one render function,
    // clicks resolved by the same layout math.
    world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add<UIYoga>()
    .set<UIFlexItem>({1.0f, 1.0f, YGAlignAuto})
    .add<ArcGridSurface>()
    .set<CustomRenderable>({0.0f, 0.0f, false, 0xFFFFFFFF, 0, 0, draw_task})
    .set<CallbackOnLeftClick>({grid_clicked})
    .set<ZIndex>({11});

    auto status_text = world.entity()
    .is_a(UIElement)
    .child_of(column)
    .add<UIYoga>()
    .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
    .set<TextRenderable>({"arc load <task_id> in the Interlocutor loads a task",
                          "CharisSIL", 16.0f, 0x9A9A9AFF})
    .set<ZIndex>({12});

    // A task on screen the moment the panel opens: the first of the dataset,
    // no typing required. `arc load <id>` in the Interlocutor switches it.
    ArcTaskView initial;
    initial.status_text = status_text;
    initial.query_row = query_row;
    if (!dataset_ids().empty()) load_task(initial, dataset_ids().front());
    else initial.status = std::string("no tasks found in ") + kArcDataDir;
    column.set<ArcTaskView>(initial);

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());

    // The panel's identity badge is also the thing you pick up to say where you
    // are speaking: drag it onto a conversation and what is said there is said
    // In this task. One object, so there is no separate "context handle" to
    // learn about.
    create_badge(badges, UIElement, "ArcTask", 0x0074D9FF)
        .set<ContextBadge>({context_entity(world)});
}

} // namespace

namespace panels {

// Left/right arrow navigation, called from main.cpp's key_callback once every
// other claimant on the arrows has passed. Wraps at both ends of the dataset.
bool arc_task_step(int delta)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return false;

    const std::vector<std::string>& ids = dataset_ids();
    if (ids.empty()) return false;

    ArcTaskView& view = ve.ensure<ArcTaskView>();
    const int n = (int)ids.size();
    int next = view.task_index < 0 ? 0 : (view.task_index + delta) % n;
    if (next < 0) next += n;

    load_task(view, ids[next]);
    return true;
}

ArcTask::ArcTask(flecs::world& world)
{
    world.module<ArcTask>();
    g_world = &world;

    world.component<ArcTaskView>();
    world.component<ArcGridSurface>();
    world.component<ArcQuerySubmit>();

    // Sends a parked query to the solver bridge. Parked rather than sent at
    // submit time so a busy client retries next frame instead of dropping;
    // no task or no server resolves to a status line, not a wedged socket.
    world.system<ArcTaskView, ArcQuerySubmit>("ArcQuerySubmitSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity panel, ArcTaskView& view, ArcQuerySubmit& submit) {
        if (view.task_id.empty()) {
            view.status = "load a task first: arc load <task_id>";
            view.dirty = true;
            panel.remove<ArcQuerySubmit>();
            return;
        }
        if (!solver_running()) {
            view.status = "solver offline -- start ARC Solver in Peach Core";
            view.dirty = true;
            panel.remove<ArcQuerySubmit>();
            return;
        }
        if (!send_solver({{"command", "nlp_query"},
                          {"task_id", view.task_id},
                          {"query", submit.query}}, nlp_query_response))
            return;   // client busy; the submit stays parked and retries
        view.status = "querying...";
        view.dirty = true;
        panel.remove<ArcQuerySubmit>();
    });

    // The vocabulary is what lets an utterance be read as a query, so the
    // panel fetches it as soon as the solver is up -- and keeps waiting
    // quietly when it isn't, rather than making the vocabulary something you
    // have to remember to ask for.
    world.system<ArcTaskView>("ArcVocabularySystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, ArcTaskView& view) {
        if (!g_vocab.empty() || g_vocab_inflight) return;
        if (view.task_id.empty()) return;

        // Says so on the panel rather than only in a log nobody is reading:
        // with no vocabulary, a word the solver would recognise stays a plain
        // word, and the reason is never visible in the sentence.
        if (!solver_running()) {
            const std::string want =
                "no vocabulary -- start ARC Solver in Peach Core to recognise terms";
            if (view.status != want) { view.status = want; view.dirty = true; }
            return;
        }
        if (!send_solver({{"command", "vocabulary"}, {"task_id", view.task_id}},
                         vocabulary_response))
            return;                      // client busy: try again next frame
        g_vocab_inflight = true;
        view.status = "reading the solver's vocabulary...";
        view.dirty = true;
    });

    // Words the comprehension named that no tag matched yet, waiting on the
    // morphology transducer -- one lookup per frame, because the morphology
    // client is shared with the Lexicon and a REQ socket carries one question
    // at a time.
    world.system<ArcTaskView>("ArcTermResolveSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity panel, ArcTaskView& view) {
        if (view.pending_words.empty() || !view.resolving.empty()) return;

        flecs::entity client = g_world->lookup(kMorphologyClient);
        if (!client.is_valid() || !server_running(kMorphologyServerId)) {
            // Without the transducer an inflected word simply cannot be read.
            // Reported as unread rather than stripped down by hand: what
            // "polyominoes" reduces to is something the Lexicon teaches.
            for (const std::string& word : view.pending_words)
                view.unresolved_words.push_back(word);
            view.pending_words.clear();
            finish_resolution(panel, view);
            return;
        }
        if (!tradewinds::try_reserve(client)) return;   // Lexicon has it; next frame

        view.resolving = view.pending_words.front();
        view.pending_words.erase(view.pending_words.begin());
        client.set<SendMapRequest>({{{"type", "analyze"}, {"word", view.resolving}}});
        client.set<AwaitResponse>({analyze_response});
    });

    // The grids read the view at draw time, so the refresh pass only has the
    // status line and the query's badges to keep current.
    world.system<ArcTaskView>("ArcStatusSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity panel, ArcTaskView& view) {
        if (!view.dirty) return;
        view.dirty = false;
        flecs::world world = panel.world();
        publish_context_detail(world, view.task_id);
        if (view.status_text.is_valid() && view.status_text.is_alive())
            view.status_text.ensure<TextRenderable>().text = view.status;
        if (view.structured != view.rendered_query) rebuild_query_row(view);
    });

    // The chat side of the interleave: typed lines starting with "arc" reach
    // here through functions::dispatch_text before the interlocutor sees
    // them, and the Result::message comes back as a transcript reply.
    functions::register_function({
        "arc",
        "ARC task control: 'load <task_id>' loads a task into the ArcTask panel, "
        "'schema' lists the query tags, anything else runs as a cell query "
        "whose matches highlight in the panel.",
        {{"line", "string", "load <task_id> | schema | a cell query", true}},
        [](const functions::Args& args) -> functions::Result {
            auto found = args.find("line");
            std::string line = found == args.end() ? "" : found->second;
            const size_t b = line.find_first_not_of(" \t");
            line = b == std::string::npos ? "" : line.substr(b);

            flecs::entity ve = view_entity();
            if (!ve.is_valid())
                return {false, "open an ArcTask panel first"};
            ArcTaskView& view = ve.ensure<ArcTaskView>();

            const size_t sp = line.find(' ');
            const std::string head = line.substr(0, sp);
            std::string tail = sp == std::string::npos ? "" : line.substr(sp + 1);
            const size_t tb = tail.find_first_not_of(" \t");
            tail = tb == std::string::npos ? "" : tail.substr(tb);

            if (head == "load") {
                if (tail.empty()) return {false, "arc load <task_id>"};
                if (!load_task(view, tail)) return {false, view.status};
                return {true, "loaded " + view.task_id + " -- " +
                              std::to_string(view.task.train.size()) + " train, " +
                              std::to_string(view.task.test.size()) + " test"};
            }
            if (head == "schema") {
                if (!solver_running())
                    return {false, "solver offline -- start ARC Solver in Peach Core"};
                if (!send_solver({{"command", "vocabulary"}, {"task_id", view.task_id}},
                                 vocabulary_response))
                    return {false, "solver client busy -- try again"};
                return {true, "schema requested; tags land in the panel status line"};
            }
            if (line.empty())
                return {false, "arc load <task_id> | arc schema | arc <query>"};

            ve.set<ArcQuerySubmit>({line});
            if (view.task_id.empty())
                return {true, "querying (no task loaded yet): " + line};
            return {true, "querying " + view.task_id + ": " + line};
        },
        nullptr,
    });

    // The panel as something a query can be RUN AGAINST. The Interlocutor's Q
    // gesture hands over the comprehension's words; this decides what they
    // name here. Three ways a word can resolve, and every one of them is
    // learned or served rather than written down: it names a tag in the
    // solver's own vocabulary, or the morphology transducer (taught in the
    // Lexicon) reduces it to one that does. A word neither reaches is reported,
    // not guessed.
    query_contexts::register_context({
        "arc", "ARC Task",

        // Ready only with a panel open, a task in it, and a vocabulary to read
        // words against -- so Q says which of those is missing instead of
        // silently selecting nothing.
        [] {
            flecs::entity ve = view_entity();
            if (!ve.is_valid()) return false;
            const ArcTaskView* view = ve.try_get<ArcTaskView>();
            return view && !view->task_id.empty() && !g_vocab.empty();
        },

        [](const query_contexts::Comprehension& query) -> query_contexts::Result {
            flecs::entity ve = view_entity();
            if (!ve.is_valid()) return {false, "no ArcTask panel open"};

            ArcTaskView& view = ve.ensure<ArcTaskView>();
            if (view.task_id.empty()) return {false, "no ARC task loaded"};
            if (g_vocab.empty())
                return {false, "the solver's vocabulary is not loaded yet "
                               "(start ARC Solver in Peach Core)"};

            view.resolved_terms.clear();
            view.unresolved_words.clear();
            view.pending_words.clear();
            view.inductive = false;

            // Roles are read as a flat conjunction here because jane_eyre's
            // query language is flat: Above is a prepositional entailment
            // computed over cells, not a two-place predicate. The roles arrive
            // intact all the same, so a context that can express direction --
            // or this one, once the solver can -- has them without the
            // annotation changing.
            for (const std::string& word : query.words()) {
                const std::string term = vocab_term(word);
                if (!term.empty()) view.resolved_terms.push_back(term);
                else view.pending_words.push_back(word);   // ask the transducer
            }

            if (view.pending_words.empty()) {
                finish_resolution(ve, view);
            } else {
                view.status = "reading " + std::to_string(view.pending_words.size()) +
                              " word(s) through morphology...";
                view.dirty = true;
            }
            return {true, "running against " + view.task_id};
        },
    });

    register_panel({EditorType::ArcTask, "arc_task", "ArcTask", create_content});
}

} // namespace panels
