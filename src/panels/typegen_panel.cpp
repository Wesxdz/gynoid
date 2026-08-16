#include "typegen_panel.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include "../components.h"
#include "../tradewinds.h"
#include "../capabilities/ui_kit.h"
#include "../capabilities/ui_metrics.h"
#include "registry.h"

using namespace tradewinds;

namespace {

// Shared with the Entities and Lattice panels; no client of this panel's own.
constexpr const char* kClientName = "EntityQueryClient";

constexpr uint32_t kSlotColor   = 0x5b8bd0ff;   // pinned entity slots
constexpr uint32_t kRelColor    = 0xa34d1aff;   // relation, the Interlocutor's brown
constexpr uint32_t kChipOn      = 0x8f7fd0ff;   // the header badge

// The seven adornments of a triple: which slots are pinned (see the memory of
// type_generalization.png). Order here is the chip order.
enum class Adorn { Triple, Ego, Outbound, Bridge, Inbound, Alter, Predicate };

// icon names live in assets/types/. Row order is Wesley's reading order:
// the relation alone, the grounded fact, then the four one-wild views paired
// by direction, then the two ends alone.
struct AdornDef { Adorn adorn; const char* name; const char* icon; bool src, rel, tgt; };
constexpr AdornDef kAdorns[] = {
    {Adorn::Predicate, "Predicate", "predicate", false, true,  false},
    {Adorn::Triple,    "Triple",    "triple",    true,  true,  true},
    {Adorn::Outbound,  "Outbound",  "outbound",  true,  true,  false},
    {Adorn::Bridge,    "Bridge",    "bridge",    true,  false, true},
    {Adorn::Inbound,   "Inbound",   "inbound",   false, true,  true},
    {Adorn::Ego,       "Ego",       "ego",       true,  false, false},
    {Adorn::Alter,     "Alter",     "alter",     false, false, true},
};

// Relations that are ECS plumbing rather than world knowledge; rows keep them
// out of every extension listing.
bool plumbing_relation(const std::string& rel)
{
    return rel == "ChildOf" || rel == "Identifier" || rel == "IsA"
        || rel.rfind("flecs", 0) == 0;
}

// The state of one TypeGen panel.
struct TypeGenView
{
    std::string src, rel, tgt;   // the seeded triple
    std::string tracked;         // last followed "src\trel\ttgt"

    // One snapshot of every pair in the world ((*, *) deduped); every
    // adornment is computed from it client-side, so all seven show at once.
    struct Row { std::string name; std::vector<std::pair<std::string, std::string>> pairs; };
    std::vector<Row> rows;

    std::string status;
    bool dirty = true;

    // Which relationship role the mouse is over in the Interlocutor (0 src,
    // 1 rel, 2 tgt, -1 none). Hovering a slot narrows the table to the
    // adornments that pin that slot -- the panel answers the hover.
    int hover = -1;
};

flecs::world* g_world = nullptr;

flecs::entity view_entity()
{
    flecs::entity found;
    if (!g_world) return found;
    g_world->query<TypeGenView>().each([&](flecs::entity e, TypeGenView&) {
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

bool request_snapshot()
{
    if (!g_world) return false;
    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) return false;
    client.set<SendMapRequest>({{{"type", "query"}, {"expr", "(*, *)"}}});
    client.set<AwaitResponse>({+[](std::map<std::string, msgpack::object>& res) {
        flecs::entity ve = view_entity();
        if (!ve.is_valid()) return;
        TypeGenView& view = ve.ensure<TypeGenView>();
        view.rows.clear();
        view.dirty = true;

        const std::string status = field_str(res, "status");
        if (status != "OK") {
            view.status = status == "OFFLINE"
                ? "world offline -- " + field_str(res, "error")
                : (status.empty() ? "no reply" : status + ": " + field_str(res, "error"));
            return;
        }
        view.status.clear();

        try {
            std::vector<std::string> names;
            std::vector<std::vector<std::string>> relations;
            auto names_it = res.find("names");
            auto rels_it = res.find("relations");
            if (names_it != res.end()) names = names_it->second.as<std::vector<std::string>>();
            if (rels_it != res.end())
                relations = rels_it->second.as<std::vector<std::vector<std::string>>>();

            // Wildcard fan-out: one row per pair, full relation list on each
            // copy -- first copy per entity wins.
            std::set<std::string> seen;
            const size_t rows = std::min(names.size(), relations.size());
            for (size_t i = 0; i < rows; i++) {
                if (!seen.insert(names[i]).second) continue;
                TypeGenView::Row row;
                row.name = names[i];
                for (const std::string& rel : relations[i]) {
                    const auto tab = rel.find('\t');
                    if (tab == std::string::npos) continue;
                    const std::string r = rel.substr(0, tab);
                    if (plumbing_relation(r)) continue;
                    row.pairs.push_back({r, rel.substr(tab + 1)});
                }
                if (!row.pairs.empty()) view.rows.push_back(std::move(row));
            }
        } catch (const std::exception& exc) {
            view.status = std::string("unreadable snapshot: ") + exc.what();
        }
    }});
    return true;
}

// One adornment's extension: every triple in the snapshot whose pinned slots
// match the seed. total reports the uncapped count; the listing is capped so
// seven sections can't bury the panel.
std::vector<std::array<std::string, 3>> extension(const TypeGenView& view,
                                                  const AdornDef& def,
                                                  size_t cap, size_t* total)
{
    std::vector<std::array<std::string, 3>> out;
    size_t count = 0;
    for (const TypeGenView::Row& row : view.rows) {
        if (def.src && row.name != view.src) continue;
        for (const auto& [r, t] : row.pairs) {
            if (def.rel && r != view.rel) continue;
            if (def.tgt && t != view.tgt) continue;
            count++;
            if (out.size() < cap) out.push_back({row.name, r, t});
        }
    }
    if (total) *total = count;
    return out;
}

// Rebuilds the whole panel content. Everything here is display -- there are
// no inputs to preserve -- so a wholesale rebuild keeps the code flat.
void rebuild(flecs::entity content, TypeGenView& view)
{
    flecs::world world = content.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid()) return;

    std::vector<flecs::entity> old;
    content.children([&](flecs::entity c) { old.push_back(c); });
    for (flecs::entity c : old) c.destruct();

    auto row = [&](float gap) {
        return world.entity()
            .is_a(UIElement)
            .child_of(content)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                                   YGJustifyCenter, YGAlignCenter, gap})
            .set<ZIndex>({12});
    };

    if (view.src.empty()) {
        world.entity().is_a(UIElement).child_of(row(4.0f)).add<UIYoga>()
            .set<TextRenderable>({"select a relationship in the Interlocutor",
                                  "CharisSIL", 16.0f, 0x6A6A6AFF})
            .set<ZIndex>({13});
        return;
    }

    // No seed row: the Triple adornment below IS the fully-pinned seed.
    if (!view.status.empty()) {
        world.entity().is_a(UIElement).child_of(row(4.0f)).add<UIYoga>()
            .set<TextRenderable>({view.status, "CharisSIL", 16.0f, 0x9A7A7AFF})
            .set<ZIndex>({13});
        return;
    }

    // Every adornment at once, as a table: sprite | name | src | rel | tgt |
    // count. Column-major, because Yoga has no grid: each column is a stack,
    // a row is the same index in every stack, and a shared fixed cell height
    // keeps the rows level -- so column spacing is genuinely shared, sized by
    // each column's widest cell. Slots are the entity's EMIST chip alone (the
    // shorthand id carries the identity; the name lives in the Interlocutor),
    // and a wild slot wears the wildcard-box sprite.
    constexpr float kCellHeight = 30.0f;

    auto table = world.entity()
        .is_a(UIElement)
        .child_of(content)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyFlexStart, YGAlignFlexStart, 16.0f})
        .set<ZIndex>({12});

    auto column = [&](YGAlign align) {
        return world.entity()
            .is_a(UIElement)
            .child_of(table)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                                   YGJustifyFlexStart, align, 0.0f})
            .set<ZIndex>({12});
    };
    auto cell = [&](flecs::entity col) {
        return world.entity()
            .is_a(UIElement)
            .child_of(col)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<UISize>({YGUndefined, kCellHeight})
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                   YGJustifyCenter, YGAlignCenter, 0.0f})
            .set<ZIndex>({12});
    };

    // Left-aligned, not centered: alter.png is wider than its siblings, and
    // centering would stagger the shared left edge the eye reads down.
    auto icon_col  = column(YGAlignFlexStart);
    auto name_col  = column(YGAlignFlexStart);
    auto src_col   = column(YGAlignCenter);
    auto rel_col   = column(YGAlignCenter);
    auto tgt_col   = column(YGAlignCenter);
    auto count_col = column(YGAlignFlexEnd);

    // Header row: one cell per column, same fixed height as every data cell
    // so the first data row stays level across columns.
    auto header_text = [&](flecs::entity col, const char* label) {
        world.entity().is_a(UIElement).child_of(cell(col))
            .add<UIYoga>()
            .set<TextRenderable>({label, "CharisSIL", 16.0f, 0x777777FF})
            .set<ZIndex>({13});
    };
    cell(icon_col);                 // empty corner under the sprite column
    header_text(name_col, "Pattern");
    header_text(src_col, "src");
    header_text(rel_col, "rel");
    header_text(tgt_col, "tgt");
    header_text(count_col, "ext");

    auto slot_cell = [&](flecs::entity col, bool pinned,
                         const std::string& name, uint32_t fallback) {
        auto c = cell(col);
        if (!pinned) {
            world.entity().is_a(UIElement).child_of(c)
                .add<UIYoga>()
                .add<UINativeImageSize>()
                .set<ImageCreator>({"types/wildcard_box.png", 1.0f, 1.0f})
                .set<ZIndex>({13});
            return;
        }
        std::string symbol;
        uint32_t color = fallback;
        if (!entity_screen_style(name, &symbol, &color)) {
            // No bound identity: the first letter stands in, tinted with the
            // slot's role colour, same convention as the triple blocks.
            symbol = name.substr(0, 1);
            color = fallback;
        }
        // Identity register: the same uppercase sprite the entity's badge
        // wears in the Interlocutor, slightly shrunk for table density.
        create_letter_chip(c, symbol, color, /*preserve_case=*/false, 0.72f);
    };

    for (const AdornDef& def : kAdorns) {
        // Hover narrows: only the adornments that pin the hovered slot.
        if (view.hover == 0 && !def.src) continue;
        if (view.hover == 1 && !def.rel) continue;
        if (view.hover == 2 && !def.tgt) continue;

        size_t total = 0;
        extension(view, def, 0, &total);

        auto icon_cell = cell(icon_col);
        // Icons center within a rail-box the width of the standard glyphs
        // (outbound et al are 24px), so the smaller triple/predicate sit
        // centered relative to their siblings. alter.png is wider than the
        // box; its cell stays auto so it grows rightward off the same rail.
        if (def.adorn != Adorn::Alter)
            icon_cell.set<UISize>({24.0f, kCellHeight});
        world.entity().is_a(UIElement).child_of(icon_cell)
            .add<UIYoga>()
            .add<UINativeImageSize>()
            .set<ImageCreator>({std::string("types/") + def.icon + ".png", 1.0f, 1.0f})
            .set<ZIndex>({13});

        world.entity().is_a(UIElement).child_of(cell(name_col))
            .add<UIYoga>()
            .set<TextRenderable>({def.name, "CharisSIL", 16.0f, 0xCCCCCCFF})
            .set<ZIndex>({13});

        slot_cell(src_col, def.src, view.src, kSlotColor);
        slot_cell(rel_col, def.rel, view.rel, kRelColor);
        slot_cell(tgt_col, def.tgt, view.tgt, kSlotColor);

        world.entity().is_a(UIElement).child_of(cell(count_col))
            .add<UIYoga>()
            .set<TextRenderable>({std::to_string(total), "CharisSIL", 16.0f,
                                  total ? 0x8f7fd0FF : 0x555555FF})
            .set<ZIndex>({13});
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
    .set<RectRenderable>({0.0f, 0.0f, false, 0x121317ff})
    .set<ZIndex>({9});

    TypeGenView initial;
    const std::string t = annotator_selected_triple();
    if (!t.empty()) {
        const auto tab1 = t.find('\t');
        const auto tab2 = t.rfind('\t');
        initial.src = t.substr(0, tab1);
        initial.rel = t.substr(tab1 + 1, tab2 - tab1 - 1);
        initial.tgt = t.substr(tab2 + 1);
        initial.tracked = t;
    }

    auto content = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFillParent>({12.0f, 12.0f, 12.0f, 12.0f})
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifyFlexStart, YGAlignStretch, 10.0f})
    .set<TypeGenView>(initial)
    .set<ZIndex>({11});
    (void)content;

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());
    create_badge(badges, UIElement, "TypeGen", kChipOn);

    if (!initial.src.empty()) request_snapshot();
}

} // namespace

namespace panels {

TypeGen::TypeGen(flecs::world& world)
{
    world.module<TypeGen>();
    g_world = &world;

    world.component<TypeGenView>();

    // Follows the annotator's selected relationship; a new triple reseeds the
    // panel and refetches the snapshot. Same tracked-vs-focus discipline as
    // the Lattice: only a *change* of selection pulls the panel.
    world.system<TypeGenView>("TypeGenFollowSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, TypeGenView& view) {
        // The narrowing role: the mouse wins while it is over a slot, and the
        // annotation cursor's own position carries it otherwise -- selecting
        // the src badge narrows exactly like hovering it.
        int narrow = annotator_hovered_slot();
        if (narrow < 0) narrow = annotator_selected_slot();
        if (narrow != view.hover) {
            view.hover = narrow;
            view.dirty = true;
        }

        const std::string t = annotator_selected_triple();
        if (t.empty() || t == view.tracked) return;
        if (!request_snapshot()) return;   // client busy: retry next frame
        const auto tab1 = t.find('\t');
        const auto tab2 = t.rfind('\t');
        view.src = t.substr(0, tab1);
        view.rel = t.substr(tab1 + 1, tab2 - tab1 - 1);
        view.tgt = t.substr(tab2 + 1);
        view.tracked = t;
        view.status = "loading...";
        view.dirty = true;
    });

    world.system<TypeGenView>("TypeGenRebuildSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity content, TypeGenView& view) {
        if (!view.dirty) return;
        view.dirty = false;
        rebuild(content, view);
    });

    register_panel({EditorType::TypeGen, "typegen", "TypeGen", create_content});
}

} // namespace panels
