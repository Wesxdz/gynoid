#include "lattice_panel.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
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

// The Entities panel's client, created centrally in main.cpp against the
// entity_query server registered in servers.json -- the Peach Core owns the
// server roster, and this panel adds no client and no socket of its own.
// Sharing is safe: every request installs its own handler, and both callers
// retry rather than clobber when the client is mid-request.
constexpr const char* kClientName = "EntityQueryClient";

constexpr uint32_t kSuperColor    = 0x8f7fd0ff;   // above: what subsumes the focus
constexpr uint32_t kFocusColor    = 0xe66c25ff;
constexpr uint32_t kSubColor      = 0x619393ff;   // below: what the focus subsumes
constexpr uint32_t kInstanceColor = 0x84a05aff;   // bottom: individuals IsA the focus

// The state of one Lattice panel. Lives on the panel's own node so it dies with
// the canvas when the panel type changes.
struct LatticeView
{
    std::string focus;
    // The interlocutor badge the panel last followed. Kept apart from focus so
    // clicking around the lattice does not fight the annotator: a badge pulls
    // focus only when it *changes*, and navigation in between is left alone.
    std::string tracked;
    std::vector<std::string> supers;    // direct supertypes (focus SubtypeOf these)
    std::vector<std::string> subs;      // direct subtypes (these SubtypeOf focus)
    std::vector<std::string> instances; // individuals IsA the focus (its extension)
    std::string status;
    bool dirty = true;                 // rebuild the diagram
    int stage = 0;                     // 0 idle, 1 awaiting lattice

    // name->symbol bindings the current badges were drawn with. Compared each
    // frame so a binding created in the Interlocutor restyles the lattice the
    // moment it exists, not on the next navigation.
    std::string style_sig;
};

// A drawn node. rank orders the three bands vertically; name is what clicking
// it navigates to.
struct LatticeNode
{
    std::string name;
    int rank = 0;                      // -1 super, 0 focus, +1 sub
};

// The element that draws the covering edges. One CustomRenderable rather than a
// LineRenderable per edge, because an edge's endpoints are two *other* entities'
// laid-out bounds, which are only known at draw time.
struct LatticeEdges {};

// A persistent band of the diagram. The bands (and the entry shell between
// supers and focus) are created once; rebuilds only replace their contents,
// so the shell keeps its draft and focus across every refresh.
struct LatticeBand { int rank = 0; };   // -1 supers, 0 focus, +1 subs

// The panel node holding the LatticeView, so handlers can find it again. A
// tradewinds response carries no context back, so the view is recovered by
// query -- the same shape entity_query_response uses in main.cpp.
flecs::world* g_world = nullptr;

flecs::entity view_entity()
{
    flecs::entity found;
    if (!g_world) return found;
    g_world->query<LatticeView>().each([&](flecs::entity e, LatticeView&) {
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

// Guarded by tradewinds' immediate reservation, never by has<AwaitResponse>:
// the component is deferred, so two panels sharing this client could both
// pass a component guard in one frame and double-send on the REQ socket. The
// reservation is released when a reply lands, before its handler runs, so a
// handler that chains a follow-up request re-reserves like anyone else.
bool request(const std::string& expr, void (*handler)(std::map<std::string, msgpack::object>&))
{
    if (!g_world) return false;
    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) return false;
    client.set<SendMapRequest>({{{"type", "query"}, {"expr", expr}}});
    client.set<AwaitResponse>({handler});
    return true;
}

void instances_response(std::map<std::string, msgpack::object>& res);

// Every entity carrying a SubtypeOf edge; one reply carries the whole
// neighbourhood. The focus's supertypes are read off its own row, and its
// subtypes are the rows whose edges point back at it. Asked this way rather
// than embedding the focus in an expr, for two reasons: a flecs expr like
// "rat" matches entities TAGGED rat (the extension), never the type entity
// NAMED rat that owns the (SubtypeOf, rodent) pair -- and intensional names
// like "cat+domesticated" would break expr parsing entirely.
constexpr const char* kSupersExpr = "(SubtypeOf, *)";

// Supertypes come from the focus's own SubtypeOf pairs, subtypes from every
// row whose SubtypeOf points back at the focus -- both read out of the one
// (SubtypeOf, *) reply. Relations arrive encoded as "relation\ttarget"; see
// describe() in scripts/entity_query.py. SubtypeOf is the lattice's one
// stored direction; IsA stays reserved for entity-instance-of-Type.
void supers_response(std::map<std::string, msgpack::object>& res)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    LatticeView& view = ve.ensure<LatticeView>();
    view.supers.clear();
    view.subs.clear();

    const std::string status = field_str(res, "status");
    if (status != "OK") {
        view.stage = 0;
        if (status == "OFFLINE") {
            // Not fatal: the world may simply not be up yet.
            view.status = "world offline -- " + field_str(res, "error");
        } else {
            view.status = status.empty() ? "no reply"
                                         : status + ": " + field_str(res, "error");
        }
        view.dirty = true;
        return;
    }

    try {
        std::vector<std::string> names;
        std::vector<std::vector<std::string>> relations;
        auto names_it = res.find("names");
        auto rels_it = res.find("relations");
        if (names_it != res.end()) names = names_it->second.as<std::vector<std::string>>();
        if (rels_it != res.end())
            relations = rels_it->second.as<std::vector<std::vector<std::string>>>();

        // A wildcard query yields one row PER PAIR, so an entity with two
        // SubtypeOf edges arrives twice, each copy carrying the full relation
        // list -- process each entity once or everything doubles.
        std::set<std::string> seen;
        const size_t rows = std::min(names.size(), relations.size());
        for (size_t i = 0; i < rows; i++) {
            if (!seen.insert(names[i]).second) continue;
            const bool is_focus = names[i] == view.focus;
            for (const std::string& rel : relations[i]) {
                const auto tab = rel.find('\t');
                if (tab == std::string::npos) continue;
                if (rel.compare(0, tab, "SubtypeOf") != 0) continue;
                const std::string target = rel.substr(tab + 1);
                if (is_focus) view.supers.push_back(target);
                else if (target == view.focus) view.subs.push_back(names[i]);
            }
        }
    } catch (const std::exception& exc) {
        view.status = std::string("unreadable relations: ") + exc.what();
    }

    view.status.clear();
    // Second hop: the focus's extension -- expr "<focus>" matches exactly the
    // entities tagged with it, which IS instance-of. An intensional name
    // can't ride an expr, so a meet focus skips the hop.
    if (view.focus.find('+') == std::string::npos) {
        view.stage = 2;
        request(view.focus, instances_response);
    } else {
        view.instances.clear();
        view.stage = 0;
        view.dirty = true;
    }
}

// The focus's individuals. Errors leave the band empty rather than alarming
// anyone -- the extension is dressing on the lattice, not its skeleton.
void instances_response(std::map<std::string, msgpack::object>& res)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    LatticeView& view = ve.ensure<LatticeView>();
    view.instances.clear();
    view.stage = 0;
    view.dirty = true;

    if (field_str(res, "status") != "OK") return;
    try {
        auto it = res.find("names");
        if (it == res.end()) return;
        for (const std::string& n : it->second.as<std::vector<std::string>>())
            if (n != view.focus) view.instances.push_back(n);
    } catch (const std::exception&) { /* leave empty */ }
}

// Reply to a manual "focus IsA <name>" annotation. On success the focus is
// reloaded in place, so the new supertype appears in the band above --
// clicking it then navigates, exactly like any other neighbour.
void supertype_response(std::map<std::string, msgpack::object>& res)
{
    flecs::entity ve = view_entity();
    if (!ve.is_valid()) return;
    LatticeView& view = ve.ensure<LatticeView>();

    const std::string status = field_str(res, "status");
    if (status != "OK") {
        view.status = (status.empty() ? "no reply" : status) + ": " + field_str(res, "error");
        view.dirty = true;
        return;
    }

    view.supers.clear();
    view.subs.clear();
    view.status = "loading...";
    view.stage = 1;
    view.dirty = true;
    request(kSupersExpr, supers_response);
}

bool send_supertype(const std::string& sub, const std::string& sup,
                    const std::string& requires_csv)
{
    if (!g_world) return false;
    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) return false;
    client.set<SendMapRequest>({{{"type", "supertype"}, {"sub", sub}, {"sup", sup},
                                 {"requires", requires_csv}}});
    client.set<AwaitResponse>({supertype_response});
    return true;
}

// "pet + domesticated + owned" -> {"pet", "domesticated", "owned"}: the head
// names the other end of the edge, the rest are its differentia.
std::vector<std::string> split_conjunction(const std::string& text)
{
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t plus = text.find('+', pos);
        std::string part = text.substr(pos, plus == std::string::npos
                                            ? std::string::npos : plus - pos);
        const size_t b = part.find_first_not_of(" \t");
        if (b != std::string::npos) {
            const size_t e = part.find_last_not_of(" \t");
            parts.push_back(part.substr(b, e - b + 1));
        }
        if (plus == std::string::npos) break;
        pos = plus + 1;
    }
    return parts;
}

std::string join_csv(const std::vector<std::string>& parts, size_t from)
{
    std::string csv;
    for (size_t i = from; i < parts.size(); i++) {
        if (!csv.empty()) csv += ",";
        csv += parts[i];
    }
    return csv;
}

bool send_instance(const std::string& kind, const std::string& name)
{
    if (!g_world) return false;
    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) return false;
    client.set<SendMapRequest>({{{"type", "instance"}, {"kind", kind}, {"name", name}}});
    client.set<AwaitResponse>({supertype_response});
    return true;
}

// An intension-based subtype of `base`: the anonymous concept base ⊓ props,
// materialized under its canonical '+'-joined name with SubtypeOf edges to
// every part. Same refresh handler as any other edit.
bool send_meet(const std::string& base, const std::string& props_csv)
{
    if (!g_world) return false;
    flecs::entity client = g_world->lookup(kClientName);
    if (!tradewinds::try_reserve(client)) return false;
    client.set<SendMapRequest>({{{"type", "meet"}, {"base", base}, {"props", props_csv}}});
    client.set<AwaitResponse>({supertype_response});
    return true;
}

// Moves the focus to `name` and starts the two-hop reload. The view is only
// touched when the request actually goes out: committing first and dropping
// the request would leave the panel saying "loading..." for a reply that was
// never asked for.
bool navigate(LatticeView& view, const std::string& name)
{
    if (name == view.focus) return true;
    if (!request(kSupersExpr, supers_response)) return false;
    view.focus = name;
    view.supers.clear();
    view.subs.clear();
    view.status = "loading...";
    view.stage = 1;
    view.dirty = true;
    return true;
}

void refocus(flecs::entity node)
{
    const LatticeNode* info = node.try_get<LatticeNode>();
    flecs::entity ve = view_entity();
    if (!info || !ve.is_valid()) return;
    navigate(ve.ensure<LatticeView>(), info->name);
}

// Covering edges, drawn from the laid-out bounds of the nodes themselves. Read
// at draw time because that is the only moment both ends are known.
void draw_edges(NVGcontext* vg, const RenderCommand*, const CustomRenderable& r)
{
    if (!g_world) return;

    UIElementBounds focus_box{0, 0, 0, 0};
    bool have_focus = false;
    std::vector<UIElementBounds> supers, subs, instances;

    g_world->query<const LatticeNode, const UIElementBounds>()
        .each([&](flecs::entity, const LatticeNode& node, const UIElementBounds& b) {
            if (b.xmax <= b.xmin) return;    // not laid out yet
            if (node.rank < 0)       supers.push_back(b);
            else if (node.rank == 1) subs.push_back(b);
            else if (node.rank > 1)  instances.push_back(b);
            else { focus_box = b; have_focus = true; }
        });
    if (!have_focus) return;

    const float fx = (focus_box.xmin + focus_box.xmax) * 0.5f;

    // Bare curves: an edge carries no conditions any more -- a conditional
    // subsumption is stored as an intensional node, never as a dressed edge.
    nvgStrokeWidth(vg, 1.0f);
    auto edge = [&](const UIElementBounds& b, float from_y, float to_y, uint32_t rgba) {
        nvgBeginPath(vg);
        nvgMoveTo(vg, fx, from_y);
        // A gentle S-curve rather than a straight line: with several nodes on a
        // rank the straight versions converge into an unreadable fan.
        const float mx = (b.xmin + b.xmax) * 0.5f;
        const float mid = (from_y + to_y) * 0.5f;
        nvgBezierTo(vg, fx, mid, mx, mid, mx, to_y);
        nvgStrokeColor(vg, nvgRGBA((rgba >> 24) & 0xFF, (rgba >> 16) & 0xFF,
                                   (rgba >> 8) & 0xFF, 0x99));
        nvgStroke(vg);
    };

    for (const UIElementBounds& b : supers)    edge(b, focus_box.ymin, b.ymax, kSuperColor);
    for (const UIElementBounds& b : subs)      edge(b, focus_box.ymax, b.ymin, kSubColor);
    for (const UIElementBounds& b : instances) edge(b, focus_box.ymax, b.ymin, kInstanceColor);
}

// Refills the three persistent bands. Cheap enough to do wholesale: a rebuild
// only happens when the focus moves, a reply lands, or a badge identity
// changes. The bands themselves (and the entry shell between them) are never
// destroyed here -- only their contents are.
void rebuild(flecs::entity panel, LatticeView& view)
{
    flecs::world world = panel.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid()) return;

    flecs::entity above, middle, below, ground;
    panel.children([&](flecs::entity c) {
        const LatticeBand* band = c.try_get<LatticeBand>();
        if (!band) return;
        if (band->rank < 0)       above = c;
        else if (band->rank == 1) below = c;
        else if (band->rank > 1)  ground = c;
        else                      middle = c;
        std::vector<flecs::entity> old;
        c.children([&](flecs::entity child) { old.push_back(child); });
        for (flecs::entity child : old) child.destruct();
    });
    if (!above.is_valid() || !middle.is_valid() || !below.is_valid()) return;

    auto node = [&](flecs::entity parent, const std::string& name, int rank, uint32_t color) {
        // An entity with an on-screen identity wears it here too: same colour,
        // same symbol chip -- the chip IS the entity's shorthand id, and a
        // lattice recolouring it would break the one-glance link back to the
        // sentence. Pure types (rodent, mammal) have no binding and keep the
        // band's rank colour instead.
        // A single-part badge, worn with its on-screen identity when it has
        // one -- the exact call the Interlocutor makes for an annotated
        // entity, so the two badges are visually the same object.
        auto part_badge = [&](flecs::entity into, const std::string& part) {
            std::string symbol;
            uint32_t identity = 0;
            if (entity_screen_style(part, &symbol, &identity))
                return create_badge(into, UIElement, part.c_str(), identity,
                                    false, false, symbol, "", 0, identity);
            return create_badge(into, UIElement, part.c_str(), color);
        };

        flecs::entity badge;
        if (name.find('+') != std::string::npos) {
            // An intensional concept is its parts, so it renders as them: the
            // member badges grouped in one outlined block, the way the query
            // frame contains its members -- each part keeps its own identity
            // colour and chip. The group as a whole is the clickable node.
            badge = world.entity()
                .is_a(UIElement)
                .child_of(parent)
                .add(flecs::OrderedChildren)
                .add<UIYoga>()
                .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                                       YGJustifyCenter, YGAlignCenter, 4.0f})
                .set<UIPadding>({3.0f, 5.0f, 3.0f, 5.0f})
                .set<RoundedRectRenderable>({0.0f, 0.0f, 6.0f, true,
                                             (color & 0xFFFFFF00) | 0xAA})
                .set<ZIndex>({12});
            for (const std::string& part : split_conjunction(name))
                part_badge(badge, part);
        } else {
            badge = part_badge(parent, name);
        }

        badge.set<LatticeNode>({name, rank});
        // Only type neighbours navigate: the focus is already here, and an
        // individual has no subsumption neighbourhood to show yet.
        if (rank == -1 || rank == 1) badge.set<CallbackOnLeftClick>({refocus});
        return badge;
    };

    // Above: what subsumes the focus. Aligned to the band's bottom so the
    // supertypes sit as close to the focus as their band allows -- rank is
    // meant to read as height in the order.
    // No focus yet: the panel waits for a badge, and says so instead of
    // pretending to load something.
    if (view.focus.empty()) {
        world.entity().is_a(UIElement).child_of(middle).add<UIYoga>()
            .set<TextRenderable>({"select a badge in the Interlocutor",
                                  "CharisSIL", 16.0f, 0x6A6A6AFF})
            .set<ZIndex>({13});
        return;
    }

    for (const std::string& s : view.supers) node(above, s, -1, kSuperColor);
    node(middle, view.focus, 0, kFocusColor);
    for (const std::string& s : view.subs) node(below, s, 1, kSubColor);
    if (ground.is_valid())
        for (const std::string& s : view.instances) node(ground, s, 2, kInstanceColor);

    // Status only when there is something to say; an empty lattice is a fact
    // about the world, not an error.
    if (!view.status.empty()) {
        world.entity().is_a(UIElement).child_of(middle).add<UIYoga>()
            .set<TextRenderable>({view.status, "CharisSIL", 16.0f, 0x9A7A7AFF})
            .set<ZIndex>({13});
    } else if (view.supers.empty() && view.subs.empty()) {
        world.entity().is_a(UIElement).child_of(middle).add<UIYoga>()
            .set<TextRenderable>({"no SubtypeOf neighbours", "CharisSIL", 16.0f, 0x6A6A6AFF})
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

    // Edges under the nodes, so a curve passes behind a badge rather than
    // across its face.
    world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add<UIYoga>()
    .set<UIFillParent>({})
    .add<LatticeEdges>()
    .set<CustomRenderable>({0.0f, 0.0f, false, 0xFFFFFFFF, 0, 0, draw_edges})
    .set<ZIndex>({10});

    // Opens on the badge selected in the Interlocutor when there is one. With
    // nothing selected the panel opens empty and waits -- no default focus,
    // no speculative query.
    std::string focus = annotator_selected_badge();

    // Three stacked bands: supertypes, focus, subtypes. A Hasse diagram read
    // top to bottom, with height in the drawing meaning height in the order.
    // The bands are persistent; rebuild() only replaces their contents.
    LatticeView initial;
    initial.focus = focus;
    initial.tracked = focus;
    initial.status = focus.empty() ? "" : "loading...";
    initial.dirty = true;
    initial.stage = focus.empty() ? 0 : 1;

    auto diagram = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFillParent>({12.0f, 12.0f, 12.0f, 12.0f})
    .set<UIFlexContainer>({YGFlexDirectionColumn, YGWrapNoWrap,
                           YGJustifySpaceBetween, YGAlignStretch, 4.0f})
    .set<LatticeView>(initial)
    .set<ZIndex>({11});

    // Every band sizes to its content; the two spacers absorb the leftover
    // height. The whole lattice clusters tightly around the focus -- supers
    // a badge's height above, subs a badge's height below -- and the empty
    // outer space stays free for deeper generalization ranks later.
    auto band = [&](int rank, YGAlign align) {
        return world.entity()
            .is_a(UIElement)
            .child_of(diagram)
            .add(flecs::OrderedChildren)
            .add<UIYoga>()
            .set<LatticeBand>({rank})
            .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
            // A wide gap between nodes, several times the tight gap inside a
            // grouped intension: proximity is what says "one concept of many
            // parts" versus "many concepts" -- cat+domesticated's members sit
            // 4px apart, so siblings in a band must sit far enough that the
            // two readings can't be confused.
            .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapWrap,
                                   YGJustifyCenter, align, 32.0f})
            .set<ZIndex>({12});
    };

    auto spacer = [&]() {
        world.entity()
            .is_a(UIElement)
            .child_of(diagram)
            .add<UIYoga>()
            .set<UIFlexItem>({1.0f, 1.0f, YGAlignAuto});
    };

    // Manual annotation, worn as the badge it will become: an empty badge
    // shell centered exactly where the new neighbour will appear -- one
    // between the supertypes and the focus, one between the focus and the
    // subtypes. Type into it, Enter asserts the edge; the shared EntityQuery
    // plumbing does the keyboard work and parks the submit on the diagram.
    auto entry_shell = [&](EntityQuery::Kind kind, const char* placeholder) {
        auto row = world.entity()
        .is_a(UIElement)
        .child_of(diagram)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignAuto})
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyCenter, YGAlignCenter, 0.0f})
        .set<ZIndex>({13});

        // The Interlocutor input box's exact dress: 0x555555 stroke outline
        // over a 0x222327 fill, radius 2 -- so every text entry in the editor
        // is recognisably the same control.
        auto shell = world.entity()
        .is_a(UIElement)
        .child_of(row)
        .add(flecs::OrderedChildren)
        .add<UIYoga>()
        .set<UISize>({YGUndefined, BADGE_HEIGHT})
        .set<UIMinSize>({110.0f, YGUndefined})
        .set<UIFlexContainer>({YGFlexDirectionRow, YGWrapNoWrap,
                               YGJustifyCenter, YGAlignCenter, 0.0f})
        .set<UIPadding>({0.0f, 10.0f, 0.0f, 10.0f})
        .set<RoundedRectRenderable>({100.0f, BADGE_HEIGHT, 2.0f, true, 0x555555FF})
        .add<AddTagOnLeftClick, FocusEntityQuery>()
        .set<ZIndex>({13});

        world.entity()
        .is_a(UIElement)
        .child_of(shell)
        .add<UIYoga>()
        .set<UIAbsoluteEdges>({0.0f, 0.0f, 0.0f, 0.0f})
        .set<RoundedRectRenderable>({10.0f, 10.0f, 2.0f, false, 0x222327FF})
        .set<ZIndex>({12});

        auto shell_text = world.entity()
        .is_a(UIElement)
        .child_of(shell)
        .add<UIYoga>()
        .add<ScissorContainer>(shell)
        .set<TextRenderable>({"", "CharisSIL", 16.0f, 0xFFFFFFFF})
        .set<ZIndex>({14});

        EntityQuery entry;
        entry.panel = diagram;
        entry.input_text = shell_text;
        entry.kind = kind;
        entry.placeholder = placeholder;
        shell.set<EntityQuery>(entry);
    };

    spacer();
    band(-1, YGAlignFlexEnd);      // supertypes, hugging the entry shell
    entry_shell(EntityQuery::Supertype, "Supertype");
    band(0, YGAlignCenter);        // focus
    entry_shell(EntityQuery::Subtype, "Subtype");
    band(1, YGAlignFlexStart);     // subtypes
    entry_shell(EntityQuery::Instance, "Instance");
    band(2, YGAlignFlexStart);     // individuals -- the focus's extension
    spacer();

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());
    create_badge(badges, UIElement, "Lattice", kSuperColor);

    if (!focus.empty()) request(kSupersExpr, supers_response);
}

} // namespace

namespace panels {

Lattice::Lattice(flecs::world& world)
{
    world.module<Lattice>();
    g_world = &world;

    world.component<LatticeView>();
    world.component<LatticeNode>();
    world.component<LatticeEdges>();
    world.component<LatticeBand>();
    world.component<SupertypeSubmit>();
    world.component<SubtypeSubmit>();
    world.component<InstanceSubmit>();

    // Names an individual of the focus: "Remy" under Rat becomes the entity
    // Remy tagged Rat -- instance-of, kept apart from the type order.
    world.system<LatticeView, InstanceSubmit>("LatticeInstanceSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity panel, LatticeView& view, InstanceSubmit& submit) {
        if (view.focus.empty()) {
            view.status = "select a focus before naming an instance";
            view.dirty = true;
            panel.remove<InstanceSubmit>();
            return;
        }
        auto parts = split_conjunction(submit.name);
        if (parts.empty()) { panel.remove<InstanceSubmit>(); return; }
        if (!send_instance(view.focus, parts[0])) return;
        view.status = parts[0] + "...";
        view.dirty = true;
        panel.remove<InstanceSubmit>();
    });

    // Sends a supertype the bar submitted: <focus> SubtypeOf <name>. Kept
    // parked while the shared client is busy, so a submit never silently
    // loses to an in-flight query -- it just goes next frame.
    world.system<LatticeView, SupertypeSubmit>("LatticeSupertypeSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity panel, LatticeView& view, SupertypeSubmit& submit) {
        if (view.focus.empty()) {
            view.status = "select a focus before adding a supertype";
            view.dirty = true;
            panel.remove<SupertypeSubmit>();
            return;
        }
        // "pet + domesticated + owned": pet becomes a supertype of the meet
        // focus+domesticated+owned -- never of the bare focus. The bridge
        // builds the intensional node; a plain name is a direct edge.
        auto parts = split_conjunction(submit.name);
        if (parts.empty()) { panel.remove<SupertypeSubmit>(); return; }
        if (!send_supertype(view.focus, parts[0], join_csv(parts, 1))) return;
        view.status = parts[0] + "...";
        view.dirty = true;
        panel.remove<SupertypeSubmit>();
    });

    // The mirror image: a subtype entered below the focus is the same edge
    // with the arguments swapped -- <name> SubtypeOf <focus>.
    world.system<LatticeView, SubtypeSubmit>("LatticeSubtypeSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity panel, LatticeView& view, SubtypeSubmit& submit) {
        if (view.focus.empty()) {
            view.status = "select a focus before adding a subtype";
            view.dirty = true;
            panel.remove<SubtypeSubmit>();
            return;
        }
        // Two gestures. A leading '+' adds properties to the focus itself --
        // "+ domesticated" makes the intensional subtype cat+domesticated,
        // named by its parts rather than by hand. Otherwise the head names
        // the subtype and the rest are conditions on its edge up to the
        // focus: "cat + domesticated" under pet.
        const size_t first = submit.name.find_first_not_of(" \t");
        const bool intension = first != std::string::npos && submit.name[first] == '+';
        auto parts = split_conjunction(submit.name);
        if (parts.empty()) { panel.remove<SubtypeSubmit>(); return; }

        bool sent;
        std::string label;
        if (intension) {
            sent = send_meet(view.focus, join_csv(parts, 0));
            label = view.focus + "+" + join_csv(parts, 0);
        } else {
            sent = send_supertype(parts[0], view.focus, join_csv(parts, 1));
            label = parts[0];
        }
        if (!sent) return;
        view.status = label + "...";
        view.dirty = true;
        panel.remove<SubtypeSubmit>();
    });

    // Mirrors the Interlocutor's selection: when the annotation cursor lands
    // on a *different* badge, the lattice refocuses on it. Deselection and
    // plain-text stops leave the panel where it is, and so does clicking
    // around the lattice while the annotator holds still -- tracked moves only
    // with the badge, never with the panel's own navigation.
    world.system<LatticeView>("LatticeFollowSelectionSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, LatticeView& view) {
        std::string badge = annotator_selected_badge();
        if (badge.empty() || badge == view.tracked) return;
        // navigate() declines while a request is in flight; tracked is left
        // unchanged so the follow retries next frame instead of losing it.
        if (badge == view.focus || navigate(view, badge)) view.tracked = badge;
    });

    // Restyles when an on-screen identity appears or changes: binding an
    // entity in the Interlocutor must recolour its lattice badge immediately.
    // Symbols only -- colour follows the symbol, and asking for it here would
    // risk a blocking colour-server round trip every frame.
    world.system<LatticeView>("LatticeStyleSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity, LatticeView& view) {
        std::string sig;
        auto add = [&](const std::string& name) {
            std::string symbol;
            if (entity_screen_style(name, &symbol, nullptr)) {
                sig += name; sig += ':'; sig += symbol; sig += ';';
            }
        };
        add(view.focus);
        for (const std::string& s : view.supers) add(s);
        for (const std::string& s : view.subs) add(s);
        for (const std::string& s : view.instances) add(s);
        if (sig != view.style_sig) {
            view.style_sig = sig;
            view.dirty = true;
        }
    });

    // Rebuilds a panel whose view changed. PreFrame, so the nodes it creates
    // are laid out in the same frame the reply landed.
    world.system<LatticeView>("LatticeRebuildSystem")
    .kind(flecs::PreFrame)
    .each([](flecs::entity panel, LatticeView& view) {
        if (!view.dirty) return;
        view.dirty = false;
        rebuild(panel, view);
    });

    register_panel({EditorType::Lattice, "lattice", "Lattice", create_content});
}

} // namespace panels
