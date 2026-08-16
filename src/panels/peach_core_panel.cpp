#include "peach_core_panel.h"

#include <cmath>
#include <string>

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "../components.h"
#include "../capabilities/ui_kit.h"
#include "../server_registry.h"
#include "registry.h"

namespace {

// Status colours, carried by a bar spanning the icon's full width rather than
// by a 10px dot. Hue alone is a weak signal against a painted ground -- these
// icons are classical paintings, and a small coloured disc disappears into
// them. The bar is roughly eight times the area and sits where minilm's "384"
// caption band does, so it reads as part of the same visual language.
// RGB only; the alpha below is applied to all of them.
constexpr uint32_t kBarOffline  = 0x6e6a66;     // nothing running
constexpr uint32_t kBarStarting = 0xfab257;     // spawned, inside the warmup grace
constexpr uint32_t kBarRunning  = 0x5aa9e6;     // process is up, has not announced
constexpr uint32_t kBarReady    = 0x61c300;     // the server said it is ready
constexpr uint32_t kBarForeign  = 0x9a7fd0;     // up, but not ours to stop

// The band is a glaze, not a UI chip: the painting stays visible through it, so
// it reads as part of the artwork rather than as a widget parked on top. One
// knob for all five states.
constexpr uint32_t kBarAlpha = 0xB4;            // ~70%

// The band is proportioned off minilm.png's painted caption strip, so the two
// read as the same element: in that 81x97 icon the grey runs rows 78-95, 18px
// tall, inset 1px from the sides and 1px up from the bottom by the orange
// keyline. Held as a ratio because the strip shrinks icons to fit the panel --
// a fixed pixel height would go proportionally fat as the icons get small.
constexpr float kBarRatio = 18.0f / 97.0f;
constexpr float kBarMinHeight = 2.0f;

// Matches the 1px keyline the art already frames its caption band with.
constexpr float kIconInset = 1.0f;

// What the painting itself is multiplied by. Offline servers go dark, so the
// whole strip reads at a glance: a lit panel is a live one. This is the signal
// you see from across the room; the bar is the precise readout.
constexpr uint32_t kArtLit = 0xffffffff;
constexpr uint32_t kArtDim = 0x59595fff;

NVGcolor unpack(uint32_t rgba)
{
    return nvgRGBA((rgba >> 24) & 0xFF, (rgba >> 16) & 0xFF, (rgba >> 8) & 0xFF, rgba & 0xFF);
}

uint32_t barColor(const ServerState& state)
{
    uint32_t rgb = kBarOffline;
    switch (state.status) {
    case ServerStatus::Offline:  rgb = kBarOffline; break;
    case ServerStatus::Starting: rgb = kBarStarting; break;
    case ServerStatus::Running:  rgb = state.owned ? kBarRunning : kBarForeign; break;
    case ServerStatus::Ready:    rgb = state.owned ? kBarReady : kBarForeign; break;
    }
    return (rgb << 8) | kBarAlpha;
}

// Brightness the icon art is multiplied by. Starting breathes between dim and
// lit: the click-to-start feedback question is "did anything happen", and a
// moving icon answers it before any colour is decoded.
NVGcolor artTint(const ServerState& state, double now)
{
    switch (state.status) {
    case ServerStatus::Offline:
        return unpack(kArtDim);
    case ServerStatus::Starting: {
        const float pulse = 0.5f + 0.5f * (float)std::sin(now * 5.0f);
        const auto mix = [pulse](uint32_t a, uint32_t b, int shift) {
            const float av = (float)((a >> shift) & 0xFF);
            const float bv = (float)((b >> shift) & 0xFF);
            return (uint8_t)(av + (bv - av) * pulse);
        };
        return nvgRGBA(mix(kArtDim, kArtLit, 24), mix(kArtDim, kArtLit, 16),
                       mix(kArtDim, kArtLit, 8), 255);
    }
    case ServerStatus::Running:
    case ServerStatus::Ready:
        return unpack(kArtLit);
    }
    return unpack(kArtLit);
}

const char* statusWord(const ServerState& state)
{
    switch (state.status) {
    case ServerStatus::Offline:  return "offline";
    case ServerStatus::Starting: return "starting";
    case ServerStatus::Running:  return state.owned ? "running" : "running (external)";
    case ServerStatus::Ready:    return state.owned ? "ready" : "ready (external)";
    }
    return "unknown";
}

// What the hover overlay says under the server's name: enough to tell why a
// click did nothing, which is the question this panel exists to answer.
std::string statusLine(const ServerDef& def, const ServerState& state)
{
    std::string line = statusWord(state);
    if (state.pid > 0) line += "  pid " + std::to_string(state.pid);
    if (!def.conda_env.empty()) line += "  env " + def.conda_env;
    if (state.status == ServerStatus::Offline) {
        line += "   -   click to start";
    } else if (state.owned) {
        line += "   -   click to stop";
    } else {
        line += "   -   started outside Thornfield, not ours to stop";
    }
    return line;
}

// Tag for the overlay's three text rows, so the refresh system can find them
// without depending on child order.
struct OverlayLabel {};
struct OverlayStatus {};
struct OverlayCommand {};

// An icon in the strip that is not a server. The overlay describes it from
// this instead of from a ServerDef, so a non-server emblem can sit in the row
// without the hover text going blank over it.
struct EmblemInfo
{
    std::string label;
    std::string description;
};

// The band under an icon. Tagged so the status system can recolour it without
// caring where it sits in the icon's children.
struct ServerStatusBar {};

void create_content(flecs::entity leaf, flecs::entity UIElement)
{
    flecs::world world = leaf.world();
    auto canvas = leaf.target<EditorCanvas>();

    // Server icon strip. A Yoga row filling the panel: each icon stretches to
    // the row height, takes its width from the image's native aspect ratio, and
    // shrinks uniformly if the strip would overflow.
    //
    // Anonymous, deliberately. As a named root entity this resolved to the same
    // node every time the panel was built, so reopening Peach Core reparented
    // the existing strip onto the new canvas and appended a second full set of
    // icons beneath the first.
    auto server_hud = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFillParent>({4.0f, 4.0f, 4.0f, 4.0f})
    .set<Position, Local>({4.0f, 4.0f})
    .set<UIFlexContainer>({
        YGFlexDirectionRow, YGWrapNoWrap,
        YGJustifyCenter, YGAlignStretch, 0.0f});

    // Hover description backdrop: its own Yoga root covering the panel, holding
    // a column of text rows the refresh system fills from whichever icon is
    // hovered. ServerDescription::selected is what the existing hover observers
    // in main.cpp write.
    auto panel_overlay = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add<UIYoga>()
    .set<UIFillParent>({})
    .set<RectRenderable>({0.0f, 0.0f, false, 0x000000DD})
    .add<ServerDescription>()
    .set<ZIndex>({15});

    auto overlay_text = world.entity()
    .is_a(UIElement)
    .child_of(panel_overlay)
    .add(flecs::OrderedChildren)
    .add<UIYoga>()
    .set<UIFillParent>({12.0f, 12.0f, 12.0f, 12.0f})
    .set<UIFlexContainer>({
        YGFlexDirectionColumn, YGWrapNoWrap,
        YGJustifyCenter, YGAlignFlexStart, 4.0f})
    .set<ZIndex>({16});

    world.entity().is_a(UIElement).child_of(overlay_text).add<UIYoga>()
        .add<OverlayLabel>()
        .set<TextRenderable>({"", "CharisSIL", 18.0f, 0xFFFFFFFF})
        .set<ZIndex>({16});

    world.entity().is_a(UIElement).child_of(overlay_text).add<UIYoga>()
        .add<OverlayStatus>()
        .set<TextRenderable>({"", "CharisSIL", 14.0f, 0xBFBFBFFF})
        .set<ZIndex>({16});

    // The exact command line, monospaced: when a conda env or a path in
    // servers.json is wrong, this is the line that shows it.
    world.entity().is_a(UIElement).child_of(overlay_text).add<UIYoga>()
        .add<OverlayCommand>()
        .set<TextRenderable>({"", "JetBrainsMono", 11.0f, 0x8A8A8AFF})
        .set<ZIndex>({16});

    // The root of trust, leftmost and ahead of every server. Not a process and
    // not startable: it stands for the gynoid's own identity -- a PUF key, the
    // thing the servers downstream are trusted *by*. It carries no status band
    // for the same reason. When attestation is real, that band is where its
    // state belongs, and it would be the one icon in the row reporting on the
    // robot rather than on something the robot talks to.
    world.entity()
    .is_a(UIElement)
    .child_of(server_hud)
    .add<UIYoga>()
    .add<UIMaxNativeImageSize>()
    .set<UIFlexItem>({0.0f, 1.0f, YGAlignAuto})
    .set<UIFlexContainer>({
        YGFlexDirectionColumn, YGWrapNoWrap,
        YGJustifyFlexEnd, YGAlignCenter, 0.0f})
    .set<UIPadding>({0.0f, kIconInset, kIconInset, kIconInset})
    .set<EmblemInfo>({"Peach Core",
                      "Root of trust. The gynoid's own identity key -- what the "
                      "servers beside it are trusted by. No attestation wired up yet."})
    .add<ServerHUDOverlay>(panel_overlay)
    .add<AddTagOnHoverEnter, ShowServerHUDOverlay>()
    .add<AddTagOnHoverExit, HideServerHUDOverlay>()
    .set<ImageCreator>({"../assets/server_hud/peach_core.png", 1.0f, 1.0f})
    .set<ZIndex>({10});

    // One icon per registry entry, in config order. No hardcoded icon list:
    // what the panel shows is exactly what servers.json declares.
    flecs::entity scope = world.lookup("Servers");
    if (!scope) return;

    scope.children([&](flecs::entity server)
    {
        const ServerDef* def = server.try_get<ServerDef>();
        if (!def) return;

        flecs::entity server_icon = world.entity()
        .is_a(UIElement)
        .child_of(server_hud)
        .add<UIYoga>()
        // Height comes from the row's AlignStretch, width from the image
        // aspect ratio; shrink lets the whole strip scale down to fit. Capped
        // at 1:1 so a tall panel doesn't upscale the icons.
        .add<UIMaxNativeImageSize>()
        .set<UIFlexItem>({0.0f, 1.0f, YGAlignAuto})
        // Column so the status dot sits centred along the bottom edge.
        .set<UIFlexContainer>({
            YGFlexDirectionColumn, YGWrapNoWrap,
            YGJustifyFlexEnd, YGAlignCenter, 0.0f})
        // Inset the content box so the status band lands inside the art's own
        // keyline rather than flush against the icon's edges.
        .set<UIPadding>({0.0f, kIconInset, kIconInset, kIconInset})
        .add<ServerOf>(server)
        .add<ServerHUDOverlay>(panel_overlay)
        .add<AddTagOnHoverEnter, ShowServerHUDOverlay>()
        .add<AddTagOnHoverExit, HideServerHUDOverlay>()
        .set<CallbackOnLeftClick>({[](flecs::entity icon) {
            servers::toggle(icon.target<ServerOf>());
        }})
        .set<ImageCreator>({"../assets/server_hud/" + def->icon + ".png", 1.0f, 1.0f})
        .set<ZIndex>({10});

        // The status band, along the icon's bottom edge. AlignStretch overrides
        // the column's AlignCenter so it spans the icon's full width whatever
        // the strip has shrunk the icon to; only the height is fixed. Recoloured
        // per frame by ServerStatusSystem, which finds it by ServerStatusBar.
        world.entity()
        .is_a(UIElement)
        .child_of(server_icon)
        .add<UIYoga>()
        .add<ServerStatusBar>()
        .set<UISize>({YGUndefined, kBarRatio * 97.0f})
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignStretch})
        .set<RectRenderable>({0.0f, kBarRatio * 97.0f, false, (kBarOffline << 8) | kBarAlpha})
        .set<ZIndex>({12});
    });
}

// Repaint the icon and its status band from the server's current state, every
// frame. Cheap -- a handful of icons, one component read apiece -- and it keeps
// the strip honest the moment a process dies rather than on the next rebuild.
void register_status_system(flecs::world& world)
{
    // The icon art: dimmed while its server is down, breathing while starting.
    world.system<ImageRenderable>("ServerIconTintSystem")
    .with<ServerOf>(flecs::Wildcard)
    .kind(flecs::PreUpdate)
    .each([](flecs::entity e, ImageRenderable& img)
    {
        flecs::entity server = e.target<ServerOf>();
        if (!server.is_valid() || !server.is_alive()) return;
        if (const ServerState* state = server.try_get<ServerState>()) {
            img.tint = artTint(*state, e.world().get_info()->world_time_total);
        }
    });

    // The band. Its own query rather than a branch inside the one above,
    // because the bar is a RectRenderable and the icon an ImageRenderable.
    world.system<RectRenderable>("ServerStatusBarSystem")
    .with<ServerStatusBar>()
    .kind(flecs::PreUpdate)
    .each([](flecs::entity e, RectRenderable& bar)
    {
        // The bar hangs off the icon, which is what carries ServerOf.
        flecs::entity icon = e.parent();
        flecs::entity server = icon.target<ServerOf>();
        if (!server.is_valid() || !server.is_alive()) return;

        if (const ServerState* state = server.try_get<ServerState>()) {
            bar.color = barColor(*state);
        }

        // Track the icon's laid-out height, so the band keeps minilm's
        // proportion however far the row has shrunk the icons. Writing UISize
        // is what Yoga reads; RectRenderable's height is overwritten by
        // readback anyway, but keeping it in step avoids a one-frame flash on
        // the first layout.
        if (const UIElementSize* icon_size = icon.try_get<UIElementSize>()) {
            if (icon_size->height > 0.0f) {
                const float h = std::max(kBarMinHeight,
                                         std::round(icon_size->height * kBarRatio));
                UISize& size = e.ensure<UISize>();
                if (size.h != h) {
                    size.h = h;
                    bar.height = h;
                }
            }
        }
    });
}

// Fill the hover overlay from whichever icon the pointer is on. The overlay's
// visibility is already handled in main.cpp (a system toggling its ZIndex from
// ServerDescription::selected); this only supplies the words.
void register_overlay_system(flecs::world& world)
{
    world.system<const ServerDescription>("ServerOverlayTextSystem")
    .kind(flecs::PreUpdate)
    .each([](flecs::entity overlay, const ServerDescription& desc)
    {
        flecs::world w = overlay.world();

        std::string label, status, command;
        flecs::entity icon = w.entity(desc.selected);
        if (desc.selected && icon.is_valid() && icon.is_alive()) {
            if (const EmblemInfo* emblem = icon.try_get<EmblemInfo>()) {
                label = emblem->label;
                status = emblem->description;
            }
            flecs::entity server = icon.target<ServerOf>();
            const ServerDef* def = server ? server.try_get<ServerDef>() : nullptr;
            const ServerState* state = server ? server.try_get<ServerState>() : nullptr;
            if (def && state) {
                label = def->label;
                status = statusLine(*def, *state);
                command = def->description;
                if (!def->script.empty()) {
                    command += command.empty() ? "" : "\n";
                    command += servers::command_line(*def, "~/miniconda3");
                }
            }
        }

        // Walk once, writing whichever rows are present.
        overlay.children([&](flecs::entity row) {
            row.children([&](flecs::entity text_row) {
                TextRenderable* text = text_row.try_get_mut<TextRenderable>();
                if (!text) return;
                if (text_row.has<OverlayLabel>()) text->text = label;
                else if (text_row.has<OverlayStatus>()) text->text = status;
                else if (text_row.has<OverlayCommand>()) text->text = command;
            });
        });
    });
}

} // namespace

namespace panels {

PeachCore::PeachCore(flecs::world& world)
{
    world.module<PeachCore>();
    world.component<OverlayLabel>();
    world.component<OverlayStatus>();
    world.component<OverlayCommand>();
    world.component<ServerStatusBar>();
    world.component<EmblemInfo>();

    register_status_system(world);
    register_overlay_system(world);
    register_panel({EditorType::PeachCore, "peach_core", "Peach Core", create_content});
}

} // namespace panels
