#include "droid_panel.h"

#include <cmath>

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "../components.h"
#include "../panel3d.h"
#include "../capabilities/ui_kit.h"
#include "../capabilities/ui_metrics.h"
#include "registry.h"

namespace {

void create_content(flecs::entity leaf, flecs::entity UIElement)
{
    flecs::world world = leaf.world();

    auto canvas = leaf.target<EditorCanvas>();

    auto grey_bkg = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add<UIYoga>()
    .set<UIFillParent>({})
    .set<RectRenderable>({0.0f, 0.0f, false, 0x3b3b3bff})
    .set<ZIndex>({9});

    // The droid is a live render3d scene rather than a portrait bitmap: the
    // texture behind this ImageRenderable is the offscreen viewport in
    // panel3d.cpp, redrawn each frame before nanovg flushes. It fills the
    // canvas, and Panel3DSyncSystem keeps the render resolution matched to
    // whatever size Yoga gives it.
    //
    // UIAspectRatio with an undefined ratio opts out of the intrinsic aspect
    // ratio images normally publish -- the viewport is whatever shape the
    // panel is, not whatever shape the texture happens to be.
    world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add<UIYoga>()
    .set<UIFillParent>({})
    .add<UIAspectRatio>()
    .add<Panel3DViewport>()
    .set<ImageRenderable>({panel3d::image_handle(), 1.0f, 1.0f, 0.0f, 0.0f})
    .set<ZIndex>({10});

    world.entity()
    .is_a(UIElement)
    .set<ImageCreator>({"../assets/mnist_version.png", 1.0f, 1.0f})
    .set<ZIndex>({15})
    .child_of(grey_bkg);

    // The 3D tooltip: the same badge recipe the header wears, riding the
    // cursor over the viewport while a part is hovered. It is a plain nanovg
    // element, so it is pure emission by construction -- the scene's lighting
    // never touches it -- and screen-sized regardless of camera zoom.
    // Parented to the canvas, not the viewport: the canvas's Expand locks its
    // bounds, so parking the tooltip far off-canvas while nothing is hovered
    // cannot inflate the bounds that Panel3DSyncSystem sizes the render
    // target from. Amber to match the hover highlight it labels.
    auto part_tooltip = world.entity()
    .is_a(UIElement)
    .child_of(canvas)
    .add<DroidPartTooltip>()
    .set<Position, Local>({-10000.0f, -10000.0f})
    .set<ZIndex>({30});
    create_badge(part_tooltip, UIElement, "3D part", 0xFAB257FF);

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());

    create_badge(badges, UIElement, "Aeri Peach", 0xe66c25ff, false, false,
                 {}, {0}, {"A"}, {0xe66c25ff});
    create_badge(badges, UIElement, "Physical", 0x619393ff);
}

// The 3D viewport is the one element whose *pixels* depend on its layout, so
// the size Yoga gave it has to reach the renderer before the passes run. Runs
// after boundsCalculationSystem (also OnLoad; guaranteed by import order) so
// the bounds read here are this frame's, and before Panel3DInput on PreUpdate
// so a drag is applied in the frame it happened.
//
// With no Droid panel open the query is empty and the whole render3d chain is
// switched off rather than drawing a scene nothing samples.
void register_sync_system(flecs::world& world)
{
    auto viewports = world.query_builder<const UIElementBounds>()
        .with<Panel3DViewport>()
        .cached()
        .build();

    // A split divider being dragged means the pointer belongs to panel
    // resizing, not to the 3D scene -- without this, resizing a Droid panel
    // with the cursor over the viewport grabs the orbit camera and spins the
    // model as a side effect of the resize gesture.
    auto split_drags = world.query_builder()
        .with<Dragging>()
        .build();

    world.system("Panel3DSyncSystem")
    .kind(flecs::OnLoad)
    .run([viewports, split_drags](flecs::iter& it)
    {
        flecs::world w = it.world();
        flecs::entity glfw_state = w.lookup("GLFWState");
        const Window* win = glfw_state.is_valid() ? glfw_state.try_get<Window>() : nullptr;
        if (!win || !win->handle) return;

        bool live = false;
        viewports.each([&](flecs::entity e, const UIElementBounds& bounds)
        {
            if (live) return;   // one scene, so the first laid-out panel wins
            int panel_w = (int)std::lround(bounds.xmax - bounds.xmin);
            int panel_h = (int)std::lround(bounds.ymax - bounds.ymin);
            if (panel_w <= 0 || panel_h <= 0) return;
            live = true;

            panel3d::set_active(true);
            panel3d::resize(w, panel_w, panel_h);

            // The image handle is minted per viewport allocation, so a resize
            // invalidates whatever the panel was created with.
            ImageRenderable& img = e.ensure<ImageRenderable>();
            img.imageHandle = panel3d::image_handle();

            double cursor_x = 0.0, cursor_y = 0.0;
            if (const CursorState* cursor = glfw_state.try_get<CursorState>()) {
                cursor_x = cursor->x;
                cursor_y = cursor->y;
            }
            const bool left_down =
                glfwGetMouseButton(win->handle, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            // While a divider drag is live the viewport sees no pointer at
            // all: no orbit grab, no hover highlight flicker as the divider
            // sweeps across the 3D image.
            const bool ui_drag = split_drags.count() > 0;
            panel3d::set_pointer(cursor_x, cursor_y, left_down && !ui_drag,
                                 !ui_drag && point_in_bounds(cursor_x, cursor_y, bounds),
                                 bounds.xmin, bounds.ymin);
        });

        if (!live) panel3d::set_active(false);
    });
}

// The tooltip badge rides the cursor: bottom-left corner on the pointer,
// extending up-right, while a part is hovered. PreFrame, so the position it
// writes flows through this frame's world-position and bounds propagation --
// and the cursor read here is this frame's, polled before the pipeline ran,
// so the badge tracks live with no trailing. Only the shown/hidden state is
// the pick's, one frame behind like the highlight. The cursor is window-space
// and the holder canvas-local, so the canvas's bounds origin (stable outside
// panel rearranges) converts between them.
void register_tooltip_system(flecs::world& world)
{
    world.system<Position, const UIElementBounds>("Droid3DTooltipSystem")
    .term_at(0).second<Local>()
    .term_at(1).parent()
    .with<DroidPartTooltip>()
    .kind(flecs::PreFrame)
    .each([](flecs::entity e, Position& pos, const UIElementBounds& canvas_bounds)
    {
        flecs::entity glfw_state = e.world().lookup("GLFWState");
        const CursorState* cursor =
            glfw_state.is_valid() ? glfw_state.try_get<CursorState>() : nullptr;
        if (panel3d::has_hover() && cursor) {
            pos.x = (float)cursor->x - canvas_bounds.xmin;
            pos.y = (float)cursor->y - canvas_bounds.ymin - BADGE_HEIGHT;
        } else {
            pos.x = -10000.0f;
            pos.y = -10000.0f;
        }
    });
}

} // namespace

namespace panels {

Droid::Droid(flecs::world& world)
{
    world.module<Droid>();
    register_sync_system(world);
    register_tooltip_system(world);
    register_panel({EditorType::Droid, "droid", "Droid", create_content});
}

} // namespace panels
