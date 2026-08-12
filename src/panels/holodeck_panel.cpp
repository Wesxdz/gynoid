#include "holodeck_panel.h"

#include <cmath>
#include <cstdlib>
#include <string>

// Same GL prelude as main.cpp: components.h and vnc_struct.h assume the
// mel_spec glad is the one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "../components.h"
#include "../vnc_struct.h"
#include "../screen3d.h"
#include "../capabilities/ui_kit.h"
#include "../capabilities/vnc_sources.h"
#include "registry.h"

namespace {

// The VNC stream as a plane in a walkable room. Same source plumbing as the
// flat VNC Stream panel -- get_vnc_source refcounts, so both panels can watch
// one connection -- but the pixels land on screen3d's emissive plane instead
// of a flat ImageRenderable.
void create_content(flecs::entity leaf, flecs::entity UIElement)
{
    flecs::world world = leaf.world();

    auto badges = world.entity()
    .is_a(UIElement)
    .set<LayoutBox>({LayoutBox::Horizontal, 2.0f})
    .set<Position, Local>({84.0f, 0.0f})
    .child_of(leaf.target<EditorHeader>());

    const char* vnc_host = getenv("VNC_SERVER_HOST");
    if (!vnc_host) {
        vnc_host = "192.168.1.104";
    }
    int port = 5901;

    create_badge(badges, UIElement, "Holodeck", 0x9740f6ff);
    create_badge(badges, UIElement, vnc_host, 0xa7a7a7ff, true);
    create_badge(badges, UIElement, std::to_string(port).c_str(), 0xf64242ff, true);
    create_badge(badges, UIElement, "click to walk", 0xFAB257FF, true);

    VNCData data = get_vnc_source(leaf, vnc_host, port);

    // Same shape as the Droid viewport: a Yoga fill-parent image whose texture
    // is the module's offscreen room, resolution kept matched to the layout by
    // the sync system below. UIAspectRatio opts out of the image's intrinsic
    // ratio -- the room is whatever shape the panel is.
    world.entity()
    .is_a(UIElement)
    .child_of(leaf.target<EditorCanvas>())
    .add<UIYoga>()
    .set<UIFillParent>({})
    .add<UIAspectRatio>()
    .add<Screen3DViewport>()
    .add<IsStreamingFrom>(data.vnc_stream)
    .set<ImageRenderable>({screen3d::image_handle(), 1.0f, 1.0f, 0.0f, 0.0f})
    .set<ZIndex>({10});
}

// On Panel3DSyncSystem's exact pattern: size and activity follow the laid-out
// panel, the pointer feeds the walk (with the same divider-drag gate as the
// orbit camera), and each frame the VNC stream's current texture and native
// size reach the screen plane, so a mid-session mode switch on the remote
// side just works.
void register_sync_system(flecs::world& world)
{
    auto viewports = world.query_builder<const UIElementBounds>()
        .with<Screen3DViewport>()
        .cached()
        .build();

    auto split_drags = world.query_builder()
        .with<Dragging>()
        .build();

    world.system("Screen3DSyncSystem")
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
            if (live) return;   // one room, so the first laid-out panel wins
            int panel_w = (int)std::lround(bounds.xmax - bounds.xmin);
            int panel_h = (int)std::lround(bounds.ymax - bounds.ymin);
            if (panel_w <= 0 || panel_h <= 0) return;
            live = true;

            screen3d::set_active(true);
            screen3d::resize(panel_w, panel_h);

            ImageRenderable& img = e.ensure<ImageRenderable>();
            img.imageHandle = screen3d::image_handle();

            flecs::entity stream = e.target<IsStreamingFrom>();
            if (stream.is_valid()) {
                if (const VNCClientHandle* handle = stream.try_get<VNCClientHandle>()) {
                    const VNCClient& client = **handle;
                    screen3d::set_screen(client.vncTexture, client.width, client.height);
                }
            }

            double cursor_x = 0.0, cursor_y = 0.0;
            if (const CursorState* cursor = glfw_state.try_get<CursorState>()) {
                cursor_x = cursor->x;
                cursor_y = cursor->y;
            }
            const bool left_down =
                glfwGetMouseButton(win->handle, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            const bool ui_drag = split_drags.count() > 0;
            screen3d::set_pointer(cursor_x, cursor_y, left_down && !ui_drag,
                                  !ui_drag && point_in_bounds(cursor_x, cursor_y, bounds));
        });

        if (!live) screen3d::set_active(false);
    });
}

} // namespace

namespace panels {

Holodeck::Holodeck(flecs::world& world)
{
    world.module<Holodeck>();
    register_sync_system(world);
    register_panel({EditorType::Holodeck, "holodeck", "Holodeck", create_content});
}

} // namespace panels
