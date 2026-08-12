#pragma once

// The Holodeck panel's scene: a remote screen hung as an emissive plane in a
// dark grid room the user walks around in with FPS controls -- WASD to move,
// mouse look while the pointer is captured (click the viewport to grab it,
// Escape to let go).
//
// Deliberately not built on render3d: that module is single-scene (one
// RenderTarget, one ActiveCamera, one frame state as world singletons) and the
// Droid panel already owns it. This room is two shader programs and three
// quads, so it carries its own GL and publishes the result to nanovg the same
// way panel3d does -- an offscreen target wrapped as an image handle, drawn
// into during world.progress() before nanovg's flush samples it.

#include <flecs.h>

struct GLFWwindow;
struct NVGcontext;

namespace screen3d {

// Build the offscreen target, shaders and geometry, and register the input +
// render systems. Requires a current GL context. Returns false if GL setup
// fails, in which case nothing is registered.
bool init(flecs::world& world, NVGcontext* vg, GLFWwindow* window,
          int width, int height);

// nanovg image handle for the viewport, or -1 before a successful init.
int image_handle();

// Switch the room on only while a Holodeck panel is open. Deactivating also
// releases the pointer if the walk had it captured.
void set_active(bool active);

// Resize the offscreen target to the panel's laid-out size. No-op when
// unchanged, so it is cheap to call every frame.
void resize(int width, int height);

// The framebuffer and viewport the render pass must restore when done, so
// nanovg's flush lands where main() intended. Call once per frame.
void set_ui_target(unsigned int fbo, int width, int height);

// Pointer state in window coordinates. A left press while `hovering` captures
// the pointer and starts the walk; hovering gates the grab exactly like the
// Droid panel's orbit.
void set_pointer(double mouse_x, double mouse_y, bool left_down, bool hovering);

// The screen plane's texture: a GL texture id (the VNC stream's) with its
// native pixel size, which sets the plane's aspect. Call every frame; 0 means
// "no signal" and draws the plane dark.
void set_screen(unsigned int gl_texture, int width, int height);

// True while the walk owns the pointer (GLFW cursor disabled). The GLFW
// callbacks and the cursor-mode system consult this to keep captured mouse
// motion and WASD keys out of the rest of the UI.
bool pointer_captured();

// Release a captured pointer, restoring the cursor where the grab happened.
// Safe to call when not captured.
void release_pointer();

void shutdown();

} // namespace screen3d
