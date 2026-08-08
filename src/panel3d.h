#pragma once

// A render3d scene drawn into an offscreen target and published to nanovg as an
// image handle, so a 3D view participates in Thornfield's normal 2D layer
// ordering instead of sitting under or over the entire UI.
//
// Thornfield renders its whole UI into one framebuffer and blits that to the
// screen, and nanovg only executes its queued draws at nvgEndFrame. The 3D
// passes therefore run during world.progress() -- before that flush -- and hand
// nanovg a texture that is already up to date by the time it samples it.
//
// This header deliberately knows nothing about Thornfield's UI components; the
// caller feeds it the panel's size and pointer state each frame.

#include <string>

#include <flecs.h>

struct GLFWwindow;
struct NVGcontext;

namespace panel3d {

// Import render3d into `world`, load `stl_path`, and build the scene at an
// initial `width` x `height`. Requires a current GL context. Returns false if
// the model could not be loaded, in which case nothing is registered and the
// panel simply shows its background.
bool init(flecs::world& world, NVGcontext* vg, GLFWwindow* window,
          const std::string& stl_path, int width, int height);

// nanovg image handle for the viewport, or -1 before a successful init.
int image_handle();

// Switch the render passes on only while a Droid panel is open -- the scene
// costs a shadow map, an opaque pass and a post pass every frame otherwise.
void set_active(bool active);

// Resize the offscreen target to the size the panel was laid out at. A no-op
// when the size is unchanged, so it is cheap to call every frame.
void resize(flecs::world& world, int width, int height);

// The framebuffer and viewport the render passes must restore once they are
// done, so nanovg's flush lands where main() intended. Call once per frame,
// after binding the UI framebuffer and before world.progress().
void set_ui_target(unsigned int fbo, int width, int height);

// Pointer state for the orbit camera, in window coordinates. `hovering` gates
// the drag so a press that began elsewhere never grabs the camera.
void set_pointer(double mouse_x, double mouse_y, bool left_down, bool hovering);

// Accumulate a scroll wheel delta; applied as zoom on the next frame the panel
// is hovered. Safe to call from a GLFW callback.
void add_scroll(double delta);

// Blender-style numpad view controls, for a GLFW key callback:
//
//   1 front (ctrl: back)      4 / 6  orbit left / right 15 degrees
//   3 right (ctrl: left)      8 / 2  orbit up / down 15 degrees
//   7 top   (ctrl: bottom)    + / -  zoom in / out
//   9 opposite of the current view
//   .  reframe the model
//
// Returns true when the key was consumed, which only happens while the pointer
// is over the panel -- so numpad input anywhere else in Thornfield is
// untouched. Pass GLFW's `key` and `mods` straight through.
bool view_key(int key, int mods);

void shutdown(flecs::world& world);

} // namespace panel3d
