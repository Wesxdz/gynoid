#include "panel3d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <nanovg.h>

#include <render3d/render3d.hpp>

#include "cad_shaders.h"
#include "stl_mesh.h"

// nanovg_gl.h carries its own implementation and NANOVG_GL2_IMPLEMENTATION is
// defined for every target that links nanovg, so including it here would give
// the binary a second copy. Declaring the one entry point needed is what
// mel_spec_render.cpp does for the same reason.
extern "C" int nvglCreateImageFromHandleGL2(NVGcontext* ctx, GLuint textureId, int w, int h, int flags);

namespace panel3d {
namespace {

// Camera limits. zoom is the orthographic half-height in world units, and the
// model is normalized to kModelSize, so these frame it from "filling the panel"
// to "small in the middle".
constexpr float kModelSize = 8.0f;
constexpr float kZoomDefault = 6.0f;
constexpr float kZoomMin = 2.5f;
constexpr float kZoomMax = 24.0f;
constexpr float kCameraDistance = 40.0f;

// Clip planes fitted tightly around the model instead of render3d's +/-1000
// default, and this matters for more than precision: the post pass measures
// SSAO occlusion and silhouette depth cliffs in *world units*, by scaling
// normalized depth by (far - near). Across a 2000-unit range a 1-unit gap is
// 0.0005 in depth -- under the outline's 0.0008 threshold and far below the
// SSAO's 0.35-unit saturation, so both effects silently do nothing. The model's
// half-diagonal is kModelSize * sqrt(3) / 2 ~= 6.93, so it always sits within
// kCameraDistance +/- 7.
constexpr float kNearPlane = kCameraDistance - 10.0f;
constexpr float kFarPlane = kCameraDistance + 10.0f;

// Panel background, matching the grey the Droid canvas used to paint (0x3b3b3b)
// so the 3D image reads as part of the panel rather than a window onto a sky.
constexpr float kBackground = 0x3b / 255.0f;

// ---- CAD presentation look -------------------------------------------------
// Tuning lives here rather than in the GLSL so the look can be adjusted without
// touching the shader. Light colours are linear and pre-multiplied by their
// intensity; the fragment shader tonemaps.

// Surface finish. Low metalness with a tight-ish roughness reads as anodised or
// painted aluminium -- a fully metallic value needs a real environment probe to
// look like anything but grey mud.
constexpr float kRoughness = 0.24f;
constexpr float kMetalness = 0.10f;
constexpr float kExposure = 1.0f;

// Three-point rig, expressed in camera space and rebuilt each frame:
// x is right, y is up, z is toward the viewer. Key over the viewer's left
// shoulder, fill low on the right, rim from behind and above.
constexpr glm::vec3 kKeyAxes{-0.55f, 0.72f, 0.42f};
constexpr glm::vec3 kFillAxes{0.85f, -0.18f, 0.45f};
constexpr glm::vec3 kRimAxes{0.25f, 0.55f, -0.80f};

constexpr glm::vec3 kKeyColor{3.90f, 3.82f, 3.66f};    // neutral, a touch warm
constexpr glm::vec3 kFillColor{0.70f, 0.82f, 1.00f};   // cool, soft
constexpr glm::vec3 kRimColor{0.55f, 0.66f, 0.85f};

// Hemisphere ambient, kept low so the key light carries the form. Ambient this
// dim is what separates a CAD render from a flat clay one: the unlit side still
// reads, but it is clearly unlit.
constexpr glm::vec3 kSkyAmbient{0.155f, 0.175f, 0.215f};
constexpr glm::vec3 kGroundAmbient{0.042f, 0.040f, 0.038f};

// Studio backdrop sweep, in display space (the backdrop pass does not tonemap).
constexpr glm::vec3 kBackdropCenter{0.285f, 0.293f, 0.310f};
constexpr glm::vec3 kBackdropEdge{0.120f, 0.126f, 0.140f};

struct State
{
    NVGcontext* vg = nullptr;
    GLFWwindow* window = nullptr;

    // The offscreen target the post pass resolves into, and its nanovg handle.
    // Separate from render3d's RenderTarget, which holds the raw colour/normal/
    // depth attachments the post pass reads.
    GLuint fbo = 0;
    GLuint tex = 0;
    int width = 0;
    int height = 0;
    int nvg_image = -1;

    // Where the render passes hand control back to, so nanovg flushes into
    // Thornfield's UI framebuffer rather than whatever render3d bound last.
    GLuint ui_fbo = 0;
    int ui_width = 0;
    int ui_height = 0;

    double mouse_x = 0.0;
    double mouse_y = 0.0;
    bool left_down = false;
    bool hovering = false;
    double scroll = 0.0;

    flecs::entity camera;
    r3d::Mesh3D mesh;

    // Programs owned by this panel, replacing the module's cel-shaded opaque
    // pass and daylight sky gradient.
    GLuint cad_shader = 0;
    GLuint backdrop_shader = 0;

    // Held here rather than as a function-local static in the pass: a static
    // query outlives the world and calls ecs_query_fini() on freed memory
    // during static teardown. shutdown() releases it while the world is alive.
    flecs::query<r3d::Mesh3D, r3d::Material, r3d::Transform, r3d::Visible> cad_query;

    // Every system that costs GPU work, so the whole chain can be switched off
    // in the frames where no Droid panel is open.
    std::vector<flecs::entity> passes;
    bool active = false;
    bool ready = false;
};

State g;

// ---- Offscreen target ------------------------------------------------------

void releaseViewport()
{
    // nanovg owns the texture once it wraps it, so the image handle is the only
    // thing to release -- deleting the texture as well would double-free it.
    if (g.nvg_image != -1 && g.vg) {
        nvgDeleteImage(g.vg, g.nvg_image);
        g.nvg_image = -1;
        g.tex = 0;
    }
    if (g.fbo) {
        glDeleteFramebuffers(1, &g.fbo);
        g.fbo = 0;
    }
}

bool allocateViewport(int width, int height)
{
    // A resize can land mid-frame, after main() has bound the UI framebuffer,
    // so whatever was bound is put back before returning.
    GLint prev_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

    releaseViewport();

    g.width = width;
    g.height = height;

    glGenFramebuffers(1, &g.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g.fbo);

    glGenTextures(1, &g.tex);
    glBindTexture(GL_TEXTURE_2D, g.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.tex, 0);

    const bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    if (!complete) {
        std::cerr << "[panel3d] viewport FBO not complete" << std::endl;
        return false;
    }

    // FLIPY: GL renders bottom-up, nanovg's image pattern samples top-down.
    g.nvg_image = nvglCreateImageFromHandleGL2(g.vg, g.tex, width, height, NVG_IMAGE_FLIPY);
    return g.nvg_image != -1;
}

// ---- CAD passes ------------------------------------------------------------

// Studio backdrop, drawn on the Background phase in place of r3d.SkyPass.
void BackdropPass(flecs::iter& it)
{
    flecs::world world = it.world();
    const r3d::GraphicsContext& gfx = world.get<r3d::GraphicsContext>();
    const r3d::WindowResource& win = world.get<r3d::WindowResource>();
    if (!g.backdrop_shader) return;

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glUseProgram(g.backdrop_shader);
    glUniform3fv(glGetUniformLocation(g.backdrop_shader, "uCenterColor"), 1, glm::value_ptr(kBackdropCenter));
    glUniform3fv(glGetUniformLocation(g.backdrop_shader, "uEdgeColor"), 1, glm::value_ptr(kBackdropEdge));
    glUniform1f(glGetUniformLocation(g.backdrop_shader, "uAspect"),
                (float)win.renderWidth / (float)std::max(win.renderHeight, 1));

    glBindVertexArray(gfx.quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
}

// The CAD-lit draw, on the Opaque phase. Only picks up r3d::DeferredDraw
// entities -- the tag that told the module's own opaque pass to skip them.
void CadOpaquePass(flecs::iter& it)
{
    flecs::world world = it.world();
    const r3d::RenderFrame& fr = world.get<r3d::RenderFrame>();
    if (!fr.valid || !g.cad_shader) return;

    // Camera basis, from the camera-to-world matrix: column 2 is the camera's
    // +Z, which in GL convention points back toward the viewer.
    const glm::vec3 right = glm::normalize(glm::vec3(fr.camMatrix[0]));
    const glm::vec3 up = glm::normalize(glm::vec3(fr.camMatrix[1]));
    const glm::vec3 to_viewer = glm::normalize(glm::vec3(fr.camMatrix[2]));

    auto rig = [&](const glm::vec3& axes) {
        return glm::normalize(axes.x * right + axes.y * up + axes.z * to_viewer);
    };

    glUseProgram(g.cad_shader);
    const GLuint sh = g.cad_shader;

    // Orthographic projection, so one view direction covers the whole image.
    glUniform3fv(glGetUniformLocation(sh, "uViewDir"), 1, glm::value_ptr(to_viewer));

    const glm::vec3 key = rig(kKeyAxes);
    const glm::vec3 fill = rig(kFillAxes);
    const glm::vec3 rim = rig(kRimAxes);
    glUniform3fv(glGetUniformLocation(sh, "uKeyDir"), 1, glm::value_ptr(key));
    glUniform3fv(glGetUniformLocation(sh, "uFillDir"), 1, glm::value_ptr(fill));
    glUniform3fv(glGetUniformLocation(sh, "uRimDir"), 1, glm::value_ptr(rim));
    glUniform3fv(glGetUniformLocation(sh, "uKeyColor"), 1, glm::value_ptr(kKeyColor));
    glUniform3fv(glGetUniformLocation(sh, "uFillColor"), 1, glm::value_ptr(kFillColor));
    glUniform3fv(glGetUniformLocation(sh, "uRimColor"), 1, glm::value_ptr(kRimColor));
    glUniform3fv(glGetUniformLocation(sh, "uSkyColor"), 1, glm::value_ptr(kSkyAmbient));
    glUniform3fv(glGetUniformLocation(sh, "uGroundColor"), 1, glm::value_ptr(kGroundAmbient));
    glUniform1f(glGetUniformLocation(sh, "uRoughness"), kRoughness);
    glUniform1f(glGetUniformLocation(sh, "uMetalness"), kMetalness);
    glUniform1f(glGetUniformLocation(sh, "uExposure"), kExposure);

    const GLint mvp_loc = glGetUniformLocation(sh, "uMVP");
    const GLint model_loc = glGetUniformLocation(sh, "uModel");
    const GLint nm_loc = glGetUniformLocation(sh, "uNM");
    const GLint color_loc = glGetUniformLocation(sh, "uBaseColor");

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);

    g.cad_query.each([&](flecs::entity, r3d::Mesh3D& mesh, r3d::Material& mat,
                      r3d::Transform& trans, r3d::Visible& vis) {
        if (!vis.value) return;

        const glm::mat4 mvp = fr.viewProj * trans.matrix;
        const glm::mat3 nm = glm::mat3(glm::transpose(glm::inverse(trans.matrix)));

        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(trans.matrix));
        glUniformMatrix3fv(nm_loc, 1, GL_FALSE, glm::value_ptr(nm));
        glUniform3fv(color_loc, 1, glm::value_ptr(mat.color));

        r3d::drawMesh(mesh);
    });
}

// ---- Present ---------------------------------------------------------------

// render3d's own present pass upscales to the default framebuffer, which for
// Thornfield is the window backbuffer that gets cleared moments later. This is
// the same post pass aimed at the panel's texture instead, followed by handing
// the UI framebuffer back so nanovg's flush is unaffected.
void PresentToPanel(flecs::iter& it)
{
    flecs::world world = it.world();
    const r3d::GraphicsContext& gfx = world.get<r3d::GraphicsContext>();
    const r3d::RenderTarget& rt = world.get<r3d::RenderTarget>();

    if (g.fbo) {
        glBindFramebuffer(GL_FRAMEBUFFER, g.fbo);
        glViewport(0, 0, g.width, g.height);

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glClearColor(kBackground, kBackground, kBackground, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(gfx.postShader);

        // SSAO needs depth deltas in world units: ortho depth [0,1] spans
        // farPlane - nearPlane of the active camera.
        float depthRange = 2000.0f;
        world.each([&](flecs::entity, const r3d::Camera& cam, r3d::ActiveCamera) {
            depthRange = cam.farPlane - cam.nearPlane;
        });
        glUniform1f(glGetUniformLocation(gfx.postShader, "uDepthRange"), depthRange);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, rt.colorTex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, rt.normalTex);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, rt.depthTex);

        glBindVertexArray(gfx.quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        glActiveTexture(GL_TEXTURE0);
        glUseProgram(0);
    }

    // Hand the UI framebuffer back. nanovg restores the rest of the GL state it
    // cares about during its own flush, but not the binding or the viewport.
    glBindFramebuffer(GL_FRAMEBUFFER, g.ui_fbo);
    glViewport(0, 0, g.ui_width, g.ui_height);
}

// ---- Input -----------------------------------------------------------------

// Drives render3d's orbit controller from the pointer state the caller pushed
// in. The controller reads InputState but does not decide when a drag is
// happening -- that is the application's call, and here it means "left button
// held while the pointer is over the panel".
void PanelInput(flecs::iter& it)
{
    flecs::world world = it.world();
    if (!g.active || !g.camera.is_valid() || !g.camera.is_alive()) return;

    r3d::InputState& input = world.ensure<r3d::InputState>();
    input.mouseX = g.mouse_x;
    input.mouseY = g.mouse_y;
    input.mouseButtons[0] = g.left_down;
    world.modified<r3d::InputState>();

    r3d::CameraController& ctrl = g.camera.ensure<r3d::CameraController>();
    // A drag only starts over the panel, but once started it continues wherever
    // the pointer goes, which is what makes a fast orbit feel like one gesture.
    ctrl.orbiting = g.left_down && (ctrl.orbiting || g.hovering);
    ctrl.panning = false;

    if (g.scroll != 0.0) {
        if (g.hovering) {
            r3d::Camera& cam = g.camera.ensure<r3d::Camera>();
            cam.zoom = std::clamp(cam.zoom * std::pow(0.88f, (float)g.scroll), kZoomMin, kZoomMax);
        }
        g.scroll = 0.0;
    }
}

// ---- Setup -----------------------------------------------------------------

// render3d names its passes inside its own module scope; finding the present
// system by name avoids hard-coding the module path, which is an implementation
// detail of how the module registers itself.
flecs::entity findSystem(flecs::world& world, const char* name)
{
    flecs::entity found;
    world.query_builder().with(flecs::System).build().each([&](flecs::entity e) {
        if (!found && std::strcmp(e.name().c_str(), name) == 0) found = e;
    });
    return found;
}

void applyRenderSize(flecs::world& world, int width, int height)
{
    r3d::WindowResource& win = world.ensure<r3d::WindowResource>();
    win.width = width;
    win.height = height;
    win.renderWidth = width;
    win.renderHeight = height;
    world.modified<r3d::WindowResource>();

    r3d::RenderTarget& rt = world.get_mut<r3d::RenderTarget>();
    rt.cleanup();
    rt.init(width, height);
    world.modified<r3d::RenderTarget>();

    // uTexelSize is set once at import from the initial resolution, so the
    // outline and SSAO taps would sample at the wrong spacing after a resize.
    const r3d::GraphicsContext& gfx = world.get<r3d::GraphicsContext>();
    glUseProgram(gfx.postShader);
    glUniform2f(glGetUniformLocation(gfx.postShader, "uTexelSize"),
                1.0f / (float)width, 1.0f / (float)height);
    glUseProgram(0);
}

} // namespace

// ---- Public API ------------------------------------------------------------

bool init(flecs::world& world, NVGcontext* vg, GLFWwindow* window,
          const std::string& stl_path, int width, int height)
{
    if (g.ready) return true;

    width = std::max(width, 1);
    height = std::max(height, 1);

    StlMesh stl = loadStl(stl_path);
    if (!stl.valid) return false;
    normalizeStl(stl, kModelSize);

    g.vg = vg;
    g.window = window;
    g.ui_width = width;
    g.ui_height = height;

    world.set<r3d::WindowResource>({window, width, height, width, height});
    world.set<r3d::RendererConfig>({
        .shadowResolution = 2048,
        .registerTransformBake = true,
        .registerCameraSystems = true,
        .createRenderTarget = true,
    });
    world.import<r3d::Renderer>();

    if (!allocateViewport(width, height)) {
        std::cerr << "[panel3d] could not create the offscreen viewport" << std::endl;
        return false;
    }

    g.cad_shader = r3d::createShaderProgram(cad::CAD_VERTEX_SHADER, cad::CAD_FRAGMENT_SHADER);
    g.backdrop_shader = r3d::createShaderProgram(r3d::POST_VERTEX_SHADER, cad::BACKDROP_FRAGMENT_SHADER);
    if (!g.cad_shader || !g.backdrop_shader) {
        std::cerr << "[panel3d] CAD shaders failed to build" << std::endl;
        return false;
    }

    g.cad_query = world.query_builder<r3d::Mesh3D, r3d::Material, r3d::Transform, r3d::Visible>()
                       .with<r3d::DeferredDraw>()
                       .cached()
                       .build();

    g.mesh = r3d::uploadMesh(stl.vertices, stl.indices);

    g.camera = world.entity("Panel3DCamera")
        .set<r3d::Camera>({
            .zoom = kZoomDefault,
            .nearPlane = kNearPlane,
            .farPlane = kFarPlane,
        })
        .set<r3d::CameraController>({
            .azimuth = 315.0f,
            .elevation = 22.0f,
            .distance = kCameraDistance,
            .target = {0.0f, 0.0f, 0.0f},
        })
        .set<r3d::Position3D>({})
        .set<r3d::Transform>({})
        .add<r3d::ActiveCamera>();

    world.entity("Panel3DDroid")
        .set<r3d::Position3D>({{0.0f, 0.0f, 0.0f}})
        .set<r3d::Transform>({})
        .set<r3d::Mesh3D>(g.mesh)
        // Light neutral grey. The CAD pass reads this as the base colour, so
        // this is the one place the part's finish colour is set.
        .set<r3d::Material>({{0.76f, 0.78f, 0.82f}})
        .set<r3d::Visible>({true})
        // Opts the mesh out of the module's cel-shaded opaque pass in favour of
        // CadOpaquePass -- the extension point render3d documents for exactly
        // this. It stays in the frustum/transform machinery either way.
        .add<r3d::DeferredDraw>();

    // Module passes this panel replaces. r3d.Present goes to the default
    // framebuffer instead of the panel texture; r3d.SkyPass paints a daylight
    // gradient; r3d.ShadowPass is dead weight here, because the key light is
    // camera-relative, so its shadows fall almost entirely behind the part --
    // contact darkening comes from the post pass's SSAO instead.
    for (const char* name : {"r3d.Present", "r3d.SkyPass", "r3d.ShadowPass"}) {
        if (flecs::entity pass = findSystem(world, name)) {
            pass.disable();
        } else {
            std::cerr << "[panel3d] could not find " << name << " to disable" << std::endl;
        }
    }

    const r3d::Phases& phases = world.get<r3d::Phases>();
    // PreUpdate, so the pointer state reaches the orbit controller in the same
    // frame it was captured rather than the next one.
    world.system<>("Panel3DInput").kind(flecs::PreUpdate).run(PanelInput);

    flecs::entity backdrop =
        world.system<>("Panel3DBackdrop").kind(phases.Background).run(BackdropPass);
    flecs::entity cad =
        world.system<>("Panel3DCadOpaque").kind(phases.Opaque).run(CadOpaquePass);
    flecs::entity panel_present =
        world.system<>("Panel3DPresent").kind(phases.Present).run(PresentToPanel);

    for (const char* name : {"r3d.FrameBegin", "r3d.OpaquePass",
                             "r3d.BillboardPass", "r3d.FrameEnd"}) {
        if (flecs::entity pass = findSystem(world, name)) g.passes.push_back(pass);
    }
    g.passes.push_back(backdrop);
    g.passes.push_back(cad);
    g.passes.push_back(panel_present);
    for (flecs::entity pass : g.passes) pass.disable();

    g.ready = true;
    std::cout << "[panel3d] viewport " << width << "x" << height << std::endl;
    return true;
}

int image_handle()
{
    return g.nvg_image;
}

void set_active(bool active)
{
    if (!g.ready || active == g.active) return;
    g.active = active;
    for (flecs::entity pass : g.passes) {
        if (active) pass.enable(); else pass.disable();
    }
}

void resize(flecs::world& world, int width, int height)
{
    if (!g.ready) return;
    width = std::max(width, 1);
    height = std::max(height, 1);
    if (width == g.width && height == g.height) return;

    if (!allocateViewport(width, height)) return;
    applyRenderSize(world, width, height);
}

void set_ui_target(unsigned int fbo, int width, int height)
{
    g.ui_fbo = (GLuint)fbo;
    g.ui_width = width;
    g.ui_height = height;
}

void set_pointer(double mouse_x, double mouse_y, bool left_down, bool hovering)
{
    g.mouse_x = mouse_x;
    g.mouse_y = mouse_y;
    g.left_down = left_down;
    g.hovering = hovering;
}

void add_scroll(double delta)
{
    g.scroll += delta;
}

bool view_key(int key, int mods)
{
    if (!g.ready || !g.active || !g.hovering) return false;
    if (!g.camera.is_valid() || !g.camera.is_alive()) return false;

    r3d::CameraController& ctrl = g.camera.ensure<r3d::CameraController>();
    r3d::Camera& cam = g.camera.ensure<r3d::Camera>();
    const bool ctrl_held = (mods & GLFW_MOD_CONTROL) != 0;

    // A pole is approached but never reached: the controller builds its view
    // matrix with a world-up of +Y, which degenerates when the view direction
    // is exactly vertical.
    constexpr float kPole = 89.9f;
    constexpr float kStep = 15.0f;
    auto setView = [&](float azimuth, float elevation) {
        ctrl.azimuth = azimuth;
        ctrl.elevation = elevation;
    };

    switch (key) {
    // The camera sits at target + distance * (cos(e)sin(a), sin(e), cos(e)cos(a)),
    // so azimuth 0 looks down -Z and azimuth 90 looks down -X.
    case GLFW_KEY_KP_1: setView(ctrl_held ? 180.0f : 0.0f, 0.0f); break;
    case GLFW_KEY_KP_3: setView(ctrl_held ? 270.0f : 90.0f, 0.0f); break;
    case GLFW_KEY_KP_7: setView(ctrl.azimuth, ctrl_held ? -kPole : kPole); break;
    case GLFW_KEY_KP_9: setView(ctrl.azimuth + 180.0f, -ctrl.elevation); break;

    // Signs follow the drag: dragging right lowers azimuth, and numpad 6 is the
    // keyboard equivalent of that same nudge.
    case GLFW_KEY_KP_4: ctrl.azimuth += kStep; break;
    case GLFW_KEY_KP_6: ctrl.azimuth -= kStep; break;
    case GLFW_KEY_KP_8: ctrl.elevation = std::clamp(ctrl.elevation + kStep, -kPole, kPole); break;
    case GLFW_KEY_KP_2: ctrl.elevation = std::clamp(ctrl.elevation - kStep, -kPole, kPole); break;

    case GLFW_KEY_KP_ADD:      cam.zoom = std::clamp(cam.zoom / 1.2f, kZoomMin, kZoomMax); break;
    case GLFW_KEY_KP_SUBTRACT: cam.zoom = std::clamp(cam.zoom * 1.2f, kZoomMin, kZoomMax); break;

    case GLFW_KEY_KP_DECIMAL:
        ctrl.target = glm::vec3(0.0f);
        ctrl.targetGoal = ctrl.target;
        ctrl.panAccum = glm::vec3(0.0f);
        ctrl.interpolating = false;
        cam.zoom = kZoomDefault;
        break;

    // Numpad 5 is Blender's ortho/perspective toggle; render3d only projects
    // orthographically, so it is left unconsumed rather than silently ignored.
    default:
        return false;
    }

    return true;
}

void shutdown(flecs::world& world)
{
    if (!g.ready) return;
    releaseViewport();
    g.cad_query = {};
    if (g.cad_shader) { glDeleteProgram(g.cad_shader); g.cad_shader = 0; }
    if (g.backdrop_shader) { glDeleteProgram(g.backdrop_shader); g.backdrop_shader = 0; }
    r3d::cleanupMesh(g.mesh);
    r3d::shutdown(world);
    g.ready = false;
}

} // namespace panel3d
