#include "screen3d.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nanovg.h>

// Same single-implementation arrangement panel3d.cpp documents: nanovg_gl.h
// carries its implementation, so only the entry point is declared here.
extern "C" int nvglCreateImageFromHandleGL2(NVGcontext* ctx, GLuint textureId, int w, int h, int flags);

namespace screen3d {
namespace {

// ---- Room tuning -----------------------------------------------------------
// Metres and seconds. The screen is a home-theatre-sized plane a few steps
// from the spawn point, standing on a floor whose grid gives the walk scale.

constexpr float kEyeHeight = 1.65f;
constexpr float kWalkSpeed = 3.2f;              // m/s; Shift multiplies
constexpr float kSprintMul = 2.2f;
constexpr float kLookSens = 0.0022f;            // radians per pixel
constexpr float kPitchLimit = 1.553f;           // ~89 degrees
constexpr float kFloorHalf = 40.0f;
constexpr float kWalkHalf = 38.0f;              // keeps the floor underfoot

constexpr float kScreenHeight = 2.25f;          // plane height; width follows aspect
constexpr float kScreenLift = 0.6f;             // bottom edge above the floor
constexpr float kScreenZ = -2.4f;
constexpr float kBezel = 0.10f;                 // dark frame around the image

constexpr float kFovY = 1.2217f;                // 70 degrees
constexpr float kNear = 0.05f;
constexpr float kFar = 220.0f;

// Palette: near-black void, grid a register brighter, majors brighter still.
constexpr glm::vec3 kClearColor{0.043f, 0.045f, 0.055f};
constexpr glm::vec3 kGroundColor{0.058f, 0.060f, 0.072f};
constexpr glm::vec3 kLineColor{0.135f, 0.140f, 0.170f};
constexpr glm::vec3 kMajorColor{0.205f, 0.210f, 0.260f};
constexpr glm::vec3 kBezelColor{0.085f, 0.085f, 0.095f};
constexpr glm::vec3 kNoSignalColor{0.020f, 0.022f, 0.028f};

// ---- Shaders ---------------------------------------------------------------

const char* GRID_VS = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uVP;
out vec3 vWorld;
void main(){ vWorld = aPos; gl_Position = uVP * vec4(aPos, 1.0); }
)GLSL";

// Anti-aliased 1m grid with 5m majors, fogged toward the clear colour so the
// floor dissolves into the void instead of ending at a hard edge.
const char* GRID_FS = R"GLSL(
#version 330 core
in vec3 vWorld;
uniform vec3 uEye;
uniform vec3 uGround;
uniform vec3 uLine;
uniform vec3 uMajor;
uniform vec3 uFog;
out vec4 frag;
float gridLine(vec2 coord){
    vec2 g = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    return 1.0 - min(min(g.x, g.y), 1.0);
}
void main(){
    vec3 col = uGround;
    col = mix(col, uLine, gridLine(vWorld.xz) * 0.65);
    col = mix(col, uMajor, gridLine(vWorld.xz / 5.0) * 0.85);
    float fog = exp(-distance(vWorld, uEye) * 0.05);
    frag = vec4(mix(uFog, col, fog), 1.0);
}
)GLSL";

const char* QUAD_VS = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 uVP;
uniform mat4 uModel;
out vec2 vUV;
void main(){ vUV = aUV; gl_Position = uVP * uModel * vec4(aPos, 0.0, 1.0); }
)GLSL";

// The screen is pure emission: the stream's pixels, no lighting, no tonemap.
const char* QUAD_FS = R"GLSL(
#version 330 core
in vec2 vUV;
uniform int uTextured;
uniform vec3 uColor;
uniform sampler2D uTex;
out vec4 frag;
void main(){
    if (uTextured == 1) frag = vec4(texture(uTex, vUV).rgb, 1.0);
    else frag = vec4(uColor, 1.0);
}
)GLSL";

struct State
{
    NVGcontext* vg = nullptr;
    GLFWwindow* window = nullptr;

    GLuint fbo = 0;
    GLuint tex = 0;
    GLuint depth_rbo = 0;
    int width = 0;
    int height = 0;
    int nvg_image = -1;

    GLuint ui_fbo = 0;
    int ui_width = 0;
    int ui_height = 0;

    // Pointer state pushed in by the sync system, window coordinates.
    double mouse_x = 0.0;
    double mouse_y = 0.0;
    bool left_down = false;
    bool hovering = false;
    bool prev_left = false;

    // The walk. While captured, GLFW's virtual cursor supplies look deltas
    // and the real cursor is parked; restore_* is where it goes back.
    bool captured = false;
    double restore_x = 0.0;
    double restore_y = 0.0;
    double last_x = 0.0;
    double last_y = 0.0;
    float yaw = 0.0f;                  // 0 looks down -Z, at the screen
    float pitch = 0.0f;
    glm::vec3 eye{0.0f, kEyeHeight, 2.6f};

    // The stream on the wall.
    GLuint screen_tex = 0;
    int screen_w = 0;
    int screen_h = 0;

    GLuint grid_shader = 0;
    GLuint quad_shader = 0;
    GLuint floor_vao = 0, floor_vbo = 0;
    GLuint quad_vao = 0, quad_vbo = 0;

    flecs::entity input_system;
    flecs::entity render_system;
    bool active = false;
    bool ready = false;
};

State g;

// ---- GL setup --------------------------------------------------------------

GLuint compileProgram(const char* vs_src, const char* fs_src)
{
    auto compile = [](GLenum type, const char* src) -> GLuint {
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
            std::cerr << "[screen3d] shader: " << log << std::endl;
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    };
    GLuint vs = compile(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) { if (vs) glDeleteShader(vs); if (fs) glDeleteShader(fs); return 0; }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::cerr << "[screen3d] link: " << log << std::endl;
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

void releaseViewport()
{
    if (g.nvg_image != -1 && g.vg) {
        nvgDeleteImage(g.vg, g.nvg_image);   // nanovg owns the texture it wraps
        g.nvg_image = -1;
        g.tex = 0;
    }
    if (g.depth_rbo) { glDeleteRenderbuffers(1, &g.depth_rbo); g.depth_rbo = 0; }
    if (g.fbo) { glDeleteFramebuffers(1, &g.fbo); g.fbo = 0; }
}

bool allocateViewport(int width, int height)
{
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

    // Unlike panel3d, this pass draws real geometry, so it needs its own depth.
    glGenRenderbuffers(1, &g.depth_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, g.depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g.depth_rbo);

    const bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    if (!complete) {
        std::cerr << "[screen3d] viewport FBO not complete" << std::endl;
        return false;
    }

    // FLIPY: GL renders bottom-up, nanovg's image pattern samples top-down.
    g.nvg_image = nvglCreateImageFromHandleGL2(g.vg, g.tex, width, height, NVG_IMAGE_FLIPY);
    return g.nvg_image != -1;
}

// ---- Input -----------------------------------------------------------------

void capturePointer()
{
    g.captured = true;
    g.restore_x = g.mouse_x;
    g.restore_y = g.mouse_y;
    glfwSetInputMode(g.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(g.window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
    // Seed the delta origin so the grab itself doesn't jerk the view.
    glfwGetCursorPos(g.window, &g.last_x, &g.last_y);
}

void InputPass(flecs::iter& it)
{
    if (!g.active || !g.window) return;

    // Alt-tab while walking should not leave the app holding the mouse.
    if (g.captured && !glfwGetWindowAttrib(g.window, GLFW_FOCUSED)) {
        release_pointer();
    }

    const bool click = g.left_down && !g.prev_left;
    g.prev_left = g.left_down;

    if (!g.captured) {
        if (click && g.hovering) capturePointer();
        return;
    }

    // Mouse look, from the virtual cursor GLFW runs while the real one is
    // disabled. Screen y grows downward, so a downward delta pitches down.
    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(g.window, &cx, &cy);
    g.yaw += (float)(cx - g.last_x) * kLookSens;
    g.pitch = std::clamp(g.pitch - (float)(cy - g.last_y) * kLookSens,
                         -kPitchLimit, kPitchLimit);
    g.last_x = cx;
    g.last_y = cy;

    // WASD on the ground plane, yaw only -- looking at the ceiling while
    // holding W still walks forward, the FPS convention.
    const glm::vec3 fwd{std::sin(g.yaw), 0.0f, -std::cos(g.yaw)};
    const glm::vec3 right{std::cos(g.yaw), 0.0f, std::sin(g.yaw)};
    glm::vec3 wish{0.0f};
    if (glfwGetKey(g.window, GLFW_KEY_W) == GLFW_PRESS) wish += fwd;
    if (glfwGetKey(g.window, GLFW_KEY_S) == GLFW_PRESS) wish -= fwd;
    if (glfwGetKey(g.window, GLFW_KEY_D) == GLFW_PRESS) wish += right;
    if (glfwGetKey(g.window, GLFW_KEY_A) == GLFW_PRESS) wish -= right;

    if (wish.x != 0.0f || wish.z != 0.0f) {
        float speed = kWalkSpeed;
        if (glfwGetKey(g.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(g.window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
            speed *= kSprintMul;
        }
        g.eye += glm::normalize(wish) * speed * it.delta_time();
        g.eye.x = std::clamp(g.eye.x, -kWalkHalf, kWalkHalf);
        g.eye.z = std::clamp(g.eye.z, -kWalkHalf, kWalkHalf);
        g.eye.y = kEyeHeight;
    }
}

// ---- Render ----------------------------------------------------------------

void drawQuad(const glm::mat4& vp, const glm::mat4& model,
              bool textured, const glm::vec3& color)
{
    const GLuint sh = g.quad_shader;
    glUniformMatrix4fv(glGetUniformLocation(sh, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform1i(glGetUniformLocation(sh, "uTextured"), textured ? 1 : 0);
    glUniform3fv(glGetUniformLocation(sh, "uColor"), 1, glm::value_ptr(color));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    (void)vp;
}

void RenderPass(flecs::iter&)
{
    if (!g.active || !g.fbo || g.width <= 0 || g.height <= 0) return;

    glBindFramebuffer(GL_FRAMEBUFFER, g.fbo);
    glViewport(0, 0, g.width, g.height);
    glClearColor(kClearColor.r, kClearColor.g, kClearColor.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    // The screen plane reads from both sides (mirrored from behind, which is
    // physically what a projection screen does), so no culling anywhere.
    glDisable(GL_CULL_FACE);

    const glm::vec3 dir{std::cos(g.pitch) * std::sin(g.yaw),
                        std::sin(g.pitch),
                        -std::cos(g.pitch) * std::cos(g.yaw)};
    const float aspect = (float)g.width / (float)std::max(g.height, 1);
    const glm::mat4 vp = glm::perspective(kFovY, aspect, kNear, kFar)
                       * glm::lookAt(g.eye, g.eye + dir, glm::vec3(0, 1, 0));

    // Floor.
    glUseProgram(g.grid_shader);
    glUniformMatrix4fv(glGetUniformLocation(g.grid_shader, "uVP"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform3fv(glGetUniformLocation(g.grid_shader, "uEye"), 1, glm::value_ptr(g.eye));
    glUniform3fv(glGetUniformLocation(g.grid_shader, "uGround"), 1, glm::value_ptr(kGroundColor));
    glUniform3fv(glGetUniformLocation(g.grid_shader, "uLine"), 1, glm::value_ptr(kLineColor));
    glUniform3fv(glGetUniformLocation(g.grid_shader, "uMajor"), 1, glm::value_ptr(kMajorColor));
    glUniform3fv(glGetUniformLocation(g.grid_shader, "uFog"), 1, glm::value_ptr(kClearColor));
    glBindVertexArray(g.floor_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // The screen: bezel behind, image in front. Aspect follows the stream's
    // native size; 16:9 until the first frame reports one.
    const float screen_aspect = (g.screen_w > 0 && g.screen_h > 0)
        ? (float)g.screen_w / (float)g.screen_h
        : 16.0f / 9.0f;
    const float sw = kScreenHeight * screen_aspect;
    const glm::vec3 center{0.0f, kScreenLift + kScreenHeight * 0.5f, kScreenZ};

    glUseProgram(g.quad_shader);
    glUniformMatrix4fv(glGetUniformLocation(g.quad_shader, "uVP"), 1, GL_FALSE, glm::value_ptr(vp));
    glBindVertexArray(g.quad_vao);

    const glm::mat4 bezel = glm::scale(
        glm::translate(glm::mat4(1.0f), center + glm::vec3(0, 0, -0.01f)),
        glm::vec3(sw + kBezel, kScreenHeight + kBezel, 1.0f));
    drawQuad(vp, bezel, false, kBezelColor);

    const glm::mat4 plane = glm::scale(glm::translate(glm::mat4(1.0f), center),
                                       glm::vec3(sw, kScreenHeight, 1.0f));
    if (g.screen_tex) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g.screen_tex);
        glUniform1i(glGetUniformLocation(g.quad_shader, "uTex"), 0);
        drawQuad(vp, plane, true, kNoSignalColor);
    } else {
        drawQuad(vp, plane, false, kNoSignalColor);
    }

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);

    // Hand the UI framebuffer back for nanovg's flush.
    glBindFramebuffer(GL_FRAMEBUFFER, g.ui_fbo);
    glViewport(0, 0, g.ui_width, g.ui_height);
}

} // namespace

// ---- Public API ------------------------------------------------------------

bool init(flecs::world& world, NVGcontext* vg, GLFWwindow* window,
          int width, int height)
{
    if (g.ready) return true;

    g.vg = vg;
    g.window = window;
    g.ui_width = std::max(width, 1);
    g.ui_height = std::max(height, 1);

    if (!allocateViewport(std::max(width, 1), std::max(height, 1))) {
        std::cerr << "[screen3d] could not create the offscreen viewport" << std::endl;
        return false;
    }

    g.grid_shader = compileProgram(GRID_VS, GRID_FS);
    g.quad_shader = compileProgram(QUAD_VS, QUAD_FS);
    if (!g.grid_shader || !g.quad_shader) return false;

    // Floor: two triangles on y=0.
    const float f = kFloorHalf;
    const float floor_verts[] = {
        -f, 0, -f,   -f, 0, f,    f, 0, f,
        -f, 0, -f,    f, 0, f,    f, 0, -f,
    };
    glGenVertexArrays(1, &g.floor_vao);
    glGenBuffers(1, &g.floor_vbo);
    glBindVertexArray(g.floor_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.floor_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floor_verts), floor_verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Unit quad in XY as a triangle strip, v=0 at the top edge to match the
    // stream's top-first row order.
    const float quad_verts[] = {
        -0.5f, -0.5f, 0.0f, 1.0f,
         0.5f, -0.5f, 1.0f, 1.0f,
        -0.5f,  0.5f, 0.0f, 0.0f,
         0.5f,  0.5f, 1.0f, 0.0f,
    };
    glGenVertexArrays(1, &g.quad_vao);
    glGenBuffers(1, &g.quad_vbo);
    glBindVertexArray(g.quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // PreUpdate for input (same slot as panel3d's), PostUpdate for the draw --
    // any phase inside world.progress() works, the GL context is current and
    // nanovg only samples the texture at its flush after progress returns.
    g.input_system = world.system<>("Screen3DInput").kind(flecs::PreUpdate).run(InputPass);
    g.render_system = world.system<>("Screen3DRender").kind(flecs::PostUpdate).run(RenderPass);
    g.input_system.disable();
    g.render_system.disable();

    g.ready = true;
    std::cout << "[screen3d] viewport " << g.width << "x" << g.height << std::endl;
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
    if (!active) release_pointer();
    if (active) { g.input_system.enable(); g.render_system.enable(); }
    else { g.input_system.disable(); g.render_system.disable(); }
}

void resize(int width, int height)
{
    if (!g.ready) return;
    width = std::max(width, 1);
    height = std::max(height, 1);
    if (width == g.width && height == g.height) return;
    allocateViewport(width, height);
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

void set_screen(unsigned int gl_texture, int width, int height)
{
    g.screen_tex = (GLuint)gl_texture;
    g.screen_w = width;
    g.screen_h = height;
}

bool pointer_captured()
{
    return g.ready && g.captured;
}

void release_pointer()
{
    if (!g.captured) return;
    g.captured = false;
    if (g.window) {
        if (glfwRawMouseMotionSupported()) {
            glfwSetInputMode(g.window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
        }
        glfwSetInputMode(g.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        // Back where the grab happened, so releasing doesn't teleport the
        // cursor to wherever the virtual position wandered.
        glfwSetCursorPos(g.window, g.restore_x, g.restore_y);
    }
}

void shutdown()
{
    if (!g.ready) return;
    release_pointer();
    releaseViewport();
    if (g.grid_shader) { glDeleteProgram(g.grid_shader); g.grid_shader = 0; }
    if (g.quad_shader) { glDeleteProgram(g.quad_shader); g.quad_shader = 0; }
    if (g.floor_vbo) { glDeleteBuffers(1, &g.floor_vbo); g.floor_vbo = 0; }
    if (g.floor_vao) { glDeleteVertexArrays(1, &g.floor_vao); g.floor_vao = 0; }
    if (g.quad_vbo) { glDeleteBuffers(1, &g.quad_vbo); g.quad_vbo = 0; }
    if (g.quad_vao) { glDeleteVertexArrays(1, &g.quad_vao); g.quad_vao = 0; }
    g.ready = false;
}

} // namespace screen3d
