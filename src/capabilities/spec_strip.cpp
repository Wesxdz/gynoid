#include "spec_strip.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <vector>

#include "../components.h"
#include "audio_player.h"
#include "spectrogram.h"
#include "ui_kit.h"

namespace spec_strip {
namespace {

// Drawn at spec_gen's native resolution: one screen pixel per mel row and per
// spectrogram column, so the harmonic banding and the ordered dither land
// exactly as the palette drew them. Resampling a print-like image is what
// smears those textures, which are the whole point of the styling.
//
// kZoom scales both axes together if a strip ever needs to be bigger or
// smaller; 1.0 is native. spec_hud emits roughly 56 columns a second, so a
// clip reads at about 56px per second of speech.
constexpr float kZoom = 1.0f;
constexpr float kPlayheadWidth = 2.0f;

// The strip that owns the transport. Identity, not path: comparing wav paths
// meant every strip sharing a filename lit up together, and "the last one you
// clicked or played" is a fact about a specific element, not about a file.
flecs::entity g_active;

NVGcontext* vg_of(flecs::world world)
{
    flecs::entity graphics = world.lookup("Graphics");
    if (!graphics.is_valid()) return nullptr;
    const Graphics* g = graphics.try_get<Graphics>();
    return g ? g->vg : nullptr;
}

// Plain click is the transport; seeking is deliberately behind a modifier.
// Clicking a spectrogram to jump to that instant is a lovely affordance, but it
// was the *only* affordance, so simply hearing a line meant landing wherever
// the pointer happened to be. Playing is the common act, so it gets the plain
// click.
//
//   click            play from the start, or pause/resume if already playing
//   shift/ctrl-click seek to that point in the clip
void on_click(flecs::entity strip)
{
    const SpecStrip* clip = strip.try_get<SpecStrip>();
    if (!clip) return;

    flecs::entity glfw_state = strip.world().lookup("GLFWState");
    const Window* win = glfw_state.is_valid() ? glfw_state.try_get<Window>() : nullptr;
    const CursorState* cursor =
        glfw_state.is_valid() ? glfw_state.try_get<CursorState>() : nullptr;

    bool scrubbing = false;
    if (win && win->handle) {
        auto held = [&](int key) {
            return glfwGetKey(win->handle, key) == GLFW_PRESS;
        };
        scrubbing = held(GLFW_KEY_LEFT_SHIFT) || held(GLFW_KEY_RIGHT_SHIFT)
                 || held(GLFW_KEY_LEFT_CONTROL) || held(GLFW_KEY_RIGHT_CONTROL);
    }

    const bool active = (strip == g_active);

    if (scrubbing) {
        const UIElementBounds* bounds = strip.try_get<UIElementBounds>();
        if (!bounds || !cursor) return;
        const float width = bounds->xmax - bounds->xmin;
        if (width <= 0.0f) return;

        // The spectrogram's x axis IS time, so where you clicked is when.
        const float t = std::clamp(((float)cursor->x - bounds->xmin) / width, 0.0f, 1.0f);
        if (!active) audio::play(clip->wav_path);
        g_active = strip;
        audio::seek(t * audio::duration());
        return;
    }

    // Toggling only makes sense for the clip that holds the transport; any
    // other strip starts from the top.
    if (!active || audio::finished()) {
        play(strip, clip->wav_path);
    } else if (audio::paused()) {
        audio::resume();
    } else {
        audio::pause();
    }
}

} // namespace

void play(flecs::entity strip, const std::string& wav_path)
{
    if (!strip.is_valid() || !strip.is_alive()) return;

    // The path is taken as an argument rather than read back off the strip.
    // A strip built from inside a system is created deferred, so its SpecStrip
    // has not merged yet and try_get returns null -- which silently meant
    // "nothing to play" for every freshly synthesized line.
    std::string path = wav_path;
    if (path.empty()) {
        if (const SpecStrip* clip = strip.try_get<SpecStrip>()) path = clip->wav_path;
    }
    if (path.empty()) return;

    g_active = strip;
    audio::play(path);
}

flecs::entity create_above(flecs::entity message, const std::string& wav_path,
                           const std::string& palette)
{
    if (!message.is_valid() || !message.is_alive()) return flecs::entity();

    // "Above" is achieved by flipping the message row into a reverse column
    // rather than by reordering children. Reordering meant set_child_order and
    // suspending the deferred stage from inside another system's iteration,
    // which is what took the app down; a style change on one entity cannot.
    //
    // ColumnReverse lays the first child at the bottom, so the bubble that is
    // already there stays put and the strip appended after it lands above.
    const UIFlexContainer* flex = message.try_get<UIFlexContainer>();
    const YGAlign align = flex ? (flex->justify == YGJustifyFlexEnd
                                  ? YGAlignFlexEnd : YGAlignFlexStart)
                               : YGAlignFlexStart;
    message.set<UIFlexContainer>({
        YGFlexDirectionColumnReverse, YGWrapNoWrap,
        YGJustifyFlexStart, align, 4.0f});

    return create(message, wav_path, palette);
}

flecs::entity create(flecs::entity parent, const std::string& wav_path,
                     const std::string& palette)
{
    flecs::world world = parent.world();
    flecs::entity UIElement = world.lookup("UIElement");
    if (!UIElement.is_valid()) return flecs::entity();

    const int handle = spectrogram::image(vg_of(world), wav_path, palette);
    if (handle == -1) return flecs::entity();

    int native_w = 0, native_h = 0;
    spectrogram::size(wav_path, palette, &native_w, &native_h);
    const float strip_w = native_w * kZoom;
    const float strip_h = native_h * kZoom;

    flecs::entity strip = world.entity()
        .is_a(UIElement)
        .child_of(parent)
        .add<UIYoga>()
        .set<SpecStrip>({wav_path, palette})
        .set<UISize>({strip_w, strip_h})
        // shrink 0: a long clip is allowed to be wide. Letting the row squeeze
        // it would compress time, and the x axis here is time.
        // FlexStart regardless of which side the message bubble sits on: the
        // strip reads left-to-right as time, so it starts at the left margin.
        .set<UIFlexItem>({0.0f, 0.0f, YGAlignFlexStart})
        .set<CallbackOnLeftClick>({on_click})
        .set<ZIndex>({12});

    // Drawn by hand rather than as an ImageRenderable, because the strip is two
    // images split at the playhead: colour behind it, grey ahead of it. Colour
    // arriving as the audio reaches it makes the picture its own progress bar.
    //
    // The strip entity is captured so the draw can read its world position:
    // RenderCommand carries that, but it is only forward-declared outside
    // main.cpp, so the pointer handed to this callback cannot be dereferenced
    // here. Its size arrives through the CustomRenderable argument, which Yoga
    // writes on readback.
    const int grey_handle = spectrogram::grey(vg_of(world), wav_path, palette);
    const std::string clip_path = wav_path;

    strip.set<CustomRenderable>({strip_w, strip_h, false, 0xFFFFFFFF, 0, 0,
        [strip, clip_path, handle, grey_handle]
        (NVGcontext* vg, const RenderCommand*, const CustomRenderable& r)
        {
            if (r.width <= 0.0f || r.height <= 0.0f) return;
            const Position* at = strip.try_get<Position, World>();
            if (!at) return;

            double played = 0.0;
            if (strip == g_active && audio::duration() > 0.0) {
                played = std::clamp(audio::position() / audio::duration(), 0.0, 1.0);
            }
            const float split = at->x + (float)played * r.width;

            // Both halves paint the whole image and let the scissor decide
            // which part lands, so the two are always in register -- offsetting
            // the pattern instead would drift them by a pixel.
            auto band = [&](int image, float x, float w) {
                if (image == -1 || w <= 0.25f) return;
                nvgSave(vg);
                // Intersect, never replace: the render queue has already set
                // the transcript's crop for this command, and nvgScissor would
                // discard it -- which is how a strip escaped the panel it was
                // scrolled out of.
                nvgIntersectScissor(vg, x, at->y, w, r.height);
                NVGpaint paint = nvgImagePattern(vg, at->x, at->y,
                                                 r.width, r.height, 0.0f, image, 1.0f);
                nvgBeginPath(vg);
                nvgRect(vg, at->x, at->y, r.width, r.height);
                nvgFillPaint(vg, paint);
                nvgFill(vg);
                nvgRestore(vg);
            };

            band(grey_handle, split, at->x + r.width - split);   // still to come
            band(handle, at->x, split - at->x);                  // already heard

            // The playhead is drawn here, by the one strip that holds the
            // transport, rather than being an entity per strip that all but one
            // hides. A hidden-by-default element is only ever as correct as the
            // system doing the hiding; this cannot render twice.
            if (strip == g_active) {
                nvgBeginPath(vg);
                nvgRect(vg, split - kPlayheadWidth * 0.5f, at->y, kPlayheadWidth, r.height);
                nvgFillColor(vg, nvgRGBA(255, 255, 255, 204));
                nvgFill(vg);
            }
        }});

    return strip;
}

} // namespace spec_strip
