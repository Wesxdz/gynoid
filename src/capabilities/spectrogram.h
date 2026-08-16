#pragma once

// Styled spectrograms as nanovg images.
//
// Wraps deps/spec_hud (vendored from paphos/spec_gen), which renders a wav to
// RGB with per-harmonic palettes. This takes that buffer straight to a texture
// -- no PNG on disk in either direction -- and caches by (path, palette) so a
// strip that is on screen every frame is rendered once.

#include <string>

struct NVGcontext;

namespace spectrogram {

// Image handle for `wav_path` in `palette`, rendering it if this is the first
// ask. Returns -1 when the clip cannot be read. Repeat calls are cheap.
int image(NVGcontext* vg, const std::string& wav_path, const std::string& palette);

// Desaturated twin of the same render, drawn where a clip has not played yet.
// Built in the same pass, so asking for it never costs a second spectrogram.
int grey(NVGcontext* vg, const std::string& wav_path, const std::string& palette);

// Native size of a rendered strip, for a caller that wants its aspect ratio.
// Both zero when the pair has not been rendered.
void size(const std::string& wav_path, const std::string& palette, int* w, int* h);

// The palette used when none is named. spec_hud's first, unless overridden.
const std::string& default_palette();
void set_default_palette(const std::string& name);

void shutdown(NVGcontext* vg);

} // namespace spectrogram
