#include "spectrogram.h"

#include <iostream>
#include <map>
#include <vector>

#include <nanovg.h>
#include <spec_hud.h>

namespace spectrogram {
namespace {

struct Entry
{
    int handle = -1;        // the palette as drawn
    int grey_handle = -1;   // luminance-only twin
    int width = 0;
    int height = 0;
};

std::map<std::string, Entry> g_cache;   // key: wav path + '|' + palette
std::string g_default = "fever";

// How bright the unplayed half sits relative to the palette's own luminance.
constexpr float kGreyLevel = 0.55f;

std::string key_of(const std::string& path, const std::string& palette)
{
    return path + "|" + palette;
}

} // namespace

int image(NVGcontext* vg, const std::string& wav_path, const std::string& palette)
{
    if (!vg || wav_path.empty()) return -1;

    const std::string name = palette.empty() ? g_default : palette;
    const std::string key = key_of(wav_path, name);

    const auto cached = g_cache.find(key);
    if (cached != g_cache.end()) return cached->second.handle;

    int w = 0, h = 0;
    unsigned char* rgb = spec_hud_render(wav_path.c_str(), name.c_str(), 0, &w, &h);
    if (!rgb || w <= 0 || h <= 0) {
        std::cerr << "[spectrogram] could not render " << wav_path << std::endl;
        // Cached as a failure so a missing clip is not re-attempted every frame.
        g_cache[key] = {};
        return -1;
    }

    // nanovg wants RGBA; spec_hud produces tightly packed RGB. Opaque
    // throughout: the palettes carry their own background, and a transparent
    // one would let the panel show through the quiet parts of the image.
    //
    // The grey twin is built here rather than on demand: it is the same walk
    // over the same buffer, and it means a strip can never be caught with one
    // half of its pair missing.
    const size_t n = (size_t)w * h;
    std::vector<unsigned char> rgba(n * 4);
    std::vector<unsigned char> grey(n * 4);
    for (size_t i = 0; i < n; ++i) {
        const unsigned char r = rgb[i * 3 + 0];
        const unsigned char g = rgb[i * 3 + 1];
        const unsigned char b = rgb[i * 3 + 2];
        rgba[i * 4 + 0] = r;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = b;
        rgba[i * 4 + 3] = 255;

        // Rec. 709 luminance, then pulled down: an equally bright grey reads
        // as "already played" at a glance, and the point is that colour
        // arrives with the playhead.
        const unsigned char y =
            (unsigned char)((0.2126f * r + 0.7152f * g + 0.0722f * b) * kGreyLevel);
        grey[i * 4 + 0] = y;
        grey[i * 4 + 1] = y;
        grey[i * 4 + 2] = y;
        grey[i * 4 + 3] = 255;
    }
    spec_hud_free(rgb);

    Entry entry;
    entry.width = w;
    entry.height = h;
    entry.handle = nvgCreateImageRGBA(vg, w, h, 0, rgba.data());
    entry.grey_handle = nvgCreateImageRGBA(vg, w, h, 0, grey.data());
    g_cache[key] = entry;

    std::cout << "[spectrogram] " << wav_path << " as " << name
              << " (" << w << "x" << h << ")" << std::endl;
    return entry.handle;
}

int grey(NVGcontext* vg, const std::string& wav_path, const std::string& palette)
{
    image(vg, wav_path, palette);   // ensures the pair exists
    const auto it = g_cache.find(key_of(wav_path, palette.empty() ? g_default : palette));
    return (it == g_cache.end()) ? -1 : it->second.grey_handle;
}

void size(const std::string& wav_path, const std::string& palette, int* w, int* h)
{
    if (w) *w = 0;
    if (h) *h = 0;
    const auto it = g_cache.find(key_of(wav_path, palette.empty() ? g_default : palette));
    if (it == g_cache.end()) return;
    if (w) *w = it->second.width;
    if (h) *h = it->second.height;
}

const std::string& default_palette() { return g_default; }

void set_default_palette(const std::string& name)
{
    if (spec_hud_palette_index(name.c_str()) >= 0) g_default = name;
}

void shutdown(NVGcontext* vg)
{
    if (vg) {
        for (auto& [key, entry] : g_cache) {
            if (entry.handle != -1) nvgDeleteImage(vg, entry.handle);
            if (entry.grey_handle != -1) nvgDeleteImage(vg, entry.grey_handle);
        }
    }
    g_cache.clear();
}

} // namespace spectrogram
