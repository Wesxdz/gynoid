#pragma once

#include <flecs.h>
#include <nanovg.h>

// Component to store mel spectrogram rendering state
struct MelSpecRender {
    int nvgTextureHandle;  // NanoVG texture handle
    int width;             // Texture width
    int height;            // Texture height
    unsigned char* imageData;  // RGB image data
    bool hasUpdate;        // Flag for pending texture update
    bool enabled;          // Whether to render
    float xOffset;         // X position offset from avatar
    float yOffset;         // Y position offset from avatar
    int zIndex;            // Render layer (above avatar = 310)
    float fillProgress;    // 0.0 to 1.0, how much of the texture is filled
};

// Configuration component
struct MelSpecConfig {
    bool enabled;
    float alpha;           // Opacity
    float scale;           // Display scale
    int zIndex;
};

// Initialize mel_spec rendering module
void MelSpecRenderModule(flecs::world& world);

// Cleanup function
void CleanupMelSpec();
