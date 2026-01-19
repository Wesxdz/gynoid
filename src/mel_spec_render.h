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

    // Time-based scroll synchronization (for sync with filmstrip)
    double scrollStartTime;      // Wall clock time when scroll mode started
    size_t totalScrollCommands;  // Number of scroll commands received since start
    float renderOffset;          // Calculated X offset to sync with wall clock (in normalized 0-1)
    static constexpr float COLUMNS_PER_SECOND = 22050.0f / 256.0f;  // ~86.13 columns/sec
    static constexpr float SCROLL_DURATION = 24.0f;  // Total window duration in seconds
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
