#pragma once

#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>

/**
 * Fast frame difference detector using downscaled grayscale comparison.
 * No ML, runs at 1000+ FPS on CPU.
 */
class FrameDiff {
public:
    FrameDiff(int thumbSize = 16, int maxFrames = 10)
        : thumbSize(thumbSize), maxFrames(maxFrames) {}

    // Compute difference from rolling average (0.0 = identical, 1.0 = completely different)
    // Returns frameCount in rolling average
    float computeDiff(const uint8_t* bgraPixels, int width, int height, int pitch, int& frameCount) {
        // Downscale to thumbnail grayscale
        std::vector<float> thumb = downscaleToGray(bgraPixels, width, height, pitch);

        // Compute diff from rolling average
        float diff = 0.0f;
        if (!rollingAvg.empty()) {
            float sumDiff = 0.0f;
            for (size_t i = 0; i < thumb.size(); i++) {
                float d = thumb[i] - rollingAvg[i];
                sumDiff += d * d;
            }
            diff = std::sqrt(sumDiff / thumb.size()) / 255.0f;  // Normalize to 0-1
        }

        frameCount = (int)history.size();

        // Update rolling average
        updateRollingAvg(thumb);
        framesProcessed++;

        return diff;
    }

    uint64_t getFramesProcessed() const { return framesProcessed; }

private:
    int thumbSize;
    int maxFrames;
    std::deque<std::vector<float>> history;
    std::vector<float> rollingAvg;
    uint64_t framesProcessed = 0;

    std::vector<float> downscaleToGray(const uint8_t* bgra, int w, int h, int pitch) {
        std::vector<float> thumb(thumbSize * thumbSize);

        float scaleX = (float)w / thumbSize;
        float scaleY = (float)h / thumbSize;

        for (int ty = 0; ty < thumbSize; ty++) {
            for (int tx = 0; tx < thumbSize; tx++) {
                // Sample from center of each cell
                int sx = (int)((tx + 0.5f) * scaleX);
                int sy = (int)((ty + 0.5f) * scaleY);

                const uint8_t* p = bgra + sy * pitch + sx * 4;
                // BGRA -> grayscale (simple average)
                float gray = (p[0] + p[1] + p[2]) / 3.0f;
                thumb[ty * thumbSize + tx] = gray;
            }
        }

        return thumb;
    }

    void updateRollingAvg(const std::vector<float>& thumb) {
        history.push_back(thumb);
        while ((int)history.size() > maxFrames) {
            history.pop_front();
        }

        // Recompute average
        rollingAvg.assign(thumbSize * thumbSize, 0.0f);
        for (const auto& h : history) {
            for (size_t i = 0; i < h.size(); i++) {
                rollingAvg[i] += h[i];
            }
        }
        float scale = 1.0f / history.size();
        for (auto& v : rollingAvg) {
            v *= scale;
        }
    }
};
