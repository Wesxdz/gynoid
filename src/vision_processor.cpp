/**
 * vision_processor.cpp
 *
 * Processes contours directly from VNC streamed framebuffers.
 */

#include "vision_processor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <sys/time.h>

// LibVNC for direct VNC framebuffer access
#include <rfb/rfbclient.h>

// SDL for surface handling
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

// TILE_SIZE now comes from vision_processor.h
constexpr int MIN_CONTOUR_AREA = 10;
constexpr int MIN_WIDTH = 2;
constexpr int MIN_HEIGHT = 2;

// Threshold for mean absolute difference per tile to be considered "changed"
// Range: 0-255 (pixel value range)
// Higher = less sensitive (only large changes detected)
// Lower = more sensitive (even small changes detected)
constexpr double TILE_DIFF_THRESHOLD = 5.0;

struct TileCoord {
    int x, y;

    bool operator==(const TileCoord& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    template<>
    struct hash<TileCoord> {
        size_t operator()(const TileCoord& coord) const {
            return hash<int>()(coord.x) ^ (hash<int>()(coord.y) << 1);
        }
    };
}

class VisionProcessor {
private:
    // Per-quadrant frame storage (BGR format)
    std::unordered_map<int, cv::Mat> frameCache;
    std::unordered_map<int, cv::Mat> visionCache;
    std::mutex cacheLock;

    // Palette colors (BGR format)
    std::vector<cv::Scalar> palette;

    // Debug: store changed tiles per quadrant
    std::unordered_map<int, std::vector<TileCoord>> changedTilesDebug;
    std::mutex debugMutex;

public:
    VisionProcessor(const std::string& paletteFile) {
        loadPalette(paletteFile);
    }

    /**
     * Get the list of changed tiles for a quadrant (for debug rendering)
     */
    std::vector<TileCoord> getChangedTiles(int quadrantId) {
        std::lock_guard<std::mutex> lock(debugMutex);
        auto it = changedTilesDebug.find(quadrantId);
        if (it != changedTilesDebug.end()) {
            return it->second;
        }
        return {};
    }

    void clearChangedTiles(int quadrantId) {
        std::lock_guard<std::mutex> lock(debugMutex);
        auto it = changedTilesDebug.find(quadrantId);
        if (it != changedTilesDebug.end()) {
            it->second.clear();
        }
    }

    /**
     * Compute tile clusters from changed tiles using flood-fill
     * Groups adjacent tiles into bounding rectangles
     *
     * @param quadrantId Quadrant identifier
     * @param minClusterSize Minimum tiles per cluster (default 2)
     * @return Vector of TileCluster structs
     */
    std::vector<TileCluster> computeTileClusters(int quadrantId, int minClusterSize = 2) {
        std::vector<TileCoord> tiles = getChangedTiles(quadrantId);
        std::vector<TileCluster> clusters;

        if (tiles.empty()) {
            return clusters;
        }

        // Convert tiles to a set for O(1) lookup
        std::unordered_set<TileCoord> tileSet(tiles.begin(), tiles.end());
        std::unordered_set<TileCoord> visited;

        // Flood-fill to find connected components
        for (const auto& startTile : tiles) {
            if (visited.count(startTile) > 0) {
                continue;
            }

            // Start a new cluster with BFS/flood-fill
            std::vector<TileCoord> clusterTiles;
            std::vector<TileCoord> queue = {startTile};
            visited.insert(startTile);

            while (!queue.empty()) {
                TileCoord current = queue.back();
                queue.pop_back();
                clusterTiles.push_back(current);

                // Check 4-connected neighbors (up, down, left, right)
                TileCoord neighbors[4] = {
                    {current.x - 1, current.y},
                    {current.x + 1, current.y},
                    {current.x, current.y - 1},
                    {current.x, current.y + 1}
                };

                for (const auto& neighbor : neighbors) {
                    if (tileSet.count(neighbor) > 0 && visited.count(neighbor) == 0) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            // Only include clusters with at least minClusterSize tiles
            if (clusterTiles.size() >= static_cast<size_t>(minClusterSize)) {
                // Compute bounding rectangle
                int minX = clusterTiles[0].x;
                int maxX = clusterTiles[0].x;
                int minY = clusterTiles[0].y;
                int maxY = clusterTiles[0].y;

                for (const auto& tile : clusterTiles) {
                    minX = std::min(minX, tile.x);
                    maxX = std::max(maxX, tile.x);
                    minY = std::min(minY, tile.y);
                    maxY = std::max(maxY, tile.y);
                }

                TileCluster cluster;
                cluster.minX = minX;
                cluster.minY = minY;
                cluster.maxX = maxX;
                cluster.maxY = maxY;
                cluster.tileCount = static_cast<int>(clusterTiles.size());

                clusters.push_back(cluster);
            }
        }

        return clusters;
    }

    /**
     * Load hex palette file and convert to BGR format
     */
    void loadPalette(const std::string& paletteFile) {
        std::ifstream file(paletteFile);
        if (!file.is_open()) {
            std::cerr << "[VISION] Failed to open palette file: " << paletteFile << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            // Remove '#' if present
            if (line[0] == '#') line = line.substr(1);

            // Parse hex color
            if (line.length() >= 6) {
                int r = std::stoi(line.substr(0, 2), nullptr, 16);
                int g = std::stoi(line.substr(2, 2), nullptr, 16);
                int b = std::stoi(line.substr(4, 2), nullptr, 16);
                palette.emplace_back(b, g, r); // BGR format for OpenCV
            }
        }

        std::cout << "[VISION] Loaded " << palette.size() << " colors from palette" << std::endl;
    }

    /**
     * Compute mean absolute difference between two tiles
     *
     * @param tile1 First tile (BGR format)
     * @param tile2 Second tile (BGR format)
     * @return Mean absolute difference (0-255 range)
     */
    double computeTileDiff(const cv::Mat& tile1, const cv::Mat& tile2) {
        if (tile1.size() != tile2.size() || tile1.type() != tile2.type()) {
            return 255.0; // Maximum difference if sizes/types don't match
        }

        // Compute absolute difference
        cv::Mat diff;
        cv::absdiff(tile1, tile2, diff);

        // Compute mean across all channels
        cv::Scalar meanDiff = cv::mean(diff);

        // Average across BGR channels
        double avgDiff = (meanDiff[0] + meanDiff[1] + meanDiff[2]) / 3.0;

        return avgDiff;
    }

    /**
     * Process a single tile and return contour overlay
     */
    cv::Mat processTile(const cv::Mat& tile) {
        // Create transparent overlay for this tile
        cv::Mat tileOverlay = cv::Mat::zeros(tile.rows, tile.cols, CV_8UC4);

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(tile, gray, cv::COLOR_BGR2GRAY);

        // Blur and edge detection
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);

        cv::Mat edged;
        cv::Canny(blurred, edged, 20, 150);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edged, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        // Y-coordinate to color mapping
        std::unordered_map<int, cv::Scalar> yColorMap;

        // Draw contours
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Rect bbox = cv::boundingRect(contours[i]);

            // Assign color based on Y coordinate
            if (yColorMap.find(bbox.y) == yColorMap.end()) {
                if (!palette.empty()) {
                    yColorMap[bbox.y] = palette[yColorMap.size() % palette.size()];
                } else {
                    yColorMap[bbox.y] = cv::Scalar(255, 255, 255);
                }
            }

            cv::Scalar color = yColorMap[bbox.y];

            // Filter small contours
            if (bbox.width * bbox.height >= MIN_CONTOUR_AREA &&
                bbox.width > MIN_WIDTH &&
                bbox.height > MIN_HEIGHT) {

                // Draw contour with transparency
                cv::drawContours(tileOverlay, contours, i,
                               cv::Scalar(color[0], color[1], color[2], 255), 1);
                cv::drawContours(tileOverlay, contours, i,
                               cv::Scalar(color[0], color[1], color[2], 64), 3);
            }
        }

        return tileOverlay;
    }

    /**
     * Process VNC framebuffer directly with tile-based incremental updates
     *
     * @param framebuffer Raw pixel data from VNC client
     * @param width Framebuffer width
     * @param height Framebuffer height
     * @param pitch Row stride in bytes
     * @param quadrantId Quadrant identifier (0-3)
     * @param outputPath Path to save vision overlay PNG
     * @return true if processing succeeded
     */
    bool processFramebuffer(const uint8_t* framebuffer, int width, int height,
                           int pitch, int quadrantId, const std::string& outputPath) {
        if (!framebuffer || width <= 0 || height <= 0) {
            std::cerr << "[VISION] Invalid framebuffer parameters" << std::endl;
            return false;
        }

        // Convert framebuffer to OpenCV Mat (assume BGRA format from SDL surface)
        cv::Mat screenshot(height, width, CV_8UC4, (void*)framebuffer, pitch);

        // Convert BGRA to BGR for processing
        cv::Mat bgrScreenshot;
        cv::cvtColor(screenshot, bgrScreenshot, cv::COLOR_BGRA2BGR);

        // Calculate tile grid
        int tilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
        int tilesY = (height + TILE_SIZE - 1) / TILE_SIZE;

        std::vector<TileCoord> changedTiles;

        // Check which tiles have changed using frame differencing
        {
            std::lock_guard<std::mutex> lock(cacheLock);

            // Get previous frame if it exists
            auto frameIt = frameCache.find(quadrantId);
            bool hasPreviousFrame = (frameIt != frameCache.end() &&
                                     !frameIt->second.empty() &&
                                     frameIt->second.size() == bgrScreenshot.size());

            if (hasPreviousFrame) {
                const cv::Mat& previousFrame = frameIt->second;

                // Compare tiles between current and previous frame
                for (int ty = 0; ty < tilesY; ty++) {
                    for (int tx = 0; tx < tilesX; tx++) {
                        int tileX = tx * TILE_SIZE;
                        int tileY = ty * TILE_SIZE;
                        int tileW = std::min(TILE_SIZE, width - tileX);
                        int tileH = std::min(TILE_SIZE, height - tileY);

                        cv::Rect tileRect(tileX, tileY, tileW, tileH);
                        cv::Mat currentTile = bgrScreenshot(tileRect);
                        cv::Mat previousTile = previousFrame(tileRect);

                        // Compute mean absolute difference
                        double diff = computeTileDiff(currentTile, previousTile);

                        // Mark as changed if difference exceeds threshold
                        if (diff > TILE_DIFF_THRESHOLD) {
                            TileCoord coord = {tx, ty};
                            changedTiles.push_back(coord);
                        }
                    }
                }
            } else {
                // No previous frame - mark all tiles as changed
                for (int ty = 0; ty < tilesY; ty++) {
                    for (int tx = 0; tx < tilesX; tx++) {
                        TileCoord coord = {tx, ty};
                        changedTiles.push_back(coord);
                    }
                }
            }

            // Store current frame for next comparison
            frameCache[quadrantId] = bgrScreenshot.clone();
        }

        // Store changed tiles for debug rendering
        {
            std::lock_guard<std::mutex> lock(debugMutex);
            changedTilesDebug[quadrantId] = changedTiles;
        }

        if (changedTiles.empty()) {
            std::cout << "[VISION] Quadrant " << quadrantId
                      << " no tiles changed, skipping" << std::endl;
            return true;
        }

        std::cout << "[VISION] Quadrant " << quadrantId
                  << " processing " << changedTiles.size()
                  << "/" << (tilesX * tilesY) << " changed tiles" << std::endl;

        // Get or create vision overlay
        cv::Mat visionOverlay;
        {
            std::lock_guard<std::mutex> lock(cacheLock);
            auto it = visionCache.find(quadrantId);
            if (it != visionCache.end()) {
                visionOverlay = it->second.clone();
            } else {
                visionOverlay = cv::Mat::zeros(height, width, CV_8UC4);
            }
        }

        // Process only changed tiles
        for (const auto& coord : changedTiles) {
            int tileX = coord.x * TILE_SIZE;
            int tileY = coord.y * TILE_SIZE;
            int tileW = std::min(TILE_SIZE, width - tileX);
            int tileH = std::min(TILE_SIZE, height - tileY);

            // Extract tile from BGR screenshot
            cv::Rect tileRect(tileX, tileY, tileW, tileH);
            cv::Mat tile = bgrScreenshot(tileRect);

            // Process tile
            cv::Mat tileOverlay = processTile(tile);

            // Update vision overlay
            tileOverlay.copyTo(visionOverlay(tileRect));
        }

        // Cache the updated overlay
        {
            std::lock_guard<std::mutex> lock(cacheLock);
            visionCache[quadrantId] = visionOverlay;
        }

        // Save output with atomic write using SDL_image (like the VNC screenshot saving)
        std::string tempPath = outputPath + ".tmp";

        // Convert visionOverlay (CV_8UC4 BGRA) to SDL_Surface for PNG writing
        SDL_Surface* outputSurface = SDL_CreateRGBSurfaceFrom(
            visionOverlay.data,
            visionOverlay.cols,
            visionOverlay.rows,
            32,  // bits per pixel (BGRA = 8*4 = 32)
            visionOverlay.step[0],  // pitch
            0x00FF0000,  // R mask (BGRA format)
            0x0000FF00,  // G mask
            0x000000FF,  // B mask
            0xFF000000   // A mask
        );

        if (!outputSurface) {
            std::cerr << "[VISION ERROR] Failed to create SDL surface for quadrant "
                      << quadrantId << std::endl;
            return false;
        }

        // Save using SDL_image (same as VNC screenshot saving)
        if (IMG_SavePNG(outputSurface, tempPath.c_str()) != 0) {
            std::cerr << "[VISION ERROR] Failed to save PNG for quadrant "
                      << quadrantId << ": " << SDL_GetError() << std::endl;
            SDL_FreeSurface(outputSurface);
            return false;
        }

        SDL_FreeSurface(outputSurface);

        // Atomic rename
        if (rename(tempPath.c_str(), outputPath.c_str()) != 0) {
            std::cerr << "[VISION ERROR] Failed to rename temp file for quadrant "
                      << quadrantId << std::endl;
            return false;
        }

        std::cout << "[VISION] Quadrant " << quadrantId
                  << " processed -> " << outputPath << std::endl;

        return true;
    }

    /**
     * Process from SDL surface (convenience wrapper)
     */
    bool processFromSurface(SDL_Surface* surface, int quadrantId,
                           const std::string& outputPath) {
        if (!surface || !surface->pixels) {
            std::cerr << "[VISION] Invalid SDL surface" << std::endl;
            return false;
        }

        return processFramebuffer(
            static_cast<const uint8_t*>(surface->pixels),
            surface->w,
            surface->h,
            surface->pitch,
            quadrantId,
            outputPath
        );
    }
};

// Shared vision processor instance - single source of truth
static VisionProcessor* g_visionProcessor = nullptr;
static std::string g_lastPalette;

// Get or create the shared vision processor instance
static VisionProcessor* getSharedProcessor(const char* paletteFile = nullptr) {
    if (paletteFile && (!g_visionProcessor || g_lastPalette != paletteFile)) {
        delete g_visionProcessor;
        g_visionProcessor = new VisionProcessor(paletteFile);
        g_lastPalette = paletteFile;
        std::cout << "[VISION] Created shared processor with palette: " << paletteFile << std::endl;
    }

    // Initialize with default palette if not yet initialized
    if (!g_visionProcessor) {
        g_visionProcessor = new VisionProcessor("assets/palettes/resurrect-64.hex");
        g_lastPalette = "assets/palettes/resurrect-64.hex";
        std::cout << "[VISION] Created shared processor with default palette" << std::endl;
    }

    return g_visionProcessor;
}

// Example usage function that can be called from main.cpp
extern "C" {
    /**
     * Process VNC framebuffer and generate vision overlay
     *
     * @param framebuffer Raw pixel data (BGRA format)
     * @param width Framebuffer width
     * @param height Framebuffer height
     * @param pitch Row stride in bytes
     * @param quadrantId Quadrant identifier (0-3)
     * @param paletteFile Path to palette hex file
     * @param outputPath Path to save vision overlay PNG
     * @return 1 on success, 0 on failure
     */
    int process_vnc_vision(const uint8_t* framebuffer, int width, int height,
                          int pitch, int quadrantId,
                          const char* paletteFile, const char* outputPath) {
        VisionProcessor* processor = getSharedProcessor(paletteFile);
        return processor->processFramebuffer(
            framebuffer, width, height, pitch,
            quadrantId, outputPath
        ) ? 1 : 0;
    }

    /**
     * Process from SDL surface
     */
    int process_vnc_vision_from_surface(SDL_Surface* surface, int quadrantId,
                                       const char* paletteFile,
                                       const char* outputPath) {
        VisionProcessor* processor = getSharedProcessor(paletteFile);
        return processor->processFromSurface(surface, quadrantId, outputPath) ? 1 : 0;
    }

    /**
     * Get changed tiles for debug rendering
     * Returns number of changed tiles, fills output arrays
     */
    int get_changed_tiles(int quadrantId, int* tileX, int* tileY, int maxTiles) {
        VisionProcessor* processor = getSharedProcessor();

        auto tiles = processor->getChangedTiles(quadrantId);
        int count = std::min((int)tiles.size(), maxTiles);

        for (int i = 0; i < count; i++) {
            tileX[i] = tiles[i].x;
            tileY[i] = tiles[i].y;
        }

        return count;
    }

    /**
     * Clear changed tiles for a quadrant (after consuming them)
     */
    void clear_changed_tiles(int quadrantId) {
        VisionProcessor* processor = getSharedProcessor();
        processor->clearChangedTiles(quadrantId);
    }

    /**
     * Get tile clusters for a quadrant
     * Returns number of clusters found, fills output array
     */
    int get_tile_clusters(int quadrantId, TileCluster* clusters, int maxClusters, int minClusterSize) {
        VisionProcessor* processor = getSharedProcessor();

        auto clusterVec = processor->computeTileClusters(quadrantId, minClusterSize);
        int count = std::min((int)clusterVec.size(), maxClusters);

        for (int i = 0; i < count; i++) {
            clusters[i] = clusterVec[i];
        }

        return count;
    }

    /**
     * Get the vision processor instance (for accessing methods)
     */
    void* get_vision_processor_instance() {
        return getSharedProcessor();
    }
}
