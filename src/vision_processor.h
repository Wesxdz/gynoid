/**
 * vision_processor.h
 *
 * C++ header for VNC framebuffer contour processing
 */

#ifndef VISION_PROCESSOR_H
#define VISION_PROCESSOR_H

#include <SDL2/SDL.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Process VNC framebuffer and generate vision overlay
 *
 * @param framebuffer Raw pixel data (BGRA format from SDL surface)
 * @param width Framebuffer width
 * @param height Framebuffer height
 * @param pitch Row stride in bytes
 * @param quadrantId Quadrant identifier (0-3)
 * @param paletteFile Path to palette hex file (e.g., "assets/palettes/resurrect-64.hex")
 * @param outputPath Path to save vision overlay PNG (e.g., "/tmp/vision_quad_0.png")
 * @return 1 on success, 0 on failure
 */
int process_vnc_vision(const uint8_t* framebuffer, int width, int height,
                      int pitch, int quadrantId,
                      const char* paletteFile, const char* outputPath);

/**
 * Process from SDL surface (convenience wrapper)
 *
 * @param surface SDL surface containing VNC framebuffer
 * @param quadrantId Quadrant identifier (0-3)
 * @param paletteFile Path to palette hex file
 * @param outputPath Path to save vision overlay PNG
 * @return 1 on success, 0 on failure
 */
int process_vnc_vision_from_surface(SDL_Surface* surface, int quadrantId,
                                   const char* paletteFile,
                                   const char* outputPath);

/**
 * Get changed tiles for debug rendering
 *
 * @param quadrantId Quadrant identifier (0-3)
 * @param tileX Output array for tile X coordinates
 * @param tileY Output array for tile Y coordinates
 * @param maxTiles Maximum number of tiles to return
 * @return Number of changed tiles
 */
int get_changed_tiles(int quadrantId, int* tileX, int* tileY, int maxTiles);

/**
 * Clear changed tiles for a quadrant (after consuming them)
 *
 * @param quadrantId Quadrant identifier (0-3)
 */
void clear_changed_tiles(int quadrantId);

/**
 * Tile cluster bounding rectangle (in tile coordinates)
 */
struct TileCluster {
    int minX, minY;  // Top-left corner in tile coords
    int maxX, maxY;  // Bottom-right corner in tile coords (inclusive)
    int tileCount;   // Number of tiles in this cluster
};

/**
 * Get tile clusters for a quadrant (groups of adjacent changed tiles)
 *
 * @param quadrantId Quadrant identifier (0-3)
 * @param clusters Output array for tile clusters
 * @param maxClusters Maximum number of clusters to return
 * @param minClusterSize Minimum number of tiles for a cluster (default 2)
 * @return Number of clusters found
 */
int get_tile_clusters(int quadrantId, TileCluster* clusters, int maxClusters, int minClusterSize);

/**
 * Get vision processor instance
 */
void* get_vision_processor_instance();

// Tile size constant (must match vision_processor.cpp)
#define TILE_SIZE 16

#ifdef __cplusplus
}
#endif

#endif // VISION_PROCESSOR_H
