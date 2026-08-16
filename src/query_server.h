#pragma once

#include <flecs.h>
#include "spatial_index.h"
#include <functional>
#include <string>
#include <unordered_map>

namespace query_server {

// Spatial query handler callback for application-specific 2D queries
// Takes: query_string, spatial_index
// Returns: set of entity IDs matching the spatial query, or empty set if not handled
using SpatialQueryHandler = std::function<std::unordered_set<uint64_t>(
    const char*, spatial::SpatialIndexManager*)>;

// Initialize the query server with world and spatial index
void initialize(flecs::world* world, spatial::SpatialIndexManager* spatial_index);

// Register a spatial query handler for application-specific 2D spatial queries
// This allows the application to provide component-type-specific spatial query implementations
void register_spatial_handler(SpatialQueryHandler handler);

// Start the socket server on specified port
// This function spawns a background thread and returns immediately
void start_server(int port);

// Answer every request that has arrived since the last call. Must be called
// regularly (once per frame) from the thread that owns the world: connection
// threads only read requests and park them -- all world access happens here,
// so the server never races the host's own frame.
void process_pending();

} // namespace query_server
