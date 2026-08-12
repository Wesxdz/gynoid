#pragma once

// The VNC connection pool's public face. get_vnc_source refcounts connections
// by host:port, so any number of panels (flat stream, Holodeck, future
// agent-desktop views) can watch one session over one socket.
//
// The definition still lives in main.cpp; this header lets panel modules
// consume the pool without including the monolith.

#include <string>

#include <flecs.h>

#include "../vnc_struct.h"

struct VNCData
{
    flecs::entity vnc_stream;
    VNCClientHandle client;  // Stable handle that survives ECS moves
};

VNCData get_vnc_source(flecs::entity user_state_leaf, const std::string& host, int port);
