#pragma once

// Minimal STL reader producing render3d's vertex format. render3d deliberately
// ships no asset loaders, so this is the one place that knows about files.
//
// Triangles are emitted unwelded (three vertices per facet, all carrying the
// facet's normal), which is what gives an STL its flat, faceted shading. A
// welded mesh would need smoothing-group heuristics STL does not carry.

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <render3d/mesh.hpp>

struct StlMesh
{
    std::vector<r3d::Vertex> vertices;
    std::vector<uint32_t> indices;

    // Bounds in the file's own units, before any normalization.
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};

    bool valid = false;

    glm::vec3 center() const { return (min + max) * 0.5f; }
    glm::vec3 extent() const { return max - min; }
};

// Reads ASCII or binary STL. On failure returns a mesh with valid == false and
// logs the reason; callers are expected to carry on without the model.
StlMesh loadStl(const std::string& path);

// Recentre the mesh on the origin and scale it so its largest axis spans
// `target_size` world units. Bounds are updated to match.
void normalizeStl(StlMesh& mesh, float target_size);
