#include "stl_mesh.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace {

// A facet's normal is recomputed from its winding rather than trusted from the
// file: exporters emit zero or denormalized normals often enough that lighting
// goes black in patches, and STL already mandates counter-clockwise winding.
// The stored normal is the fallback for degenerate (zero-area) triangles.
glm::vec3 facetNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
                      const glm::vec3& stored)
{
    glm::vec3 n = glm::cross(b - a, c - a);
    float len = glm::length(n);
    if (len > 1e-12f) return n / len;

    float slen = glm::length(stored);
    return (slen > 1e-6f) ? stored / slen : glm::vec3(0.0f, 1.0f, 0.0f);
}

void pushFacet(StlMesh& mesh, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
               const glm::vec3& stored)
{
    const glm::vec3 n = facetNormal(a, b, c, stored);
    const uint32_t base = (uint32_t)mesh.vertices.size();

    mesh.vertices.push_back({a, n});
    mesh.vertices.push_back({b, n});
    mesh.vertices.push_back({c, n});
    mesh.indices.push_back(base);
    mesh.indices.push_back(base + 1);
    mesh.indices.push_back(base + 2);

    for (const glm::vec3& v : {a, b, c}) {
        mesh.min = glm::min(mesh.min, v);
        mesh.max = glm::max(mesh.max, v);
    }
}

// Advance past the next occurrence of `word`, returning the position just after
// it, or nullptr at end of buffer. Scanning for keywords rather than tokenizing
// line by line keeps a 4 MB ASCII file well under a frame's worth of work.
const char* seek(const char* p, const char* end, const char* word, size_t wordLen)
{
    while (p + wordLen <= end) {
        const char* hit = (const char*)memchr(p, word[0], (size_t)(end - p) - wordLen + 1);
        if (!hit) return nullptr;
        if (memcmp(hit, word, wordLen) == 0) return hit + wordLen;
        p = hit + 1;
    }
    return nullptr;
}

bool readVec3(const char*& p, const char* end, glm::vec3& out)
{
    char* next = nullptr;
    for (int i = 0; i < 3; ++i) {
        out[i] = strtof(p, &next);
        if (next == p) return false;
        p = next;
        if (p > end) return false;
    }
    return true;
}

// `buf` must be NUL-terminated (strtof needs it); `end` excludes the terminator.
bool parseAscii(const std::vector<char>& buf, StlMesh& mesh)
{
    const char* p = buf.data();
    const char* end = buf.data() + buf.size() - 1;

    while (true) {
        const char* n = seek(p, end, "normal", 6);
        if (!n) break;

        glm::vec3 stored{0.0f};
        if (!readVec3(n, end, stored)) break;
        p = n;

        glm::vec3 v[3];
        bool complete = true;
        for (int i = 0; i < 3; ++i) {
            const char* vp = seek(p, end, "vertex", 6);
            if (!vp || !readVec3(vp, end, v[i])) { complete = false; break; }
            p = vp;
        }
        if (!complete) break;

        pushFacet(mesh, v[0], v[1], v[2], stored);
    }

    return !mesh.vertices.empty();
}

bool parseBinary(const std::vector<char>& buf, StlMesh& mesh)
{
    // 80-byte header, uint32 triangle count, then 50 bytes per triangle.
    if (buf.size() < 84) return false;

    uint32_t count = 0;
    memcpy(&count, buf.data() + 80, sizeof(count));
    if (buf.size() < 84 + (size_t)count * 50) return false;

    mesh.vertices.reserve((size_t)count * 3);
    mesh.indices.reserve((size_t)count * 3);

    const char* p = buf.data() + 84;
    for (uint32_t i = 0; i < count; ++i, p += 50) {
        float f[12];
        memcpy(f, p, sizeof(f));
        pushFacet(mesh,
                  {f[3], f[4], f[5]},
                  {f[6], f[7], f[8]},
                  {f[9], f[10], f[11]},
                  {f[0], f[1], f[2]});
    }
    return count > 0;
}

// "solid" is not a reliable ASCII marker -- plenty of binary exporters write it
// into the 80-byte header -- so the triangle count is checked against the file
// size instead, which is exact for binary and essentially never matches ASCII.
bool looksBinary(const std::vector<char>& buf)
{
    if (buf.size() < 84) return false;
    uint32_t count = 0;
    memcpy(&count, buf.data() + 80, sizeof(count));
    return buf.size() == 84 + (size_t)count * 50;
}

} // namespace

StlMesh loadStl(const std::string& path)
{
    StlMesh mesh;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[stl] cannot open " << path << std::endl;
        return mesh;
    }

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf((size_t)size);
    if (size <= 0 || !file.read(buf.data(), size)) {
        std::cerr << "[stl] cannot read " << path << std::endl;
        return mesh;
    }

    mesh.min = glm::vec3(std::numeric_limits<float>::max());
    mesh.max = glm::vec3(std::numeric_limits<float>::lowest());

    bool ok;
    if (looksBinary(buf)) {
        ok = parseBinary(buf, mesh);
    } else {
        buf.push_back('\0');   // parseAscii runs strtof over this buffer
        ok = parseAscii(buf, mesh);
    }
    if (!ok) {
        std::cerr << "[stl] no triangles parsed from " << path << std::endl;
        return {};
    }

    mesh.valid = true;
    std::cout << "[stl] " << path << ": " << (mesh.indices.size() / 3) << " triangles" << std::endl;
    return mesh;
}

void normalizeStl(StlMesh& mesh, float target_size)
{
    if (!mesh.valid || mesh.vertices.empty()) return;

    const glm::vec3 center = mesh.center();
    const glm::vec3 extent = mesh.extent();
    const float largest = glm::max(extent.x, glm::max(extent.y, extent.z));
    const float scale = (largest > 1e-6f) ? (target_size / largest) : 1.0f;

    for (r3d::Vertex& v : mesh.vertices) {
        v.pos = (v.pos - center) * scale;
    }

    // Normals are unaffected: a uniform scale about the centroid preserves them.
    const glm::vec3 half = extent * scale * 0.5f;
    mesh.min = -half;
    mesh.max = half;
}
