
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <functional>

#define GLAD_GL_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>


#include "thorn_vfx.h"

// Shader sources for 3D plane rendering
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aBary;
layout (location = 3) in float aGlow;
layout (location = 4) in vec2 aCentroidOffset;

out vec2 TexCoord;
out vec3 Bary;
out float Glow;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int glowPass;  // 0 = normal, 1 = outer glow pass
uniform float glowExpand;  // How much to expand for glow (e.g., 0.015)

void main()
{
    vec3 pos = aPos;

    // In glow pass, expand vertices outward from centroid
    // Glow encoding: 0-2 = normal glow, 10+ = central pulse (10 = start, 12 = fully expanded)
    if (glowPass == 1 && aGlow > 0.0) {
        vec2 expandDir = normalize(aCentroidOffset);
        float expandAmount;

        if (aGlow >= 10.0) {
            // Central pulse: decode scale and progress
            // Format: glow = 10.0 + (scale-0.6)*5.0 + progress*0.5
            float encoded = aGlow - 10.0;
            float scaleEnc = floor(encoded / 0.5) * 0.5;  // Quantized scale portion
            float pulseProgress = clamp((encoded - scaleEnc) / 0.5, 0.0, 1.0);
            float pulseScale = scaleEnc / 5.0 + 0.6;  // Recover 0.6-1.4 range

            // Ease out for smooth deceleration as it expands
            float easedProgress = 1.0 - (1.0 - pulseProgress) * (1.0 - pulseProgress);

            // Add rotational asymmetry using centroid offset angle
            float angle = atan(aCentroidOffset.y, aCentroidOffset.x);
            float wobble = 1.0 + 0.3 * sin(angle * 3.0 + pulseScale * 10.0);  // Asymmetric shape

            expandAmount = glowExpand * pulseScale * 2.0 * easedProgress * 20.0 * wobble;
        } else {
            expandAmount = glowExpand * aGlow;
        }
        pos.xy += expandDir * expandAmount;
    }

    gl_Position = projection * view * model * vec4(pos, 1.0);
    TexCoord = aTexCoord;
    Bary = aBary;
    Glow = aGlow;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Bary;
in float Glow;

uniform sampler2D uiTexture;
uniform int glowPass;  // 0 = normal, 1 = outer glow pass
uniform float chromaStrength;  // Chromatic aberration intensity

void main()
{
    if (glowPass == 1) {
        // Outer glow pass: soft diffuse glow
        if (Glow <= 0.0) discard;

        // Distance from center using barycentric (0.33 at center, 0 at edges)
        float minBary = min(min(Bary.x, Bary.y), Bary.z);

        // Check if this is a central pulse (glow >= 10.0)
        if (Glow >= 10.0) {
            // Central pulse with rounded corners - decode scale and progress
            float encoded = Glow - 10.0;
            float scaleEnc = floor(encoded / 0.5) * 0.5;
            float pulseProgress = clamp((encoded - scaleEnc) / 0.5, 0.0, 1.0);
            float pulseScale = scaleEnc / 5.0 + 0.6;

            // Create rounded corners by using a smooth distance from edges
            // Transform barycentric to a rounded shape - vary by scale
            float cornerRadius = 0.1 + pulseScale * 0.1;  // Bigger pulses = rounder corners
            float smoothEdge = smoothstep(0.0, cornerRadius, minBary);

            // Radial falloff from center for soft edge blur
            float centerDist = 1.0 - minBary * 3.0;  // 0 at center, 1 at edges
            float edgeSoftness = 0.3;
            float radialFalloff = 1.0 - smoothstep(1.0 - edgeSoftness, 1.0, centerDist);

            // Combine rounded corners with radial falloff
            float shapeMask = smoothEdge * radialFalloff;

            // Intensity fades as pulse expands outward - vary by scale
            float fadeIntensity = 1.0 - pulseProgress * (0.5 + pulseScale * 0.3);
            // Add a bright leading edge that travels outward - vary ring width by scale
            float ringWidth = 0.15 + pulseScale * 0.15;
            float ringPos = pulseProgress;
            float normalizedDist = centerDist;
            float ringIntensity = exp(-pow((normalizedDist - ringPos) / ringWidth, 2.0) * 2.0);

            float glowIntensity = (fadeIntensity * 0.15 + ringIntensity * 0.3) * shapeMask;
            glowIntensity *= 0.5;  // Subtle enough to see character beneath

            // Warm shield pulse color (slightly cyan-shifted for energy feel)
            vec3 glowColor = mix(vec3(1.0, 0.95, 0.7), vec3(0.7, 0.95, 1.0), pulseProgress * 0.3);

            FragColor = vec4(glowColor * glowIntensity, glowIntensity * 0.4);
        } else {
            // Normal glow for outer triangles
            // Soft radial falloff - bright at outer edge, fading inward
            float edgeness = 1.0 - minBary * 3.0;  // 1 at edges, 0 at center

            // Very gentle cubic falloff for soft blur
            float t = clamp(edgeness, 0.0, 1.0);
            float glowIntensity = t * t * (3.0 - 2.0 * t);  // Smooth hermite
            glowIntensity *= Glow * 0.35;  // Subtle intensity

            // Soft warm yellow glow
            vec3 glowColor = vec3(1.0, 0.92, 0.6);

            FragColor = vec4(glowColor * glowIntensity, glowIntensity * 0.6);
        }
    } else {
        // Normal pass: render textured triangle
        // Use transparency for UVs outside the 0-1 range (edge triangles)
        if (TexCoord.x < 0.0 || TexCoord.x > 1.0 || TexCoord.y < 0.0 || TexCoord.y > 1.0) {
            FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        } else {
            // Chromatic aberration: offset RGB channels radially from center
            vec2 center = vec2(0.5, 0.5);
            vec2 dir = TexCoord - center;
            float dist = length(dir);
            vec2 offset = dir * chromaStrength * dist;  // Stronger at edges

            // Sample each channel at slightly different positions
            float r = texture(uiTexture, TexCoord + offset).r;
            float g = texture(uiTexture, TexCoord).g;
            float b = texture(uiTexture, TexCoord - offset).b;
            float a = texture(uiTexture, TexCoord).a;

            FragColor = vec4(r, g, b, a);
        }
    }
}
)";

// Compile shader helper
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
    }

    return shader;
}

// Helper function to check if a point is inside a triangle using barycentric coordinates
bool pointInTriangle(float px, float py,
                     float ax, float ay, float bx, float by, float cx, float cy) {
    float v0x = cx - ax, v0y = cy - ay;
    float v1x = bx - ax, v1y = by - ay;
    float v2x = px - ax, v2y = py - ay;

    float dot00 = v0x * v0x + v0y * v0y;
    float dot01 = v0x * v1x + v0y * v1y;
    float dot02 = v0x * v2x + v0y * v2y;
    float dot11 = v1x * v1x + v1y * v1y;
    float dot12 = v1x * v2x + v1y * v2y;

    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

// Generate flat triangle grid pattern (alternating up/down triangles)
// Each triangle has 3 vertices and 1 face
void generateTriangularGrid(std::vector<float>& vertices, std::vector<unsigned int>& indices,
                           float width, float height, int subdivisionsX, int subdivisionsY) {
    vertices.clear();
    indices.clear();

    int vertexIndex = 0;

    // Calculate triangle dimensions for triangular tiling
    float triWidth = width / subdivisionsX;
    float triHeight = triWidth * 0.866025f; // sqrt(3)/2 for equilateral triangles

    // Giant triangle boundary (upward pointing - flat bottom edge, point at top)
    float triSize = std::min(width, height) * 0.4f;
    // Snap bottom Y to nearest row boundary for clean flat edge alignment
    float giantBottomY = -triSize * 0.4f;
    giantBottomY = floor(giantBottomY / triHeight) * triHeight;  // Snap to grid row
    float giantTopY = giantBottomY + triSize * 0.866025f;  // Point at top
    float halfBase = (giantTopY - giantBottomY) / 0.866025f * 0.5f;  // Half base width

    float giantTriAx = -halfBase, giantTriAy = giantBottomY;   // Bottom left (flat edge)
    float giantTriBx =  halfBase, giantTriBy = giantBottomY;   // Bottom right (flat edge)
    float giantTriCx =  0.0f,     giantTriCy = giantTopY;      // Top center (point)

    // Adjust number of rows based on height
    int numRows = (int)(height / triHeight) + 1;

    // Generate triangle tiling pattern
    for (int row = 0; row < numRows; row++) {
        // Calculate Y position for this row
        float rowY = (row * triHeight) - height * 0.5f;

        // Determine if this is an even or odd row (for offset)
        bool isEvenRow = (row % 2 == 0);

        // Number of triangles in this row (add extra for edge coverage)
        int numTrisInRow = subdivisionsX * 2 + 2;

        for (int col = -1; col < numTrisInRow; col++) {
            // Determine if this is an upward or downward pointing triangle
            bool isUpward = (col % 2 == 0);

            // Calculate base X position (start one column earlier to fill left gap)
            float baseX = (col * triWidth * 0.5f) - width * 0.5f;
            if (!isEvenRow) {
                baseX -= triWidth * 0.25f; // Offset odd rows left to fill gap
            }

            float x0, x1, x2, y0, y1, y2;

            if (isUpward) {
                // Upward pointing triangle (△)
                x0 = baseX;                    // left
                x1 = baseX + triWidth;         // right
                x2 = baseX + triWidth * 0.5f;  // top (center)

                y0 = rowY;
                y1 = rowY;
                y2 = rowY + triHeight;
            } else {
                // Downward pointing triangle (▽)
                x0 = baseX;                    // left
                x1 = baseX + triWidth;         // right
                x2 = baseX + triWidth * 0.5f;  // bottom (center)

                y0 = rowY + triHeight;
                y1 = rowY + triHeight;
                y2 = rowY;
            }

            // Calculate UV coordinates
            float u0 = (x0 + width * 0.5f) / width;
            float u1 = (x1 + width * 0.5f) / width;
            float u2 = (x2 + width * 0.5f) / width;

            float v0 = (y0 + height * 0.5f) / height;
            float v1 = (y1 + height * 0.5f) / height;
            float v2 = (y2 + height * 0.5f) / height;

            // Calculate centroid for offset computation
            float centX = (x0 + x1 + x2) / 3.0f;
            float centY = (y0 + y1 + y2) / 3.0f;

            // Vertex 0 (barycentric: 1,0,0)
            vertices.push_back(x0);
            vertices.push_back(y0);
            vertices.push_back(0.0f);  // Flat on Z=0
            vertices.push_back(u0);
            vertices.push_back(v0);
            vertices.push_back(1.0f);  // baryX
            vertices.push_back(0.0f);  // baryY
            vertices.push_back(0.0f);  // baryZ
            vertices.push_back(0.0f);  // glow
            vertices.push_back(x0 - centX);  // offsetX (direction from centroid)
            vertices.push_back(y0 - centY);  // offsetY

            // Vertex 1 (barycentric: 0,1,0)
            vertices.push_back(x1);
            vertices.push_back(y1);
            vertices.push_back(0.0f);
            vertices.push_back(u1);
            vertices.push_back(v1);
            vertices.push_back(0.0f);  // baryX
            vertices.push_back(1.0f);  // baryY
            vertices.push_back(0.0f);  // baryZ
            vertices.push_back(0.0f);  // glow
            vertices.push_back(x1 - centX);  // offsetX
            vertices.push_back(y1 - centY);  // offsetY

            // Vertex 2 (barycentric: 0,0,1)
            vertices.push_back(x2);
            vertices.push_back(y2);
            vertices.push_back(0.0f);
            vertices.push_back(u2);
            vertices.push_back(v2);
            vertices.push_back(0.0f);  // baryX
            vertices.push_back(0.0f);  // baryY
            vertices.push_back(1.0f);  // baryZ
            vertices.push_back(0.0f);  // glow
            vertices.push_back(x2 - centX);  // offsetX
            vertices.push_back(y2 - centY);  // offsetY

            // Single triangle face
            indices.push_back(vertexIndex);
            indices.push_back(vertexIndex + 1);
            indices.push_back(vertexIndex + 2);

            vertexIndex += 3; // 3 vertices per triangle
        }
    }
}

// Apply rotation to a point around origin (Rodrigues' rotation formula simplified for unit axis)
void rotatePoint(float& x, float& y, float& z, float axisX, float axisY, float axisZ, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    float dot = x * axisX + y * axisY + z * axisZ;
    float crossX = axisY * z - axisZ * y;
    float crossY = axisZ * x - axisX * z;
    float crossZ = axisX * y - axisY * x;

    float newX = x * c + crossX * s + axisX * dot * (1 - c);
    float newY = y * c + crossY * s + axisY * dot * (1 - c);
    float newZ = z * c + crossZ * s + axisZ * dot * (1 - c);

    x = newX;
    y = newY;
    z = newZ;
}

// Initialize particle animation - FTL deceleration into Thornfield
// Screen tetrahedrons move with velocities, calculated to collide at their target positions
void initializeParticles(Graphics& graphics, const std::vector<float>& targetVertices,
                         float width, float height, float duration = 3.0f) {
    // Re-seed each time for different pattern on every spawn
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    static std::mt19937 gen(seed);
    gen.seed(seed);

    // Velocity distribution - coming from far away (negative Z) towards camera/grid (positive Z)
    std::uniform_real_distribution<float> vxDist(-0.3f, 0.3f);   // Small lateral drift
    std::uniform_real_distribution<float> vyDist(-0.3f, 0.3f);   // Small vertical drift
    std::uniform_real_distribution<float> vzDist(1.5f, 4.0f);    // Moving towards camera (+Z direction)
    std::uniform_real_distribution<float> rotAngleDist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> collisionTimeDist(0.2f, 1.0f);  // Central triangle timing (fast)
    std::uniform_real_distribution<float> outerCollisionTimeDist(1.2f, 2.5f);  // Outer grid delayed

    // Calculate giant triangle bounds (same as in generateTriangularGrid)
    // Upward pointing - flat bottom edge, point at top
    float triWidth = width / 160;  // Match subdivisions (80x2)
    float triHeight = triWidth * 0.866025f;
    float triSize = std::min(width, height) * 0.4f;
    float bottomY = -triSize * 0.4f;
    bottomY = floor(bottomY / triHeight) * triHeight;
    float topY = bottomY + triSize * 0.866025f;
    float halfBase = (topY - bottomY) / 0.866025f * 0.5f;
    float giantTriAx = -halfBase, giantTriAy = bottomY;   // Bottom left (flat edge)
    float giantTriBx =  halfBase, giantTriBy = bottomY;   // Bottom right (flat edge)
    float giantTriCx =  0.0f,     giantTriCy = topY;      // Top center (point)

    graphics.particles.clear();
    graphics.gridVertices = targetVertices;

    // Each vertex has 11 floats: x, y, z, u, v, baryX, baryY, baryZ, glow, offsetX, offsetY
    int numVertices = targetVertices.size() / 11;
    // 3 vertices per triangle
    int numTriangles = numVertices / 3;

    for (int t = 0; t < numTriangles; t++) {
        // Calculate triangle's centroid and find min Y vertex
        float centroidX = 0, centroidY = 0, centroidZ = 0;
        float minVertY = 1e10f;
        for (int v = 0; v < 3; v++) {
            int i = t * 3 + v;
            float vy = targetVertices[i * 11 + 1];
            centroidX += targetVertices[i * 11 + 0];
            centroidY += vy;
            centroidZ += targetVertices[i * 11 + 2];
            minVertY = std::min(minVertY, vy);
        }
        centroidX /= 3.0f;
        centroidY /= 3.0f;
        centroidZ /= 3.0f;

        // Check if this triangle is inside the central giant triangle
        bool isInCentralTriangle = pointInTriangle(centroidX, centroidY,
                                                    giantTriAx, giantTriAy,
                                                    giantTriBx, giantTriBy,
                                                    giantTriCx, giantTriCy);

        // Exclude downward-pointing triangles at the bottom edge (they have points, not flat edges)
        // Downward triangles have centroid above their minimum Y vertex
        bool isDownwardPointing = (centroidY > minVertY + triHeight * 0.2f);
        bool isAtBottomEdge = (minVertY < bottomY + triHeight * 0.5f);
        if (isInCentralTriangle && isDownwardPointing && isAtBottomEdge) {
            isInCentralTriangle = false;  // Exclude from central triangle
        }

        // Calculate collision time based on position
        float collisionTime;
        float vx, vy, vz;

        if (isInCentralTriangle) {
            // Central triangle loads first with random timing
            collisionTime = collisionTimeDist(gen);
            // Random velocity for central triangles
            vx = vxDist(gen);
            vy = vyDist(gen);
            vz = vzDist(gen);
        } else {
            // Outer triangles (torus/thorns) stay stable until central triangle forms
            // Then "wither away" - triangles near center extract first like decay/fire

            float distFromCenter = sqrt(centroidX * centroidX + centroidY * centroidY);
            float maxDist = sqrt(width * width + height * height) * 0.5f;  // Half diagonal
            float normalizedDist = std::min(distFromCenter / maxDist, 1.0f);

            // Withering timing: wait for central triangle to form (1.5s), then decay
            // Closer to center = extract sooner (inverse of before)
            // Narrow time window (0.8s) for gradual peeling effect
            float witherStart = 1.5f;  // Start after central triangle is mostly formed
            float witherDuration = 0.8f;  // Narrow window for decay effect

            // Invert: closer triangles (low normalizedDist) wither first
            float witherOrder = 1.0f - normalizedDist;

            std::uniform_real_distribution<float> jitterDist(-0.05f, 0.05f);
            float baseTime = witherStart + witherOrder * witherDuration;
            collisionTime = baseTime + jitterDist(gen);

            // Calculate spawn position on thorny stem torus around central triangle
            // Triangles cluster at discrete thorn positions to form visible spikes

            // Torus parameters
            float majorRadius = width * 0.7f;   // Main ring radius
            float minorRadius = height * 0.1f;  // Stem tube thickness

            // Number of thorns around the torus
            int numThorns = 16;

            // Decide if this triangle is part of stem or a thorn
            std::uniform_real_distribution<float> partDist(0.0f, 1.0f);
            bool isStem = partDist(gen) < 0.25f;  // 25% form the stem, 75% form thorns

            float torusX, torusY, torusZ;

            if (isStem) {
                // Stem triangles - distributed along the torus surface
                std::uniform_real_distribution<float> uDist(0.0f, 2.0f * M_PI);
                std::uniform_real_distribution<float> vDist(0.0f, 2.0f * M_PI);
                float u = uDist(gen);
                float v = vDist(gen);

                torusX = (majorRadius + minorRadius * cos(v)) * cos(u);
                torusY = (majorRadius + minorRadius * cos(v)) * sin(u);
                torusZ = minorRadius * sin(v);
            } else {
                // Thorn triangles - cluster at discrete thorn positions
                std::uniform_real_distribution<float> thornIndexDist(0.0f, (float)numThorns);
                int thornIndex = (int)thornIndexDist(gen);

                // Various thorn sizes - some small, some large
                std::uniform_real_distribution<float> thornSizeDist(0.1f, 0.35f);
                float thornLength = thornSizeDist(gen) * height;

                // Thorn profile: 0 = spikey isosceles, 1 = fat equilateral
                std::uniform_real_distribution<float> profileDist(0.0f, 1.0f);
                float thornProfile = profileDist(gen);

                // Position around main ring for this thorn
                float u = (thornIndex / (float)numThorns) * 2.0f * M_PI;

                // Each thorn has a fixed outward direction (v angle)
                std::uniform_real_distribution<float> vVariation(-0.2f, 0.2f);
                float v = (thornIndex % 6) * (M_PI / 3.0f) + vVariation(gen);  // 6 directions

                // Base position on torus surface
                float baseX = (majorRadius + minorRadius * cos(v)) * cos(u);
                float baseY = (majorRadius + minorRadius * cos(v)) * sin(u);
                float baseZ = minorRadius * sin(v);

                // Thorn direction (outward from tube surface)
                float thornDirX = cos(v) * cos(u);
                float thornDirY = cos(v) * sin(u);
                float thornDirZ = sin(v);

                // Position along the thorn (0 = base, 1 = tip)
                std::uniform_real_distribution<float> alongThorn(0.0f, 1.0f);
                float tPos = alongThorn(gen);

                // Thorn profile affects spread vs length ratio
                // Spikey (profile=0): narrow spread, elongated
                // Equilateral (profile=1): wide spread, shorter effective length
                float baseSpread = 0.03f + thornProfile * 0.12f;  // 0.03 to 0.15
                float lengthScale = 1.0f - thornProfile * 0.4f;   // 1.0 to 0.6
                thornLength *= lengthScale;

                // Thorn tapers - spread decreases toward tip (more dramatic for spikey)
                float taperPower = 1.0f + (1.0f - thornProfile) * 1.5f;  // 1.0 to 2.5
                float spread = pow(1.0f - tPos, taperPower) * baseSpread * height;
                std::uniform_real_distribution<float> spreadDist(-1.0f, 1.0f);

                // Calculate perpendicular directions for spread
                float perpX1 = -sin(u);
                float perpY1 = cos(u);
                float perpZ1 = 0.0f;
                float perpX2 = thornDirY * perpZ1 - thornDirZ * perpY1;
                float perpY2 = thornDirZ * perpX1 - thornDirX * perpZ1;
                float perpZ2 = thornDirX * perpY1 - thornDirY * perpX1;

                float spreadOffset1 = spreadDist(gen) * spread;
                float spreadOffset2 = spreadDist(gen) * spread;

                torusX = baseX + thornDirX * tPos * thornLength + perpX1 * spreadOffset1 + perpX2 * spreadOffset2;
                torusY = baseY + thornDirY * tPos * thornLength + perpY1 * spreadOffset1 + perpY2 * spreadOffset2;
                torusZ = baseZ + thornDirZ * tPos * thornLength + perpZ1 * spreadOffset1 + perpZ2 * spreadOffset2;
            }

            // Tilt the torus around the X-axis (planetary ring angle)
            float tiltAngle = -0.45f;  // About 25 degrees
            float cosT = cos(tiltAngle);
            float sinT = sin(tiltAngle);
            float arcX = torusX;
            float arcY = torusY * cosT - torusZ * sinT;
            float arcZ = torusY * sinT + torusZ * cosT;

            // Offset to position the ring
            arcY += height * 0.15f;

            // Z position - push spawn further back
            float spawnZ = arcZ - 4.0f;

            // Calculate velocity to travel from arc spawn to target in collision time
            vx = (centroidX - arcX) / collisionTime;
            vy = (centroidY - arcY) / collisionTime;
            vz = (0.0f - spawnZ) / collisionTime;  // Target Z is 0
        }

        // Random rotation axis (normalized) for initial orientation
        float axisX = axisDist(gen);
        float axisY = axisDist(gen);
        float axisZ = axisDist(gen);
        float axisLen = sqrt(axisX*axisX + axisY*axisY + axisZ*axisZ);
        if (axisLen > 0.001f) {
            axisX /= axisLen;
            axisY /= axisLen;
            axisZ /= axisLen;
        } else {
            axisX = 0; axisY = 0; axisZ = 1;
        }
        float rotAngle = rotAngleDist(gen);

        // Apply to all 3 vertices of this triangle
        for (int v = 0; v < 3; v++) {
            int i = t * 3 + v;
            TriangleParticle p;

            // Target position from grid (where it will collide)
            p.targetX = targetVertices[i * 11 + 0];
            p.targetY = targetVertices[i * 11 + 1];
            p.targetZ = targetVertices[i * 11 + 2];

            // Local offset from centroid (maintains triangle shape)
            float localX = p.targetX - centroidX;
            float localY = p.targetY - centroidY;
            float localZ = p.targetZ - centroidZ;

            // Apply random rotation to local offset (tumbling debris orientation)
            rotatePoint(localX, localY, localZ, axisX, axisY, axisZ, rotAngle);

            // Store rotated local offset (for maintaining shape during flight)
            p.localX = localX;
            p.localY = localY;
            p.localZ = localZ;

            // Velocity (same for all vertices of this triangle)
            p.vx = vx;
            p.vy = vy;
            p.vz = vz;

            // Collision time
            p.collisionTime = collisionTime;

            // UVs don't animate
            p.u = targetVertices[i * 11 + 3];
            p.v = targetVertices[i * 11 + 4];

            // Barycentric coordinates (from grid generation)
            p.baryX = targetVertices[i * 11 + 5];
            p.baryY = targetVertices[i * 11 + 6];
            p.baryZ = targetVertices[i * 11 + 7];

            p.elapsedTime = 0.0f;
            p.hitTime = -1.0f;  // Not hit yet
            p.vertexIndex = i;
            p.locked = false;
            p.isCentral = isInCentralTriangle;

            // Pulse variation based on debris velocity magnitude and rotation
            float velocityMag = sqrt(vx*vx + vy*vy + vz*vz);
            p.pulseScale = 0.6f + (velocityMag / 4.0f) * 0.8f;  // Faster debris = bigger pulse
            p.pulseScale = std::min(p.pulseScale, 1.4f);
            p.pulseRotation = rotAngle;  // Use debris rotation for pulse asymmetry

            graphics.particles.push_back(p);
        }
    }
}

// Update particle positions based on velocity trajectories toward collision points
void updateParticles(Graphics& graphics, float deltaTime) {
    for (auto& p : graphics.particles) {
        p.elapsedTime += deltaTime;

        float x, y, z;
        float glow = 0.0f;

        if (p.locked || p.elapsedTime >= p.collisionTime) {
            // Track first hit time for glow effect
            if (!p.locked) {
                p.hitTime = p.elapsedTime;
            }
            p.locked = true;

            // Time since impact
            float timeSinceHit = p.elapsedTime - p.hitTime;

            // Impact overshoot effect - deflects backward then bounces back
            // Central triangles have stronger overshoot (debris impact on shield)
            float overshootAmount = p.isCentral ? 0.425f : 0.112f;
            float overshootDuration = p.isCentral ? 1.0f : 0.48f;
            float overshootZ = 0.0f;

            if (timeSinceHit < overshootDuration) {
                // Damped spring oscillation for bounce-back effect
                // z(t) = A * sin(ωt) * e^(-bt) where ω gives ~1.5 oscillations
                float t = timeSinceHit / overshootDuration;
                float omega = 4.5f * M_PI;  // ~1.5 oscillations
                float damping = 4.0f;
                overshootZ = overshootAmount * sin(omega * t) * exp(-damping * t);
            }

            // Position with overshoot (negative Z = pushed back toward viewer)
            x = p.targetX;
            y = p.targetY;
            z = p.targetZ - overshootZ;

            // Compute glow based on triangle type
            if (p.isCentral) {
                // Central pulse: encode scale and progress in glow value
                // Format: glow = 10.0 + (scale-0.6)*5.0 + progress*0.5
                // Scale range 0.6-1.4 -> 0-4, progress 0-1 -> 0-0.5
                // Total range: 10.0 to 14.5
                float pulseDuration = 0.8f;
                if (timeSinceHit < pulseDuration) {
                    float pulseProgress = timeSinceHit / pulseDuration;
                    float scaleEnc = (p.pulseScale - 0.6f) * 5.0f;  // 0-4
                    glow = 10.0f + scaleEnc + pulseProgress * 0.5f;
                } else {
                    glow = 0.0f;  // Pulse complete
                }
            } else {
                // Normal glow for outer triangles
                // float glowDuration = 0.8f;
                // if (timeSinceHit < glowDuration) {
                //     glow = exp(-3.0f * timeSinceHit / glowDuration);
                // }
                glow = 0.0f;
                // float pulseDuration = 0.1f;
                // if (timeSinceHit < pulseDuration) {
                //     float pulseProgress = timeSinceHit / pulseDuration;
                //     float scaleEnc = (p.pulseScale - 0.6f) * 1.0f;  // 0-4
                //     glow = 30.0f + scaleEnc + pulseProgress * 0.5f;
                // } else {
                //     glow = 0.0f;  // Pulse complete
                // }
                float glowDuration = 0.5f;
                glow = exp(-5.0f * timeSinceHit / glowDuration);

            }
        } else {
            // Flying towards collision point along velocity trajectory
            float timeToCollision = p.collisionTime - p.elapsedTime;
            float t = p.elapsedTime / p.collisionTime;  // 0 at start, 1 at collision

            // Centroid follows trajectory: starts far, arrives at target centroid
            float centroidX = p.targetX - p.vx * timeToCollision;
            float centroidY = p.targetY - p.vy * timeToCollision;
            float centroidZ = p.targetZ - p.vz * timeToCollision;

            // Rotated local offset blends out as we approach collision (tumbling -> aligned)
            float localBlend = 1.0f - t;
            x = centroidX + p.localX * localBlend;
            y = centroidY + p.localY * localBlend;
            z = centroidZ + p.localZ * localBlend;
        }

        // Update in grid vertices buffer (11 floats per vertex: x,y,z, u,v, baryX,baryY,baryZ, glow, offsetX,offsetY)
        int offset = p.vertexIndex * 11;
        graphics.gridVertices[offset + 0] = x;
        graphics.gridVertices[offset + 1] = y;
        graphics.gridVertices[offset + 2] = z;
        // UV (3,4) and barycentric (5,6,7) stay the same
        graphics.gridVertices[offset + 8] = glow;
    }
}

// Upload updated vertices to GPU
void uploadParticleVertices(Graphics& graphics) {
    glBindBuffer(GL_ARRAY_BUFFER, graphics.gridVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, graphics.gridVertices.size() * sizeof(float), graphics.gridVertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Initialize noise tetrahedrons - grey debris all around (we've just hit the debris field)
void initializeNoiseTetrahedrons(Graphics& graphics, int count = 300) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    // Spread debris all around in 3D space
    std::uniform_real_distribution<float> xDist(-8.0f, 8.0f);
    std::uniform_real_distribution<float> yDist(-6.0f, 6.0f);
    std::uniform_real_distribution<float> zDistFar(-15.0f, -3.0f);   // Far debris field
    std::uniform_real_distribution<float> zDistNear(-3.0f, 2.0f);    // Near-camera debris (passes through early)
    std::uniform_real_distribution<float> velocityDist(4.0f, 10.0f);  // FTL speeds
    std::uniform_real_distribution<float> scaleDist(0.015f, 0.06f);
    std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * M_PI);

    // Initialize FTL deceleration state
    graphics.decelerationTime = 0.0f;
    graphics.decelerationDuration = 5.0f;  // 5 seconds to decelerate

    graphics.noiseParticles.clear();
    graphics.noiseParticles.reserve(count);

    // Spawn 40% of particles near the camera so they pass through early
    int nearCount = count * 4 / 10;

    for (int i = 0; i < count; i++) {
        NoiseTetrahedron n;
        // Distribute throughout 3D space around the viewer
        n.x = xDist(gen);
        n.y = yDist(gen);
        // Near particles pass through camera area early, before screen forms
        n.z = (i < nearCount) ? zDistNear(gen) : zDistFar(gen);
        n.vz = velocityDist(gen);
        n.scale = scaleDist(gen);

        // Random rotation axis (normalized)
        n.axisX = axisDist(gen);
        n.axisY = axisDist(gen);
        n.axisZ = axisDist(gen);
        float axisLen = sqrt(n.axisX*n.axisX + n.axisY*n.axisY + n.axisZ*n.axisZ);
        if (axisLen > 0.001f) {
            n.axisX /= axisLen;
            n.axisY /= axisLen;
            n.axisZ /= axisLen;
        } else {
            n.axisX = 0; n.axisY = 0; n.axisZ = 1;
        }
        n.rotAngle = angleDist(gen);

        graphics.noiseParticles.push_back(n);
    }

    // Pre-allocate vertex buffer (4 verts * 5 floats per tetrahedron)
    graphics.noiseVertices.resize(count * 4 * 5);
    graphics.noiseVertexCount = count * 12; // 4 faces * 3 indices
}

// Respawn a noise tetrahedron at far distance
void respawnNoiseTetrahedron(NoiseTetrahedron& n) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(-8.0f, 8.0f);
    std::uniform_real_distribution<float> yDist(-6.0f, 6.0f);
    std::uniform_real_distribution<float> velocityDist(4.0f, 10.0f);
    std::uniform_real_distribution<float> scaleDist(0.015f, 0.06f);
    std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * M_PI);

    n.x = xDist(gen);
    n.y = yDist(gen);
    n.z = -20.0f;  // Respawn far away
    n.vz = velocityDist(gen);
    n.scale = scaleDist(gen);

    // New random rotation
    n.axisX = axisDist(gen);
    n.axisY = axisDist(gen);
    n.axisZ = axisDist(gen);
    float axisLen = sqrt(n.axisX*n.axisX + n.axisY*n.axisY + n.axisZ*n.axisZ);
    if (axisLen > 0.001f) {
        n.axisX /= axisLen;
        n.axisY /= axisLen;
        n.axisZ /= axisLen;
    } else {
        n.axisX = 0; n.axisY = 0; n.axisZ = 1;
    }
    n.rotAngle = angleDist(gen);
}

// Update noise tetrahedrons - move towards camera with FTL deceleration
void updateNoiseTetrahedrons(Graphics& graphics, float deltaTime) {
    // Update deceleration time
    graphics.decelerationTime += deltaTime;

    // Calculate deceleration factor: starts at 1.0 (full speed), decays towards 0.1 (crawl)
    float t = std::min(graphics.decelerationTime / graphics.decelerationDuration, 1.0f);
    // Exponential decay for deceleration feel
    float speedMultiplier = 0.1f + 0.9f * exp(-3.0f * t);

    for (auto& n : graphics.noiseParticles) {
        // Apply decelerated velocity
        n.z += n.vz * speedMultiplier * deltaTime;

        // Respawn if past camera (but only if still decelerating fast enough)
        if (n.z > 5.0f) {
            if (speedMultiplier > 0.15f) {
                respawnNoiseTetrahedron(n);
            } else {
                // At near-stop, just keep them drifting slowly past
                n.z = 5.1f; // Park them just past camera
            }
        }
    }
}

// Generate vertices for noise tetrahedrons
void generateNoiseVertices(Graphics& graphics) {
    int idx = 0;
    for (const auto& n : graphics.noiseParticles) {
        float s = n.scale;
        float h = s * 0.8f;  // Apex height

        // Base triangle vertices (local, centered at origin)
        float lx0 = -s,    ly0 = -s * 0.577f, lz0 = 0;
        float lx1 = s,     ly1 = -s * 0.577f, lz1 = 0;
        float lx2 = 0,     ly2 = s * 1.155f,  lz2 = 0;
        float lx3 = 0,     ly3 = 0,           lz3 = h;  // Apex

        // Apply rotation to each local vertex
        rotatePoint(lx0, ly0, lz0, n.axisX, n.axisY, n.axisZ, n.rotAngle);
        rotatePoint(lx1, ly1, lz1, n.axisX, n.axisY, n.axisZ, n.rotAngle);
        rotatePoint(lx2, ly2, lz2, n.axisX, n.axisY, n.axisZ, n.rotAngle);
        rotatePoint(lx3, ly3, lz3, n.axisX, n.axisY, n.axisZ, n.rotAngle);

        // Translate to world position
        float x0 = n.x + lx0, y0 = n.y + ly0, z0 = n.z + lz0;
        float x1 = n.x + lx1, y1 = n.y + ly1, z1 = n.z + lz1;
        float x2 = n.x + lx2, y2 = n.y + ly2, z2 = n.z + lz2;
        float x3 = n.x + lx3, y3 = n.y + ly3, z3 = n.z + lz3;

        // Grey UV
        float u = 0.5f, v = 0.5f;

        // Vertex 0
        graphics.noiseVertices[idx++] = x0;
        graphics.noiseVertices[idx++] = y0;
        graphics.noiseVertices[idx++] = z0;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
        // Vertex 1
        graphics.noiseVertices[idx++] = x1;
        graphics.noiseVertices[idx++] = y1;
        graphics.noiseVertices[idx++] = z1;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
        // Vertex 2
        graphics.noiseVertices[idx++] = x2;
        graphics.noiseVertices[idx++] = y2;
        graphics.noiseVertices[idx++] = z2;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
        // Vertex 3 (apex)
        graphics.noiseVertices[idx++] = x3;
        graphics.noiseVertices[idx++] = y3;
        graphics.noiseVertices[idx++] = z3;
        graphics.noiseVertices[idx++] = u;
        graphics.noiseVertices[idx++] = v;
    }
}

// Upload noise vertices to GPU
void uploadNoiseVertices(Graphics& graphics) {
    glBindBuffer(GL_ARRAY_BUFFER, graphics.noiseVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, graphics.noiseVertices.size() * sizeof(float), graphics.noiseVertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Initialize 3D rendering resources
void initialize3DRendering(Graphics& graphics, int width, int height) {
    graphics.uiWidth = width;
    graphics.uiHeight = height;
    graphics.tiltAngle = 0.0f;

    // Create framebuffer for UI rendering
    glGenFramebuffers(1, &graphics.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, graphics.fbo);

    // Create texture to render UI to
    glGenTextures(1, &graphics.fboTexture);
    glBindTexture(GL_TEXTURE_2D, graphics.fboTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, graphics.fboTexture, 0);

    // Create depth and stencil renderbuffer
    glGenRenderbuffers(1, &graphics.fboDepthRenderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, graphics.fboDepthRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, graphics.fboDepthRenderBuffer);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer is not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Create shader program
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    graphics.shaderProgram = glCreateProgram();
    glAttachShader(graphics.shaderProgram, vertexShader);
    glAttachShader(graphics.shaderProgram, fragmentShader);
    glLinkProgram(graphics.shaderProgram);

    GLint success;
    glGetProgramiv(graphics.shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(graphics.shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create plane geometry
    float aspectRatio = (float)width / (float)height;
    float planeWidth = 2.0f * aspectRatio;
    float planeHeight = 2.0f;

    float vertices[] = {
        // positions                              // tex     // bary (unused)   // glow  // centroid offset
        -planeWidth/2, -planeHeight/2, 0.0f,   0.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom left
         planeWidth/2, -planeHeight/2, 0.0f,   1.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom right
         planeWidth/2,  planeHeight/2, 0.0f,   1.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // top right
        -planeWidth/2,  planeHeight/2, 0.0f,   0.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f   // top left
    };

    unsigned int indices[] = {
        0, 1, 2,
        0, 2, 3
    };

    glGenVertexArrays(1, &graphics.planeVAO);
    glGenBuffers(1, &graphics.planeVBO);
    glGenBuffers(1, &graphics.planeEBO);

    glBindVertexArray(graphics.planeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphics.planeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Barycentric coords attribute (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Glow factor attribute (location 3)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);

    // Centroid offset attribute (location 4)
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(9 * sizeof(float)));
    glEnableVertexAttribArray(4);

    glBindVertexArray(0);

    // Create triangular grid geometry (20x20 subdivisions for smooth tessellation)
    std::vector<float> gridVertices;
    std::vector<unsigned int> gridIndices;
    generateTriangularGrid(gridVertices, gridIndices, planeWidth, planeHeight, 80, 80);

    graphics.gridVertexCount = gridIndices.size();
    // Straight to plane mode: the triangle spawn animation is skipped at
    // startup. Set true to bring the intro effect back.
    graphics.useGridMode = false;
    graphics.gridModeTransitionTimer = 0.0f;
    graphics.allParticlesLocked = true;

    glGenVertexArrays(1, &graphics.gridVAO);
    glGenBuffers(1, &graphics.gridVBO);
    glGenBuffers(1, &graphics.gridEBO);

    glBindVertexArray(graphics.gridVAO);

    // Initialize particles with random positions that will animate to grid
    initializeParticles(graphics, gridVertices, planeWidth, planeHeight, 3.0f);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.gridVBO);
    // Use DYNAMIC_DRAW since we'll update vertices each frame during animation
    glBufferData(GL_ARRAY_BUFFER, graphics.gridVertices.size() * sizeof(float), graphics.gridVertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphics.gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndices.size() * sizeof(unsigned int), gridIndices.data(), GL_STATIC_DRAW);

    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Barycentric coords attribute (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Glow factor attribute (location 3)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);

    // Centroid offset attribute (location 4)
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(9 * sizeof(float)));
    glEnableVertexAttribArray(4);

    glBindVertexArray(0);

    // Create grey texture for noise tetrahedrons
    unsigned char greyPixel[4] = {100, 100, 100, 255};  // Dark grey RGBA
    glGenTextures(1, &graphics.greyTexture);
    glBindTexture(GL_TEXTURE_2D, graphics.greyTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, greyPixel);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Initialize noise tetrahedrons
    initializeNoiseTetrahedrons(graphics, 300);

    // Generate indices for noise tetrahedrons (4 faces * 3 verts each)
    std::vector<unsigned int> noiseIndices;
    for (int t = 0; t < (int)graphics.noiseParticles.size(); t++) {
        unsigned int base = t * 4;
        // Face 0: base (v0, v1, v2)
        noiseIndices.push_back(base + 0);
        noiseIndices.push_back(base + 1);
        noiseIndices.push_back(base + 2);
        // Face 1: side (v0, v1, apex)
        noiseIndices.push_back(base + 0);
        noiseIndices.push_back(base + 1);
        noiseIndices.push_back(base + 3);
        // Face 2: side (v1, v2, apex)
        noiseIndices.push_back(base + 1);
        noiseIndices.push_back(base + 2);
        noiseIndices.push_back(base + 3);
        // Face 3: side (v2, v0, apex)
        noiseIndices.push_back(base + 2);
        noiseIndices.push_back(base + 0);
        noiseIndices.push_back(base + 3);
    }
    graphics.noiseVertexCount = noiseIndices.size();

    glGenVertexArrays(1, &graphics.noiseVAO);
    glGenBuffers(1, &graphics.noiseVBO);
    GLuint noiseEBO;
    glGenBuffers(1, &noiseEBO);

    glBindVertexArray(graphics.noiseVAO);

    // Generate initial vertices
    generateNoiseVertices(graphics);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.noiseVBO);
    glBufferData(GL_ARRAY_BUFFER, graphics.noiseVertices.size() * sizeof(float), graphics.noiseVertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, noiseEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, noiseIndices.size() * sizeof(unsigned int), noiseIndices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// TODO: Remove these and just integrate a standard GFX programming math lib like GLM

// Helper function to create a 4x4 identity matrix
void mat4Identity(float* mat) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
}

// Helper function to create a perspective projection matrix
void mat4Perspective(float* mat, float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    mat4Identity(mat);
    mat[0] = 1.0f / (aspect * tanHalfFov);
    mat[5] = 1.0f / tanHalfFov;
    mat[10] = -(far + near) / (far - near);
    mat[11] = -1.0f;
    mat[14] = -(2.0f * far * near) / (far - near);
    mat[15] = 0.0f;
}

// Helper function to create an orthographic projection matrix
void mat4Ortho(float* mat, float left, float right, float bottom, float top, float near, float far) {
    mat4Identity(mat);
    mat[0] = 2.0f / (right - left);
    mat[5] = 2.0f / (top - bottom);
    mat[10] = -2.0f / (far - near);
    mat[12] = -(right + left) / (right - left);
    mat[13] = -(top + bottom) / (top - bottom);
    mat[14] = -(far + near) / (far - near);
}

// Helper function to create a look-at view matrix
void mat4LookAt(float* mat, float eyeX, float eyeY, float eyeZ,
                float centerX, float centerY, float centerZ,
                float upX, float upY, float upZ) {
    // Calculate forward vector
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;
    float fLen = sqrt(fx*fx + fy*fy + fz*fz);
    fx /= fLen; fy /= fLen; fz /= fLen;

    // Calculate right vector
    float rx = fy * upZ - fz * upY;
    float ry = fz * upX - fx * upZ;
    float rz = fx * upY - fy * upX;
    float rLen = sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rLen; ry /= rLen; rz /= rLen;

    // Calculate up vector
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    mat4Identity(mat);
    mat[0] = rx; mat[4] = ry; mat[8] = rz;
    mat[1] = ux; mat[5] = uy; mat[9] = uz;
    mat[2] = -fx; mat[6] = -fy; mat[10] = -fz;
    mat[12] = -(rx * eyeX + ry * eyeY + rz * eyeZ);
    mat[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
    mat[14] = (fx * eyeX + fy * eyeY + fz * eyeZ);
}

// Helper function to create a rotation matrix around Y axis
void mat4RotateY(float* mat, float angle) {
    mat4Identity(mat);
    float c = cos(angle);
    float s = sin(angle);
    mat[0] = c;
    mat[2] = s;
    mat[8] = -s;
    mat[10] = c;
}

// Resize framebuffer and plane when window size changes
void resize3DRendering(Graphics& graphics, int width, int height) {
    if (width == graphics.uiWidth && height == graphics.uiHeight) {
        return; // No change, skip resize
    }

    graphics.uiWidth = width;
    graphics.uiHeight = height;

    // Resize framebuffer texture
    glBindTexture(GL_TEXTURE_2D, graphics.fboTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Resize depth/stencil renderbuffer
    glBindRenderbuffer(GL_RENDERBUFFER, graphics.fboDepthRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Update plane geometry to match aspect ratio
    float aspectRatio = (float)width / (float)height;
    float planeWidth = 2.0f * aspectRatio;
    float planeHeight = 2.0f;

    float vertices[] = {
        // positions                              // tex     // bary (unused)   // glow  // centroid offset
        -planeWidth/2, -planeHeight/2, 0.0f,   0.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom left
         planeWidth/2, -planeHeight/2, 0.0f,   1.0f, 0.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // bottom right
         planeWidth/2,  planeHeight/2, 0.0f,   1.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f,  // top right
        -planeWidth/2,  planeHeight/2, 0.0f,   0.0f, 1.0f,   0.33f, 0.33f, 0.34f, 0.0f,  0.0f, 0.0f   // top left
    };

    glBindBuffer(GL_ARRAY_BUFFER, graphics.planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Update grid geometry to match aspect ratio
    std::vector<float> gridVertices;
    std::vector<unsigned int> gridIndices;
    generateTriangularGrid(gridVertices, gridIndices, planeWidth, planeHeight, 80, 80);

    graphics.gridVertexCount = gridIndices.size();

    // Reinitialize particles for new grid dimensions
    initializeParticles(graphics, gridVertices, planeWidth, planeHeight, 4.0f);

    glBindBuffer(GL_ARRAY_BUFFER, graphics.gridVBO);
    glBufferData(GL_ARRAY_BUFFER, graphics.gridVertices.size() * sizeof(float), graphics.gridVertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphics.gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndices.size() * sizeof(unsigned int), gridIndices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Cleanup 3D resources
void cleanup3DRendering(Graphics& graphics) {
    glDeleteFramebuffers(1, &graphics.fbo);
    glDeleteTextures(1, &graphics.fboTexture);
    glDeleteRenderbuffers(1, &graphics.fboDepthRenderBuffer);
    glDeleteVertexArrays(1, &graphics.planeVAO);
    glDeleteBuffers(1, &graphics.planeVBO);
    glDeleteBuffers(1, &graphics.planeEBO);
    glDeleteVertexArrays(1, &graphics.gridVAO);
    glDeleteBuffers(1, &graphics.gridVBO);
    glDeleteBuffers(1, &graphics.gridEBO);
    glDeleteProgram(graphics.shaderProgram);
}