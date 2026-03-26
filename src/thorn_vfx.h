#pragma once

#include <GLFW/glfw3.h>
#include <vector>
#include <string>

#include "components.h"
/**
 * CORE VFX PIPELINE FUNCTIONS
 * Implementation found in thorn_vfx.cpp
 */

// Lifecycle and Resource Management
void initialize3DRendering(Graphics& graphics, int width, int height);
void resize3DRendering(Graphics& graphics, int width, int height);
void cleanup3DRendering(Graphics& graphics);

// Shader Compilation
GLuint compileShader(GLenum type, const char* source);

// Geometry Generation
void generateTriangularGrid(std::vector<float>& vertices, std::vector<unsigned int>& indices,
                            float width, float height, int subdivisionsX, int subdivisionsY);

// UI Particle System (The Assembly Effect)
void initializeParticles(Graphics& graphics, const std::vector<float>& targetVertices,
                         float width, float height, float duration);
void updateParticles(Graphics& graphics, float deltaTime);
void uploadParticleVertices(Graphics& graphics);

// Background Debris System (Noise Field)
void initializeNoiseTetrahedrons(Graphics& graphics, int count);
void updateNoiseTetrahedrons(Graphics& graphics, float deltaTime);
void generateNoiseVertices(Graphics& graphics);
void uploadNoiseVertices(Graphics& graphics);
void respawnNoiseTetrahedron(NoiseTetrahedron& n);

/**
 * MATH & UTILITY FUNCTIONS
 */

// Collision and Geometry Logic
bool pointInTriangle(float px, float py,
                     float ax, float ay, float bx, float by, float cx, float cy);

void rotatePoint(float& x, float& y, float& z,
                 float axisX, float axisY, float axisZ, float angle);

// Matrix Math Helpers (GLM Alternatives)
void mat4Identity(float* mat);
void mat4Perspective(float* mat, float fov, float aspect, float near, float far);
void mat4Ortho(float* mat, float left, float right, float bottom, float top, float near, float far);
void mat4LookAt(float* mat, float eyeX, float eyeY, float eyeZ,
                float centerX, float centerY, float centerZ,
                float upX, float upY, float upZ);
void mat4RotateY(float* mat, float angle);
