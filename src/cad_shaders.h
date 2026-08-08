#pragma once

// GLSL for the Droid panel's CAD presentation look.
//
// render3d's stock opaque pass is a deliberately stylized 4-step posterized cel
// shader -- great for the voxel scenes it was written for, wrong for showing a
// machined part. These replace it through the module's own extension point:
// the mesh is tagged r3d::DeferredDraw so the stock pass skips it, and the
// passes in panel3d.cpp draw it on the Opaque and Background phases instead.
//
// The rig is a studio three-point setup built in *camera* space each frame, so
// orbiting never swings the model into a badly lit angle -- the lights follow
// you the way they do in a turntable render. Shading is GGX over a hemisphere
// ambient, which is what gives a matte part its readable form.
//
// Everything here is linear-light and tonemaps at the end, because the colour
// buffer render3d's post pass samples is plain RGBA8 display space.

namespace cad {

static const char* CAD_VERTEX_SHADER = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNM;
out vec3 vNormal;
void main(){
    vNormal = normalize(uNM * aNorm);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)GLSL";

static const char* CAD_FRAGMENT_SHADER = R"GLSL(
#version 330 core
in vec3 vNormal;
layout(location=0) out vec4 FragColor;
layout(location=1) out vec4 FragNormal;

uniform vec3  uBaseColor;
uniform float uRoughness;
uniform float uMetalness;

// The projection is orthographic, so every fragment shares one view direction
// and it can be a uniform instead of a per-pixel normalize.
uniform vec3 uViewDir;

uniform vec3 uKeyDir;
uniform vec3 uFillDir;
uniform vec3 uRimDir;
uniform vec3 uKeyColor;
uniform vec3 uFillColor;
uniform vec3 uRimColor;

uniform vec3  uSkyColor;      // hemisphere ambient from above
uniform vec3  uGroundColor;   // bounce from below
uniform float uExposure;

const float PI = 3.14159265;

float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Height-correlated Smith visibility (the G term already divided by 4·NoL·NoV).
float V_Smith(float NoV, float NoL, float a) {
    float a2 = a * a;
    float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

vec3 F_Schlick(vec3 f0, float u) {
    return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
}

vec3 shadeLight(vec3 N, vec3 V, vec3 L, vec3 lightColor,
                vec3 albedo, vec3 f0, float a, float NoV) {
    float NoL = dot(N, L);
    if (NoL <= 0.0) return vec3(0.0);

    vec3 H = normalize(L + V);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);

    vec3 F = F_Schlick(f0, VoH);
    vec3 spec = F * (D_GGX(NoH, a) * V_Smith(NoV, NoL, a));
    vec3 diff = (1.0 - F) * albedo / PI;

    return (diff + spec) * lightColor * NoL;
}

void main(){
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uViewDir);
    float NoV = max(dot(N, V), 1e-4);

    // Perceptual roughness -> the alpha the GGX terms want.
    float a = max(uRoughness * uRoughness, 1e-3);

    vec3 albedo = uBaseColor * (1.0 - uMetalness);
    vec3 f0 = mix(vec3(0.055), uBaseColor, uMetalness);

    vec3 color = vec3(0.0);
    color += shadeLight(N, V, normalize(uKeyDir),  uKeyColor,  albedo, f0, a, NoV);
    color += shadeLight(N, V, normalize(uFillDir), uFillColor, albedo, f0, a, NoV);

    // Hemisphere ambient. The gradient from sky to bounce is what keeps an
    // unlit face from reading as a flat silhouette -- more important on a matte
    // part than any single light.
    vec3 skyAmbient = mix(uGroundColor, uSkyColor, N.y * 0.5 + 0.5);
    color += skyAmbient * albedo;

    // Ambient specular against the same hemisphere: a cheap stand-in for an
    // environment probe, and the reason the part reads as a hard surface rather
    // than as clay.
    vec3 R = reflect(-V, N);
    vec3 envColor = mix(uGroundColor, uSkyColor, R.y * 0.5 + 0.5);
    vec3 envF = f0 + (max(vec3(1.0 - uRoughness), f0) - f0) * pow(1.0 - NoV, 5.0);
    color += envColor * envF;

    // Back rim, gated on facing away from the viewer: separates the silhouette
    // from the background without washing out the front faces.
    float rim = pow(1.0 - NoV, 3.0) * max(dot(N, normalize(uRimDir)), 0.0);
    color += uRimColor * rim;

    // Exposure tonemap, then gamma. Exponential rather than Reinhard: it keeps
    // more separation in the upper mids, which is where a grey part lives.
    color = 1.0 - exp(-color * uExposure);
    color = pow(max(color, 0.0), vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
    FragNormal = vec4(N * 0.5 + 0.5, 1.0);
}
)GLSL";

// Studio backdrop: a soft radial sweep, brighter behind the subject and falling
// off to the corners. Replaces the module's sky gradient, which is a daylight
// zenith-to-horizon ramp and reads as an outdoor scene behind a part.
//
// Depth is untouched (the pass runs with depth test off and no depth write), so
// the post pass still sees a full depth cliff at the silhouette and draws its
// outline halo.
static const char* BACKDROP_FRAGMENT_SHADER = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform vec3  uCenterColor;
uniform vec3  uEdgeColor;
uniform float uAspect;

void main(){
    // Centred slightly above the middle, where the subject sits.
    vec2 p = (vUV - vec2(0.5, 0.55)) * vec2(uAspect, 1.0);
    float r = clamp(length(p) / 0.72, 0.0, 1.0);
    FragColor = vec4(mix(uCenterColor, uEdgeColor, smoothstep(0.0, 1.0, r)), 1.0);
}
)GLSL";

} // namespace cad
