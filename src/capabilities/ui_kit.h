#pragma once

// Badge/chip factory and small UI helpers, declared for panel modules.
//
// Definitions still live in main.cpp; moving them into a capability TU is a
// later extraction step -- this header exists so panels can leave the monolith
// *now* without waiting for the services they call to leave first.
//
// main.cpp must NOT include this header: the default arguments here are
// repeated on the definitions there, and a translation unit may declare a
// default for a parameter only once.

#include <cstdint>
#include <string>

#include <flecs.h>

#include "../components.h"

flecs::entity create_badge(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule = false, bool is_double_arrow = false,
                           std::string postfix_symbol = "", std::string prefix_symbol = "",
                           uint32_t prefix_tint = 0, uint32_t postfix_tint = 0);

flecs::entity create_letter_chip(flecs::entity parent, const std::string& symbol,
                                 uint32_t color);

bool point_in_bounds(float x, float y, UIElementBounds bounds);
