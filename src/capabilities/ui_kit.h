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
#include <vector>

#include <flecs.h>

#include "../components.h"

flecs::entity create_badge(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule = false, bool is_double_arrow = false,
                           std::string postfix_symbol = "", std::string prefix_symbol = "",
                           uint32_t prefix_tint = 0, uint32_t postfix_tint = 0);

// Vector-based version for sets of prefix/postfix glyphs.
flecs::entity create_badge(flecs::entity parent, flecs::entity UIElement,
                           const char* text, uint32_t base_color,
                           bool is_capsule, bool is_double_arrow,
                           const std::vector<std::string>& prefix_ids,
                           const std::vector<uint32_t>& prefix_tints,
                           const std::vector<std::string>& postfix_ids,
                           const std::vector<uint32_t>& postfix_tints);

// The chip form of a symbol: gradient ground, sprite glyph. Two registers,
// Thornfield-wide: an entity's IDENTITY chip is always the uppercase sheet --
// the same sprite its badge postfix wears in the Interlocutor, so the
// shorthand is one glyph everywhere (pass preserve_case=false) -- while a
// CHARACTER chip keeps its case, because in the Lexicon the case is data.
flecs::entity create_letter_chip(flecs::entity parent, const std::string& symbol,
                                 uint32_t color, bool preserve_case = true,
                                 float scale = 0.9f);

bool point_in_bounds(float x, float y, UIElementBounds bounds);
