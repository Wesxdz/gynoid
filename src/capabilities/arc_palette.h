#pragma once

// The ten ARC colours, in the order the tasks number them: cell value N is
// arc_palette[N]. One copy, because a colour is an identity here -- a Sun badge
// in the Interlocutor, a Sun cell in a grid and a Sun tile in a processor's
// readout are the same yellow or they are lying about being the same thing.
//
// Presentation lives here rather than in the solver: jane_eyre declares what a
// term MEANS and hands over an index, and this decides how it looks.

#include <cstdint>

// RRGGBBAA, the convention every colour in components.h uses.
constexpr uint32_t arc_palette[10] = {
    0x555555FF, // 0 slate
    0x0074D9FF, // 1 zen
    0xFF4136FF, // 2 rose
    0x2ECC40FF, // 3 grass
    0xFFDC00FF, // 4 sun
    0xAAAAAAFF, // 5 grey
    0xF012BEFF, // 6 purse
    0xFF851BFF, // 7 tang
    0x7FDBFFFF, // 8 sky
    0x985898FF, // 9 space
};

constexpr const char* arc_color_names[10] = {
    "slate", "zen", "rose", "grass", "sun",
    "grey", "purse", "tang", "sky", "space",
};
