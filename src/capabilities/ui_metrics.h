#pragma once

// Badge layout metrics shared between the monolith's badge factory and panel
// modules that position UI relative to badges (e.g. the Droid tooltip riding
// the cursor with its bottom edge on the pointer).

// Calibrated badge height; the factory in main.cpp builds to this, so
// anything aligning against a badge's edge can rely on it.
inline constexpr float BADGE_HEIGHT = 25.0f;
