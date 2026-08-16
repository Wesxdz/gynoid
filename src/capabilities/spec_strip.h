#pragma once

// The spectrogram strip: a clip you can look at and play.
//
// Builds the element -- styled spectrogram image, playhead bar, click-to-seek --
// and registers the system that sweeps the playhead. Used by the Interlocutor to
// hang a strip under every spoken message, but it knows nothing about chat: give
// it a parent and a wav and it works anywhere.

#include <string>

#include <flecs.h>

namespace spec_strip {

// Build a strip for `wav_path` under `parent`. `palette` empty uses the
// default. Returns the strip entity, or an invalid entity if the clip could
// not be rendered.
flecs::entity create(flecs::entity parent, const std::string& wav_path,
                     const std::string& palette = "");

// Build a strip and place it directly above `message` in that message's list,
// rather than inside the message row. An utterance reads as the thing the line
// produced, so it wants its own band in the transcript, at full width -- inside
// the row it would be squeezed beside the bubble.
flecs::entity create_above(flecs::entity message, const std::string& wav_path,
                           const std::string& palette = "");

// Play `strip` from the start and hand it the transport. The playhead and the
// colour follow whichever strip was last given it, so this is what a caller
// uses instead of audio::play when the clip belongs to a strip.
// `wav_path` may be omitted only when the strip's SpecStrip is known to have
// merged -- a strip created inside a system has not, so callers there must
// pass the path they already hold.
void play(flecs::entity strip, const std::string& wav_path = "");

} // namespace spec_strip
