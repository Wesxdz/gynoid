#pragma once

// Audio playback, so Thornfield knows where in a clip it is.
//
// The chatter server can play its own output through pydub, but a process that
// plays a file cannot tell a UI what its playhead is doing, and cannot be told
// to jump. Owning playback here is what makes the spectrogram strip in the
// Interlocutor a player rather than a picture: the position is measured from
// the device queue rather than from wall clock, so it stays true if the device
// stalls, and seeking is just requeueing from an offset.
//
// One clip at a time, deliberately. This is a voice, not a mixer.

#include <string>

namespace audio {

// Opens the device lazily on first play. Safe to call repeatedly.
bool play(const std::string& wav_path);

void stop();

// Hold the device without discarding the queue, so resume() continues from
// exactly where it stopped. position() freezes while paused, which is what
// makes a paused playhead sit still rather than snap back.
void pause();
void resume();
bool paused();

// True once the clip has run to its end. Distinguishes "finished, so a click
// should start it again" from "stopped partway".
bool finished();

// Restart the current clip from `seconds`. Ignored when nothing is loaded.
void seek(double seconds);

// Seconds into the clip, or 0 when idle. Derived from what the device has
// actually consumed, not from elapsed time.
double position();

// Length of the loaded clip in seconds, 0 when nothing is loaded.
double duration();

bool playing();

// Path of the loaded clip, empty when idle. Lets a UI tell which strip owns
// the playhead without tracking it separately.
const std::string& current();

void shutdown();

} // namespace audio
