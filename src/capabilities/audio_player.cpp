#include "audio_player.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <SDL2/SDL.h>

namespace audio {
namespace {

struct State
{
    SDL_AudioDeviceID device = 0;
    SDL_AudioSpec spec{};

    // The whole clip, converted to the device's format once at load. Kept so a
    // seek can requeue from an offset without touching the file again.
    Uint8* pcm = nullptr;
    Uint32 length = 0;

    std::string path;
    Uint32 queued_from = 0;   // byte offset the current queue started at
    bool open = false;

    // The device reports its position in chunks, so a playhead driven straight
    // off it steps rather than sweeps. These anchor the last observed device
    // position to a wall-clock instant so the value can be carried forward
    // smoothly between updates -- the device stays the authority, the clock
    // only fills the gaps.
    double anchor_pos = 0.0;
    Uint64 anchor_ms = 0;
    double last_raw = -1.0;
    bool is_paused = false;
};

State g;

double bytesPerSecond()
{
    if (!g.open) return 0.0;
    return (double)g.spec.freq * g.spec.channels * (SDL_AUDIO_BITSIZE(g.spec.format) / 8);
}

void freeClip()
{
    if (g.pcm) { SDL_FreeWAV(g.pcm); g.pcm = nullptr; }
    g.length = 0;
    g.path.clear();
    g.queued_from = 0;
}

// Opens the device to match the first clip loaded. Reopened when a later clip
// has a different format, which is cheaper and far simpler than running a
// resampler for what is only ever one voice at a time.
bool openDevice(const SDL_AudioSpec& want)
{
    if (g.open && g.spec.freq == want.freq && g.spec.format == want.format
        && g.spec.channels == want.channels) {
        return true;
    }
    if (g.device) { SDL_CloseAudioDevice(g.device); g.device = 0; g.open = false; }

    if (!SDL_WasInit(SDL_INIT_AUDIO) && SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        std::cerr << "[audio] cannot init SDL audio: " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_AudioSpec spec = want;
    // A large device buffer means coarse position steps. 1024 frames keeps the
    // gaps the clock has to bridge down to about 20ms.
    if (spec.samples == 0 || spec.samples > 1024) spec.samples = 1024;

    SDL_AudioSpec have{};
    // No format changes allowed: the queue-depth arithmetic that produces the
    // playhead assumes the bytes queued are the bytes loaded.
    g.device = SDL_OpenAudioDevice(nullptr, 0, &spec, &have, 0);
    if (!g.device) {
        std::cerr << "[audio] cannot open device: " << SDL_GetError() << std::endl;
        return false;
    }
    g.spec = have;
    g.open = true;
    return true;
}

bool load(const std::string& wav_path)
{
    if (g.path == wav_path && g.pcm) return true;

    SDL_AudioSpec want{};
    Uint8* pcm = nullptr;
    Uint32 length = 0;
    if (!SDL_LoadWAV(wav_path.c_str(), &want, &pcm, &length)) {
        std::cerr << "[audio] cannot load " << wav_path << ": " << SDL_GetError() << std::endl;
        return false;
    }

    if (!openDevice(want)) { SDL_FreeWAV(pcm); return false; }

    freeClip();
    g.pcm = pcm;
    g.length = length;
    g.path = wav_path;
    return true;
}

} // namespace

bool play(const std::string& wav_path)
{
    if (!load(wav_path)) return false;
    seek(0.0);
    return true;
}

void stop()
{
    if (!g.open) return;
    SDL_PauseAudioDevice(g.device, 1);
    SDL_ClearQueuedAudio(g.device);
    g.queued_from = g.length;   // reads as finished rather than as position 0
}

void seek(double seconds)
{
    if (!g.open || !g.pcm) return;

    const double bps = bytesPerSecond();
    Uint32 offset = (bps > 0.0) ? (Uint32)std::max(0.0, seconds * bps) : 0;
    // Align to a frame boundary; a mid-sample offset shifts the channels and
    // turns stereo into noise.
    const int frame = g.spec.channels * (SDL_AUDIO_BITSIZE(g.spec.format) / 8);
    if (frame > 0) offset -= offset % frame;
    offset = std::min(offset, g.length);

    SDL_ClearQueuedAudio(g.device);
    g.queued_from = offset;
    g.last_raw = -1.0;          // force a re-anchor on the next position()
    g.is_paused = false;
    if (offset < g.length) {
        SDL_QueueAudio(g.device, g.pcm + offset, g.length - offset);
    }
    SDL_PauseAudioDevice(g.device, 0);
}

void pause()
{
    if (!g.open || g.is_paused) return;
    SDL_PauseAudioDevice(g.device, 1);
    g.is_paused = true;
}

void resume()
{
    if (!g.open || !g.is_paused) return;
    g.is_paused = false;
    g.last_raw = -1.0;              // re-anchor; the clock moved while stopped
    SDL_PauseAudioDevice(g.device, 0);
}

bool paused() { return g.is_paused; }

bool finished()
{
    return g.open && g.pcm && SDL_GetQueuedAudioSize(g.device) == 0;
}

double position()
{
    if (!g.open || !g.pcm) return 0.0;
    const double bps = bytesPerSecond();
    if (bps <= 0.0) return 0.0;

    // What the device has left to consume tells us where it is. That is the
    // truth, but it arrives one buffer at a time.
    const Uint32 remaining = SDL_GetQueuedAudioSize(g.device);
    const double raw = std::max(0.0, ((double)g.length - (double)remaining) / bps);
    const Uint64 now = SDL_GetTicks64();

    // Re-anchor whenever the device actually moves; between those moments the
    // clock carries the value forward. Wall clock alone would drift on an
    // underrun, and the device alone steps -- together they do neither.
    if (raw != g.last_raw) {
        g.last_raw = raw;
        g.anchor_pos = raw;
        g.anchor_ms = now;
    }

    if (g.is_paused) return std::clamp(g.anchor_pos, 0.0, duration());

    const double elapsed = (double)(now - g.anchor_ms) / 1000.0;
    // Never run further ahead than the next update could possibly be, so a
    // stalled device shows a stalled playhead rather than a sprinting one.
    const double chunk = (double)g.spec.samples / (double)std::max(g.spec.freq, 1);
    const double lead = std::min(elapsed, chunk * 2.0);

    return std::clamp(g.anchor_pos + lead, 0.0, duration());
}

double duration()
{
    const double bps = bytesPerSecond();
    return (bps > 0.0) ? (double)g.length / bps : 0.0;
}

bool playing()
{
    return g.open && g.pcm && SDL_GetQueuedAudioSize(g.device) > 0;
}

const std::string& current()
{
    return g.path;
}

void shutdown()
{
    if (g.device) { SDL_CloseAudioDevice(g.device); g.device = 0; }
    freeClip();
    g.open = false;
}

} // namespace audio
