#include "speech.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>

#include <msgpack.hpp>

#include "../components.h"
#include "../tradewinds.h"
#include "function_library.h"
#include "audio_player.h"
#include "spec_strip.h"

using namespace tradewinds;

namespace speech {
namespace {

constexpr const char* kClientName = "ChatterClient";
constexpr const char* kSocket = "ipc:///tmp/thornfield_chatter_socket";

flecs::world* g_world = nullptr;

// The message the next successful reply hangs its spectrogram strip under.
// REQ/REP allows one call in flight, so there is exactly one pending target.
flecs::entity g_strip_target;

// Every utterance gets its own file. The server defaults to output.wav, so
// without this each clip overwrote the last -- and since a strip is identified
// by its path, every strip then shared one spectrogram AND one playhead, and
// seeking any of them seeked all of them.
int g_utterance = 0;

std::string field(std::map<std::string, msgpack::object>& res, const char* key)
{
    auto it = res.find(key);
    if (it == res.end()) return {};
    try {
        return it->second.as<std::string>();
    } catch (const std::exception&) {
        return {};
    }
}

// The reply lands here, one round trip later. Speech is fire-and-forget from
// the caller's point of view, so this exists to surface failures -- a wrong
// voice name or a server that never loaded its model would otherwise be
// silence, which is indistinguishable from success for a TTS call.
void speak_response(std::map<std::string, msgpack::object>& res)
{
    const std::string status = field(res, "status");
    if (status == "OK") {
        const std::string wav = field(res, "output_path");
        std::cout << "[speak] " << wav
                  << " (voice " << field(res, "voice") << ")" << std::endl;

        // The utterance becomes something you can look at and replay, hung
        // under the line that asked for it.
        if (!wav.empty()) {
            flecs::entity strip;
            if (g_strip_target.is_valid() && g_strip_target.is_alive()) {
                strip = spec_strip::create_above(g_strip_target, wav);
            }
            // The server no longer plays its own output. Routing through the
            // strip hands it the transport, so the new utterance is the one
            // wearing the playhead.
            if (strip.is_valid()) spec_strip::play(strip, wav); else audio::play(wav);
        }
        g_strip_target = flecs::entity();
    } else if (status == "LOADING") {
        std::cerr << "[speak] the speech server is still loading its model" << std::endl;
    } else {
        std::cerr << "[speak] " << status << ": " << field(res, "error") << std::endl;
    }
}

// "as <voice>: the words" selects a reference clip from the library; anything
// else is spoken in the default voice. Kept here rather than in the registry
// because the shape of a command line is the function's own business.
functions::Args parse_speak(const std::string& tail)
{
    functions::Args args;
    std::string text = tail;

    if (text.rfind("as ", 0) == 0 || text.rfind("As ", 0) == 0) {
        const auto colon = text.find(':');
        if (colon != std::string::npos) {
            std::string voice = text.substr(3, colon - 3);
            while (!voice.empty() && voice.back() == ' ') voice.pop_back();
            if (!voice.empty()) args["voice"] = voice;
            text = text.substr(colon + 1);
        }
    }

    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.erase(0, 1);
    args["text"] = text;
    return args;
}

functions::Result speak(const functions::Args& args)
{
    const auto text_it = args.find("text");
    if (text_it == args.end() || text_it->second.empty()) {
        return {false, "nothing to say"};
    }
    if (!g_world) return {false, "speech capability not initialised"};

    flecs::entity client = g_world->lookup(kClientName);
    if (!client.is_valid()) return {false, "no speech client"};

    // REQ allows one request in flight; a second while the first is unanswered
    // would desynchronise the socket for every later call.
    if (client.has<AwaitResponse>()) {
        return {false, "still speaking; wait for the current line to finish"};
    }

    // tradewinds' SendMapRequest packs a string->string msgpack map, and the
    // server coerces. Optional keys are omitted rather than sent empty so the
    // server applies its own defaults.
    std::map<std::string, std::string> data{
        {"type", "speak"},
        {"text", text_it->second},
        {"save_path", "utterance_" + std::to_string(++g_utterance) + ".wav"},
        // Thornfield plays it, so the server must not: two players would
        // double every line, one of them without a playhead.
        {"play_audio", "false"},
    };
    for (const char* key : {"voice", "seed", "exaggeration", "cfg_weight", "temperature"}) {
        const auto it = args.find(key);
        if (it != args.end() && !it->second.empty()) {
            // The server names the reference clip audio_prompt_path; "voice" is
            // the word a caller would reach for.
            data[std::string(key) == "voice" ? "audio_prompt_path" : key] = it->second;
        }
    }

    client.set<SendMapRequest>({data});
    client.set<AwaitResponse>({speak_response});
    return {true, "speaking"};
}

} // namespace

void expect_strip_on(flecs::entity message)
{
    g_strip_target = message;
}

void register_capability(flecs::world& world)
{
    g_world = &world;

    world.entity(kClientName)
         .set<ZMQClient>({kSocket, zmq::socket_type::req});

    functions::register_function({
        "speak",
        "Say something aloud in the gynoid's voice.",
        {
            {"text", "string", "The words to speak.", true},
            {"voice", "string",
             "Reference clip from the voice library, by bare name (e.g. 'honest_meeting', "
             "'aesthetic_death'). Omit for the default voice.", false},
            {"seed", "number",
             "Reproduce a previous take exactly. Omit for a fresh delivery.", false},
            {"exaggeration", "number", "Emotional intensity, 0-1. Default 0.5.", false},
            {"cfg_weight", "number", "Adherence to the reference voice, 0-1. Default 0.5.", false},
            {"temperature", "number", "Sampling temperature. Default 0.8.", false},
        },
        speak,
        parse_speak,
    });
}

} // namespace speech
