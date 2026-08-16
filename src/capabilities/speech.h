#pragma once

// Speech as a callable capability.
//
// Registers `speak` in the function library and owns the ZMQ client that
// reaches servers/chatter_zmq_server.py. Nothing that wants the gynoid to say
// something needs to know about the socket, the voice library or msgpack --
// they call functions::invoke("speak", {{"text", ...}}) or type "Speak ..."
// into any frontend wired to functions::dispatch_text.
//
// Call after the ZMQ context exists and after the tradewinds module is
// imported; the client entity is created here.

#include <flecs.h>

namespace speech {

// Creates the chatter REQ client and registers the speak function.
void register_capability(flecs::world& world);

// Hang the next utterance's spectrogram strip under `message`. The reply
// arrives a round trip later, by which time the caller's UI exists -- so the
// target is recorded rather than passed through the call.
void expect_strip_on(flecs::entity message);

} // namespace speech
