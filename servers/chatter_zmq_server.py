# Chatterbox TTS over ZeroMQ.
#
# A duplicate of chatter_server.py with the websocket transport replaced by the
# REP/msgpack pattern the rest of Thornfield's servers use (see
# scripts/entity_semantics.py), so it can be launched, watched and stopped from
# the Peach Core panel like any other.
#
# Runs in the `chatterbox` conda environment -- assets/config/servers.json names
# it, and server_registry.cpp execs that env's interpreter directly.
#
# Two things differ from the websocket original beyond the transport:
#
#   * The socket binds before the model loads, and a loader thread flips
#     is_ready. Requests arriving during the ~30s CUDA load get {"status":
#     "LOADING"} instead of blocking the caller's frame.
#   * It announces itself to Thornfield when the model is actually up, which is
#     what turns the Peach Core status band green. Re-announced periodically so
#     the state survives a Thornfield restart while this server keeps running.
#
# Protocol
#   -> {"type": "speak", "text": ..., "audio_prompt_path": ..., "save_path": ...,
#       "play_audio": bool, "seed": int, "exaggeration": float, "cfg_weight": float,
#       "temperature": float}
#   <- {"status": "OK", "output_path": ..., "voice": ..., "seed": ...,
#       "exaggeration": ..., "cfg_weight": ..., "temperature": ...}
#      {"status": "LOADING"} | {"status": "READY_WAITING"} | {"status": "ERROR", "error": ...}
#
# audio_prompt_path may be a bare name from the voice library ("aesthetic_death")
# or a full path. Omit it for the default voice, honest_meeting.

import os
import random
import sys
import threading
import time

import msgpack
import zmq

SERVER_ID = "chatterbox"
SOCKET_ADDR = os.environ.get("THORNFIELD_CHATTER_SOCKET",
                             "ipc:///tmp/thornfield_chatter_socket")
# Thornfield binds this PULL socket; see kAnnounceSocket in src/server_registry.h.
ANNOUNCE_ADDR = os.environ.get("THORNFIELD_ANNOUNCE_SOCKET",
                               "ipc:///tmp/thornfield_server_status")
ANNOUNCE_PERIOD = 5.0

HERE = os.path.dirname(os.path.abspath(__file__))
# Both default under the project rather than the ../../runtime/speech the
# websocket original used, which resolved outside this tree.
OUTPUT_DIR = os.environ.get("THORNFIELD_SPEECH_DIR",
                            os.path.normpath(os.path.join(HERE, "..", "runtime", "speech")))
DEVICE = os.environ.get("THORNFIELD_TTS_DEVICE", "cuda")

# The voice library. It lives outside this project on purpose -- it is a corpus
# of reference clips shared by everything that speaks, not a gynoid asset.
# A request's audio_prompt_path may be a bare name ("aesthetic_death") and is
# resolved against here; anything containing a path separator is used as given.
VOICE_DIR = os.environ.get("THORNFIELD_VOICE_DIR", "/home/wesxdz/paphos/chatter")
DEFAULT_VOICE = os.environ.get("THORNFIELD_VOICE_PROMPT", "honest_meeting.wav")

model = None
is_ready = False
load_error = None       # set instead of is_ready when the model cannot load


def truthy(value, default=True):
    """Booleans arrive as strings.

    tradewinds' SendMapRequest packs a msgpack map of string->string, so a C++
    caller sending play_audio=false delivers the *string* "false" -- which is
    non-empty, and so true to Python. Numbers survive that trip because int()
    and float() parse their own text; booleans do not, and the failure is
    silent: the clip simply plays twice, once from each side.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def log(msg):
    print(f"[chatterbox] {msg}", file=sys.stderr, flush=True)


def announce(context, status="READY"):
    """Tell Thornfield our readiness. Fire and forget -- if nothing is bound the
    message is dropped when the socket closes, which is the correct outcome."""
    try:
        sock = context.socket(zmq.PUSH)
        sock.setsockopt(zmq.LINGER, 250)
        sock.connect(ANNOUNCE_ADDR)
        sock.send(msgpack.packb({"id": SERVER_ID, "status": status}))
        sock.close()
    except Exception as exc:
        log(f"announce failed: {exc}")


def announce_loop(context):
    while True:
        if is_ready:
            announce(context)
        time.sleep(ANNOUNCE_PERIOD)


def load_model(context):
    global model, is_ready, load_error
    try:
        # Imported here, not at module scope: the import alone pulls in torch
        # and costs seconds, and the socket should already be answering by then.
        from chatterbox.tts import ChatterboxTTS

        log(f"loading model on {DEVICE}...")
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
    except Exception as exc:
        # A loader that dies silently is worse than one that never started: the
        # socket stays bound, so every request gets LOADING forever and the
        # Peach Core band sits on "running". CUDA OOM is the common way in.
        load_error = f"{type(exc).__name__}: {exc}"
        log(f"model failed to load -- {load_error}")
        announce(context, "ERROR")
        return

    is_ready = True
    log("model ready")
    announce(context)


def resolve_voice(name):
    """Turn a request's voice reference into a path in the library.

    A bare name is looked up in VOICE_DIR, with .wav appended if it has no
    extension, so callers can say "aesthetic_death". Anything containing a path
    separator is taken literally.
    """
    if os.sep in name or name.startswith("."):
        return name
    if not os.path.splitext(name)[1]:
        name += ".wav"
    return os.path.join(VOICE_DIR, name)


def available_voices(limit=8):
    try:
        names = sorted(f[:-4] for f in os.listdir(VOICE_DIR) if f.endswith(".wav"))
    except OSError:
        return "voice library unreadable at " + VOICE_DIR
    return f"{len(names)} in {VOICE_DIR}, e.g. " + ", ".join(names[:limit])


def synthesize(req):
    import soundfile as sf

    text = req.get("text", "")
    if not text:
        return {"status": "ERROR", "error": "no text provided"}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, os.path.basename(
        req.get("save_path", "temp_speech.wav")))

    # Never substitute a voice silently. Chatterbox will happily synthesize in
    # its own stock voice when handed no prompt, and a missing reference clip
    # coming back as a stranger's voice is far worse than an error saying which
    # clip could not be found.
    requested = req.get("audio_prompt_path") or DEFAULT_VOICE
    prompt = resolve_voice(requested)
    if not os.path.exists(prompt):
        return {"status": "ERROR",
                "error": f"voice '{requested}' not found at {prompt} ({available_voices()})"}

    import torch

    # A seed is always chosen and always reported, even when the caller did not
    # ask for one. Sampling is stochastic (temperature 0.8 by default), so
    # without this a take that happens to sound right is unrepeatable -- you
    # could keep the file but never generate another line in the same delivery.
    seed = req.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params = {
        "exaggeration": float(req.get("exaggeration", 0.5)),
        "cfg_weight": float(req.get("cfg_weight", 0.5)),
        "temperature": float(req.get("temperature", 0.8)),
    }

    log(f"speaking as {os.path.splitext(os.path.basename(prompt))[0]} (seed {seed})")
    wav = model.generate(text, audio_prompt_path=prompt, **params)

    sf.write(save_path, wav.cpu().numpy().squeeze(), model.sr, subtype="PCM_16")
    log(f"wrote {save_path}")

    if truthy(req.get("play_audio"), True):
        threading.Thread(target=play_audio, args=(save_path,), daemon=True).start()

    # Everything needed to reproduce this take goes back with it.
    return {"status": "OK", "output_path": save_path, "voice": prompt,
            "seed": seed, **params}


def play_audio(path):
    try:
        from pydub import AudioSegment
        from pydub.playback import play
        play(AudioSegment.from_wav(path))
    except Exception as exc:
        log(f"playback error: {exc}")


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(SOCKET_ADDR)
    log(f"listening on {SOCKET_ADDR}")

    threading.Thread(target=load_model, args=(context,), daemon=True).start()
    threading.Thread(target=announce_loop, args=(context,), daemon=True).start()

    while True:
        req = msgpack.unpackb(socket.recv(), raw=False)

        if load_error:
            socket.send(msgpack.packb({
                "status": "ERROR",
                "error": f"model failed to load: {load_error}"}))
            continue

        if not is_ready:
            socket.send(msgpack.packb({"status": "LOADING"}))
            continue

        if req.get("type") != "speak":
            socket.send(msgpack.packb({"status": "READY_WAITING"}))
            continue

        try:
            reply = synthesize(req)
        except Exception as exc:  # keep the server alive for the next request
            log(f"synthesis failed: {exc}")
            reply = {"status": "ERROR", "error": str(exc)}

        socket.send(msgpack.packb(reply))


if __name__ == "__main__":
    main()
