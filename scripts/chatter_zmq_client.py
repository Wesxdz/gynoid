#!/usr/bin/env python3
# Client for servers/chatter_zmq_server.py -- the ZeroMQ counterpart to
# chatter_client.py, which speaks the websocket server's protocol.
#
# Usage
#   python3 scripts/chatter_zmq_client.py "hello there"
#   python3 scripts/chatter_zmq_client.py "hello" --wait          # sit through the model load
#   python3 scripts/chatter_zmq_client.py "hello" --no-play -o greeting.wav
#
# Only needs zmq and msgpack, so it runs in the base environment; the chatterbox
# env is the server's business, not the caller's.
#
# Two differences from the websocket client are worth knowing:
#
#   * The server binds before its CUDA model finishes loading and answers
#     {"status": "LOADING"} until it is up. --wait polls through that instead of
#     failing, which is usually what you want right after clicking the icon in
#     the Peach Core panel.
#   * An ipc:// connect succeeds whether or not anything is listening -- ZeroMQ
#     just queues. A dead server therefore shows up as a timeout, never as a
#     refused connection, so "no reply" is reported as "is it running?".

import argparse
import os
import sys

import msgpack
import zmq

SOCKET_ADDR = os.environ.get("THORNFIELD_CHATTER_SOCKET",
                             "ipc:///tmp/thornfield_chatter_socket")

# Synthesis blocks the REP socket, so the speak timeout has to cover generation,
# not just round-trip latency.
SPEAK_TIMEOUT_MS = 180_000
PROBE_TIMEOUT_MS = 3_000


def request(payload, addr=SOCKET_ADDR, timeout_ms=SPEAK_TIMEOUT_MS, context=None):
    """One request/reply. Returns the decoded reply, or None on timeout.

    A fresh socket per call, deliberately: REQ is strict lock-step, and a socket
    that has timed out mid-exchange stays wedged until it is discarded. Building
    one per request is the simplest correct thing for a CLI.
    """
    ctx = context or zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(addr)
    try:
        sock.send(msgpack.packb(payload))
        return msgpack.unpackb(sock.recv(), raw=False)
    except zmq.Again:
        return None
    finally:
        sock.close()


def probe(addr=SOCKET_ADDR, context=None):
    """Readiness check. The server answers READY_WAITING to any request type it
    does not recognise once its model is up, and LOADING before that -- so an
    unrecognised type is a probe that costs the server nothing."""
    return request({"type": "ping"}, addr, PROBE_TIMEOUT_MS, context)


def wait_until_ready(addr=SOCKET_ADDR, timeout_s=180.0, verbose=True, context=None):
    import time
    deadline = time.time() + timeout_s
    announced = False
    while time.time() < deadline:
        reply = probe(addr, context)
        if reply is None:
            if verbose and not announced:
                print("waiting for the server to come up...", file=sys.stderr, flush=True)
                announced = True
        elif reply.get("status") == "LOADING":
            if verbose and not announced:
                print("model is loading...", file=sys.stderr, flush=True)
                announced = True
        elif reply.get("status") == "ERROR":
            # The load failed outright; polling for the rest of the timeout
            # would just delay showing why.
            if verbose:
                print(reply.get("error", "server error"), file=sys.stderr)
            return False
        else:
            return True
        time.sleep(1.0)
    return False


def speak(text, audio_prompt_path=None, save_path="output.wav", play_audio=True,
          addr=SOCKET_ADDR, timeout_ms=SPEAK_TIMEOUT_MS, verbose=True, context=None,
          seed=None, exaggeration=None, cfg_weight=None, temperature=None):
    """Synthesize `text`. Returns the server's full reply, or None on failure.

    The reply carries everything needed to reproduce the take -- voice, seed and
    the three generation parameters -- which is what keep() records.
    """
    payload = {
        "type": "speak",
        "text": text,
        "save_path": save_path,
        "play_audio": play_audio,
    }
    for key, value in (("seed", seed), ("exaggeration", exaggeration),
                       ("cfg_weight", cfg_weight), ("temperature", temperature)):
        if value is not None:
            payload[key] = value
    # Omitted rather than sent empty, so the server applies its default voice.
    # It resolves bare names against the voice library and errors on a miss --
    # it will not quietly synthesize in some other voice.
    if audio_prompt_path:
        payload["audio_prompt_path"] = audio_prompt_path

    if verbose:
        print(f"-> {addr}: {text[:60]}{'...' if len(text) > 60 else ''}", file=sys.stderr, flush=True)

    reply = request(payload, addr, timeout_ms, context)

    if reply is None:
        if verbose:
            print(f"no reply within {timeout_ms // 1000}s. Is the server running? "
                  f"Start it from the Peach Core panel, or:\n"
                  f"  ~/miniconda3/envs/chatterbox/bin/python -u servers/chatter_zmq_server.py",
                  file=sys.stderr)
        return None

    status = reply.get("status")
    if status == "OK":
        if verbose:
            voice = os.path.splitext(os.path.basename(reply.get("voice", "?")))[0]
            print(f"wrote {reply['output_path']}  "
                  f"(voice: {voice}, seed: {reply.get('seed')})", file=sys.stderr)
        return reply
    if status == "LOADING":
        if verbose:
            print("server is still loading its model; retry, or pass --wait", file=sys.stderr)
        return None
    if verbose:
        print(f"server said {status}: {reply.get('error', '')}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Keeping takes
#
# runtime/speech is scratch: every synthesis overwrites output.wav. A keep
# promotes a take into voices/ under a name derived from its tags and appends a
# record to voices.jsonl -- same append-only, ts-first shape as
# annotations.jsonl and lexicon.jsonl.
#
# The record is deliberately more than a label. It carries the text, the
# reference voice, the seed and the three generation parameters, so a take you
# liked can be regenerated, or continued with a new line in the same delivery.
# Without the seed a good take is a one-off you can only ever replay.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(PROJECT_ROOT, "voices")
VOICES_INDEX = os.path.join(PROJECT_ROOT, "voices.jsonl")


def slugify(tags):
    import re
    slug = "_".join(re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t in tags if t)
    return re.sub(r"_+", "_", slug).strip("_") or "keep"


def unique_path(directory, stem, ext=".wav"):
    """First free <stem>.wav, <stem>-2.wav, ... Keeping is a reflex, and the
    same tags will recur; silently overwriting a previous keep would be the
    one unrecoverable mistake this tool could make."""
    candidate = os.path.join(directory, stem + ext)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}-{n}{ext}")
        n += 1
    return candidate


def keep(reply, tags, text, name=None, verbose=True):
    """Archive a take and record why it was worth keeping."""
    import json
    import shutil
    import time

    src = reply.get("output_path")
    if not src or not os.path.exists(src):
        # ipc:// means same host, so the server's path is readable here.
        if verbose:
            print(f"cannot keep: {src} is not readable from this machine", file=sys.stderr)
        return None

    os.makedirs(VOICES_DIR, exist_ok=True)
    dest = unique_path(VOICES_DIR, name or slugify(tags))
    shutil.copy2(src, dest)

    record = {
        "ts": int(time.time()),
        "file": os.path.relpath(dest, PROJECT_ROOT),
        "tags": tags,
        "text": text,
        "voice": os.path.splitext(os.path.basename(reply.get("voice", "")))[0],
        "seed": reply.get("seed"),
        "exaggeration": reply.get("exaggeration"),
        "cfg_weight": reply.get("cfg_weight"),
        "temperature": reply.get("temperature"),
    }
    with open(VOICES_INDEX, "a") as f:
        f.write(json.dumps(record) + "\n")

    if verbose:
        print(f"kept {record['file']}  [{', '.join(tags)}]  seed {record['seed']}",
              file=sys.stderr)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS client (ZeroMQ)")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--audio-prompt", "--voice", default=None, dest="audio_prompt",
                        help="Voice from the library, by bare name ('aesthetic_death') "
                             "or full path. Omit for the default, honest_meeting.")
    parser.add_argument("--output", "-o", default="output.wav",
                        help="Output filename, written into the server's speech directory")
    parser.add_argument("--no-play", action="store_true",
                        help="Don't play the audio after generating it")
    parser.add_argument("--socket", default=SOCKET_ADDR,
                        help=f"Server address (default: {SOCKET_ADDR})")
    parser.add_argument("--wait", action="store_true",
                        help="Poll until the model has finished loading, then send")
    parser.add_argument("--timeout", type=int, default=SPEAK_TIMEOUT_MS // 1000,
                        help="Seconds to wait for synthesis (default: %(default)s)")

    delivery = parser.add_argument_group("delivery")
    delivery.add_argument("--seed", type=int, default=None,
                          help="Reproduce a take. Omit and one is chosen, and reported, anyway.")
    delivery.add_argument("--exaggeration", type=float, default=None,
                          help="Emotional intensity (default 0.5)")
    delivery.add_argument("--cfg-weight", type=float, default=None,
                          help="Adherence to the reference voice (default 0.5)")
    delivery.add_argument("--temperature", type=float, default=None,
                          help="Sampling temperature (default 0.8)")

    keeping = parser.add_argument_group("keeping")
    keeping.add_argument("--keep", metavar="TAGS", default=None,
                         help="Archive this take into voices/ and record why, e.g. "
                              "--keep 'dramatic, rising'")
    keeping.add_argument("--name", default=None,
                         help="Filename for the kept take (default: derived from the tags)")
    args = parser.parse_args()

    if args.wait and not wait_until_ready(args.socket):
        print("server never became ready", file=sys.stderr)
        return 1

    reply = speak(
        args.text,
        audio_prompt_path=args.audio_prompt,
        save_path=args.output,
        play_audio=not args.no_play,
        addr=args.socket,
        timeout_ms=args.timeout * 1000,
        seed=args.seed,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    if not reply:
        return 1

    if args.keep is not None:
        tags = [t.strip() for t in args.keep.split(",") if t.strip()]
        keep(reply, tags, args.text, name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
