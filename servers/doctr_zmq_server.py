# docTR OCR over ZeroMQ.
#
# Wraps ~/paphos/doctr_server/dl_ocr.py's one-shot pipeline in the REP/msgpack
# pattern the rest of Thornfield's servers use (see chatter_zmq_server.py), so
# it can be launched, watched and stopped from the Peach Core panel.
#
# Same readiness discipline as chatterbox: the socket binds before the model
# loads, a loader thread flips is_ready, and requests during the load get
# {"status": "LOADING"} instead of blocking. Announces READY to Thornfield's
# inbox once the predictor is actually up, re-announced periodically so the
# state survives a Thornfield restart.
#
# Protocol
#   -> {"type": "ocr", "path": <image path>}
#   <- {"status": "OK", "count": N,
#       "words": [[text, xmin, ymin, xmax, ymax], ...]}   # absolute pixels
#      {"status": "LOADING"} | {"status": "ERROR", "error": ...}
#
# Runs in the `droid_slam` conda environment -- the one with doctr installed;
# assets/config/servers.json names it.

import os
import sys
import threading
import time

import msgpack
import zmq

SERVER_ID = "doctr"
SOCKET_ADDR = os.environ.get("THORNFIELD_DOCTR_SOCKET",
                             "ipc:///tmp/thornfield_doctr_socket")
# Thornfield binds this PULL socket; see kAnnounceSocket in src/server_registry.h.
ANNOUNCE_ADDR = os.environ.get("THORNFIELD_ANNOUNCE_SOCKET",
                               "ipc:///tmp/thornfield_server_status")
ANNOUNCE_PERIOD = 5.0

model = None
is_ready = False
load_error = None


def log(msg):
    print(f"[doctr] {msg}", file=sys.stderr, flush=True)


def announce(context, status="READY"):
    """Fire-and-forget readiness report; dropped if nothing is bound."""
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
        # Imported here, not at module scope: the import pulls in torch and
        # costs seconds, and the socket should be answering LOADING by then.
        from doctr.models import ocr_predictor

        log("loading predictor...")
        model = ocr_predictor(pretrained=True)
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        log(f"predictor failed to load -- {load_error}")
        announce(context, "ERROR")
        return

    is_ready = True
    log("predictor ready")
    announce(context)


def read_words(path):
    """One page's words with absolute pixel boxes -- the same flattening
    dl_ocr.py does, minus the drawing."""
    from doctr.io import DocumentFile

    doc = DocumentFile.from_images(path)
    export = model(doc).export()

    words = []
    for page in export["pages"]:
        h, w = page["dimensions"]
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    (x0, y0), (x1, y1) = word["geometry"]
                    words.append([word["value"],
                                  int(round(x0 * w)), int(round(y0 * h)),
                                  int(round(x1 * w)), int(round(y1 * h))])
    return words


def handle(req):
    if req.get("type") != "ocr":
        return {"status": "ERROR", "error": "unknown request type"}
    if load_error:
        return {"status": "ERROR", "error": load_error}
    if not is_ready:
        return {"status": "LOADING"}

    path = (req.get("path") or "").strip()
    if not path or not os.path.exists(path):
        return {"status": "ERROR", "error": f"no such image: {path}"}
    try:
        words = read_words(path)
        return {"status": "OK", "count": len(words), "words": words}
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(SOCKET_ADDR)
    log(f"listening on {SOCKET_ADDR}")

    threading.Thread(target=load_model, args=(context,), daemon=True).start()
    threading.Thread(target=announce_loop, args=(context,), daemon=True).start()

    while True:
        try:
            req = msgpack.unpackb(socket.recv(), raw=False)
        except Exception as exc:
            socket.send(msgpack.packb({"status": "ERROR", "error": f"bad request: {exc}"}))
            continue
        socket.send(msgpack.packb(handle(req)))


if __name__ == "__main__":
    main()
