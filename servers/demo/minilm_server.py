import zmq
import msgpack
import msgpack_numpy as m
import threading
import time

m.patch()

model = None
is_ready = False

def load_model_worker():
    global model, is_ready
    # Ensure the ZMQ server can respond quickly after startup
    from sentence_transformers import SentenceTransformer

    print("[Loader] Imports done. Loading weights...")
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    is_ready = True
    print("[Loader] Model fully ready.")

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("ipc:///tmp/minilm_socket")

threading.Thread(target=load_model_worker, daemon=True).start()

print("ZMQ Server is now responsive on IPC socket.")

while True:
    req = msgpack.unpackb(socket.recv())

    if not is_ready:
        socket.send(msgpack.packb({"status": "LOADING"}))
        continue

    if req.get("type") == "embed":
        vec = model.encode(req['content'])
        socket.send(msgpack.packb({"status": "OK", "embedding": vec}))
    else:
        socket.send(msgpack.packb({"status": "READY_WAITING"}))
