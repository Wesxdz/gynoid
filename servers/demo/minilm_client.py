import time
import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()

context = zmq.Context()

print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("ipc:///tmp/minilm_socket")

while True:
    socket.send(msgpack.packb({"type": "status"}))
    resp = msgpack.unpackb(socket.recv())

    if resp["status"] == "LOADING":
        print("Server warming up...")
        time.sleep(0.1)
        continue

    print("Server is ready! Sending task...")
    socket.send(msgpack.packb({"type": "embed", "content": "Keep it simple!"}))

    raw_message = socket.recv()
    data = msgpack.unpackb(raw_message)
    print(data)
    time.sleep(1)
