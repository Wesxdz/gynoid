# Tradewinds server
import zmq
import msgpack
import msgpack_numpy as m
import threading
import time

m.patch()

is_ready = True

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("ipc:///tmp/thornfield_interlocutor_socket")

while True:
    req = msgpack.unpackb(socket.recv())

    if not is_ready:
        socket.send(msgpack.packb({"status": "LOADING"}))
        continue

    if req.get("type") == "message":
        time.sleep(1)
        socket.send(msgpack.packb({"status": "OK", "reply": f"At {time.time()} I read the latest message you typed to me."}))
    else:
        socket.send(msgpack.packb({"status": "READY_WAITING"}))
