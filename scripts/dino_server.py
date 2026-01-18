#!/usr/bin/env python3
"""
DINO Embedding Server

TCP server that accepts VNC frame images, computes DINOv2 embeddings,
maintains a rolling average (max 10 frames with skipping), and responds
with the current rolling average embedding.

Protocol:
  Request (C++ -> Python):
    Header (16 bytes):
      - magic: 4 bytes = "DINO"
      - width: 4 bytes (uint32, little-endian)
      - height: 4 bytes (uint32, little-endian)
      - pitch: 4 bytes (uint32, little-endian)
    Body:
      - pixel_data: height * pitch bytes (BGRA format)

  Response (Python -> C++):
    Header (8 bytes):
      - magic: 4 bytes = "EMBD"
      - reserved: 4 bytes (uint32, little-endian) - reserved for future use
    Body:
      - cos_diff: 4 bytes (float32, little-endian) - cosine difference (1 - cos_sim) between current frame and rolling average
      - frame_count: 4 bytes (uint32) - number of frames in current average
"""

import argparse
import socket
import struct
import signal
import sys
from collections import deque
from threading import Lock

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoEmbeddingServer:
    MAGIC_REQUEST = b"DINO"
    MAGIC_RESPONSE = b"EMBD"
    HEADER_SIZE = 16  # 4 (magic) + 4 (width) + 4 (height) + 4 (pitch)

    def __init__(self, host='127.0.0.1', port=9999, max_frames=10):
        self.host = host
        self.port = port
        self.max_frames = max_frames

        # Rolling average state
        self.embeddings = deque(maxlen=max_frames)
        self.rolling_average = None
        self.lock = Lock()

        # Server state
        self.server_socket = None
        self.running = False

        # Load DINOv2 model
        print(f"[DINO] Loading DINOv2 model (facebook/dinov2-small)...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.model = AutoModel.from_pretrained('facebook/dinov2-small')
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"[DINO] Model loaded on {self.device}")

        # Get embedding dimension
        with torch.no_grad():
            dummy_input = self.processor(images=[Image.new('RGB', (224, 224))], return_tensors="pt")
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            dummy_output = self.model(**dummy_input)
            self.embedding_dim = dummy_output.last_hidden_state[:, 0, :].shape[1]
        print(f"[DINO] Embedding dimension: {self.embedding_dim}")

    def process_frame(self, pixel_data: bytes, width: int, height: int, pitch: int) -> np.ndarray:
        """Convert BGRA pixel data to image and compute DINOv2 embedding."""
        # Convert BGRA to RGB PIL Image
        # pixel_data is height * pitch bytes, where each row is pitch bytes
        # and each pixel is 4 bytes (BGRA)

        # Reshape pixel data accounting for pitch
        pixels = np.frombuffer(pixel_data, dtype=np.uint8)
        pixels = pixels.reshape((height, pitch))

        # Extract actual pixel data (width * 4 bytes per row)
        pixels = pixels[:, :width * 4]
        pixels = pixels.reshape((height, width, 4))

        # Convert BGRA to RGB
        rgb = pixels[:, :, [2, 1, 0]]  # BGR -> RGB (ignore alpha)

        # Create PIL Image
        image = Image.fromarray(rgb, mode='RGB')

        # Process with DINOv2
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return embedding

    def compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def update_rolling_average(self, embedding: np.ndarray) -> tuple:
        """Add embedding to queue, compute rolling average, and return cosine diff."""
        with self.lock:
            # Compute cosine diff before adding current frame (compare to existing average)
            cos_diff = 0.0
            if self.rolling_average is not None:
                cos_sim = self.compute_cosine_similarity(embedding, self.rolling_average)
                cos_diff = 1.0 - cos_sim

            # Add new embedding to queue
            self.embeddings.append(embedding)

            # Compute new average
            if len(self.embeddings) > 0:
                self.rolling_average = np.mean(list(self.embeddings), axis=0)

            return cos_diff, len(self.embeddings)

    def handle_client(self, conn: socket.socket, addr):
        """Handle a single client connection."""
        print(f"[DINO] Client connected from {addr}")

        try:
            while self.running:
                # Read header
                header = self._recv_exact(conn, self.HEADER_SIZE)
                if header is None:
                    print(f"[DINO] Client {addr} disconnected")
                    break

                # Parse header
                magic = header[:4]
                if magic != self.MAGIC_REQUEST:
                    print(f"[DINO] Invalid magic from {addr}: {magic}")
                    break

                width, height, pitch = struct.unpack('<III', header[4:16])

                # Read pixel data
                data_size = height * pitch
                pixel_data = self._recv_exact(conn, data_size)
                if pixel_data is None:
                    print(f"[DINO] Failed to receive pixel data from {addr}")
                    break

                # Process frame
                try:
                    embedding = self.process_frame(pixel_data, width, height, pitch)
                    cos_diff, frame_count = self.update_rolling_average(embedding)

                    # Send response
                    response = self._build_response(cos_diff, frame_count)
                    conn.sendall(response)

                    print(f"[DINO] Processed frame {width}x{height}, cos_diff={cos_diff:.4f}, frames={frame_count}")

                except Exception as e:
                    print(f"[DINO] Error processing frame: {e}")
                    # Send error response
                    response = self._build_response(0.0, 0)
                    conn.sendall(response)

        except Exception as e:
            print(f"[DINO] Error handling client {addr}: {e}")
        finally:
            conn.close()
            print(f"[DINO] Connection closed for {addr}")

    def _recv_exact(self, conn: socket.socket, size: int) -> bytes:
        """Receive exactly size bytes from socket."""
        data = b''
        while len(data) < size:
            try:
                chunk = conn.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.error:
                return None
        return data

    def _build_response(self, cos_diff: float, frame_count: int) -> bytes:
        """Build response packet with cosine difference and frame count."""
        # Header: magic (4) + reserved (4)
        header = self.MAGIC_RESPONSE + struct.pack('<I', 0)

        # Body: cos_diff (4) + frame_count (4)
        body = struct.pack('<f', cos_diff) + struct.pack('<I', frame_count)

        return header + body

    def start(self):
        """Start the server and listen for connections."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow periodic check for shutdown

        self.running = True
        print(f"[DINO] Server listening on {self.host}:{self.port}")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                self.handle_client(conn, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[DINO] Accept error: {e}")

    def stop(self):
        """Stop the server gracefully."""
        print("[DINO] Shutting down...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("[DINO] Server stopped")


def main():
    parser = argparse.ArgumentParser(description='DINO Embedding Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=9999, help='Port to listen on')
    parser.add_argument('--max-frames', type=int, default=10, help='Max frames for rolling average')
    args = parser.parse_args()

    server = DinoEmbeddingServer(host=args.host, port=args.port, max_frames=args.max_frames)

    # Handle shutdown signals
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.start()


if __name__ == '__main__':
    main()
