#!/usr/bin/env python3
"""
Entity color server - keeps MiniLM model loaded for fast semantic color assignment.
Communicates via Unix socket.
"""

import os
import sys
import json
import socket
import numpy as np
from pathlib import Path

# Load model once at startup (L3 is faster than L6)
print("Loading MiniLM model...", file=sys.stderr)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
print("Model loaded!", file=sys.stderr)

# Load palette
palette_path = Path(__file__).parent.parent / "assets" / "palettes" / "resurrect-64.hex"
colors = []
with open(palette_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and len(line) == 6:
            colors.append(line)

# Semantic descriptors
COLOR_SEMANTICS = {
    "2e222f": "dark shadow night void mystery death evil darkness abyss",
    "3e3546": "dark shadow dusk twilight mysterious hidden secret stealth",
    "625565": "grey stone rock mountain cave ancient wisdom old",
    "313638": "dark charcoal ash smoke shadow industrial metal",
    "966c6c": "dusty rose mauve faded old antique vintage cloth fabric",
    "ab947a": "tan sand desert earth clay pottery ancient dried",
    "694f62": "plum dusk purple brown earth soil dirt mud",
    "4c3e24": "brown earth soil mud dirt ground wood bark tree forest",
    "676633": "olive khaki military earth swamp marsh bog moss",
    "7f708a": "lavender grey mist fog cloud spirit ghost ethereal",
    "9babb2": "silver grey metal steel armor knight cold winter ice",
    "c7dcd0": "pale mint frost ice snow winter cold light pure",
    "ffffff": "white light pure holy divine sacred angel heaven bright",
    "6e2727": "blood dark crimson death danger evil demon sin",
    "b33831": "red blood rust iron anger rage war battle",
    "ea4f36": "red fire flame hot danger warning alert",
    "f57d4a": "orange fire sunset warm autumn fall harvest",
    "ae2334": "crimson blood ruby gem precious jewel royalty",
    "e83b3b": "red hot fire danger warning alert emergency",
    "fb6b1d": "orange flame fire burning hot heat energy",
    "f79617": "orange amber gold warm sunset honey autumn",
    "f9c22b": "yellow gold sun bright light treasure wealth coin",
    "7a3045": "maroon wine burgundy rich royal noble blood",
    "9e4539": "rust copper bronze metal autumn fall earth",
    "cd683d": "copper orange rust autumn harvest pumpkin",
    "e6904e": "peach orange soft warm gentle kind friendly",
    "fbb954": "gold yellow sun bright happy joy treasure",
    "a2a947": "lime green grass spring fresh new growth life",
    "d5e04b": "yellow green chartreuse spring bright vibrant energy",
    "fbff86": "pale yellow cream light soft gentle warm sunny",
    "165a4c": "dark green forest deep jungle mysterious nature",
    "239063": "green forest nature tree plant leaf life growth",
    "1ebc73": "emerald green bright vibrant life nature magic",
    "91db69": "lime green bright spring fresh grass meadow",
    "cddf6c": "yellow green spring new growth fresh young",
    "374e4a": "dark green forest pine evergreen nature wood",
    "547e64": "green sage herb plant nature forest calm",
    "92a984": "sage green soft calm peaceful nature garden",
    "b2ba90": "pale green soft muted calm peaceful gentle",
    "0b5e65": "dark teal ocean deep sea water mysterious",
    "0b8a8f": "teal ocean sea water aquatic marine fish",
    "0eaf9b": "cyan turquoise tropical ocean beach paradise",
    "30e1b9": "bright cyan aqua water magic spell ice",
    "8ff8e2": "pale cyan ice frost snow cold winter magic",
    "323353": "dark blue night sky space void cosmic",
    "484a77": "blue purple dusk twilight night mysterious",
    "4d65b4": "blue sky day clear calm peaceful serene",
    "4d9be6": "bright blue sky day light clear water",
    "8fd3ff": "pale blue sky light air wind cloud",
    "45293f": "dark purple night mystery magic arcane shadow",
    "6b3e75": "purple violet magic arcane mystic spell",
    "905ea9": "purple magic spell mystic wizard mage",
    "a884f3": "bright purple magic spell energy power",
    "eaaded": "pale lavender light magic fairy ethereal",
    "753c54": "dark pink magenta berry fruit sweet",
    "a24b6f": "pink magenta flower bloom spring love",
    "cf657f": "pink rose flower love romance beauty",
    "ed8099": "pink soft gentle love kind caring",
    "831c5d": "dark magenta purple berry fruit wine",
    "c32454": "magenta pink red hot passion love",
    "f04f78": "bright pink magenta flower bloom spring",
    "f68181": "coral pink soft warm gentle kind",
    "fca790": "peach pink soft warm gentle skin flesh",
    "fdcbb0": "pale peach cream soft warm gentle light",
}

# Pre-compute color embeddings
descriptions = [COLOR_SEMANTICS.get(c, "neutral grey default") for c in colors]
color_embeddings = model.encode(descriptions, convert_to_numpy=True)
print(f"Pre-computed embeddings for {len(colors)} colors", file=sys.stderr)

def get_entity_color(entity_text, taken_colors):
    """Get semantically appropriate color."""
    entity_embedding = model.encode([entity_text], convert_to_numpy=True)[0]

    # Cosine similarities
    similarities = np.dot(color_embeddings, entity_embedding) / (
        np.linalg.norm(color_embeddings, axis=1) * np.linalg.norm(entity_embedding)
    )

    sorted_indices = np.argsort(similarities)[::-1]
    taken_lower = [c.lower() for c in taken_colors]

    for idx in sorted_indices:
        color = colors[idx]
        if color.lower() not in taken_lower:
            return {"color": color, "index": int(idx), "similarity": float(similarities[idx])}

    # All taken, return best anyway
    best_idx = sorted_indices[0]
    return {"color": colors[best_idx], "index": int(best_idx), "similarity": float(similarities[best_idx])}

def main():
    socket_path = "/tmp/entity_color.sock"

    # Remove old socket
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    os.chmod(socket_path, 0o777)

    print(f"Server listening on {socket_path}", file=sys.stderr)

    while True:
        conn, _ = server.accept()
        try:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                continue

            request = json.loads(data)
            entity_text = request.get("text", "")
            taken_colors = request.get("taken", [])

            result = get_entity_color(entity_text, taken_colors)
            conn.sendall(json.dumps(result).encode('utf-8'))
        except Exception as e:
            conn.sendall(json.dumps({"error": str(e)}).encode('utf-8'))
        finally:
            conn.close()

if __name__ == "__main__":
    main()
