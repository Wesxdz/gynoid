#!/usr/bin/env python3
"""
Assign entity colors based on semantic similarity.
Uses pre-computed embeddings + fast model for quick lookup.
"""

import sys
import json
import numpy as np
from pathlib import Path

def get_entity_color(entity_text, taken_colors=None):
    """Get semantically appropriate color using pre-computed embeddings."""
    if taken_colors is None:
        taken_colors = []

    # Load pre-computed color embeddings
    embeddings_path = Path(__file__).parent / "color_embeddings.npz"
    if not embeddings_path.exists():
        return {"color": "4d9be6", "index": 47, "error": "run precompute_colors.py first"}

    data = np.load(embeddings_path, allow_pickle=True)
    color_embeddings = data['embeddings']
    colors = list(data['colors'])

    # Use smaller L3 model for faster inference (about 2x faster)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
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

    best_idx = sorted_indices[0]
    return {"color": colors[best_idx], "index": int(best_idx), "similarity": float(similarities[best_idx])}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: entity_color.py <entity_text> [taken_colors_json]")
        sys.exit(1)

    entity_text = sys.argv[1]
    taken_colors = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
    print(json.dumps(get_entity_color(entity_text, taken_colors)))
