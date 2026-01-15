#!/usr/bin/env python3
"""Pre-compute color embeddings for fast lookup."""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Semantic descriptors for colors (same as entity_color.py)
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

def main():
    print("Loading model...")
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    # Load palette
    palette_path = Path(__file__).parent.parent / "assets" / "palettes" / "resurrect-64.hex"
    colors = []
    with open(palette_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and len(line) == 6:
                colors.append(line)

    print(f"Loaded {len(colors)} colors")

    # Get descriptions for each color
    descriptions = []
    for color in colors:
        desc = COLOR_SEMANTICS.get(color, "neutral grey default")
        descriptions.append(desc)

    print("Computing embeddings...")
    embeddings = model.encode(descriptions, convert_to_numpy=True)

    # Save embeddings and color list
    output_path = Path(__file__).parent / "color_embeddings.npz"
    np.savez(output_path, embeddings=embeddings, colors=colors)
    print(f"Saved to {output_path}")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
