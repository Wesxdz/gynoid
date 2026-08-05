import os
import networkx as nx
from PIL import Image, ImageDraw

# 1. Define the BFO 2.0 Hierarchy Edges
bfo_edges = [
    ('entity', 'continuant'),
    ('entity', 'occurrent'),

    ('continuant', 'independent_continuant'),
    ('continuant', 'specifically_dependent_continuant'),
    ('continuant', 'generically_dependent_continuant'),

    ('independent_continuant', 'material_entity'),
    ('independent_continuant', 'immaterial_entity'),

    ('material_entity', 'object'),
    ('material_entity', 'fiat_object_part'),
    ('material_entity', 'object_aggregate'),

    ('immaterial_entity', 'site'),
    ('immaterial_entity', 'spatial_region'),
    ('immaterial_entity', 'continuant_fiat_boundary'),

    ('specifically_dependent_continuant', 'quality'),
    ('specifically_dependent_continuant', 'realizable_entity'),

    ('realizable_entity', 'disposition'),
    ('realizable_entity', 'role'),

    ('disposition', 'function'),

    ('occurrent', 'process'),
    ('occurrent', 'process_boundary'),
    ('occurrent', 'spatiotemporal_region'),
    ('occurrent', 'temporal_interval'),

    # ('spatiotemporal_region', 'spatiotemporal_region_unicolor')
]

# 2. Build the Directed Graph
G = nx.DiGraph()
G.add_edges_from(bfo_edges)

# 3. Setup Pixel Spacing Constraints (Tight & Compact)
SPRITE_SIZE = 23
X_GAP = 12       # Exact pixel gap between sprite edges horizontally
Y_GAP = 32       # Exact pixel gap between sprite rows vertically
PADDING = 16     # Canvas margin padding

X_SPACING = SPRITE_SIZE + X_GAP
Y_SPACING = SPRITE_SIZE + Y_GAP

# 4. Calculate Leaf-Centric Layout in Pure Pixel Coordinates
leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
leaf_x_coords = {leaf: i * X_SPACING for i, leaf in enumerate(leaves)}

pos = {}
def walk_tree(node, depth):
    if G.out_degree(node) == 0:
        pos[node] = (leaf_x_coords[node], depth * Y_SPACING)
        return leaf_x_coords[node]
    else:
        child_xs = [walk_tree(child, depth + 1) for child in G.neighbors(node)]
        avg_x = sum(child_xs) / len(child_xs)
        pos[node] = (avg_x, depth * Y_SPACING)
        return avg_x

# Run coordinates generation from root
walk_tree('entity', 0)

# Translate coordinates out of negative space and find exact canvas dimensions
xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]

min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

width = int(max_x - min_x + (2 * PADDING) + SPRITE_SIZE)
height = int(max_y - min_y + (2 * PADDING) + SPRITE_SIZE)

# 5. Initialize Pure Black Canvas
canvas = Image.new('RGBA', (width, height), color=(0, 0, 0, 255))
draw = ImageDraw.Draw(canvas)

# Normalize layout positions with canvas padding
adjusted_pos = {}
for node, (x, y) in pos.items():
    adj_x = x - min_x + PADDING
    adj_y = y - min_y + PADDING
    adjusted_pos[node] = (adj_x, adj_y)

# 6. Draw Clean 1-Pixel Connecting Lines (Bottom-Center to Top-Center)
HALF_SPRITE = SPRITE_SIZE // 2
for parent, child in G.edges():
    px, py = adjusted_pos[parent]
    cx, cy = adjusted_pos[child]

    # Anchor lines exactly at sprite boundaries to keep them crisp
    parent_bottom = (int(px + HALF_SPRITE), int(py + SPRITE_SIZE - 1))
    child_top = (int(cx + HALF_SPRITE), int(cy))

    # Semi-transparent white line (alpha=100) so it doesn't overpower the pixel art
    draw.line([parent_bottom, child_top], fill=(255, 255, 255, 100), width=1)

# 7. Paste Sprites at 1:1 Native Resolution
for node, (x, y) in adjusted_pos.items():
    img_filename = f"{node}.png"
    if os.path.exists(img_filename):
        try:
            sprite = Image.open(img_filename)
            # Use alpha channel as a paste mask if it exists to preserve transparency
            if sprite.mode in ('RGBA', 'LA') or (sprite.mode == 'P' and 'transparency' in sprite.info):
                canvas.paste(sprite, (int(x), int(y)), sprite.convert('RGBA'))
            else:
                canvas.paste(sprite, (int(x), int(y)))
        except Exception as e:
            print(f"Error loading {img_filename}: {e}")
    else:
        print(f"Missing file: {img_filename}")

# 8. Save Final Image
output_filename = "bfo_tree_pixel_perfect.png"
canvas.save(output_filename, "PNG")
print(f"Success: Compact, native-scaled tree saved to '{output_filename}' ({width}x{height}px)")
