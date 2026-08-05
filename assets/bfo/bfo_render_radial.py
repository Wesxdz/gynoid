import os
import math
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
]

# 2. Build the Directed Graph
G = nx.DiGraph()
G.add_edges_from(bfo_edges)

# 3. Setup Layout Constraints
SPRITE_SIZE = 23
X_GAP = 12
X_SPACING = SPRITE_SIZE + X_GAP

# 4. Setup Radial & Canvas Geometry (1080px Diameter Circle)
DIAMETER = 1080
width = DIAMETER
height = DIAMETER
CENTER_X = width // 2
CENTER_Y = height // 2

# Apple Slice Configuration - Facing Directly Downward
# 90° points straight down. 45° to 135° creates a perfect 90-degree wide wedge.
START_ANGLE = 45
END_ANGLE = 135

# Depth layer spacing offsets (Radii bounds)
R_MIN = 120       # Radius where the root ('entity') sits closer to the center
R_MAX = 500       # Radius where the deepest leaves sit near the canvas bottom edge
MAX_DEPTH = 5     # Total depth levels in BFO 2.0

# 5. Calculate Base Leaf-Centric Linear Positions
leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
leaf_x_coords = {leaf: i * X_SPACING for i, leaf in enumerate(leaves)}

linear_pos = {}
def walk_tree_linear(node, depth):
    if G.out_degree(node) == 0:
        linear_pos[node] = (leaf_x_coords[node], depth)
        return leaf_x_coords[node]
    else:
        child_xs = [walk_tree_linear(child, depth + 1) for child in G.neighbors(node)]
        avg_x = sum(child_xs) / len(child_xs)
        linear_pos[node] = (avg_x, depth)
        return avg_x

# Run initial topological sort pass
walk_tree_linear('entity', 0)

# 6. Transform Linear Coordinates into Polar/Radial Center Locations
adjusted_pos = {}
max_linear_x = (len(leaves) - 1) * X_SPACING

for node, (lx, depth) in linear_pos.items():
    # Normalize the horizontal leaf distribution between 0.0 and 1.0
    t = lx / max_linear_x if max_linear_x > 0 else 0.5

    # Map t to the specified downward arc
    angle_deg = START_ANGLE + t * (END_ANGLE - START_ANGLE)
    angle_rad = math.radians(angle_deg)

    # Calculate radius offset proportional to the node's depth layer
    r = R_MIN + (depth / MAX_DEPTH) * (R_MAX - R_MIN)

    # Convert polar coordinates to Cartesian canvas positions
    cx = CENTER_X + r * math.cos(angle_rad)
    cy = CENTER_Y + r * math.sin(angle_rad)

    adjusted_pos[node] = (cx, cy)

# 7. Initialize Pure Black Canvas
canvas = Image.new('RGBA', (width, height), color=(0, 0, 0, 255))
draw = ImageDraw.Draw(canvas)

# 8. Draw Clean 1-Pixel Radial Connecting Lines (Center to Center)
for parent, child in G.edges():
    px, py = adjusted_pos[parent]
    cx, cy = adjusted_pos[child]

    # Connect centers directly; sprite textures will cleanly overlay the endpoints
    draw.line([(int(px), int(py)), (int(cx), int(cy))], fill=(255, 255, 255, 100), width=1)

# 9. Paste Sprites centered perfectly at 1:1 Native Resolution
HALF_SPRITE = SPRITE_SIZE // 2
for node, (cx, cy) in adjusted_pos.items():
    # Offset center position to find top-left point required for pasting
    x = int(cx - HALF_SPRITE)
    y = int(cy - HALF_SPRITE)

    img_filename = f"{node}.png"
    if os.path.exists(img_filename):
        try:
            sprite = Image.open(img_filename)
            # Preserve alpha channels if transparency exists
            if sprite.mode in ('RGBA', 'LA') or (sprite.mode == 'P' and 'transparency' in sprite.info):
                canvas.paste(sprite, (x, y), sprite.convert('RGBA'))
            else:
                canvas.paste(sprite, (x, y))
        except Exception as e:
            print(f"Error loading {img_filename}: {e}")
    else:
        print(f"Missing file: {img_filename}")

# 10. Save Final Image
output_filename = "bfo_tree_radial_downward.png"
canvas.save(output_filename, "PNG")
print(f"Success: Downward radial tree saved to '{output_filename}' ({width}x{height}px)")
