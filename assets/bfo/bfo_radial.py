import os
import math
import networkx as nx
from PIL import Image, ImageDraw

# ============================================================
#  CONFIG  — the knobs you'll actually touch
# ============================================================
DIAMETER     = 1080      # the circle you're fitting the tree into
SPRITE_SIZE  = 23        # native sprite size (px), pasted 1:1
PADDING      = 16        # gap between the outermost layer and the rim

ARC_DEGREES  = 70.0      # angular width of the "apple slice" wedge
BISECTOR_DEG = 90.0      # direction the slice points (PIL: 0=right, 90=down)
APEX_OFFSET  = 0.0     # how far the apex sits from circle-center, toward the
                         # NEAR edge (0 = centered pie-slice; larger = long sliver)
R_INNER      = 24.0      # radius of the root from the apex (apex stand-off)
LEAVES_ON_RIM = True     # True: leaves snap to the circle rim along their ray
                         # False: leaf radius = its own depth (organic, honest depth)

LINE_ALPHA   = 100       # connector line opacity (0-255)
DRAW_GUIDES  = True      # draw the bounding circle + wedge outline (False = clean art)

OUTPUT       = "bfo_tree_radial.png"

# ============================================================
#  1. BFO 2.0 hierarchy (unchanged from your original)
# ============================================================
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
    ('occurrent', 'temporal_region'),
]

G = nx.DiGraph()
G.add_edges_from(bfo_edges)
ROOT = 'entity'

# ============================================================
#  2. Geometry
# ============================================================
R_HALF      = SPRITE_SIZE / 2.0
R_CIRCLE    = DIAMETER / 2.0
CIRCLE_C    = (R_CIRCLE, R_CIRCLE)                 # bounding circle center (canvas center)

bx, by = math.cos(math.radians(BISECTOR_DEG)), math.sin(math.radians(BISECTOR_DEG))
APEX    = (CIRCLE_C[0] - APEX_OFFSET * bx,         # apex retreats toward the near edge
           CIRCLE_C[1] - APEX_OFFSET * by)

def ray_to_circle(angle):
    """Distance from APEX to the bounding circle along `angle` (rad)."""
    dx, dy = math.cos(angle), math.sin(angle)
    fx, fy = APEX[0] - CIRCLE_C[0], APEX[1] - CIRCLE_C[1]
    b = dx * fx + dy * fy
    c = fx * fx + fy * fy - R_CIRCLE * R_CIRCLE     # < 0 since apex is inside
    return -b + math.sqrt(b * b - c)                # forward intersection distance

def tree_depth(node):
    kids = list(G.neighbors(node))
    return 0 if not kids else 1 + max(tree_depth(k) for k in kids)

MAX_DEPTH = tree_depth(ROOT)                        # this is your "k" (layers - 1)

# Leaf order via DFS so sibling subtrees fan out without crossing.
ordered_leaves = []
def collect(node):
    kids = list(G.neighbors(node))
    if not kids:
        ordered_leaves.append(node)
    for k in kids:
        collect(k)
collect(ROOT)

n = len(ordered_leaves)
arc0 = math.radians(BISECTOR_DEG - ARC_DEGREES / 2.0)
arc1 = math.radians(BISECTOR_DEG + ARC_DEGREES / 2.0)
leaf_angle = {
    leaf: (arc0 if n == 1 else arc0 + (arc1 - arc0) * i / (n - 1))
    for i, leaf in enumerate(ordered_leaves)
}

# Concentric rings for internal layers must clear the circle at EVERY angle in
# the arc, so size the step off the tightest ray (the arc extremes).
R_OUTER_MIN = min(ray_to_circle(arc0), ray_to_circle(arc1)) - PADDING - R_HALF
RADIAL_STEP = (R_OUTER_MIN - R_INNER) / MAX_DEPTH    # per-layer radial offset

# radius: leaves snap to their own ray's rim (LEAVES_ON_RIM) else depth ring;
# internal nodes sit on concentric depth rings. angle = centroid of children.
polar = {}
def walk(node, depth):
    kids = list(G.neighbors(node))
    if not kids:
        ang = leaf_angle[node]
        if LEAVES_ON_RIM:
            r = ray_to_circle(ang) - PADDING - R_HALF
        else:
            r = R_INNER + depth * RADIAL_STEP
    else:
        ang = sum(walk(k, depth + 1) for k in kids) / len(kids)
        r = R_INNER + depth * RADIAL_STEP
    polar[node] = (r, ang)
    return ang
walk(ROOT, 0)

# polar (about APEX) -> pixel (sprite CENTER point)
center_px = {
    node: (APEX[0] + r * math.cos(a), APEX[1] + r * math.sin(a))
    for node, (r, a) in polar.items()
}

# ============================================================
#  3. Canvas
# ============================================================
canvas = Image.new('RGBA', (DIAMETER, DIAMETER), (0, 0, 0, 255))
draw   = ImageDraw.Draw(canvas)

if DRAW_GUIDES:
    g = (255, 255, 255, 40)
    draw.ellipse([0, 0, DIAMETER - 1, DIAMETER - 1], outline=g, width=1)
    # two straight cuts from the apex out to where each arc edge meets the rim
    rim_pts = []
    for a in (arc0, arc1):
        t = ray_to_circle(a)
        p = (APEX[0] + t * math.cos(a), APEX[1] + t * math.sin(a))
        rim_pts.append(p)
        draw.line([APEX, p], fill=g, width=1)
    # close the slice with the matching arc of the bounding circle
    a_start = math.degrees(math.atan2(rim_pts[0][1] - CIRCLE_C[1],
                                      rim_pts[0][0] - CIRCLE_C[0]))
    a_end   = math.degrees(math.atan2(rim_pts[1][1] - CIRCLE_C[1],
                                      rim_pts[1][0] - CIRCLE_C[0]))
    draw.arc([0, 0, DIAMETER - 1, DIAMETER - 1], a_start, a_end, fill=g, width=1)

# connectors: parent center -> child center (sprites paste over the joints)
for parent, child in G.edges():
    draw.line([center_px[parent], center_px[child]],
              fill=(255, 255, 255, LINE_ALPHA), width=1)

# ============================================================
#  4. Paste sprites centered on each node
# ============================================================
for node, (cx, cy) in center_px.items():
    x, y = int(round(cx - R_HALF)), int(round(cy - R_HALF))
    fn = f"{node}.png"
    if os.path.exists(fn):
        try:
            spr = Image.open(fn)
            if spr.mode in ('RGBA', 'LA') or (spr.mode == 'P' and 'transparency' in spr.info):
                canvas.paste(spr, (x, y), spr.convert('RGBA'))
            else:
                canvas.paste(spr, (x, y))
        except Exception as e:
            print(f"Error loading {fn}: {e}")
    else:
        print(f"Missing file: {fn}")

canvas.save(OUTPUT, "PNG")
print(f"Saved {OUTPUT}  ({DIAMETER}x{DIAMETER})  layers={MAX_DEPTH + 1}  step={RADIAL_STEP:.1f}px")
