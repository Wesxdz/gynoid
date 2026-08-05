import os
from PIL import Image, ImageDraw, ImageFont

# Minimal working set: 15 BFO relations, no inverses, no proper variants,
# no type-specialized duplicates. The substrate (Flecs) handles reverse
# queries; the taxonomy handles target-type discrimination.
sections = [
    ("SECTION", "Instantiation"),
    ('particular', 'instance_of', 'universal'),

    ("SECTION", "Time"),
    ('entity', 'exists_at', 'temporal_region'),
    ('occurrent', 'precedes', 'occurrent'),

    ("SECTION", "Parthood"),
    ('entity', 'part_of', 'entity'),
    ('occurrent', 'temporal_part_of', 'occurrent'),

    ("SECTION", "Location"),
    ('independent_continuant', 'located_in', 'independent_continuant'),
    ('occurrent', 'occupies_region', 'region'),

    ("SECTION", "Dependence"),
    ('specifically_dependent_continuant', 'inheres_in', 'independent_continuant'),
    ('dependent_continuant', 'specifically_depends_on', 'independent_continuant'),
    ('generically_dependent_continuant', 'generically_depends_on', 'independent_continuant'),
    ('specifically_dependent_continuant', 'concretizes', 'generically_dependent_continuant'),

    ("SECTION", "Participation & Realization"),
    ('material_entity', 'participates_in', 'process'),
    ('realizable_entity', 'realized_in', 'process'),
    ('process', 'occurs_in', 'material_entity_or_site'),

    ("SECTION", "History"),
    ('material_entity', 'has_history', 'history'),
]

# Layout geometry
ROW_HEIGHT = 42
SECTION_HEIGHT = 34
PADDING = 20
SPRITE_SIZE = 23

COL_WIDTHS = [260, 280, 260]
COL_X = [
    PADDING,
    PADDING + COL_WIDTHS[0],
    PADDING + COL_WIDTHS[0] + COL_WIDTHS[1],
]

num_data_rows = sum(1 for s in sections if s[0] != "SECTION")
num_sections = sum(1 for s in sections if s[0] == "SECTION")

canvas_width = sum(COL_WIDTHS) + (PADDING * 2)
canvas_height = (
    PADDING * 2
    + ROW_HEIGHT
    + num_data_rows * ROW_HEIGHT
    + num_sections * SECTION_HEIGHT
)

canvas = Image.new('RGBA', (canvas_width, canvas_height), color=(0, 0, 0, 255))
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype("JetBrainsMono-Regular.ttf", 16)

def clean_name(text):
    return text.replace('_', ' ').title()

def draw_cell_content(draw_obj, img_canvas, node_name, start_x, current_y):
    img_filename = f"{node_name}.png"
    sprite_y = current_y + (ROW_HEIGHT - SPRITE_SIZE) // 2
    text_y = current_y + (ROW_HEIGHT - 10) // 2

    if os.path.exists(img_filename):
        try:
            sprite = Image.open(img_filename)
            if sprite.mode in ('RGBA', 'LA') or (sprite.mode == 'P' and 'transparency' in sprite.info):
                img_canvas.paste(sprite, (start_x + 10, sprite_y), sprite.convert('RGBA'))
            else:
                img_canvas.paste(sprite, (start_x + 10, sprite_y))
        except Exception as e:
            print(f"Error loading {img_filename}: {e}")
            draw_obj.rectangle(
                [start_x + 10, sprite_y, start_x + 10 + SPRITE_SIZE, sprite_y + SPRITE_SIZE],
                fill=(255, 0, 0, 100),
            )

    draw_obj.text((start_x + 42, text_y), clean_name(node_name), fill=(220, 220, 220, 255), font=font)

headers = ["Source Sprite / Class", "Relationship Type", "Target Sprite / Class"]
header_text_y = PADDING + (ROW_HEIGHT - 10) // 2
for i, text in enumerate(headers):
    x_offset = 10 if i != 1 else 0
    draw.text((COL_X[i] + x_offset, header_text_y), text, fill=(255, 165, 0, 255), font=font)
draw.line(
    [PADDING, PADDING + ROW_HEIGHT, canvas_width - PADDING, PADDING + ROW_HEIGHT],
    fill=(60, 60, 60, 255), width=1,
)

y_cursor = PADDING + ROW_HEIGHT

for entry in sections:
    if entry[0] == "SECTION":
        title = entry[1]
        draw.rectangle(
            [PADDING, y_cursor, canvas_width - PADDING, y_cursor + SECTION_HEIGHT],
            fill=(20, 20, 30, 255),
        )
        text_y = y_cursor + (SECTION_HEIGHT - 10) // 2
        draw.text((PADDING + 10, text_y), title.upper(), fill=(180, 220, 140, 255), font=font)
        draw.line(
            [PADDING, y_cursor + SECTION_HEIGHT, canvas_width - PADDING, y_cursor + SECTION_HEIGHT],
            fill=(60, 60, 60, 255), width=1,
        )
        y_cursor += SECTION_HEIGHT
    else:
        source, rel, target = entry
        draw_cell_content(draw, canvas, source, COL_X[0], y_cursor)

        rel_text_y = y_cursor + (ROW_HEIGHT - 10) // 2
        draw.text((COL_X[1], rel_text_y), f"  --[{rel}]-->", fill=(0, 210, 255, 255), font=font)

        draw_cell_content(draw, canvas, target, COL_X[2], y_cursor)

        draw.line(
            [PADDING, y_cursor + ROW_HEIGHT, canvas_width - PADDING, y_cursor + ROW_HEIGHT],
            fill=(35, 35, 35, 255), width=1,
        )
        y_cursor += ROW_HEIGHT

output_filename = "bfo_relationship_matrix_minimal.png"
canvas.save(output_filename, "PNG")
print(f"Success: Matrix sheet compiled and saved to '{output_filename}' ({canvas_width}x{canvas_height}px)")
print(f"Rows: {num_data_rows} relations across {num_sections} sections")
