#!/usr/bin/env python3
"""Tiny drawing program for MNIST-style punctuation sprites.

Iterates through every punctuation mark the interface needs, prompting for each
in turn. Draw with the mouse on the black canvas against the baseline and
x-height guides; the canvas downsamples 10:1 to a 28x28 white-on-black sprite,
preserving the mark's size and position exactly -- for punctuation those ARE
the identity (a period is small and low, an apostrophe small and high), which
is why the classic MNIST crop-and-recentre is deliberately not applied.

Keys:
  S / Enter   save current drawing, advance to the next mark
  Space / N   skip this mark (keeps any existing sprite)
  C           clear the canvas
  B / V       brush bigger / smaller
  Q / Escape  quit

Usage:
  python3 draw_punctuation.py            # fills assets/punctuation/set_01
  python3 draw_punctuation.py --set 2    # a second style set
  python3 draw_punctuation.py --redo     # revisit marks already drawn
"""

import argparse
import sys
from pathlib import Path

import tkinter as tk
from PIL import Image, ImageDraw

# Filenames are names, not characters -- '?' can't be a filename. This map is
# the contract with the C++ side: create_letter_chip resolves a punctuation
# char to punctuation/set_NN/<name>.png with the same table.
PUNCTUATION = [
    ("'",  "apostrophe"),
    ("-",  "hyphen"),
    (".",  "period"),
    (",",  "comma"),
    ("?",  "question"),
    ("!",  "exclaim"),
    (":",  "colon"),
    (";",  "semicolon"),
    ('"',  "quote"),
    ("(",  "lparen"),
    (")",  "rparen"),
    ("&",  "ampersand"),
    ("/",  "slash"),
    ("@",  "at"),
    ("#",  "hash"),
    ("%",  "percent"),
    ("+",  "plus"),
    ("=",  "equals"),
    ("_",  "underscore"),
    ("~",  "tilde"),
]

CANVAS = 280          # drawing surface, 10x the output for smooth strokes
OUTPUT = 28           # MNIST field
ZOOM = 5              # reference / preview magnification (28 -> 140)

# 20px on the 280 canvas downsamples to a ~2px stroke at 28 -- the weight the
# existing MNIST/letter sprites carry. The LANCZOS downsample then supplies the
# same soft anti-aliased edge those sprites got from their own downsampling.
DEFAULT_BRUSH = 20

# Existing sprites shown beside the live preview, so weight, alias and
# placement are matched by eye against the family the new marks must join.
REFERENCES = [
    "assets/mnist/set_0/2.png",
    "assets/mnist/set_0/0.png",
    "assets/mnist/set_0/5.png",
    "assets/letter_sets/set_01/A.png",
    "assets/letter_sets/set_01/s.png",
]


def pil_to_ppm(image, zoom):
    """PIL grayscale -> PPM bytes tk.PhotoImage accepts, pixel-doubled with
    NEAREST so the 28x28 alias structure stays visible instead of re-smoothed."""
    big = image.convert("RGB").resize(
        (image.width * zoom, image.height * zoom), Image.NEAREST)
    header = f"P6 {big.width} {big.height} 255 ".encode()
    return header + big.tobytes()


def spriteify(image):
    """Uniform downscale of the whole canvas, nothing else.

    Classic MNIST preprocessing (scale to a 20px box, recentre on the centre
    of mass) is deliberately NOT used here: it is right for digits and wrong
    for punctuation, whose identity lives in size and position. A period is
    small and low, an apostrophe small and high, a hyphen thin and central --
    recentring and rescaling would collapse them into the same blob. What you
    draw in the big canvas, placed against the guide lines, is exactly what
    the sprite becomes.
    """
    if image.getbbox() is None:
        return None                      # nothing drawn
    return image.resize((OUTPUT, OUTPUT), Image.LANCZOS)


class Drawer:
    def __init__(self, out_dir, marks):
        self.out_dir = out_dir
        self.marks = marks
        self.index = 0
        self.brush = DEFAULT_BRUSH

        self.root = tk.Tk()

        # Reference strip: real sprites from the family, then the live preview
        # of the current drawing at the same zoom -- similarity is judged
        # side by side, not remembered.
        strip = tk.Frame(self.root, bg="#111111")
        strip.pack(fill="x")
        self._photos = []          # tk keeps no reference; we must
        repo = Path(__file__).resolve().parent.parent
        for rel in REFERENCES:
            path = repo / rel
            if not path.exists():
                continue
            photo = tk.PhotoImage(data=pil_to_ppm(Image.open(path).convert("L"), ZOOM))
            self._photos.append(photo)
            tk.Label(strip, image=photo, bg="#111111").pack(side="left", padx=2, pady=4)

        self.preview = tk.Label(strip, bg="#111111",
                                highlightthickness=2,
                                highlightbackground="#3f6d8a")
        self.preview.pack(side="left", padx=10, pady=4)

        self.canvas = tk.Canvas(self.root, width=CANVAS, height=CANVAS,
                                bg="black", highlightthickness=0)
        self.canvas.pack()
        self.status = tk.Label(self.root, font=("monospace", 11), anchor="w")
        self.status.pack(fill="x")

        # The PIL image is the ground truth; the tk canvas only mirrors it.
        self.image = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last = None

        self.canvas.bind("<Button-1>", self.pen_down)
        self.canvas.bind("<B1-Motion>", self.pen_move)
        self.canvas.bind("<ButtonRelease-1>", self.pen_up)
        for key in ("s", "S", "<Return>"):
            self.root.bind(key, self.save_and_next)
        for key in ("n", "N", "<space>"):
            self.root.bind(key, self.skip)
        self.root.bind("c", self.clear)
        self.root.bind("C", self.clear)
        self.root.bind("b", lambda e: self.set_brush(+2))
        self.root.bind("B", lambda e: self.set_brush(+2))
        self.root.bind("v", lambda e: self.set_brush(-2))
        self.root.bind("V", lambda e: self.set_brush(-2))
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("Q", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.draw_guides()
        self.update_preview()
        self.refresh()

    def current(self):
        return self.marks[self.index]

    def refresh(self):
        char, name = self.current()
        done = self.index
        self.root.title(f"draw:  {char}   ({name})")
        self.status.config(text=(
            f"[{done + 1}/{len(self.marks)}]  draw   {char}   ({name})   "
            f"brush {self.brush}  |  S save   N skip   C clear   B/V brush   Q quit"))

    # -- pen ---------------------------------------------------------------
    def pen_down(self, event):
        self.last = (event.x, event.y)
        self.dot(event.x, event.y)

    def pen_move(self, event):
        if self.last is None:
            self.last = (event.x, event.y)
        x0, y0 = self.last
        self.canvas.create_line(x0, y0, event.x, event.y,
                                fill="white", width=self.brush,
                                capstyle="round", joinstyle="round")
        self.draw.line([x0, y0, event.x, event.y],
                       fill=255, width=self.brush)
        self.dot(event.x, event.y)
        self.last = (event.x, event.y)

    def pen_up(self, event):
        self.last = None
        self.update_preview()

    def update_preview(self):
        sprite = spriteify(self.image) or Image.new("L", (OUTPUT, OUTPUT), 0)
        self._preview_photo = tk.PhotoImage(data=pil_to_ppm(sprite, ZOOM))
        self.preview.config(image=self._preview_photo)

    def dot(self, x, y):
        r = self.brush / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    # -- actions -----------------------------------------------------------
    def set_brush(self, delta):
        self.brush = max(12, min(32, self.brush + delta))
        self.refresh()

    def clear(self, _event=None):
        self.canvas.delete("all")
        self.image.paste(0, (0, 0, CANVAS, CANVAS))
        self.last = None
        self.draw_guides()
        self.update_preview()

    def draw_guides(self):
        """Baseline and x-height lines, display only -- placement is identity
        for punctuation, so the guides keep it consistent across marks."""
        baseline = CANVAS * 0.75
        x_height = CANVAS * 0.40
        for y, label in ((x_height, "x-height"), (baseline, "baseline")):
            self.canvas.create_line(0, y, CANVAS, y, fill="#333333")
            self.canvas.create_text(6, y - 8, text=label, fill="#555555",
                                    anchor="w", font=("monospace", 8))

    def save_and_next(self, _event=None):
        sprite = spriteify(self.image)
        if sprite is None:
            self.status.config(text="nothing drawn -- S saves, N skips")
            return
        char, name = self.current()
        path = self.out_dir / f"{name}.png"
        sprite.save(path)
        print(f"saved {char}  ->  {path}")
        self.advance()

    def skip(self, _event=None):
        self.advance()

    def advance(self):
        self.clear()
        self.index += 1
        if self.index >= len(self.marks):
            print("all marks done")
            self.root.destroy()
            return
        self.refresh()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=int, default=1, help="style set number")
    parser.add_argument("--redo", action="store_true",
                        help="also prompt marks that already have sprites")
    args = parser.parse_args()

    out_dir = (Path(__file__).resolve().parent.parent
               / "assets" / "punctuation" / f"set_{args.set:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    marks = PUNCTUATION if args.redo else [
        (c, n) for c, n in PUNCTUATION if not (out_dir / f"{n}.png").exists()]
    if not marks:
        print(f"{out_dir} already has every mark -- use --redo to redraw")
        sys.exit(0)

    print(f"drawing {len(marks)} marks into {out_dir}")
    Drawer(out_dir, marks).run()


if __name__ == "__main__":
    main()
