"""Pictures of ARC grids for Qwen3.8's vision tower (the vision plan, docs/reports/1022_0010_vision_harness_plan/).

Standalone (numpy + Pillow only), so a notebook cell can write it as `arc_vision.py` beside `arc_skills.py`.

The model sees an image as a grid of visual tokens of 32 x 32 pixels (16-pixel patches merged 2 x 2). A cell drawn at a
multiple of 32 pixels therefore covers whole tokens, at its own row and column of the model's 2-D positions: 32 px is
one token per cell (the 10-17 probe), 64 px four, 96 px nine. 16 px puts four cells in one token.

What a picture can carry besides the colours (all optional, per `Style`):
- `lines`: a thin dark line on each cell's top and left edge (10-17: without lines, same-colour neighbours merge);
- `major`: a heavier line every N cells, for counting;
- `rulers`: row and column indices in a one-cell margin on the top and left, so a position can be read off the picture
  and matched to the digits (a crop keeps the indices of the full grid);
- `digits`: the cell's digit drawn inside it;
- `marks`: cells outlined in a colour (e.g. the cells where two grids differ) and boxes (e.g. objects).

`tokens()` gives the number of visual tokens the image processor will make (Qwen2-VL's smart resize: sides rounded to
multiples of 32, the area kept within 65,536 to 16,777,216 pixels); `render()` raises the cell size itself when a
picture would be small enough to be enlarged, so cells stay on token boundaries.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ARC's usual colours for 0-9, in the order the prompt's colour line names them
PALETTE = np.array([(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170),
                    (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37)], np.uint8)
COLOURS = '0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, 7 orange, 8 light blue, 9 maroon'
DARK = {0, 1, 2, 6, 9}         # colours a digit is written on in white; the others get black
TOKEN_PX = 32                  # one visual token covers 32 x 32 pixels
MIN_PIXELS, MAX_PIXELS = 65536, 16777216   # Qwen3.8's preprocessor_config.json: shortest_edge / longest_edge (area)
WHITE, LINE, MAJOR, RULER_BG, RULER_FG = (255, 255, 255), (64, 64, 64), (24, 24, 24), (245, 245, 245), (40, 40, 40)


@dataclass(frozen=True)
class Style:
    cell: int = 32         # pixels per cell; a multiple of 32 keeps every cell on whole visual tokens
    lines: bool = True     # thin cell lines
    major: int = 0         # a heavier line every `major` cells (0: none)
    rulers: bool = False   # row / column indices in a one-cell margin on the top and left
    digits: bool = False   # each cell's digit drawn in it
    gap: int = 1           # blank cells between grids shown side by side


STYLES = {   # the perception bench's renderings (plan section 3)
    'c32': Style(32),
    'c64': Style(64),
    'c96': Style(96),
    'c32r': Style(32, major=5, rulers=True),
    'c64r': Style(64, major=5, rulers=True),
    'c64d': Style(64, digits=True),
    'c64rd': Style(64, major=5, rulers=True, digits=True),
}


def smart_resize(h: int, w: int, factor: int = TOKEN_PX, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """The size the image processor resizes an h x w image to (Qwen2-VL's smart_resize)."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def tokens(img: np.ndarray) -> int:
    """Visual tokens the processor makes from an RGB array."""
    hb, wb = smart_resize(img.shape[0], img.shape[1])
    return (hb // TOKEN_PX) * (wb // TOKEN_PX)


def _font(px: int):
    try:
        return ImageFont.load_default(size=max(8, px))
    except TypeError:   # Pillow < 10.1 has no scalable default font
        return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], s: str, size: int, fill) -> None:
    """`s` centred on `xy`."""
    font = _font(size)
    x0, y0, x1, y1 = draw.textbbox((0, 0), s, font=font)
    draw.text((xy[0] - (x1 - x0) / 2 - x0, xy[1] - (y1 - y0) / 2 - y0), s, font=font, fill=fill)


def _fit_cell(style: Style, cells_h: int, cells_w: int) -> Style:
    """Keep the picture inside the processor's area limits without resizing: raise the cell size (in steps of 32 px)
    while the picture would be enlarged, lower it while it would be shrunk. Either resize would move cells off token
    boundaries."""
    cell = style.cell
    while cells_h * cells_w * cell * cell < MIN_PIXELS:
        cell += TOKEN_PX if cell >= TOKEN_PX else cell
    while cell > TOKEN_PX and cells_h * cells_w * cell * cell > MAX_PIXELS:
        cell -= TOKEN_PX
    return replace(style, cell=cell) if cell != style.cell else style


def render(*grids, style: Style = Style(), origin: list[tuple[int, int]] | None = None,
           marks: list[dict] | None = None) -> np.ndarray:
    """Grids side by side (top-aligned, `style.gap` blank cells apart) as an RGB uint8 array whose sides are multiples
    of 32 pixels (padded white), so every cell sits on whole visual tokens.

    origin: per grid, the (row, column) its top-left cell has in a larger grid (a crop), for the rulers.
    marks: per grid, {'cells': {(r, c), ...}, 'colour': (r, g, b), 'boxes': [(r0, c0, r1, c1, (r, g, b)), ...]};
    cells are outlined, boxes drawn around inclusive cell ranges.
    """
    gs = [np.clip(np.asarray(g, dtype=np.int64), 0, 9) for g in grids]
    if not gs or any(g.ndim != 2 or g.size == 0 for g in gs):
        raise ValueError('render needs non-empty 2-D grids')
    margin = 1 if style.rulers else 0
    rows = max(g.shape[0] for g in gs) + margin
    cols = sum(g.shape[1] for g in gs) + style.gap * (len(gs) - 1) + margin
    style = _fit_cell(style, rows, cols)
    cell = style.cell
    H, W = rows * cell, cols * cell
    canvas = np.full((math.ceil(H / TOKEN_PX) * TOKEN_PX, math.ceil(W / TOKEN_PX) * TOKEN_PX, 3), WHITE, np.uint8)
    thin, heavy = max(1, cell // 32), max(2, cell // 16)
    placed = []   # (grid, y0, x0) in pixels
    x = margin
    for g in gs:
        y0, x0 = margin * cell, x * cell
        h, w = g.shape
        px = PALETTE[g].repeat(cell, 0).repeat(cell, 1)
        if style.lines:
            for k in range(thin):
                px[k::cell, :] = LINE
                px[:, k::cell] = LINE
        canvas[y0:y0 + h * cell, x0:x0 + w * cell] = px
        placed.append((g, y0, x0))
        x += w + style.gap
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for i, (g, y0, x0) in enumerate(placed):
        h, w = g.shape
        r_off, c_off = origin[i] if origin else (0, 0)
        if style.major:
            for r in range(0, h + 1):
                if (r + r_off) % style.major == 0:
                    draw.rectangle([x0, y0 + r * cell - heavy // 2, x0 + w * cell - 1, y0 + r * cell + (heavy - 1) // 2],
                                   fill=MAJOR)
            for c in range(0, w + 1):
                if (c + c_off) % style.major == 0:
                    draw.rectangle([x0 + c * cell - heavy // 2, y0, x0 + c * cell + (heavy - 1) // 2, y0 + h * cell - 1],
                                   fill=MAJOR)
        if style.digits:
            for r in range(h):
                for c in range(w):
                    v = int(g[r, c])
                    _text(draw, (x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell), str(v), int(cell * 0.55),
                          (255, 255, 255) if v in DARK else (0, 0, 0))
        if style.rulers:
            draw.rectangle([x0, 0, x0 + w * cell - 1, cell - 1], fill=RULER_BG)
            for c in range(w):
                _text(draw, (x0 + (c + 0.5) * cell, cell / 2), str(c + c_off), int(cell * 0.42), RULER_FG)
            if i == 0:
                draw.rectangle([0, y0, cell - 1, y0 + h * cell - 1], fill=RULER_BG)
                for r in range(h):
                    _text(draw, (cell / 2, y0 + (r + 0.5) * cell), str(r + r_off), int(cell * 0.42), RULER_FG)
        m = (marks or [None] * len(placed))[i] if marks else None
        if m:
            colour, wide = tuple(m.get('colour', (255, 0, 255))), max(2, cell // 10)
            for r, c in sorted(m.get('cells', ())):
                draw.rectangle([x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1],
                               outline=colour, width=wide)
            for r0, c0, r1, c1, col in m.get('boxes', ()):
                draw.rectangle([x0 + c0 * cell, y0 + r0 * cell, x0 + (c1 + 1) * cell - 1, y0 + (r1 + 1) * cell - 1],
                               outline=tuple(col), width=wide)
    return np.asarray(img)


def diff_cells(a, b) -> set[tuple[int, int]]:
    """Cells where two same-shape grids differ (empty set when the shapes differ: show both instead)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return set()
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(a != b))}


def crop(grid, r0: int, c0: int, r1: int, c1: int):
    """Rows r0..r1 and columns c0..c1 (inclusive, clipped) and their origin, for render(..., origin=[...])."""
    g = np.asarray(grid)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(g.shape[0] - 1, r1), min(g.shape[1] - 1, c1)
    if r0 > r1 or c0 > c1:
        raise ValueError('empty crop')
    return g[r0:r1 + 1, c0:c1 + 1].tolist(), (r0, c0)


def data_uri(img: np.ndarray) -> str:
    """The array as a PNG data URI (what SGLang's OpenAI-style image_url and /generate image_data accept)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def picture(*grids, style: Style = Style(), **kw) -> tuple[str, int]:
    """render() as (PNG data URI, visual tokens)."""
    img = render(*grids, style=style, **kw)
    return data_uri(img), tokens(img)
