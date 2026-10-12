"""Generates the board texture: checker plus rank/file labels.

MuJoCo cannot draw text, so the coordinates are baked into an image.  It is
generated rather than shipped as a PNG so the squares follow
`sim.board_layout` whenever the board dimensions change.

Row 0 is the rank-8 edge, column 0 the a-file edge.
"""

import pathlib

import cv2
import numpy as np

from . import board_layout as L

TEXTURE_PATH = pathlib.Path(__file__).parent / "assets" / "board.png"
RESOLUTION = 1024

# Equal to the flat colours the perception thresholds were calibrated against.
LIGHT_RGB = (0.80, 0.70, 0.48)
DARK_RGB = (0.16, 0.40, 0.26)
FRAME_RGB = (0.33, 0.21, 0.12)
LABEL_RGB = (0.86, 0.83, 0.76)

FONT = cv2.FONT_HERSHEY_DUPLEX


def _bgr(rgb):
    return tuple(int(round(c * 255)) for c in reversed(rgb))


def _centre_text(img, text, cx, cy, scale, color, thickness, rotate=None):
    """Draw text centred on (cx, cy), optionally turned a quarter or half turn.

    Each edge's labels read correctly from outside that edge.  Centring is on
    the cap box rather than the glyph tile: a descender ('g') makes the tile
    taller, which would sit the letter high and push it off the texture edge.
    """
    (w, h), descent = cv2.getTextSize(text, FONT, scale, thickness)
    pad = thickness * 2 + 4
    tile = np.zeros((h + descent + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
    mask = np.zeros(tile.shape[:2], dtype=np.uint8)
    cv2.putText(tile, text, (pad, pad + h), FONT, scale, color, thickness,
                cv2.LINE_AA)
    cv2.putText(mask, text, (pad, pad + h), FONT, scale, 255, thickness,
                cv2.LINE_AA)

    anchor = (pad + w / 2.0, pad + h / 2.0)          # centre of the cap box
    th, tw = mask.shape
    if rotate == cv2.ROTATE_180:
        anchor = (tw - anchor[0], th - anchor[1])
    elif rotate == cv2.ROTATE_90_CLOCKWISE:
        anchor = (th - anchor[1], anchor[0])
    elif rotate == cv2.ROTATE_90_COUNTERCLOCKWISE:
        anchor = (anchor[1], tw - anchor[0])
    if rotate is not None:
        tile, mask = cv2.rotate(tile, rotate), cv2.rotate(mask, rotate)

    th, tw = mask.shape
    y0, x0 = int(round(cy - anchor[1])), int(round(cx - anchor[0]))
    # Clip rather than discard: a glyph near the edge loses a pixel instead
    # of vanishing.
    sy0, sx0 = max(0, -y0), max(0, -x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1 = min(img.shape[0], y0 + th)
    dx1 = min(img.shape[1], x0 + tw)
    if dy1 <= dy0 or dx1 <= dx0:
        return
    sub_mask = mask[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    sub_tile = tile[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    region = img[dy0:dy1, dx0:dx1]
    sel = sub_mask > 0
    region[sel] = sub_tile[sel]


def build(resolution=RESOLUTION, labels=True):
    """Return the board texture as a BGR image, camera-side up."""
    n = resolution
    px_per_m = n / (2 * L.PLATE_HALF)
    img = np.full((n, n, 3), _bgr(FRAME_RGB), dtype=np.uint8)

    def to_px(x, y):
        """Board metres -> texture pixels (col, row)."""
        return (L.PLATE_HALF + x) * px_per_m, (L.PLATE_HALF - y) * px_per_m

    for fi, file in enumerate(L.FILES):
        for ri, rank in enumerate(L.RANKS):
            cx, cy = L.square_to_xy(file + rank)
            x0, y1 = to_px(cx - L.SQUARE / 2, cy - L.SQUARE / 2)
            x1, y0 = to_px(cx + L.SQUARE / 2, cy + L.SQUARE / 2)
            # a1 is a dark square, and so is every square of the same parity.
            colour = DARK_RGB if (fi + ri) % 2 == 0 else LIGHT_RGB
            cv2.rectangle(img, (int(round(x0)), int(round(y0))),
                          (int(round(x1)), int(round(y1))), _bgr(colour), -1)

    if labels:
        border_px = L.BORDER * px_per_m
        scale = border_px / 46.0
        thickness = max(1, int(round(border_px / 22)))
        colour = _bgr(LABEL_RGB)
        near = L.BOARD_HALF + L.BORDER / 2
        for file in L.FILES:
            cx, _ = L.square_to_xy(file + "1")
            _centre_text(img, file, *to_px(cx, -near), scale, colour, thickness)
            _centre_text(img, file, *to_px(cx, near), scale, colour, thickness,
                         cv2.ROTATE_180)
        for rank in L.RANKS:
            _, cy = L.square_to_xy("a" + rank)
            _centre_text(img, rank, *to_px(-near, cy), scale, colour, thickness,
                         cv2.ROTATE_90_CLOCKWISE)
            _centre_text(img, rank, *to_px(near, cy), scale, colour, thickness,
                         cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def write(path=TEXTURE_PATH, **kwargs):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), build(**kwargs))
    return path


if __name__ == "__main__":
    print(write())
