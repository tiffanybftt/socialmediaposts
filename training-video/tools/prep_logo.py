"""Cut the AI Summit Group logo out of its flat off-white background.

Writes transparent PNG layers into assets/ so the video can animate the mark
and the wordmark separately and sit them on any background colour.
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "ai-summit-group-original.jpg"
BG = np.array([250.0, 249.0, 246.0])  # sampled from the original image
SPLIT_X = 468  # mark (and its left rule) | wordmark (and its rules)
MARK_BOX = (140, 236, 496, 472)  # tight crop of the mountain mark
SCALE = 1.5  # 1280px source -> 1920px stage


def color_to_alpha(rgb: np.ndarray) -> np.ndarray:
    alpha = np.clip((BG - rgb) / BG, 0, 1).max(axis=2)
    alpha[alpha < 0.035] = 0  # drop JPEG noise around the artwork
    safe = np.maximum(alpha, 1e-3)[..., None]
    color = np.clip(BG + (rgb - BG) / safe, 0, 255)
    out = np.dstack([color, alpha * 255]).astype(np.uint8)
    out[alpha == 0] = 0
    return out


def save(arr: np.ndarray, name: str, scale: float = 1.0) -> None:
    im = Image.fromarray(arr, "RGBA")
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    im.save(ROOT / "assets" / name)
    print("wrote", name, im.size)


rgb = np.asarray(Image.open(SRC).convert("RGB")).astype(float)
rgba = color_to_alpha(rgb)

mark = rgba.copy()
mark[:, SPLIT_X:] = 0
word = rgba.copy()
word[:, :SPLIT_X] = 0
save(mark, "logo-mark-layer.png", SCALE)
save(word, "logo-wordmark-layer.png", SCALE)

x0, y0, x1, y1 = MARK_BOX
tight = rgba.copy()
tight[:416, 470:] = 0  # sliver of the "A" in the wordmark
tight[416:440, 455:] = 0  # start of the rule beside "Powered by"
tight[400:416, :205] = 0  # left rule leading into the mark
save(tight[y0:y1, x0:x1], "logo-mark.png", 2.0)

# sanity check: composite back on the background and compare with the source
a = rgba[..., 3:4] / 255.0
back = rgba[..., :3] * a + BG * (1 - a)
print("max abs diff vs original:", int(np.abs(back - rgb).max()), "mean:", round(float(np.abs(back - rgb).mean()), 3))
