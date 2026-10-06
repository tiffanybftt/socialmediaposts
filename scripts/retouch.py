"""Natural-looking portrait retouching on a full-resolution photo.

Everything is mask-based and blended at moderate strength: skin tone is evened (texture and
freckles stay), teeth lose some yellow, eyes gain a little clarity, hair gets a light polish.
Coordinates in a spec are pixels of the ORIGINAL photo.

    spec = {
        "faces": [{
            "face":  (cx, cy, rx, ry),        # ellipse around the skin to even out
            "avoid": [(cx, cy, rx, ry), ...], # eyes/brows/lips: left untouched by skin smoothing
            "teeth": (cx, cy, rx, ry),        # search area for teeth (whitening is color-gated)
            "tooth_L": (lo, hi),              # optional: lightness gate (shadowed teeth need lower)
            "teeth_lift": 7.0,                # optional: lightness added to teeth
            "teeth_yellow": 0.4,              # optional: fraction of yellow removed
            "eyes":  [(cx, cy), ...],         # eye centers
            "lum_limit": y or None,           # smooth skin luminance only above this y (spares beards)
        }],
        "hair": [[(x, y), ...], ...],         # polygons
    }
"""
import cv2
import numpy as np
from PIL import Image

# Strengths: 0 = off, 1 = full. Kept moderate so the result stays natural.
DENOISE = (3, 5)         # luminance, color strength for denoise; NOT applied to faces/hair (it erases skin texture)
SKIN_TONE = 0.60         # evens redness/blotchy color (large-scale only)
SKIN_SMOOTH = 0.45       # softens blotchy shine/blemishes (mid-scale only); pores & freckles stay
TEETH_YELLOW = 0.38      # fraction of yellow removed from teeth (keep it believable)
TEETH_LIFT = 3.5         # lightness added to teeth (L units, 0-100); shading is preserved
EYE_CLARITY = 0.35
HAIR_SMOOTH = 0.35
HAIR_SHINE = 0.35


def _smooth(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _ellipse(shape, spec, feather):
    cx, cy, rx, ry = spec
    m = np.zeros(shape, np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), feather)


def _poly(shape, pts, feather):
    m = np.zeros(shape, np.float32)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0)
    return cv2.GaussianBlur(m, (0, 0), feather)


def _masked_blur(ch, w, sigma):
    """Blur ch using only pixels where w is high, so background colors do not bleed in."""
    return cv2.GaussianBlur(ch * w, (0, 0), sigma) / (cv2.GaussianBlur(w, (0, 0), sigma) + 1e-6)


def _skin_color(bgr8):
    ycc = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr, cb = ycc[..., 1], ycc[..., 2]
    m = _smooth(cr, 132, 140) * (1 - _smooth(cr, 172, 182)) * _smooth(cb, 70, 80) * (1 - _smooth(cb, 125, 135))
    return cv2.GaussianBlur(m, (0, 0), 3)


def retouch(img, spec, debug=None):
    """img: PIL RGB at full resolution. Returns a retouched PIL RGB image."""
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if DENOISE[0] or DENOISE[1]:
        # clean shirts, shadows and backgrounds; keep faces and hair as shot so skin keeps its texture
        keep = np.zeros(bgr.shape[:2], np.float32)
        for f in spec.get("faces", []):
            keep = np.maximum(keep, _ellipse(keep.shape, f["face"], 14))
        for pts in spec.get("hair", []):
            keep = np.maximum(keep, _poly(keep.shape, pts, 10))
        clean = cv2.fastNlMeansDenoisingColored(bgr, None, DENOISE[0], DENOISE[1], 7, 21)
        bgr = (bgr * keep[..., None] + clean * (1 - keep[..., None])).astype(np.uint8)
    skin_color = _skin_color(bgr)
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    L, a, b = (lab[..., i].copy() for i in range(3))
    shape = L.shape
    dbg = {}

    for f in spec.get("faces", []):
        skin = _ellipse(shape, f["face"], 14) * skin_color
        for av in f.get("avoid", []):
            skin *= 1 - _ellipse(shape, av, 8)
        # Frequency separation: only the slow, large-scale layer is corrected. Fine detail (pores,
        # freckles, stubble) is the "high band" and is never touched, so the skin stays real.
        for ch in (a, b):
            low = cv2.GaussianBlur(ch, (0, 0), 3)
            even = _masked_blur(ch, skin + 1e-3, 25)
            ch += SKIN_TONE * skin * (even - low)
        lum = skin.copy()
        if f.get("lum_limit"):
            ys = np.arange(shape[0], dtype=np.float32)[:, None]
            lum *= 1 - _smooth(ys, f["lum_limit"] - 15, f["lum_limit"] + 15)
        low = cv2.GaussianBlur(L, (0, 0), 3)
        L += SKIN_SMOOTH * lum * (cv2.bilateralFilter(low, 9, 6, 6) - low)

        # teeth: gated on color (yellow-leaning, unlike red lips/gums) and lightness (not the dark
        # gaps), so only actual teeth change
        lo, hi = f.get("tooth_L", (40, 52))
        zone = _ellipse(shape, f["teeth"], 3)
        tooth = zone * _smooth(b - a, -3, 3) * _smooth(L, lo, hi)
        tooth = cv2.GaussianBlur(tooth, (0, 0), 1.0)
        b -= f.get("teeth_yellow", TEETH_YELLOW) * tooth * np.clip(b - 5, 0, None)
        a -= 0.25 * tooth * np.clip(a - 3, 0, None)   # less brown/orange cast
        L += f.get("teeth_lift", TEETH_LIFT) * tooth
        L = np.minimum(L, np.maximum(L - 0.0, 0) * (1 - tooth) + 82 * tooth)  # never blow teeth out to flat white
        dbg.setdefault("teeth", []).append(tooth)
        dbg.setdefault("skin", []).append(skin)

        # eyes: a touch of local contrast and brighter whites
        for (ex, ey) in f.get("eyes", []):
            em = _ellipse(shape, (ex, ey, 42, 15), 5)
            L += EYE_CLARITY * em * (L - cv2.GaussianBlur(L, (0, 0), 2.5))
            L += 3.0 * em * _smooth(L, 58, 80)

    for pts in spec.get("hair", []):
        hm = _poly(shape, pts, 10)
        L += HAIR_SMOOTH * hm * (cv2.bilateralFilter(L, 7, 6, 5) - L)
        L += HAIR_SHINE * hm * (L - cv2.GaussianBlur(L, (0, 0), 10))
        dbg.setdefault("hair", []).append(hm)

    lab = np.stack([np.clip(L, 0, 100), a, b], -1)
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)
    if debug is not None:
        debug.update(dbg)
    return Image.fromarray(cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_BGR2RGB))
