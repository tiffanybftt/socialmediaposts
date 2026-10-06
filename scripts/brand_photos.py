#!/usr/bin/env python3
"""Brand photos as 1080x1080 social posts: photo on top, branded band below.

Usage:  python3 scripts/brand_photos.py            # renders every slide in SLIDES
Edit BRAND and SLIDES below; everything else is layout code.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retouch import retouch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# --- Brand settings: black, white, industrial grey, orange. The exact hex values and fonts are
# PLACEHOLDERS until the official brand guide is dropped in. ---
BRAND = {
    "name": "BUILT FOR THE TRADES",
    "logo": ROOT / "brand" / "logo-original.webp",  # black art on transparent background
    "band": (14, 14, 14),         # band background (black)
    "text": (255, 255, 255),      # headline + logo (white)
    "muted": (168, 170, 172),     # subline (industrial grey)
    "accent": (242, 105, 33),     # orange: label text + rule above the band
    "font_head": "/usr/share/fonts/opentype/inter/Inter-Black.otf",
    "font_body": "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
    "font_label": "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
}

SIZE = 1080       # square post
# Photo polish (applied at full resolution, before downscaling)
MIDTONES = 0.86          # gamma < 1 brightens midtones (faces); 1 = off
SHADOW_LIFT = 0.20       # brightens shadows (dark shirts, tables); 0 = off
HIGHLIGHT_EASE = 0.10    # pulls blown highlights (windows) back; 0 = off
WARMTH = (1.04, 1.00, 0.94)   # per-channel gain (R, G, B): subtle warm cast
SATURATION = 1.08
CLARITY = 22             # local-contrast percent (large-radius unsharp); 0 = off
SHARPEN = 70             # final detail sharpening percent, applied after downscale
JPEG_QUALITY = 95
RULE_H = 6               # orange rule between photo and band
BAND_H = 190             # height of the branded footer band
HEADER_H = 98            # height of the header bar (people); 0 for no header
PAD = 48                 # side padding

# Retouch regions for the selfie, in pixels of the ORIGINAL photo (see scripts/retouch.py).
# Faces: Jacob first, then Julia.
SELFIE_RETOUCH = {
    "faces": [
        {   # Jacob
            "face": (765, 795, 122, 100),
            "avoid": [(730, 732, 78, 34), (845, 742, 72, 34), (770, 882, 80, 26)],
            "teeth": (770, 883, 46, 11),
            "tooth_L": (24, 34),
            "teeth_lift": 5.0,
            "eyes": [(722, 737), (840, 748)],
            "lum_limit": 870,
        },
        {   # Julia
            "face": (1180, 1100, 172, 238),
            "avoid": [(1065, 1030, 82, 44), (1240, 1030, 82, 44), (1142, 1213, 95, 34)],
            "teeth": (1142, 1214, 58, 18),
            "tooth_L": (34, 48),
            "teeth_yellow": 0.55,
            "teeth_lift": 4.0,
            "eyes": [(1065, 1048), (1240, 1048)],
            "lum_limit": None,
        },
    ],
    "hair": [
        [(905, 1500), (915, 1000), (960, 880), (1010, 862), (1008, 1000), (1000, 1150), (985, 1300), (980, 1500)],
        [(1350, 870), (1400, 900), (1450, 1000), (1500, 1100), (1500, 1500), (1380, 1450), (1350, 1300), (1355, 1100)],
        [(1010, 870), (1030, 830), (1100, 805), (1250, 800), (1330, 830), (1360, 880), (1300, 885), (1180, 865), (1070, 880)],
    ],
}

# focus = (x, y) in the ORIGINAL photo that the crop should be centered on.
SLIDES = [
    {
        "photo": ROOT / "photos" / "lunch-selfie-01.jpg",
        "focus": (875, 932),     # centers Jacob and Julia
        "zoom": 1.2,             # crop in 20%: trims the empty keg/window space on the left
        "header": [              # (name, role, side): who is in the photo
            ("JACOB STULTZ", "OWNER, STULTZ PLUMBING", "left"),
            ("JULIA ADAMS", "OPERATIONS COACH", "right"),
        ],
        "retouch": SELFIE_RETOUCH,
        "headline": "LEADERSHIP LUNCH",
        "subline": "Good food. Real conversations. Better businesses.",
        "out": ROOT / "output" / "leadership-lunch.png",
    },
]


def crop_box(img, w, h, focus, zoom=1.0):
    """Largest w:h box that fits in img (shrunk by zoom), centered on focus, clamped to the edges."""
    iw, ih = img.size
    target = w / h
    cw, ch = (iw, int(iw / target)) if iw / ih < target else (int(ih * target), ih)
    cw, ch = int(cw / zoom), int(ch / zoom)
    left = min(max(int(focus[0] - cw / 2), 0), iw - cw)
    top = min(max(int(focus[1] - ch / 2), 0), ih - ch)
    return (left, top, left + cw, top + ch)


def polish(img):
    """Fix backlit/dim phone photos: levels, shadow lift, highlight ease, warmth, clarity."""
    img = ImageOps.autocontrast(img, cutoff=0.3)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = x ** MIDTONES
    x = x + SHADOW_LIFT * (1 - x) ** 2          # strongest in the darks, none at white
    x = x - HIGHLIGHT_EASE * x ** 4             # only touches the brightest areas
    x = x * np.array(WARMTH, dtype=np.float32)
    img = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(SATURATION)
    if CLARITY:
        img = img.filter(ImageFilter.UnsharpMask(radius=img.width / 40, percent=CLARITY, threshold=0))
    return img


def fit_photo(img, w, h, focus, zoom=1.0):
    """Crop, polish at full resolution, downscale, then sharpen for the smaller size."""
    img = polish(img.crop(crop_box(img, w, h, focus, zoom))).resize((w, h), Image.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=SHARPEN, threshold=2))


def tinted_logo(path, height, color):
    """Recolor the logo's alpha mask to a single color, trimmed and scaled to height."""
    alpha = Image.open(path).convert("RGBA").split()[3]
    alpha = alpha.crop(alpha.getbbox())
    w = round(alpha.width * height / alpha.height)
    alpha = alpha.resize((w, height), Image.LANCZOS)
    logo = Image.new("RGBA", alpha.size, color + (255,))
    logo.putalpha(alpha)
    return logo


def draw_tracked(draw, xy, text, font, fill, tracking):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def fit_font(path, text, max_w, start, draw):
    size = start
    while size > 20:
        font = ImageFont.truetype(path, size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 2
    return ImageFont.truetype(path, 20)


def draw_header(canvas, people, brand):
    """Header bar across the top: each person's name over their role, ending in the accent rule.

    people: [(name, role, side), ...] where side is "left" or "right" (match where they stand).
    """
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, SIZE, HEADER_H), fill=brand["band"])
    draw.rectangle((0, HEADER_H - RULE_H, SIZE, HEADER_H - 1), fill=brand["accent"])
    f_name = ImageFont.truetype(brand["font_head"], 30)
    f_role = ImageFont.truetype(brand["font_label"], 17)
    track = 2.6
    width = lambda t, f, tr: sum(draw.textlength(c, font=f) + tr for c in t) - tr
    gap = 8
    top = (HEADER_H - RULE_H - (f_name.size + gap + f_role.size)) // 2 - 2
    for name, role, side in people:
        w = max(width(name, f_name, 1.5), width(role, f_role, track))
        x = PAD if side == "left" else SIZE - PAD - w
        draw_tracked(draw, (x, top), name, f_name, brand["text"], 1.5)
        draw_tracked(draw, (x, top + f_name.size + gap), role, f_role, brand["accent"], track)


def render(slide):
    b = BRAND
    canvas = Image.new("RGB", (SIZE, SIZE), b["band"])
    photo = Image.open(slide["photo"]).convert("RGB")
    if slide.get("retouch"):
        photo = retouch(photo, slide["retouch"])
    photo_h = SIZE - BAND_H - HEADER_H
    canvas.paste(fit_photo(photo, SIZE, photo_h, slide["focus"], slide.get("zoom", 1.0)), (0, HEADER_H))
    if HEADER_H and slide.get("header"):
        draw_header(canvas, slide["header"], b)

    # Logo, vertically centered in the band
    logo_h = BAND_H - 60
    logo = tinted_logo(b["logo"], logo_h, b["text"])
    band_top = SIZE - BAND_H
    ImageDraw.Draw(canvas).rectangle((0, band_top, SIZE, band_top + RULE_H - 1), fill=b["accent"])
    canvas.paste(logo, (PAD, band_top + (BAND_H - logo_h) // 2), logo)

    # Text block to the right of the logo
    draw = ImageDraw.Draw(canvas)
    tx = PAD + logo.width + 36
    max_w = SIZE - PAD - tx
    label = ImageFont.truetype(b["font_label"], 21)
    head = fit_font(b["font_head"], slide["headline"], max_w, 62, draw)
    sub = fit_font(b["font_body"], slide["subline"], max_w, 27, draw)

    lh, hh, sh = 21, head.size, sub.size
    gap1, gap2 = 14, 12
    y = band_top + (BAND_H - (lh + gap1 + hh + gap2 + sh)) // 2
    draw_tracked(draw, (tx, y), b["name"], label, b["accent"], 4)
    draw.text((tx, y + lh + gap1 - 6), slide["headline"], font=head, fill=b["text"])
    draw.text((tx, y + lh + gap1 + hh + gap2 - 4), slide["subline"], font=sub, fill=b["muted"])
    return canvas


def main():
    for slide in SLIDES:
        slide["out"].parent.mkdir(parents=True, exist_ok=True)
        post = render(slide)
        post.save(slide["out"], optimize=True)
        post.save(slide["out"].with_suffix(".jpg"), quality=JPEG_QUALITY, subsampling=0, optimize=True)
        print("wrote", slide["out"].relative_to(ROOT), "(+ .jpg)")


if __name__ == "__main__":
    main()
