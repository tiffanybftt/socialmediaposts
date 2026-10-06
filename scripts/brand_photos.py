#!/usr/bin/env python3
"""Brand photos as 1080x1080 social posts: photo on top, branded band below.

Usage:  python3 scripts/brand_photos.py            # renders every slide in SLIDES
Edit BRAND and SLIDES below; everything else is layout code.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

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
BAND_H = 190             # height of the branded band
PAD = 48                 # side padding

# focus = (x, y) in the ORIGINAL photo that the crop should be centered on.
SLIDES = [
    {
        "photo": ROOT / "photos" / "lunch-selfie-01.jpg",
        "focus": (750, 1000),
        "headline": "BUILDING RELATIONSHIPS",
        "subline": "Good food. Real conversations. Better businesses.",
        "out": ROOT / "output" / "leadership-lunch.png",
    },
]


def crop_box(img, w, h, focus):
    """Largest w:h box that fits in img, centered on focus (clamped to the edges)."""
    iw, ih = img.size
    target = w / h
    cw, ch = (iw, int(iw / target)) if iw / ih < target else (int(ih * target), ih)
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


def fit_photo(img, w, h, focus):
    """Crop, polish at full resolution, downscale, then sharpen for the smaller size."""
    img = polish(img.crop(crop_box(img, w, h, focus))).resize((w, h), Image.LANCZOS)
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


def render(slide):
    b = BRAND
    canvas = Image.new("RGB", (SIZE, SIZE), b["band"])
    photo = Image.open(slide["photo"]).convert("RGB")
    canvas.paste(fit_photo(photo, SIZE, SIZE - BAND_H, slide["focus"]), (0, 0))

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
