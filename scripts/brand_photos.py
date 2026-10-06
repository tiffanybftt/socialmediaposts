#!/usr/bin/env python3
"""Brand photos as 1080x1080 social posts: photo on top, branded band below.

Usage:  python3 scripts/brand_photos.py            # renders every slide in SLIDES
Edit BRAND and SLIDES below; everything else is layout code.
"""
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont

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
WARM = (255, 218, 178)   # multiplied over the photo to push highlights/skin warm
SHADOW = (38, 16, 6)     # warm near-black the photo is blended toward
TINT = 0.28              # how far toward SHADOW (0 = none, 1 = solid)
CONTRAST = 1.12
FADE = 0.45              # darkening at the bottom edge, fading into the band
FADE_H = 0.40            # fraction of the photo height the fade covers
EMBER = 0.28             # orange glow strength along the bottom of the photo
RULE_H = 6               # orange rule between photo and band
BAND_H = 190             # height of the branded band
PAD = 48          # side padding

# focus = (x, y) in the ORIGINAL photo that the crop should be centered on.
SLIDES = [
    {
        "photo": ROOT / "photos" / "lunch-selfie-01.jpg",
        "focus": (750, 1000),
        "headline": "LEADERSHIP LUNCH",
        "subline": "With Jacob Stultz, Stultz Plumbing",
        "out": ROOT / "output" / "leadership-lunch-jacob-stultz-1.png",
    },
    {
        "photo": ROOT / "photos" / "lunch-table-02.jpg",
        "focus": (720, 895),
        "headline": "BUILDING RELATIONSHIPS",
        "subline": "Good food. Real conversations. Better businesses.",
        "out": ROOT / "output" / "leadership-lunch-jacob-stultz-2.png",
    },
]


def cover_crop(img, w, h, focus):
    """Scale-free crop of the largest w:h box that fits, centered on focus, then resize."""
    iw, ih = img.size
    target = w / h
    cw, ch = (iw, int(iw / target)) if iw / ih < target else (int(ih * target), ih)
    left = min(max(int(focus[0] - cw / 2), 0), iw - cw)
    top = min(max(int(focus[1] - ch / 2), 0), ih - ch)
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def grade(img, band, accent):
    """Dark, warm grade: warm cast + shadow tint + orange ember at the base, fading into the band."""
    img = ImageEnhance.Contrast(ImageChops.multiply(img, Image.new("RGB", img.size, WARM))).enhance(CONTRAST)
    img = Image.blend(img, Image.new("RGB", img.size, SHADOW), TINT)
    fade_h = int(img.height * FADE_H)
    ramp = Image.linear_gradient("L").resize((img.width, fade_h))
    img.paste(Image.new("RGB", (img.width, fade_h), accent), (0, img.height - fade_h), ramp.point(lambda v: int(v * EMBER)))
    img.paste(Image.new("RGB", (img.width, fade_h), band), (0, img.height - fade_h), ramp.point(lambda v: int(v * FADE)))
    return img


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
    canvas.paste(grade(cover_crop(photo, SIZE, SIZE - BAND_H, slide["focus"]), b["band"], b["accent"]), (0, 0))

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
        render(slide).save(slide["out"], optimize=True)
        print("wrote", slide["out"].relative_to(ROOT))


if __name__ == "__main__":
    main()
