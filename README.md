# socialmediaposts

Branded 1080x1080 social posts for Built for the Trades.

```
pip install -r requirements.txt
python3 scripts/brand_photos.py      # renders every post in SLIDES to output/ (.png + upload-ready .jpg)
```

- `scripts/brand_photos.py`: layout, palette, copy and crop for each post (edit `BRAND` and `SLIDES`).
- `scripts/retouch.py`: natural portrait retouching (skin tone, teeth, eyes, hair) driven by regions listed per photo.
- `photos/`: source photos. `brand/`: logo. `output/`: finished posts and captions.
