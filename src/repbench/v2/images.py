"""One image-decoding rule for every v2 model and audit (owner decision, 2026-11-19).

Images with transparency are composited onto white (POG PNGs otherwise decode with
arbitrary, often black, backgrounds — a hidden cross-dataset domain shift); all
other images are converted to RGB unchanged (no-op for the JPEG datasets).
"""

from __future__ import annotations

from PIL import Image

#: Uniform pixel budget: no model sees more than 384x384 pixels (decision F).
MAX_PIXELS = 384 * 384
#: Qwen3-VL: one visual token covers 32x32 pixels -> 144 tokens.
QWEN_MAX_PIXELS = MAX_PIXELS
QWEN_MIN_PIXELS = 4 * 32 * 32


def to_rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    return image.convert("RGB")


def load_rgb(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return to_rgb_on_white(value)
    with Image.open(value) as opened:
        opened.load()
        return to_rgb_on_white(opened)
