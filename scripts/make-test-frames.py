"""Write synthetic camera frames for the development camera and flow check.

Each frame shows one person-like figure in the centre of a busy background, so
a working matte keeps the centre and removes both sides. No real photos ship
with the repository.

Usage: python make-test-frames.py OUTPUT_DIR LIVE_WxH STILL_WxH
"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def parse_size(value: str) -> tuple[int, int]:
    width, height = (int(part) for part in value.lower().split("x"))
    if not (64 <= width <= 8192 and 64 <= height <= 8192):
        raise ValueError(f"Unsupported frame size {value}")
    return width, height


def frame(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        draw.line([(0, y), (width, y)], fill=(40 + y * 60 // height, 90, 140 + y * 40 // height))
    rng = random.Random(1)
    for _ in range(60):
        x, y = rng.randrange(width), rng.randrange(height // 2)
        shade = rng.randrange(60, 200)
        draw.rectangle([x, y, x + width // 20, y + height // 6], fill=(shade, shade, shade))
    cx, s = width / 2, height / 720
    draw.ellipse([cx - 70 * s, 140 * s, cx + 70 * s, 320 * s], fill=(224, 172, 140))
    draw.rectangle([cx - 60 * s, 300 * s, cx + 60 * s, 330 * s], fill=(210, 160, 130))
    draw.polygon([(cx - 190 * s, height), (cx - 150 * s, 360 * s), (cx - 60 * s, 325 * s),
                  (cx + 60 * s, 325 * s), (cx + 150 * s, 360 * s), (cx + 190 * s, height)], fill=(30, 30, 35))
    draw.ellipse([cx - 75 * s, 120 * s, cx + 75 * s, 220 * s], fill=(40, 25, 15))
    return image.filter(ImageFilter.GaussianBlur(1))


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for name, size in (("live.jpg", sys.argv[2]), ("still.jpg", sys.argv[3])):
        frame(*parse_size(size)).save(output / name, quality=85)
    return 0


if __name__ == "__main__":
    sys.exit(main())
