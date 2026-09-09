"""Camera-free CPU calibration using the same decode/matting path as live input."""
import argparse
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rvm_worker import OnnxRvmEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()
    image = Image.new("RGB", (1920, 1080), (85, 104, 124))
    draw = ImageDraw.Draw(image)
    draw.ellipse((650, 90, 1270, 700), fill=(185, 137, 112))
    draw.polygon([(590, 510), (1330, 510), (1650, 1080), (270, 1080)], fill=(26, 39, 69))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=90)
    jpeg = output.getvalue()
    for threads in [int(value) for value in args.candidates.split(",")]:
        engine = OnnxRvmEngine(Path(args.model), "cpu", args.ratio, .5, threads)
        engine.process(jpeg)
        print(json.dumps({"event": "candidate", "threads": threads}), flush=True)
        samples = []
        for _ in range(7):
            started = time.perf_counter()
            mask, width, height = engine.process(jpeg)
            # Include alpha-buffer preparation performed by stream consumers.
            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = 255
            rgba[:, :, 3] = np.frombuffer(mask, dtype=np.uint8).reshape(height, width)
            samples.append((time.perf_counter() - started) * 1000)
        print(json.dumps({"event": "result", "threads": threads, "medianMs": round(statistics.median(samples), 2), "p95Ms": round(sorted(samples)[-1], 2)}), flush=True)
        del engine


if __name__ == "__main__":
    main()
