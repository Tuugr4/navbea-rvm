"""Export the checksum-pinned GPL-3.0 YOLOv5 v7 segmentation checkpoint.

Run in an isolated build environment with the versions in models/subjects.json.
Runtime installations need only the exported ONNX, never torch or this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((Path(__file__).resolve().parents[1] / "models/subjects.json").read_text())
for name, filename in [("source", args.source), ("checkpoint", args.checkpoint)]:
    if hashlib.sha256(filename.read_bytes()).hexdigest() != manifest[name]["sha256"]:
        raise ValueError(f"{name} checksum mismatch")
if args.output.exists(): raise ValueError("Choose a new output path")
with tempfile.TemporaryDirectory(prefix="rvm-subject-export-") as scratch:
    root = Path(scratch)
    with zipfile.ZipFile(args.source) as archive: archive.extractall(root)
    checkpoint = root / "yolov5n-seg.pt"
    shutil.copyfile(args.checkpoint, checkpoint)
    exporter = root / ("yolov5-" + manifest["revision"]) / "export.py"
    subprocess.run([sys.executable, str(exporter), "--weights", str(checkpoint), "--include", "onnx", "--dynamic", "--opset", "12", "--device", "cpu", "--imgsz", "640"], check=True, env={**os.environ, "YOLOv5_AUTOINSTALL": "false"})
    result = checkpoint.with_suffix(".onnx")
    if hashlib.sha256(result.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Export does not match the pinned ONNX; check build dependency versions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result, args.output)
