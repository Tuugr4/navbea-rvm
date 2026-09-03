"""GPL-3.0 NVF1 streaming worker for Robust Video Matting.

The worker has no socket, cloud or camera-discovery privileges. It reads MJPEG
frames from stdin and writes sequence-aligned GRAY8 masks to stdout.
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import time
from pathlib import Path

MAGIC = b"NVF1"
VERSION = 1
HEADER = struct.Struct(">4sHHIQQIIIHH16sI")
TYPE_FRAME = 1
CODEC_MJPEG = 1
CODEC_GRAY8 = 2
MAX_PAYLOAD = 128 * 1024 * 1024


def select_onnx_providers(available: list[str], device_mode: str) -> list[str]:
    """Choose only release-supported execution providers.

    Prefer CUDA, then DirectML, then the portable CPU baseline. Native stills
    are handled separately by :class:`OnnxRvmEngine`, because DirectML cannot
    reliably expand RVM recurrent tensors across a live-to-native resolution
    transition.
    """
    if device_mode == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider was requested but is unavailable")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device_mode == "directml":
        if "DmlExecutionProvider" not in available:
            raise RuntimeError("DmlExecutionProvider was requested but is unavailable")
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def read_exact(stream, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream):
    raw = read_exact(stream, HEADER.size)
    magic, version, message_type, flags, sequence, timestamp, width, height, length, codec, _reserved, stream_id, _tail = HEADER.unpack(raw)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported NVF1 header")
    if length > MAX_PAYLOAD:
        raise ValueError("Frame payload exceeds limit")
    return {
        "type": message_type, "flags": flags, "sequence": sequence, "timestamp": timestamp,
        "width": width, "height": height, "codec": codec, "stream_id": stream_id,
        "payload": read_exact(stream, length),
    }


def write_frame(stream, frame, payload: bytes, width: int, height: int):
    header = HEADER.pack(
        MAGIC, VERSION, TYPE_FRAME, 0, frame["sequence"], time.monotonic_ns(),
        width, height, len(payload), CODEC_GRAY8, 0, frame["stream_id"], 0,
    )
    stream.write(header)
    stream.write(payload)
    stream.flush()


class RvmEngine:
    def __init__(self, model_path: Path, device_mode: str = "auto", downsample_ratio: float = 0.5):
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        if not model_path.is_file():
            raise FileNotFoundError(f"RVM TorchScript model not found: {model_path}")
        if device_mode == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.np, self.torch, self.functional, self.Image = np, torch, functional, Image
        self.device = torch.device("cuda" if device_mode == "cuda" or (device_mode == "auto" and torch.cuda.is_available()) else "cpu")
        if self.device.type == "cpu": torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
        self.downsample_ratio = downsample_ratio
        self.rec = [None] * 4
        self.warmup()

    def warmup(self):
        frame = self.torch.zeros((1, 3, 360, 640), device=self.device)
        rec = [None] * 4
        with self.torch.inference_mode():
            for _ in range(3):
                output = self.model(frame, *rec, self.downsample_ratio)
                rec = list(output[2:6])

    def reset(self): self.rec = [None] * 4

    def process(self, payload: bytes, native: bool = False) -> tuple[bytes, int, int]:
        image = self.Image.open(io.BytesIO(payload)).convert("RGB")
        if not native and (image.width > 1280 or image.height > 720):
            scale = min(1280 / image.width, 720 / image.height)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), self.Image.Resampling.BILINEAR)
        width, height = image.size
        array = self.np.asarray(image, dtype=self.np.float32)
        tensor = self.torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device) / 255.0
        autocast = self.torch.amp.autocast(device_type="cuda", dtype=self.torch.float16) if self.device.type == "cuda" else __import__("contextlib").nullcontext()
        with self.torch.inference_mode(), autocast:
            output = self.model(tensor, *self.rec, self.downsample_ratio)
            self.rec = list(output[2:6])
            alpha = self.functional.interpolate(output[1], size=(height, width), mode="bilinear", align_corners=False)
            mask = (alpha[0, 0].clamp(0, 1) * 255.0).round().byte().cpu().numpy().tobytes()
        return mask, width, height


class OnnxRvmEngine:
    def __init__(self, model_path: Path, device_mode: str = "auto", downsample_ratio: float = 0.5):
        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        if not model_path.is_file():
            raise FileNotFoundError(f"RVM ONNX model not found: {model_path}")
        available = ort.get_available_providers()
        providers = select_onnx_providers(available, device_mode)
        self.np, self.Image, self.ort = np, Image, ort
        self.model_path = str(model_path)
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.primary_provider = providers[0]
        self.cpu_session = None
        self.downsample_ratio = np.asarray([downsample_ratio], dtype=np.float32)
        self.rec = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]
        self.rec_initialized = False

    def reset(self):
        self.rec = [self.np.zeros((1, 1, 1, 1), dtype=self.np.float32) for _ in range(4)]
        self.rec_initialized = False

    def _cpu(self):
        if self.cpu_session is None:
            self.cpu_session = self.ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        return self.cpu_session

    def _run(self, session, src, rec):
        return session.run(None, {
            "src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2],
            "r4i": rec[3], "downsample_ratio": self.downsample_ratio,
        })

    def process(self, payload: bytes, native: bool = False) -> tuple[bytes, int, int]:
        image = self.Image.open(io.BytesIO(payload)).convert("RGB")
        if not native and (image.width > 1280 or image.height > 720):
            scale = min(1280 / image.width, 720 / image.height)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), self.Image.Resampling.BILINEAR)
        width, height = image.size
        src = self.np.asarray(image, dtype=self.np.float32).transpose(2, 0, 1)[None] / 255.0
        if native and self.primary_provider == "DmlExecutionProvider":
            # Keep the fast DirectML live session intact. Native stills use a
            # fresh CPU recurrence so their different resolution cannot crash
            # the DirectML Expand nodes.
            native_rec = [self.np.zeros((1, 1, 1, 1), dtype=self.np.float32) for _ in range(4)]
            outputs = self._run(self._cpu(), src, native_rec)
        else:
            try:
                # DirectML cannot expand the conventional 1x1 placeholder
                # recurrence used by the ONNX export. Seed the correctly sized
                # recurrent tensors once on CPU, then keep the live sequence on
                # DirectML. All following preview frames stay accelerated.
                if self.primary_provider == "DmlExecutionProvider" and not self.rec_initialized:
                    outputs = self._run(self._cpu(), src, self.rec)
                else:
                    outputs = self._run(self.session, src, self.rec)
            except Exception:
                if self.primary_provider != "DmlExecutionProvider":
                    raise
                # A driver-specific DirectML failure is recoverable. Permanently
                # move this worker to CPU without killing the supervisor pipe.
                self.session = self._cpu()
                self.primary_provider = "CPUExecutionProvider"
                self.reset()
                outputs = self._run(self.session, src, self.rec)
            self.rec = list(outputs[2:6])
            self.rec_initialized = True
        alpha = self.np.clip(outputs[1][0, 0], 0, 1)
        return (alpha * 255.0).round().astype(self.np.uint8).tobytes(), width, height


class FixtureEngine:
    def process(self, payload: bytes, native: bool = False) -> tuple[bytes, int, int]:
        del payload
        width, height = ((1920, 1080) if native else (1280, 720))
        return bytes([255]) * (width * height), width, height


def serve(engine, input_stream=None, output_stream=None) -> int:
    input_stream = input_stream or sys.stdin.buffer
    output_stream = output_stream or sys.stdout.buffer
    while True:
        try: frame = read_frame(input_stream)
        except EOFError: return 0
        if frame["type"] != TYPE_FRAME or frame["codec"] not in {CODEC_MJPEG, 3}: continue
        native_still = bool(frame["flags"] & 1)
        # Live-view recurrent tensors are sized for the preview frame. A native
        # still is normally larger, so reset before inference as well as after;
        # DirectML cannot expand the preview state into the still resolution.
        if native_still and hasattr(engine, "reset"):
            engine.reset()
        mask, width, height = engine.process(frame["payload"], native_still)
        # A native still can use a different resolution from the live stream.
        # Recurrent tensors are resolution-dependent, so never carry still
        # state back into the next live frame.
        if native_still and hasattr(engine, "reset"):
            engine.reset()
        write_frame(output_stream, frame, mask, width, height)


def self_test() -> int:
    stream_id = bytes(range(16))
    payload = b"jpeg"
    encoded = HEADER.pack(MAGIC, VERSION, TYPE_FRAME, 0, 7, 9, 1280, 720, len(payload), CODEC_MJPEG, 0, stream_id, 0) + payload
    frame = read_frame(io.BytesIO(encoded))
    assert frame["sequence"] == 7 and frame["payload"] == payload and HEADER.size == 64

    class TrackingEngine:
        def __init__(self): self.events = []
        def reset(self): self.events.append("reset")
        def process(self, _payload, native=False):
            self.events.append(f"process:{native}")
            return b"\xff", 1, 1

    native = HEADER.pack(MAGIC, VERSION, TYPE_FRAME, 1, 8, 10, 1920, 1080, len(payload), CODEC_MJPEG, 0, stream_id, 0) + payload
    live = HEADER.pack(MAGIC, VERSION, TYPE_FRAME, 0, 9, 11, 1280, 720, len(payload), CODEC_MJPEG, 0, stream_id, 0) + payload
    tracker = TrackingEngine()
    assert serve(tracker, io.BytesIO(native + live), io.BytesIO()) == 0
    assert tracker.events == ["reset", "process:True", "reset", "process:False"]
    assert select_onnx_providers(["DmlExecutionProvider", "CPUExecutionProvider"], "auto") == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers(["CUDAExecutionProvider", "CPUExecutionProvider"], "auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers(["DmlExecutionProvider", "CPUExecutionProvider"], "directml") == ["DmlExecutionProvider", "CPUExecutionProvider"]
    print("RVM worker transport self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("RVM_MODEL_PATH", ""))
    parser.add_argument("--device", choices=("auto", "cuda", "directml", "cpu"), default=os.environ.get("RVM_DEVICE", "auto"))
    parser.add_argument("--downsample-ratio", type=float, default=0.5)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: return self_test()
    model_path = Path(args.model).resolve()
    engine = FixtureEngine() if args.fixture else OnnxRvmEngine(model_path, args.device, args.downsample_ratio) if model_path.suffix.lower() == ".onnx" else RvmEngine(model_path, args.device, args.downsample_ratio)
    return serve(engine)


if __name__ == "__main__": raise SystemExit(main())
