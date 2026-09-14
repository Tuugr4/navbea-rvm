"""GPL-3.0 NVF1 streaming worker for Robust Video Matting.

The worker has no socket, cloud or camera-discovery privileges. It reads MJPEG
frames from stdin and writes sequence-aligned GRAY8 masks to stdout.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

# Explicit sibling import also works in the isolated packaged Python (-I).
sys.path.insert(0, str(Path(__file__).resolve().parent))

MAGIC = b"NVF1"
VERSION = 1
HEADER = struct.Struct(">4sHHIQQIIIHH16sI")
TYPE_FRAME = 1
TYPE_RESET = 5  # Private supervisor/worker control; never published to clients.
CODEC_MJPEG = 1
CODEC_GRAY8 = 2
MAX_PAYLOAD = 128 * 1024 * 1024
_native_allocator_ready = False


def register_native_allocator(ort):
    global _native_allocator_ready
    if not _native_allocator_ready:
        info = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
        config = ort.OrtArenaCfg({"max_mem": 3 * 1024**3, "arena_extend_strategy": 1})
        ort.create_and_register_allocator_v2("CPUExecutionProvider", info, {}, config)
        _native_allocator_ready = True


def select_onnx_providers(available: list[str], device_mode: str) -> list[str]:
    """Choose only release-supported execution providers.

    Prefer CUDA, then DirectML, then the portable CPU baseline. Native stills
    are handled separately by :class:`OnnxRvmEngine`, because DirectML cannot
    reliably expand RVM recurrent tensors across a live-to-native resolution
    transition.
    """
    if device_mode == "cpu":
        return ["CPUExecutionProvider"]
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


def write_frame(stream, frame, payload: bytes, width: int, height: int, codec=CODEC_GRAY8, message_type=TYPE_FRAME):
    parts = payload if isinstance(payload, tuple) else (payload,)
    length = sum(len(part) for part in parts)
    if length > MAX_PAYLOAD:
        raise ValueError("Worker output exceeds the protocol limit")
    header = HEADER.pack(
        MAGIC, VERSION, message_type, 0, frame["sequence"], frame["timestamp"],
        width, height, length, codec, 0, frame["stream_id"], 0,
    )
    stream.write(header)
    for part in parts:
        stream.write(part)
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
    def __init__(self, model_path: Path, device_mode: str = "auto", downsample_ratio: float = 0.375, still_ratio: float = 0.5, threads: int = 0, still_max_edge: int = 1024, still_threads: int = 0):
        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        if not model_path.is_file():
            raise FileNotFoundError(f"RVM ONNX model not found: {model_path}")
        available = ort.get_available_providers()
        providers = select_onnx_providers(available, device_mode)
        self.np, self.Image, self.ort = np, Image, ort
        self.model_path = str(model_path)
        self.session_options = ort.SessionOptions()
        self.session_options.intra_op_num_threads = max(1, min(threads or os.cpu_count() or 1, os.cpu_count() or 1))
        self.session_options.inter_op_num_threads = 1
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session_options.enable_mem_pattern = providers[0] != "DmlExecutionProvider"
        self.session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session_options.add_session_config_entry("session.use_env_allocators", "0")
        self.run_options = ort.RunOptions()
        self.run_options.only_execute_path_to_fetches = True
        self.session = ort.InferenceSession(self.model_path, sess_options=self.session_options, providers=providers)
        self.primary_provider = providers[0]
        self.cpu_session = None
        self.native_session = None
        self.still_threads = still_threads or self.session_options.intra_op_num_threads
        self.downsample_ratio = np.asarray([downsample_ratio], dtype=np.float32)
        self.still_ratio = np.asarray([still_ratio], dtype=np.float32)
        # Bound the encoder's working resolution for high-MP stills. The guided
        # refinement, foreground and alpha output retain the original resolution.
        self.still_max_edge = still_max_edge
        self.foreground_jpeg = None
        self.rec = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]
        self.rec_initialized = False

    def reset(self):
        self.rec = [self.np.zeros((1, 1, 1, 1), dtype=self.np.float32) for _ in range(4)]
        self.rec_initialized = False
        self.foreground_jpeg = None
        if self.native_session is not None:
            # Run a tiny empty-person probe after native outputs have been
            # consumed. Shrink only the native arena, leaving live latency and
            # loaded model weights intact between people.
            trim = self.ort.RunOptions()
            trim.only_execute_path_to_fetches = True
            trim.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
            self.native_session.run(["pha"], {
                "src": self.np.zeros((1, 3, 128, 128), dtype=self.np.float32),
                "r1i": self.rec[0], "r2i": self.rec[1], "r3i": self.rec[2], "r4i": self.rec[3],
                "downsample_ratio": self.still_ratio,
            }, trim)

    def _cpu(self):
        if self.cpu_session is None:
            self.cpu_session = self.ort.InferenceSession(self.model_path, sess_options=self.session_options, providers=["CPUExecutionProvider"])
        return self.cpu_session

    def _native(self):
        if self.primary_provider == "CUDAExecutionProvider":
            return self.session
        if self.native_session is None:
            register_native_allocator(self.ort)
            options = self.ort.SessionOptions()
            options.intra_op_num_threads = self.still_threads
            options.inter_op_num_threads = 1
            options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
            # Exact-growth, 3 GiB scratch budget; independent from the live
            # allocator and reclaimed on the acknowledged person-state reset.
            options.enable_cpu_mem_arena = True
            options.enable_mem_pattern = False
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.use_env_allocators", "1")
            self.native_session = self.ort.InferenceSession(self.model_path, sess_options=options, providers=["CPUExecutionProvider"])
        return self.native_session

    def _run(self, session, src, rec, ratio=None):
        feeds = {
            "src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2],
            "r4i": rec[3], "downsample_ratio": self.downsample_ratio if ratio is None else ratio,
        }
        if ratio is None:
            # Live subscribers consume alpha only. Do not fetch the full RGB
            # foreground output until a native still actually needs it.
            return [None, *session.run(["pha", "r1o", "r2o", "r3o", "r4o"], feeds, self.run_options)]
        # Still recurrence is discarded. Fetch only the two native outputs.
        return session.run(["fgr", "pha"], feeds, self.run_options)

    def _bytes(self, values):
        # ORT owns these independent output arrays. They are no longer consumed
        # by inference; keep rounding/clipping identical while avoiding 3 large
        # temporary float tensors. Never apply this to recurrent output arrays.
        self.np.clip(values, 0, 1, out=values)
        self.np.multiply(values, 255.0, out=values)
        self.np.rint(values, out=values)
        return values.astype(self.np.uint8)

    def process(self, payload: bytes, native: bool = False) -> tuple[bytes, int, int]:
        image = self.Image.open(io.BytesIO(payload)).convert("RGB")
        if not native and (image.width > 1280 or image.height > 720):
            scale = min(1280 / image.width, 720 / image.height)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), self.Image.Resampling.BILINEAR)
        width, height = image.size
        src = self.np.empty((1, 3, height, width), dtype=self.np.float32)
        src[0] = self.np.asarray(image).transpose(2, 0, 1)
        src *= 1.0 / 255.0
        image.close()
        self.foreground_jpeg = None
        if native:
            # Keep the fast DirectML live session intact. Native stills use a
            # fresh CPU recurrence so their different resolution cannot crash
            # the DirectML Expand nodes.
            native_rec = [self.np.zeros((1, 1, 1, 1), dtype=self.np.float32) for _ in range(4)]
            session = self._native()
            ratio = self.np.asarray([min(float(self.still_ratio[0]), self.still_max_edge / max(width, height))], dtype=self.np.float32) if self.still_max_edge else self.still_ratio
            outputs = self._run(session, src, native_rec, ratio)
            del src
            foreground = self._bytes(outputs[0][0].transpose(1, 2, 0))
            outputs[0] = None
            encoded = io.BytesIO()
            self.Image.fromarray(foreground).save(encoded, "JPEG", quality=97, subsampling=0)
            self.foreground_jpeg = encoded.getvalue()
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
        return self._bytes(outputs[1][0, 0]).tobytes(), width, height


class FixtureEngine:
    def process(self, payload: bytes, native: bool = False) -> tuple[bytes, int, int]:
        del payload
        width, height = ((1920, 1080) if native else (1280, 720))
        return bytes([255]) * (width * height), width, height


def serve(engine, input_stream=None, output_stream=None, selector=None) -> int:
    input_stream = input_stream or sys.stdin.buffer
    output_stream = output_stream or sys.stdout.buffer
    while True:
        try: frame = read_frame(input_stream)
        except EOFError: return 0
        if frame["type"] == TYPE_RESET:
            if frame["payload"] or frame["codec"] != 4:
                raise ValueError("Invalid reset request")
            if hasattr(engine, "reset"):
                engine.reset()
            if selector: selector.reset()
            write_frame(output_stream, frame, b"", 0, 0, 4, TYPE_RESET)
            continue
        if frame["type"] == 7:
            from subjects import SubjectBlocked
            try:
                if selector is None: raise SubjectBlocked("SUBJECT_MODEL_UNAVAILABLE")
                command = json.loads(frame["payload"])
                if command["action"] == "configure": selector.configure(command["policy"])
                elif command["action"] == "lock": selector.lock()
                elif command["action"] == "unlock": selector.unlock()
                else: raise SubjectBlocked("SUBJECT_INVALID_CONTROL")
                write_frame(output_stream, frame, json.dumps(selector.state).encode(), 0, 0, 4, 7)
            except SubjectBlocked as error:
                write_frame(output_stream, frame, json.dumps({"code": str(error), "message": str(error)}).encode(), 0, 0, 4, 4)
            continue
        if frame["type"] != TYPE_FRAME or frame["codec"] not in {CODEC_MJPEG, 3}: continue
        native_still = bool(frame["flags"] & 1)
        support = metadata = None
        if selector and selector.policy:
            from subjects import SubjectBlocked
            from PIL import Image
            try:
                with Image.open(io.BytesIO(frame["payload"])) as source:
                    support, metadata = selector.evaluate(source.convert("RGB"), native_still, frame["timestamp"])
            except SubjectBlocked as error:
                write_frame(output_stream, frame, json.dumps({"code": str(error), "message": str(error)}).encode(), 0, 0, 4, 4)
                continue
        # Live-view recurrent tensors are sized for the preview frame. A native
        # still is normally larger, so reset before inference as well as after;
        # DirectML cannot expand the preview state into the still resolution.
        if native_still and not isinstance(engine, OnnxRvmEngine) and hasattr(engine, "reset"):
            engine.reset()
        mask, width, height = engine.process(frame["payload"], native_still)
        # A native still can use a different resolution from the live stream.
        # Recurrent tensors are resolution-dependent, so never carry still
        # state back into the next live frame.
        if native_still and not isinstance(engine, OnnxRvmEngine) and hasattr(engine, "reset"):
            engine.reset()
        foreground = getattr(engine, "foreground_jpeg", None)
        if metadata is not None:
            if support is not None:
                mask = selector.filter_alpha(mask, width, height, support)
            meta = json.dumps(metadata, separators=(",", ":")).encode()
            write_frame(output_stream, frame, (struct.pack(">II", len(meta), len(mask)), meta, mask, foreground or b""), width, height, 6)
        elif native_still and frame["flags"] & 2 and foreground:
            # Private worker codec 5: uint32 mask length, GRAY8 mask, foreground JPEG.
            # Public stream subscribers continue to receive only GRAY8 frames.
            write_frame(output_stream, frame, (struct.pack(">I", len(mask)), mask, foreground), width, height, 5)
        else:
            write_frame(output_stream, frame, mask, width, height)
        if hasattr(engine, "foreground_jpeg"):
            engine.foreground_jpeg = None
        del mask, foreground, frame


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
    assert select_onnx_providers(["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"], "cpu") == ["CPUExecutionProvider"]
    assert select_onnx_providers(["DmlExecutionProvider", "CPUExecutionProvider"], "directml") == ["DmlExecutionProvider", "CPUExecutionProvider"]
    print("RVM worker transport self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("RVM_MODEL_PATH", ""))
    parser.add_argument("--subject-model", default="")
    parser.add_argument("--device", choices=("auto", "cuda", "directml", "cpu"), default=os.environ.get("RVM_DEVICE", "auto"))
    parser.add_argument("--downsample-ratio", type=float, default=0.375)
    parser.add_argument("--still-ratio", type=float, default=0.5)
    parser.add_argument("--still-max-edge", type=int, choices=(0, 512, 768, 1024), default=1024)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--still-threads", type=int, default=0)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--startup-id", default="")
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()
    if not 0 < args.downsample_ratio <= 1 or not 0 < args.still_ratio <= 1 or not 0 <= args.threads <= (os.cpu_count() or 1) or not 0 <= args.still_threads <= (os.cpu_count() or 1):
        parser.error("Ratios must be in (0, 1]; threads must be 0 (automatic) or an available processor count")
    if args.self_test: return self_test()
    model_path = Path(args.model).resolve()
    engine = FixtureEngine() if args.fixture else OnnxRvmEngine(model_path, args.device, args.downsample_ratio, args.still_ratio, args.threads, args.still_max_edge, args.still_threads) if model_path.suffix.lower() == ".onnx" else RvmEngine(model_path, args.device, args.downsample_ratio)
    if not args.fixture:
        from PIL import Image
        sample = io.BytesIO()
        Image.new("RGB", (128, 128), (80, 100, 120)).save(sample, format="JPEG")
        mask, width, height = engine.process(sample.getvalue())
        if width != 128 or height != 128 or len(mask) != width * height:
            raise RuntimeError("RVM startup inference returned invalid dimensions")
        engine.reset()
    selector = None
    if args.subject_model:
        from subjects import PersonDetector, SubjectSelector
        selector = SubjectSelector(PersonDetector(args.subject_model, args.threads or 2))
        selector.detector.detect(Image.new("RGB", (128, 128)))
    print(json.dumps({"event": "rvm-ready", "protocol": 1, "startupId": args.startup_id,
                      "inferenceVerified": not args.fixture, "provider": getattr(engine, "primary_provider", args.device)}), file=sys.stderr, flush=True)
    if args.check_runtime: return 0
    return serve(engine, selector=selector)


if __name__ == "__main__": raise SystemExit(main())
