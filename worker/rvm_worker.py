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

from still_matte import StillMatteEngine, emit, estimate_foreground  # noqa: E402

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


def gpu_devices(ort) -> list[dict]:
    """Execution devices as onnxruntime reports them (name, discrete, video memory)."""
    devices = []
    try:
        for device in ort.get_ep_devices():
            metadata = dict(device.device.metadata)
            devices.append({"provider": device.ep_name, "name": metadata.get("Description", ""), "discrete": metadata.get("Discrete") == "1",
                            "videoMemory": metadata.get("DxgiVideoMemory"), "adapter": metadata.get("DxgiAdapterNumber"), "highPerformanceIndex": metadata.get("DxgiHighPerformanceIndex")})
    except Exception:
        pass
    return devices


def dml_adapter(ort) -> tuple[int, str]:
    """DirectML's default adapter 0 is the integrated GPU on hybrid systems; use the high-performance one."""
    for device in gpu_devices(ort):
        if device["provider"] == "DmlExecutionProvider" and device["highPerformanceIndex"] == "0":
            return int(device["adapter"] or 0), device["name"]
    return 0, ""


def with_adapter(ort, providers: list[str]) -> list:
    return [("DmlExecutionProvider", {"device_id": dml_adapter(ort)[0]}) if name == "DmlExecutionProvider" else name for name in providers]


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
    def __init__(self, model_path: Path, device_mode: str = "auto", downsample_ratio: float = 0.375, still_ratio: float = 0.5, threads: int = 0, still_max_edge: int = 1024, still_threads: int = 0, still_warmup: int = 4, still_rvm_model: Path | None = None, native_device: str = "cpu"):
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
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options=self.session_options, providers=with_adapter(ort, providers))
        except Exception:
            if device_mode != "auto" or providers[0] == "CPUExecutionProvider":
                raise
            # The runtime can advertise DirectML without a usable adapter/driver.
            # Auto mode must still start on computers without compatible GPU hardware.
            self.session_options.enable_mem_pattern = True
            self.session = ort.InferenceSession(self.model_path, sess_options=self.session_options, providers=["CPUExecutionProvider"])
            print(json.dumps({"event":"provider-change","provider":"CPUExecutionProvider","reason":"GPU initialization fallback"}),file=sys.stderr,flush=True)
        self.input_types = {item.name: item.type for item in self.session.get_inputs()}
        if self.input_types.get("src") not in ("tensor(float)", "tensor(float16)"):
            raise ValueError("RVM model requires FP32 or FP16 inputs")
        self.tensor_dtype = np.float16 if self.input_types["src"] == "tensor(float16)" else np.float32
        self.primary_provider = self.session.get_providers()[0]
        self.cpu_session = None
        self.native_session = None
        # A larger RVM (e.g. ResNet50) can matte the native photo while the live
        # preview keeps the fast model; it runs as the native pass itself.
        self.native_model_path = str(still_rvm_model) if still_rvm_model else self.model_path
        self.native_model_name = Path(still_rvm_model).stem if still_rvm_model else None
        self.native_input_types = self.input_types
        # "auto": the native pass follows the live session onto CUDA/DirectML. DirectML cannot expand the 1x1
        # recurrent placeholder, so the first warm-up step is seeded on the CPU; any GPU failure moves photos to CPU.
        self.native_gpu = native_device == "auto" and self.primary_provider in ("CUDAExecutionProvider", "DmlExecutionProvider")
        self.native_seed = None
        self.still_threads = still_threads or self.session_options.intra_op_num_threads
        self.downsample_ratio = np.asarray([downsample_ratio], dtype=np.float32)
        self.still_ratio = np.asarray([still_ratio], dtype=np.float32)
        # Bound the encoder's working resolution for high-MP stills. The guided
        # refinement, foreground and alpha output retain the original resolution.
        self.still_max_edge = still_max_edge
        # RVM is recurrent: from a zero state a single frame leaves gaps between
        # people and see-through clothing. Encoder-sized passes fill the state.
        self.still_warmup = still_warmup
        self.still_engine = None
        self.still_deadline_ms = 5000
        self.last_still = None
        self.foreground_jpeg = None
        self.rec = [np.zeros((1, 1, 1, 1), dtype=self.tensor_dtype) for _ in range(4)]
        self.rec_initialized = False

    def reset(self):
        self.rec = [self.np.zeros((1, 1, 1, 1), dtype=self.tensor_dtype) for _ in range(4)]
        self.rec_initialized = False
        self.foreground_jpeg = None
        if self.native_session is not None and not self.native_gpu:
            # Run a tiny empty-person probe after native outputs have been
            # consumed. Shrink only the native arena, leaving live latency and
            # loaded model weights intact between people.
            trim = self.ort.RunOptions()
            trim.only_execute_path_to_fetches = True
            trim.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
            self.native_session.run(["pha"], self._feeds(self.np.zeros((1, 3, 128, 128), dtype=self.tensor_dtype), self.rec, self.still_ratio, self.native_input_types), trim)

    def _cpu(self):
        if self.cpu_session is None:
            self.cpu_session = self.ort.InferenceSession(self.model_path, sess_options=self.session_options, providers=["CPUExecutionProvider"])
        return self.cpu_session

    def _native(self):
        if self.primary_provider == "CUDAExecutionProvider" and self.native_model_name is None:
            return self.session
        if self.native_session is None and self.native_gpu:
            options = self.ort.SessionOptions()
            options.intra_op_num_threads = self.still_threads
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            providers = with_adapter(self.ort, [self.primary_provider, "CPUExecutionProvider"])
            self.native_session = self.ort.InferenceSession(self.native_model_path, sess_options=options, providers=providers)
            self.native_input_types = {item.name: item.type for item in self.native_session.get_inputs()}
            if self.primary_provider == "DmlExecutionProvider":
                self.native_seed = self.ort.InferenceSession(self.native_model_path, sess_options=self.session_options, providers=["CPUExecutionProvider"])
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
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.primary_provider == "CUDAExecutionProvider" else ["CPUExecutionProvider"]
            self.native_session = self.ort.InferenceSession(self.native_model_path, sess_options=options, providers=providers)
            self.native_input_types = {item.name: item.type for item in self.native_session.get_inputs()}
        return self.native_session

    def _feeds(self, src, rec, ratio, types=None):
        feeds = {"src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3], "downsample_ratio": ratio}
        types = types or self.input_types
        for name, value in feeds.items():
            dtype = self.np.float16 if types.get(name) == "tensor(float16)" else self.np.float32
            if value.dtype != dtype:
                feeds[name] = value.astype(dtype, copy=False)
        return feeds

    def _warm_still(self, session, rgb, ratio):
        """Recurrent state for a native still: the photo at the encoder's working
        size, run at ratio 1 so the state matches the full-resolution pass."""
        rec = [self.np.zeros((1, 1, 1, 1), dtype=self.tensor_dtype) for _ in range(4)]
        if not self.still_warmup and self.native_seed is None:
            return rec
        height, width = rgb.shape[:2]
        size = (max(1, int(width * float(ratio[0]))), max(1, int(height * float(ratio[0]))))
        small = self.np.asarray(self.Image.fromarray(rgb).resize(size, self.Image.Resampling.BILINEAR))
        src = self.np.empty((1, 3, size[1], size[0]), dtype=self.tensor_dtype)
        src[0] = small.transpose(2, 0, 1)
        src *= 1.0 / 255.0
        one = self.np.asarray([1.0], dtype=self.np.float32)
        steps = self.still_warmup
        if self.native_seed is not None and session is self.native_session:
            rec = self.native_seed.run(["r1o", "r2o", "r3o", "r4o"], self._feeds(src, rec, one, self.native_input_types), self.run_options)
            steps -= 1
        for _ in range(steps):
            rec = session.run(["r1o", "r2o", "r3o", "r4o"], self._feeds(src, rec, one, self._types(session)), self.run_options)
        return rec

    def _native_on_cpu(self, error):
        """A GPU photo pass failed: this and every later photo run on the CPU."""
        self.native_gpu = False; self.native_session = None; self.native_seed = None
        print(json.dumps({"event": "provider-change", "scope": "photo", "provider": "CPUExecutionProvider", "reason": f"GPU photo pass failed: {str(error)[:160]}"}), file=sys.stderr, flush=True)
        return self._native()

    def native_provider(self):
        if self.native_gpu: return self.primary_provider
        return "CUDAExecutionProvider" if self.primary_provider == "CUDAExecutionProvider" and self.native_model_name is None else "CPUExecutionProvider"

    def _types(self, session):
        return self.native_input_types if session is self.native_session else self.input_types

    def _run(self, session, src, rec, ratio=None):
        feeds = self._feeds(src, rec, self.downsample_ratio if ratio is None else ratio, self._types(session))
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
        if values.dtype == self.np.float16:
            values = values.astype(self.np.float32)
        self.np.clip(values, 0, 1, out=values)
        self.np.multiply(values, 255.0, out=values)
        self.np.rint(values, out=values)
        return values.astype(self.np.uint8)

    def _still_alpha(self, job, width: int, height: int, started: float):
        """Native alpha from the still model, or None to keep RVM's result."""
        engine = self.still_engine
        report = {"model": engine.name, "provider": engine.provider, "deadlineMs": self.still_deadline_ms, "used": "rvm"}
        self.last_still = report
        if job is None:
            report["fallback"] = engine.status() or "unavailable"
            return None
        remaining = self.still_deadline_ms / 1000 - (time.perf_counter() - started)
        if not job.done.wait(max(0.0, remaining)):
            job.cancel()
            report["fallback"] = "timeout"
            return None
        if job.error is not None:
            report.update(fallback="error", error=job.error)
            return None
        try:
            alpha = engine.alpha(job, width, height)
        except Exception as error:
            report.update(fallback="error", error=str(error)[:300])
            return None
        report.update(used=engine.name, inferenceMs=job.inference_ms)
        return alpha

    def process(self, payload: bytes, native: bool = False) -> tuple[bytes, int, int]:
        image = self.Image.open(io.BytesIO(payload)).convert("RGB")
        if not native and (image.width > 1280 or image.height > 720):
            scale = min(1280 / image.width, 720 / image.height)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), self.Image.Resampling.BILINEAR)
        width, height = image.size
        rgb = self.np.asarray(image)
        src = self.np.empty((1, 3, height, width), dtype=self.tensor_dtype)
        src[0] = rgb.transpose(2, 0, 1)
        src *= 1.0 / 255.0
        image.close()
        self.foreground_jpeg = None
        self.last_still = None
        if native:
            started = time.perf_counter()
            job = self.still_engine.start(rgb) if self.still_engine else None
            # Keep the fast live session intact. Native stills use their own
            # session (CPU, or the GPU with a CPU-seeded recurrence) so their
            # different resolution cannot crash the DirectML Expand nodes.
            try:
                session = self._native()
            except Exception as error:
                session = self._native_on_cpu(error)
            ratio = self.np.asarray([min(float(self.still_ratio[0]), self.still_max_edge / max(width, height))], dtype=self.np.float32) if self.still_max_edge else self.still_ratio
            try:
                outputs = self._run(session, src, self._warm_still(session, rgb, ratio), ratio)
            except Exception as error:
                if self.native_gpu:
                    # The GPU pass failed: this photo is redone on the CPU.
                    session = self._native_on_cpu(error)
                    outputs = self._run(session, src, self._warm_still(session, rgb, ratio), ratio)
                elif not self.still_warmup:
                    raise
                else:
                    print(json.dumps({"event": "still-warmup-skipped", "error": str(error)[:200]}), file=sys.stderr, flush=True)
                    outputs = self._run(session, src, [self.np.zeros((1, 1, 1, 1), dtype=self.tensor_dtype) for _ in range(4)], ratio)
            del src
            if self.native_model_name is not None:
                self.last_still = {"model": self.native_model_name, "provider": session.get_providers()[0], "used": self.native_model_name, "totalMs": round((time.perf_counter() - started) * 1000)}
            if self.still_engine is not None:
                alpha = self._still_alpha(job, width, height, started)
                if alpha is not None:
                    outputs = None
                    try:
                        foreground = estimate_foreground(self.np, self.Image, rgb, alpha)
                    except Exception as error:
                        # RVM's foreground is already released; keep the model's
                        # alpha with the camera's own colours.
                        foreground = rgb
                        self.last_still["foreground"] = f"camera ({str(error)[:120]})"
                    encoded = io.BytesIO()
                    self.Image.fromarray(foreground).save(encoded, "JPEG", quality=97, subsampling=0)
                    self.foreground_jpeg = encoded.getvalue()
                    self.last_still["totalMs"] = round((time.perf_counter() - started) * 1000)
                    return alpha.tobytes(), width, height
            del rgb
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
                print(json.dumps({"event":"provider-change","provider":self.primary_provider,"reason":"DirectML inference fallback"}),file=sys.stderr,flush=True)
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
        matte = getattr(engine, "last_still", None) if native_still else None
        if matte:
            emit("still-matte", sequence=frame["sequence"], **matte)
            if metadata is not None:
                metadata = {**metadata, "matte": matte}
            engine.last_still = None
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


def check_still_model(model_path: Path, device_mode: str, threads: int) -> int:
    """Load a still model and time one full native-size matte (24 MP, 3:2)."""
    import numpy as np
    from PIL import Image

    engine = StillMatteEngine(model_path, device_mode, select_onnx_providers, threads, block=True)
    if engine.session is None:
        raise RuntimeError(engine.error or "Still model failed to load")
    width, height = 6000, 4000
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[...] = (80, 100, 120)
    rgb[height // 4: height, width // 3: 2 * width // 3] = (200, 160, 140)
    started = time.perf_counter()
    job = engine.start(rgb)
    job.done.wait()
    if job.error:
        raise RuntimeError(job.error)
    alpha = engine.alpha(job, width, height)
    if alpha.shape != (height, width):
        raise RuntimeError("Still model returned an invalid alpha size")
    foreground = estimate_foreground(np, Image, rgb, alpha)
    Image.fromarray(foreground).save(io.BytesIO(), "JPEG", quality=97, subsampling=0)
    print(json.dumps({
        "verified": True, "provider": engine.provider, "warmupMs": engine.warmup_ms, "probeMs": job.inference_ms, "totalMs": round((time.perf_counter() - started) * 1000),
        "input": list(engine.size), "width": width, "height": height,
    }))
    return 0


def benchmark(engine, selector, size: str) -> dict:
    """Times the configured pipeline on synthetic frames: live mask, photo mask at the camera's photo size, and
    person scoring. Content does not change these timings, so no camera or customer image is needed."""
    import numpy as np
    from PIL import Image
    width, height = (int(v) for v in size.lower().split("x"))
    def jpeg(w, h):
        rng = np.random.default_rng(7)
        y, x = np.mgrid[0:h, 0:w]
        base = np.stack([x * 255 // max(1, w - 1), y * 255 // max(1, h - 1), (x + y) * 127 // max(1, w + h)], -1).astype(np.int16)
        pixels = np.clip(base + rng.integers(-20, 20, base.shape), 0, 255).astype(np.uint8)
        buffer = io.BytesIO(); Image.fromarray(pixels).save(buffer, "JPEG", quality=92); return buffer.getvalue()
    def timed(action):
        started = time.perf_counter(); action(); return (time.perf_counter() - started) * 1000
    result = {"liveProvider": getattr(engine, "primary_provider", None), "photoProvider": engine.native_provider() if isinstance(engine, OnnxRvmEngine) else None,
              "photoModel": getattr(engine, "native_model_name", None), "photoSize": [width, height]}
    live = jpeg(1280, 720)
    for _ in range(5): engine.process(live)
    samples = sorted(timed(lambda: engine.process(live)) for _ in range(40))
    result["live"] = {"meanMs": round(sum(samples) / len(samples), 1), "p95Ms": round(samples[int(len(samples) * .95) - 1], 1), "fps": round(1000 / (sum(samples) / len(samples)), 1)}
    engine.reset()
    still = getattr(engine, "still_engine", None)
    if still is not None: still.ready.wait(120)
    photo = jpeg(width, height); runs = []
    for _ in range(4):
        runs.append(timed(lambda: engine.process(photo, native=True))); engine.reset()
    result["photo"] = {"firstSeconds": round(runs[0] / 1000, 2), "seconds": round(sorted(runs[1:])[1] / 1000, 2), "runsSeconds": [round(r / 1000, 2) for r in runs]}
    if still is not None: result["photo"]["stillModel"] = {"name": still.name, "provider": still.provider, "report": getattr(engine, "last_still", None), "error": still.error}
    if isinstance(engine, OnnxRvmEngine): result["photoProvider"] = engine.native_provider()
    if selector is not None:
        image = Image.open(io.BytesIO(photo)).convert("RGB")
        detect = sorted(timed(lambda: selector.detector.detect(image, True)) for _ in range(3))[1]
        entry = {"detectMs": round(detect)}
        if selector.scorer is not None:
            entry["scoreMs"] = round(sorted(timed(lambda: selector.scorer.observe(image, 364)) for _ in range(3))[1])
        result["subjects"] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("RVM_MODEL_PATH", ""))
    parser.add_argument("--subject-model", default="")
    parser.add_argument("--depth-model", default="")
    parser.add_argument("--face-model", default="")
    parser.add_argument("--device", choices=("auto", "cuda", "directml", "cpu"), default=os.environ.get("RVM_DEVICE", "auto"))
    parser.add_argument("--downsample-ratio", type=float, default=0.375)
    parser.add_argument("--still-ratio", type=float, default=0.5)
    parser.add_argument("--still-max-edge", type=int, choices=(0, 512, 768, 1024), default=1024)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--still-threads", type=int, default=0)
    parser.add_argument("--still-warmup", type=int, choices=range(0, 9), default=4)
    parser.add_argument("--still-model", default="")
    parser.add_argument("--still-rvm-model", default="")
    parser.add_argument("--native-device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--still-device", choices=("auto", "cuda", "directml", "cpu"), default="auto")
    parser.add_argument("--still-deadline-ms", type=int, default=5000)
    parser.add_argument("--check-still-model", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--startup-id", default="")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--benchmark", default="", help="WIDTHxHEIGHT of the camera photo: time the configured pipeline and exit")
    parser.add_argument("--inspect-runtime", action="store_true")
    args = parser.parse_args()
    if args.inspect_runtime:
        import onnxruntime as ort
        print(json.dumps({"providers":ort.get_available_providers(),"onnxruntime":ort.__version__,"devices":gpu_devices(ort),"dmlAdapter":dml_adapter(ort)[1] if "DmlExecutionProvider" in ort.get_available_providers() else None}))
        return 0
    if not 0 < args.downsample_ratio <= 1 or not 0 < args.still_ratio <= 1 or not 0 <= args.threads <= (os.cpu_count() or 1) or not 0 <= args.still_threads <= (os.cpu_count() or 1):
        parser.error("Ratios must be in (0, 1]; threads must be 0 (automatic) or an available processor count")
    if not 500 <= args.still_deadline_ms <= 30000:
        parser.error("--still-deadline-ms must be between 500 and 30000")
    if args.self_test: return self_test()
    if args.check_still_model:
        return check_still_model(Path(args.still_model).resolve(), args.still_device, args.threads)
    model_path = Path(args.model).resolve()
    engine = FixtureEngine() if args.fixture else OnnxRvmEngine(model_path, args.device, args.downsample_ratio, args.still_ratio, args.threads, args.still_max_edge, args.still_threads, args.still_warmup, Path(args.still_rvm_model).resolve() if args.still_rvm_model else None, args.native_device) if model_path.suffix.lower() == ".onnx" else RvmEngine(model_path, args.device, args.downsample_ratio)
    if not args.fixture:
        from PIL import Image
        sample = io.BytesIO()
        Image.new("RGB", (128, 128), (80, 100, 120)).save(sample, format="JPEG")
        mask, width, height = engine.process(sample.getvalue())
        if width != 128 or height != 128 or len(mask) != width * height:
            raise RuntimeError("RVM startup inference returned invalid dimensions")
        engine.reset()
        if isinstance(engine, OnnxRvmEngine) and engine.native_gpu:
            # Compile and verify the GPU photo pass now, not on the first customer photo.
            sample = io.BytesIO(); Image.new("RGB", (320, 180), (80, 100, 120)).save(sample, format="JPEG")
            engine.process(sample.getvalue(), native=True); engine.reset()
        if args.check_model:
            started = time.perf_counter()
            sample = io.BytesIO()
            Image.new("RGB", (1280,720), (80,100,120)).save(sample,format="JPEG")
            live, width, height = engine.process(sample.getvalue())
            if (width,height)!=(1280,720) or len(live)!=width*height:
                raise RuntimeError("Model live validation failed")
            native, width, height = engine.process(sample.getvalue(),native=True)
            if len(native)!=width*height or not engine.foreground_jpeg:
                raise RuntimeError("Model native validation failed")
            engine.reset()
            print(json.dumps({"verified":True,"provider":engine.primary_provider,"precision":"fp16" if engine.input_types["src"]=="tensor(float16)" else "fp32","probeMs":round((time.perf_counter()-started)*1000),"width":width,"height":height,"providers":engine.ort.get_available_providers()}))
            return 0
    if args.still_model and isinstance(engine, OnnxRvmEngine):
        engine.still_engine = StillMatteEngine(Path(args.still_model).resolve(), args.still_device, select_onnx_providers, args.still_threads)
        engine.still_deadline_ms = args.still_deadline_ms
    selector = None
    if args.subject_model:
        from subjects import PersonDetector, SceneScorer, SubjectSelector
        scorer = SceneScorer(args.depth_model, args.face_model, args.threads or 2) if args.depth_model and args.face_model else None
        selector = SubjectSelector(PersonDetector(args.subject_model, args.threads or 2), scorer=scorer)
        selector.detector.detect(Image.new("RGB", (128, 128)))
        if scorer: scorer.observe(Image.new("RGB", (128, 72)), 252)
    import onnxruntime as _ort
    if args.benchmark:
        started = time.perf_counter()
        report = benchmark(engine, selector, args.benchmark)
        report["totalSeconds"] = round(time.perf_counter() - started, 1)
        print(json.dumps({"benchmark": report}), flush=True)
        return 0
    print(json.dumps({"event": "rvm-ready", "protocol": 1, "startupId": args.startup_id,
                      "inferenceVerified": not args.fixture, "provider": getattr(engine, "primary_provider", args.device),
                      "photoProvider": engine.native_provider() if isinstance(engine, OnnxRvmEngine) else None,
                      "adapter": dml_adapter(_ort)[1] if getattr(engine, "primary_provider", "") == "DmlExecutionProvider" else None}), file=sys.stderr, flush=True)
    if args.check_runtime: return 0
    return serve(engine, selector=selector)


if __name__ == "__main__": raise SystemExit(main())
