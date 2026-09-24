"""High-quality alpha matting for native photographs.

The live preview keeps using RVM. A still model (BiRefNet ONNX export) runs in
parallel with RVM's own native pass. When it misses its deadline or fails, the
RVM result is used unchanged, so a capture never waits longer than the deadline
and never fails because of this model.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FOREGROUND_EDGE = 1024


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}), file=sys.stderr, flush=True)


def box_mean(np, values, radius: int):
    """Edge-normalised box mean over a (2r+1)^2 window (separable running sums)."""
    for axis in (0, 1):
        length = values.shape[axis]
        index = np.arange(length)
        low, high = np.clip(index - radius, 0, length), np.clip(index + radius + 1, 0, length)
        shape = [1] * values.ndim
        shape[axis] = length
        sums = np.cumsum(values, axis=axis, dtype=np.float32)
        sums = np.concatenate([np.zeros_like(sums.take([0], axis=axis)), sums], axis=axis)
        values = sums.take(high, axis=axis) - sums.take(low, axis=axis)
        values /= (high - low).reshape(shape).astype(np.float32)
    return values


def blur_fusion(np, image, foreground, background, alpha, radius: int):
    """One pass of Forte & Pitie's approximate fast foreground colour estimation."""
    a = alpha[..., None]
    blurred_alpha = box_mean(np, alpha, radius)[..., None]
    blurred_fg = box_mean(np, foreground * a, radius) / (blurred_alpha + 1e-5)
    blurred_bg = box_mean(np, background * (1 - a), radius) / ((1 - blurred_alpha) + 1e-5)
    estimate = np.clip(blurred_fg + a * (image - a * blurred_fg - (1 - a) * blurred_bg), 0, 1)
    return estimate, blurred_fg, blurred_bg


def estimate_foreground(np, Image, rgb, alpha):
    """Return native RGB with background colour removed from semi-transparent pixels.

    Colour fields are estimated at a bounded resolution; only the native edge band
    (0 < alpha < 1) is rewritten, opaque pixels keep the camera's original detail.
    """
    height, width = alpha.shape
    band = (alpha > 2) & (alpha < 253)
    if not band.any():
        return rgb
    scale = min(1.0, FOREGROUND_EDGE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    small_alpha = np.asarray(Image.fromarray(alpha).resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    first, _, first_bg = blur_fusion(np, small, small, small, small_alpha, 90)
    _, fg_field, bg_field = blur_fusion(np, small, first, first_bg, small_alpha, 6)
    ys, xs = np.nonzero(band)
    sy = np.minimum((ys * size[1]) // height, size[1] - 1)
    sx = np.minimum((xs * size[0]) // width, size[0] - 1)
    a = (alpha[ys, xs].astype(np.float32) / 255.0)[:, None]
    observed = rgb[ys, xs].astype(np.float32) / 255.0
    fg, bg = fg_field[sy, sx], bg_field[sy, sx]
    corrected = np.clip(fg + a * (observed - a * fg - (1 - a) * bg), 0, 1)
    output = rgb.copy()
    output[ys, xs] = np.rint(corrected * 255.0).astype(np.uint8)
    return output


class StillJob:
    def __init__(self, ort):
        self.done = threading.Event()
        self.options = ort.RunOptions()
        self.result = None
        self.error = None
        self.inference_ms = None

    def cancel(self) -> None:
        self.options.terminate = True


class StillMatteEngine:
    """Owns one still-model session; loads and warms it off the capture path."""

    def __init__(self, model_path, device_mode: str = "auto", select_providers=None, threads: int = 0, block: bool = False):
        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        self.np, self.ort, self.Image = np, ort, Image
        self.model_path = Path(model_path)
        self.name = self.model_path.stem
        self.device_mode = device_mode
        self.select_providers = select_providers
        self.threads = threads
        self.session = None
        self.provider = None
        self.error = None
        self.warmup_ms = None
        self.busy = False
        self.ready = threading.Event()
        if block:
            self._load()
        else:
            threading.Thread(target=self._load, name="still-matte-load", daemon=True).start()

    def _adapter(self) -> int:
        """DXGI index of the high-performance GPU; DirectML defaults to adapter 0,
        which is the integrated GPU on hybrid systems."""
        try:
            for device in self.ort.get_ep_devices():
                metadata = dict(device.device.metadata)
                if device.ep_name == "DmlExecutionProvider" and metadata.get("DxgiHighPerformanceIndex") == "0":
                    return int(metadata.get("DxgiAdapterNumber", 0))
        except Exception:
            pass
        return 0

    def _session(self, providers):
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(self.threads or os.cpu_count() or 1, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if providers[0] == "DmlExecutionProvider":
            options.enable_mem_pattern = False
            # Fused BiRefNet graphs exhaust GPU memory (E_OUTOFMEMORY even with
            # 6 GB) and run ~5x slower than unfused FP16 kernels.
            options.add_session_config_entry("ep.dml.disable_graph_fusion", "1")
            providers = [("DmlExecutionProvider", {"device_id": self._adapter()}), *providers[1:]]
        return self.ort.InferenceSession(str(self.model_path), sess_options=options, providers=providers)

    def _load(self) -> None:
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Still model not found: {self.model_path}")
            providers = self.select_providers(self.ort.get_available_providers(), self.device_mode) if self.select_providers else ["CPUExecutionProvider"]
            if self.device_mode == "auto" and providers[0] == "CPUExecutionProvider":
                # A CPU run takes >10 s and would only slow RVM's own CPU pass.
                raise RuntimeError("No GPU execution provider is available")
            session = self._session(providers)
            source = session.get_inputs()[0]
            height, width = source.shape[2], source.shape[3]
            self.size = (width if isinstance(width, int) and width > 0 else 1024, height if isinstance(height, int) and height > 0 else 1024)
            self.input_name = source.name
            self.dtype = self.np.float16 if source.type == "tensor(float16)" else self.np.float32
            started = time.perf_counter()
            session.run(None, {self.input_name: self.np.zeros((1, 3, self.size[1], self.size[0]), dtype=self.dtype)})
            self.warmup_ms = round((time.perf_counter() - started) * 1000)
            self.session = session
            self.provider = session.get_providers()[0]
            emit("still-matte-ready", model=self.name, provider=self.provider, warmupMs=self.warmup_ms, input=list(self.size))
        except Exception as error:
            self.error = str(error)[:300]
            emit("still-matte-unavailable", model=self.name, reason=self.error)
        finally:
            self.ready.set()

    def status(self) -> str | None:
        """Why a capture cannot use the model right now, or None when it can."""
        if not self.ready.is_set():
            return "loading"
        if self.session is None:
            return "unavailable"
        if self.busy:
            return "busy"
        return None

    def start(self, rgb) -> StillJob | None:
        if self.status():
            return None
        self.busy = True
        job = StillJob(self.ort)
        threading.Thread(target=self._infer, args=(job, rgb), name="still-matte-run", daemon=True).start()
        return job

    def _infer(self, job: StillJob, rgb) -> None:
        np = self.np
        try:
            started = time.perf_counter()
            resized = np.asarray(self.Image.fromarray(rgb).resize(self.size, self.Image.Resampling.BILINEAR), dtype=np.float32)
            resized *= 1.0 / 255.0
            resized -= np.asarray(MEAN, dtype=np.float32)
            resized /= np.asarray(STD, dtype=np.float32)
            tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None]).astype(self.dtype, copy=False)
            # BiRefNet exports list side outputs first; the final prediction is last.
            job.result = self.session.run(None, {self.input_name: tensor}, job.options)[-1]
            job.inference_ms = round((time.perf_counter() - started) * 1000)
        except Exception as error:
            job.error = str(error)[:300]
        finally:
            # An abandoned, terminated run must still finish before the next one.
            self.busy = False
            job.done.set()

    def alpha(self, job: StillJob, width: int, height: int):
        np = self.np
        logits = np.asarray(job.result, dtype=np.float32).reshape(job.result.shape[-2:])
        probability = 1.0 / (1.0 + np.exp(-logits))
        small = np.rint(np.clip(probability, 0, 1) * 255.0).astype(np.uint8)
        return np.asarray(self.Image.fromarray(small).resize((width, height), self.Image.Resampling.BICUBIC))
