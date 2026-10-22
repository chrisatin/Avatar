from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.common.config import FACE_LANDMARKER_PATH
from core.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrackResult:
    present: bool = False
    blendshapes: dict[str, float] = field(default_factory=dict)
    landmarks: list[list[float]] = field(default_factory=list)
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    inference_ms: float = 0.0


class BaseTracker:
    name = "base"

    def process(self, image_bgr: np.ndarray) -> TrackResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _preload_headless_gl() -> None:
    """Make a user-local GLES/EGL install visible to MediaPipe's dlopen.

    ``LD_LIBRARY_PATH`` is read by the dynamic linker at process start, so
    exporting it after Python is running is not enough. Preloading the
    libraries with ``RTLD_GLOBAL`` satisfies MediaPipe's ``DT_NEEDED`` entries
    for the CLI/benchmark paths that don't go through ``run_demo.sh``.
    """
    import ctypes
    import ctypes.util
    import sys

    if not sys.platform.startswith("linux"):
        return
    try:
        if ctypes.util.find_library("GLESv2"):
            return
    except Exception:
        pass
    prefix = Path(
        os.environ.get("GL_PREFIX", str(Path.home() / ".local" / "gl-libs"))
    ) / "usr" / "lib" / "x86_64-linux-gnu"
    if not prefix.is_dir():
        return
    for name in (
        "libGLdispatch.so.0",
        "libGLX.so.0",
        "libGL.so.1",
        "libEGL.so.1",
        "libGLESv2.so.2",
    ):
        lib = prefix / name
        if not lib.exists():
            continue
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            logger.debug("could not preload %s: %s", lib, exc)


class MediaPipeTracker(BaseTracker):
    name = "mediapipe"

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ) -> None:
        _preload_headless_gl()
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions as BO
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        model_path = Path(model_path or FACE_LANDMARKER_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"face_landmarker.task not found at {model_path}; "
                "run: python scripts/download_models.py"
            )
        options = FaceLandmarkerOptions(
            base_options=BO(model_asset_path=str(model_path)),
            running_mode=RunningMode[running_mode],
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._last_result: Any = None

    def process(self, image_bgr: np.ndarray) -> TrackResult:
        import mediapipe as mp

        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._last_result = result
        if not result.face_landmarks:
            return TrackResult(inference_ms=elapsed)
        face = result.face_landmarks[0]
        h, w = image_bgr.shape[:2]
        landmarks = [[lm.x * w, lm.y * h, lm.z * w] for lm in face]
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                name = cat.category_name
                if name and name != "_neutral":
                    blendshapes[name] = float(cat.score)
        rotation = [1.0, 0.0, 0.0, 0.0]
        translation = [0.0, 0.0, 0.0]
        if result.facial_transformation_matrixes:
            m = np.asarray(result.facial_transformation_matrixes[0].data, dtype=np.float64)
            rotation = _mat_to_quat(m.reshape(4, 4)[:3, :3]).tolist()
            translation = m.reshape(4, 4)[:3, 3].tolist()
        return TrackResult(
            present=True,
            blendshapes=blendshapes,
            landmarks=landmarks,
            rotation=rotation,
            translation=translation,
            inference_ms=elapsed,
        )

    def close(self) -> None:
        self._landmarker.close()


class SyntheticTracker(BaseTracker):
    name = "synthetic"

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self._i = 0

    def process(self, image_bgr: np.ndarray) -> TrackResult:
        t0 = time.perf_counter()
        t = self._i / self.fps
        self._i += 1
        blink = max(0.0, math.sin(t * 2.0 * math.pi * 0.35) ** 12)
        w = {
            "jawOpen": 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * 0.8),
            "mouthSmileLeft": 0.4 + 0.3 * math.sin(t * 2.0 * math.pi * 0.25),
            "mouthSmileRight": 0.4 + 0.3 * math.sin(t * 2.0 * math.pi * 0.25),
            "browInnerUp": 0.3 + 0.2 * math.sin(t * 2.0 * math.pi * 0.15),
            "eyeBlinkLeft": blink,
            "eyeBlinkRight": blink,
            "eyeLookInLeft": 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * 0.1),
            "eyeLookInRight": 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * 0.1),
            "noseSneerLeft": 0.2 * abs(math.sin(t * 2.0 * math.pi * 0.12)),
            "noseSneerRight": 0.2 * abs(math.sin(t * 2.0 * math.pi * 0.12)),
        }
        elapsed = (time.perf_counter() - t0) * 1000.0
        return TrackResult(
            present=True,
            blendshapes=w,
            landmarks=_synthetic_landmarks(w),
            rotation=_mat_to_quat(_euler_to_matrix(0.0, math.sin(t) * 0.25, math.sin(t * 0.5) * 0.1)).tolist(),
            translation=[0.0, 0.0, math.sin(t * 0.7) * 0.02],
            inference_ms=elapsed,
        )


def create_tracker(backend: str = "auto", **kwargs: Any) -> BaseTracker:
    backend = backend.lower()
    if backend == "synthetic":
        return SyntheticTracker(**kwargs)
    if backend == "mediapipe":
        return MediaPipeTracker(**kwargs)
    if backend == "auto":
        try:
            return MediaPipeTracker(**kwargs)
        except Exception as exc:
            logger.warning("MediaPipe unavailable (%s); falling back to synthetic tracker", exc)
            return SyntheticTracker()
    raise ValueError(f"unknown tracker backend: {backend}")


def load_w_matrix(path: str | Path | None = None) -> np.ndarray | None:
    from core.common.config import MAT_FINAL_PATH

    p = Path(path or MAT_FINAL_PATH)
    if not p.exists():
        return None
    m = np.load(p).astype(np.float32)
    if m.shape[0] != 51 and m.shape[1] == 51:
        m = m.T
    if m.shape[0] != 51:
        logger.warning("unexpected W matrix shape %s", m.shape)
        return None
    return m


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) or 1.0)


def _euler_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


def _synthetic_landmarks(weights: dict[str, float]) -> list[list[float]]:
    jaw = weights.get("jawOpen", 0.0)
    smile = 0.5 * (weights.get("mouthSmileLeft", 0.0) + weights.get("mouthSmileRight", 0.0))
    pts: list[list[float]] = []
    for i in range(478):
        u = (i % 24) / 23.0
        v = (i // 24 % 20) / 19.0
        x = (u - 0.5) * 0.14
        y = (0.5 - v) * 0.18
        z = 0.02 * math.cos(u * math.pi) * math.cos(v * math.pi)
        if 0.35 < u < 0.65 and v < 0.35:
            y += jaw * 0.03 * (1.0 - abs(u - 0.5) / 0.15)
        if 0.3 < u < 0.7 and 0.45 < v < 0.55:
            x *= 1.0 + smile * 0.1
        pts.append([x, y, z])
    return pts
