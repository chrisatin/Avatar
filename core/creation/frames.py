from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(eq=False)
class ScoredFrame:
    index: int
    image: np.ndarray
    sharpness: float
    brightness: float
    yaw: float
    score: float


def sharpness(img: np.ndarray) -> float:
    gray = img if img.ndim == 2 else img.mean(axis=2)
    gy, gx = np.gradient(gray.astype(np.float32))
    return float(np.sqrt(gx.var() + gy.var()))


def brightness(img: np.ndarray) -> float:
    return float(img.mean())


def estimate_yaw(landmarks: np.ndarray, width: int) -> float:
    if landmarks.size == 0:
        return 0.0
    nose = landmarks[1]
    left = landmarks[234]
    right = landmarks[454]
    center = 0.5 * (left[0] + right[0])
    span = max(abs(right[0] - left[0]), 1e-6)
    return float((nose[0] - center) / span)


def face_bbox(landmarks: np.ndarray, margin: float = 0.35) -> tuple[int, int, int, int]:
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = x1 - x0
    h = y1 - y0
    x0 -= w * margin
    x1 += w * margin
    y0 -= h * margin * 1.2
    y1 += h * margin * 0.8
    return int(max(0, x0)), int(max(0, y0)), int(x1), int(y1)


def crop_face(
    img: np.ndarray,
    landmarks: np.ndarray | None = None,
    margin: float = 0.35,
    min_size: int = 96,
) -> np.ndarray:
    crop, _, _ = crop_face_with_offset(img, landmarks, margin, min_size)
    return crop


def crop_face_with_offset(
    img: np.ndarray,
    landmarks: np.ndarray | None = None,
    margin: float = 0.35,
    min_size: int = 96,
) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    if landmarks is not None and landmarks.size:
        x0, y0, x1, y1 = face_bbox(landmarks, margin=margin)
    else:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        x1, y1 = x0 + side, y0 + side
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, max(x1, x0 + min_size))
    y1 = min(h, max(y1, y0 + min_size))
    return img[y0:y1, x0:x1].copy(), x0, y0


def select_keyframes(
    frames: list[np.ndarray],
    landmarks_per_frame: list[np.ndarray | None],
    target: int,
    min_frames: int = 6,
    max_yaw: float = 0.45,
    min_brightness: float = 25.0,
) -> list[ScoredFrame]:
    scored: list[ScoredFrame] = []
    for i, (img, lms) in enumerate(zip(frames, landmarks_per_frame)):
        s = sharpness(img)
        b = brightness(img)
        yaw = estimate_yaw(lms, img.shape[1]) if lms is not None and lms.size else 0.0
        yaw_penalty = max(0.0, abs(yaw) - max_yaw) * 3.0
        score = s / 100.0 - yaw_penalty - (0.0 if b > min_brightness else 2.0)
        scored.append(ScoredFrame(i, img, s, b, yaw, score))
    scored.sort(key=lambda f: f.score, reverse=True)
    top = scored[: max(target * 2, min_frames * 2)]
    top.sort(key=lambda f: f.index)
    picked: list[ScoredFrame] = []
    min_gap = max(1, len(frames) // max(target * 2, 1))
    last = -10**9
    for f in top:
        if f.index - last >= min_gap or not picked:
            picked.append(f)
            last = f.index
        if len(picked) >= target:
            break
    if len(picked) < min_frames:
        picked_idx = {f.index for f in picked}
        for f in top:
            if f.index not in picked_idx:
                picked.append(f)
                picked_idx.add(f.index)
            if len(picked) >= min_frames:
                break
    picked.sort(key=lambda f: f.index)
    logger.info("selected %d/%d keyframes", len(picked), len(frames))
    return picked


def save_keyframes(scored: list[ScoredFrame], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import cv2

        for f in scored:
            p = out_dir / f"key_{f.index:05d}.png"
            cv2.imwrite(str(p), f.image)
            paths.append(p)
    except ImportError:
        from PIL import Image

        for f in scored:
            p = out_dir / f"key_{f.index:05d}.png"
            Image.fromarray(f.image[:, :, ::-1]).save(p)
            paths.append(p)
    return paths
