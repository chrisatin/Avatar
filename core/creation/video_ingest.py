from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.common.logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class FrameBundle:
    frames: list[np.ndarray]
    paths: list[Path]
    fps: float
    source: str


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffprobe_duration(path: Path) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def extract_video_frames(
    video_path: str | Path,
    out_dir: str | Path | None = None,
    sample_fps: float = 25.0,
) -> FrameBundle:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    frames: list[np.ndarray] = []
    paths: list[Path] = []
    fps = sample_fps
    if _ffmpeg_available():
        out_dir = Path(out_dir) if out_dir else video_path.parent / f"{video_path.stem}_frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        duration = _ffprobe_duration(video_path)
        if duration > 0:
            fps = min(sample_fps, max(1.0, 240.0 / max(duration, 0.1)))
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-vf", f"fps={fps}",
                str(out_dir / "frame_%05d.png"),
            ],
            check=True,
        )
        paths = sorted(out_dir.glob("frame_*.png"))
        frames = [_read_image(p) for p in paths]
    else:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
        step = max(1, int(round(native_fps / sample_fps)))
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(img)
            idx += 1
        cap.release()
        fps = native_fps / step if step else sample_fps
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return FrameBundle(frames=frames, paths=paths, fps=fps, source=str(video_path))


def load_frames_from_dir(frames_dir: str | Path, sample_fps: float = 25.0) -> FrameBundle:
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)
    paths = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise ValueError(f"no image frames found in {frames_dir}")
    frames = [_read_image(p) for p in paths]
    return FrameBundle(frames=frames, paths=paths, fps=sample_fps, source=str(frames_dir))


def load_source(
    source: str | Path,
    workdir: str | Path,
    sample_fps: float = 25.0,
) -> FrameBundle:
    source = Path(source)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        return load_frames_from_dir(source, sample_fps=sample_fps)
    return extract_video_frames(source, out_dir=workdir / "frames", sample_fps=sample_fps)


def _read_image(path: Path) -> np.ndarray:
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except ImportError:
        pass
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
