from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
FLAME_DIR = MODELS_DIR / "flame"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CLIENT_DIR = REPO_ROOT / "client"
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
MAT_FINAL_URL = (
    "https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools/resolve/main/mat_final.npy"
)
MAT_FINAL_PATH = MODELS_DIR / "mat_final.npy"


@dataclass
class IngestConfig:
    target_frames: int = 24
    min_frames: int = 6
    max_frames: int = 48
    sample_fps: float = 25.0
    face_crop_margin: float = 0.35
    min_face_size: int = 96
    sharpness_threshold: float = 40.0
    remove_background: bool = False


@dataclass
class FitConfig:
    backend: str = "auto"
    num_shape: int = 100
    num_expr: int = 50
    iterations: int = 60
    lr: float = 0.02
    texture_size: int = 1024
    template_subdivisions: int = 32


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 200
    job_workers: int = 2
    tracker_max_fps: float = 30.0
    jpeg_quality: int = 85


@dataclass
class PipelineConfig:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    device: str = "cuda"
    outputs_dir: str = str(OUTPUTS_DIR)
    models_dir: str = str(MODELS_DIR)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    cfg_path = Path(path) if path else CONFIG_PATH
    data: dict[str, Any] = {}
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    device = data.get("device", "cuda" if _cuda_available() else "cpu")
    if os.environ.get("AVATAR_FORCE_CPU"):
        device = "cpu"
    elif device == "cuda" and not _cuda_available():
        device = "cpu"
    return PipelineConfig(
        ingest=IngestConfig(**data.get("ingest", {})),
        fit=FitConfig(**data.get("fit", {})),
        server=ServerConfig(**data.get("server", {})),
        device=device,
        outputs_dir=data.get("outputs_dir", str(OUTPUTS_DIR)),
        models_dir=data.get("models_dir", str(MODELS_DIR)),
    )


def _cuda_available() -> bool:
    if not os.environ.get("AVATAR_FORCE_CPU"):
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False
    return False
