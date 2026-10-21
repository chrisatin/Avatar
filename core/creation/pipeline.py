from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from core.common.config import PipelineConfig, load_config
from core.common.logging_utils import get_logger
from core.creation.blendshapes import bake_flame_arkit_deltas, mesh_stats
from core.creation.export import write_glb
from core.creation.face_track import track_frames
from core.creation.flame_fit import (
    FlameBackendUnavailable,
    fit_flame_sequence,
    flame_assets_present,
)
from core.creation.frames import save_keyframes, select_keyframes
from core.creation.template_head import create_avatar_mesh
from core.creation.texture import neutral_texture, splat_albedo
from core.creation.video_ingest import load_source

logger = get_logger(__name__)

ProgressFn = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    pass


def pick_backend(config: PipelineConfig) -> str:
    backend = config.fit.backend
    if backend != "auto":
        return backend
    if not flame_assets_present():
        return "template"
    try:
        import smplx  # noqa: F401

        return "flame"
    except ImportError:
        logger.warning("smplx not installed; using template backend")
        return "template"


def run_creation_pipeline(
    source: str | Path,
    out_dir: str | Path,
    config: PipelineConfig | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    config = config or load_config()
    progress = progress or _noop
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    progress("ingest", 0.0)
    t0 = time.perf_counter()
    bundle = load_source(source, workdir=out_dir / "work", sample_fps=config.ingest.sample_fps)
    timings["ingest"] = time.perf_counter() - t0
    logger.info("decoded %d frames from %s", len(bundle.frames), bundle.source)

    progress("tracking", 0.05)
    t0 = time.perf_counter()
    tracker = None
    try:
        from core.animation.tracker import MediaPipeTracker

        tracker = MediaPipeTracker()
    except Exception as exc:
        logger.warning("MediaPipe tracker unavailable at creation time: %s", exc)
    tracked = track_frames(
        _to_scored(bundle.frames),
        tracker=tracker,
        crop_margin=config.ingest.face_crop_margin,
    )
    if tracker is not None:
        tracker.close()
    timings["tracking"] = time.perf_counter() - t0

    progress("keyframes", 0.3)
    t0 = time.perf_counter()
    landmarks_list = [t.landmarks for t in tracked]
    scored = [t.scored for t in tracked]
    keyframes = select_keyframes(
        [t.scored.image for t in tracked],
        landmarks_list,
        target=config.ingest.target_frames,
        min_frames=config.ingest.min_frames,
    )
    key_indices = {f.index for f in keyframes}
    key_tracked = [t for t in tracked if t.scored.index in key_indices]
    save_keyframes(keyframes, out_dir / "keyframes")
    timings["keyframes"] = time.perf_counter() - t0

    progress("fitting", 0.4)
    backend = pick_backend(config)
    t0 = time.perf_counter()
    primary_lmk = _primary_landmarks(key_tracked)
    mesh = None
    if backend == "flame":
        try:
            seq = [t.landmarks for t in key_tracked if t.landmarks is not None]
            mesh = fit_flame_sequence(
                seq,
                num_betas=config.fit.num_shape,
                num_expr=config.fit.num_expr,
                iterations=config.fit.iterations,
                lr=config.fit.lr,
                device=config.device if config.device != "cuda" or _cuda() else "cpu",
            )
        except FlameBackendUnavailable as exc:
            logger.warning("FLAME backend unavailable (%s); falling back to template", exc)
            backend = "template"
    if mesh is None:
        mesh = create_avatar_mesh(
            landmarks=primary_lmk,
            subdivisions=config.fit.template_subdivisions,
        )
    timings["fitting"] = time.perf_counter() - t0

    progress("texture", 0.7)
    t0 = time.perf_counter()
    key_lmk = [t.landmarks for t in key_tracked]
    if any(l is not None for l in key_lmk):
        texture = splat_albedo(
            mesh,
            [t.scored for t in key_tracked],
            key_lmk,
            size=config.fit.texture_size,
        )
    else:
        texture = neutral_texture(config.fit.texture_size)
    mesh.texture = texture
    timings["texture"] = time.perf_counter() - t0

    progress("bake", 0.85)
    t0 = time.perf_counter()
    if backend == "flame" and not mesh.blendshape_deltas:
        from core.animation.tracker import load_w_matrix

        w = load_w_matrix()
        if w is not None:
            mesh = bake_flame_arkit_deltas(mesh, w, num_expr=config.fit.num_expr)
        else:
            logger.warning("mat_final.npy missing; FLAME mesh exported without morphs")
    timings["bake"] = time.perf_counter() - t0

    progress("export", 0.95)
    t0 = time.perf_counter()
    glb_path = write_glb(out_dir / "model.glb", mesh, name="Avatar", texture=texture)
    timings["export"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_all

    stats = mesh_stats(mesh)
    meta = {
        "backend": backend,
        "source": str(source),
        "frames_decoded": len(bundle.frames),
        "keyframes": len(keyframes),
        "tracked_keyframes": sum(1 for t in key_tracked if t.landmarks is not None),
        "glb": str(glb_path),
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "mesh": stats,
        "device": config.device,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    progress("done", 1.0)
    logger.info(
        "creation finished: backend=%s total=%.1fs verts=%d morphs=%d",
        backend,
        timings["total"],
        stats["vertices"],
        stats["blendshapes"],
    )
    return meta


def _to_scored(frames: list[np.ndarray]):
    from core.creation.frames import ScoredFrame, brightness, sharpness

    out = []
    for i, img in enumerate(frames):
        out.append(
            ScoredFrame(
                index=i,
                image=img,
                sharpness=sharpness(img),
                brightness=brightness(img),
                yaw=0.0,
                score=0.0,
            )
        )
    return out


def _primary_landmarks(tracked) -> np.ndarray | None:
    best = None
    best_score = -1e9
    for t in tracked:
        if t.landmarks is None:
            continue
        s = t.scored.sharpness - abs(t.scored.yaw) * 100.0
        if s > best_score:
            best_score = s
            best = t.landmarks
    return best


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
