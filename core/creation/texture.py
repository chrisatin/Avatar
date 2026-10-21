from __future__ import annotations

import numpy as np

from core.common.logging_utils import get_logger
from core.creation.frames import ScoredFrame, estimate_yaw, face_bbox
from core.creation.template_head import AvatarMesh, HEAD_AXES

logger = get_logger(__name__)

DEFAULT_SKIN = np.array([196.0, 158.0, 132.0], dtype=np.float32)


def splat_albedo(
    mesh: AvatarMesh,
    scored_frames: list[ScoredFrame],
    landmarks_per_frame: list[np.ndarray | None],
    size: int = 1024,
    max_yaw: float = 0.4,
) -> np.ndarray:
    tex = np.zeros((size, size, 3), dtype=np.float32)
    weight = np.zeros((size, size, 1), dtype=np.float32)
    verts = mesh.vertices
    uvs = mesh.uvs
    px = np.clip((uvs[:, 0] * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip(((1.0 - uvs[:, 1]) * (size - 1)).astype(np.int32), 0, size - 1)
    for f, lms in zip(scored_frames, landmarks_per_frame):
        if lms is None or len(lms) == 0:
            continue
        yaw = estimate_yaw(lms, f.image.shape[1])
        if abs(yaw) > max_yaw:
            continue
        x0, y0, x1, y1 = face_bbox(lms, margin=0.15)
        bw = max(x1 - x0, 1)
        bh = max(y1 - y0, 1)
        s = bw / max(2.0 * float(HEAD_AXES[0]), 1e-6)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1) - 0.05 * bh
        scale_w = (1.0 - abs(yaw) / max(max_yaw, 1e-6)) * (f.sharpness / 100.0 + 0.25)
        front = verts[:, 2] > 0.0
        u_px = np.clip((cx + verts[:, 0] * s).astype(np.int32), 0, f.image.shape[1] - 1)
        v_px = np.clip((cy - verts[:, 1] * s).astype(np.int32), 0, f.image.shape[0] - 1)
        colors = f.image[v_px, u_px].astype(np.float32)
        valid = front & (u_px > 0) & (v_px > 0)
        tex[py[valid], px[valid]] += colors[valid] * scale_w
        weight[py[valid], px[valid]] += scale_w
    filled = weight[..., 0] > 1e-6
    tex[filled] /= weight[filled]
    skin = DEFAULT_SKIN.copy()
    if filled.any():
        skin = tex[filled].reshape(-1, 3).mean(axis=0)
    tex[~filled] = skin
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    tex = _dilate(tex, filled, iterations=4)
    logger.info("albedo splat: %.1f%% texels sampled from %d frames", 100.0 * filled.mean(), len(scored_frames))
    return tex


def _dilate(tex: np.ndarray, mask: np.ndarray, iterations: int = 4) -> np.ndarray:
    out = tex.copy()
    m = mask.copy()
    for _ in range(iterations):
        if m.all():
            break
        shifted = np.zeros_like(out)
        cnt = np.zeros(mask.shape, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                rolled_m = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
                out = np.where(rolled_m[..., None] & (~m[..., None]), shifted, out)
                m = m | rolled_m
                cnt += rolled_m.astype(np.float32)
        m = mask | (cnt > 0)
    return out


def neutral_texture(size: int = 1024) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    shade = 1.0 - 0.15 * (yy - 0.5) ** 2 * 4.0
    tex = DEFAULT_SKIN[None, None, :] * shade[..., None]
    return np.clip(tex, 0, 255).astype(np.uint8)
