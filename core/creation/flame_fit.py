from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core.common.config import FLAME_DIR
from core.common.logging_utils import get_logger
from core.creation.template_head import AvatarMesh

logger = get_logger(__name__)


class FlameBackendUnavailable(RuntimeError):
    pass


def flame_assets_present(model_dir: str | Path | None = None) -> bool:
    d = Path(model_dir or FLAME_DIR)
    if not d.exists():
        return False
    hits = list(d.glob("**/*.pkl")) + list(d.glob("**/*.npz"))
    return len(hits) > 0


def _find_model_file(model_dir: Path) -> Path:
    preferred = [
        "FLAME2020/generic_model.pkl",
        "FLAME_2020/generic_model.pkl",
        "generic_model.pkl",
        "FLAME2023/flame2023.pkl",
        "flame2023.pkl",
    ]
    for rel in preferred:
        p = model_dir / rel
        if p.exists():
            return p
    pkls = sorted(model_dir.glob("**/*.pkl"))
    if pkls:
        return pkls[0]
    raise FlameBackendUnavailable(f"no FLAME .pkl under {model_dir}")


def _load_landmark_embedding(model_dir: Path) -> np.ndarray | None:
    for name in ("landmark_embedding.npz", "mediapipe_landmark_embedding.npz"):
        for p in model_dir.glob(f"**/{name}"):
            data = np.load(p)
            for key in ("landmarks", "embedding", "lmk", "landmark_ids"):
                if key in data:
                    arr = np.asarray(data[key])
                    if arr.ndim == 1 and arr.dtype.kind in "iu":
                        return arr.astype(np.int64)
                    if arr.ndim == 2 and arr.shape[1] == 3:
                        return arr.astype(np.float32)
            logger.warning("unrecognized landmark embedding format in %s", p)
    return None


def _create_flame(model_file: Path, num_betas: int, num_expr: int):
    import smplx

    try:
        return smplx.create(
            str(model_file.parent),
            model_type="flame",
            gender="neutral",
            num_betas=num_betas,
            num_expression_coeffs=num_expr,
            use_face_contour=False,
            flat_shape=True,
        )
    except Exception as exc:
        raise FlameBackendUnavailable(f"smplx FLAME init failed: {exc}") from exc


def fit_flame_sequence(
    landmarks_seq: list[np.ndarray],
    model_dir: str | Path | None = None,
    num_betas: int = 100,
    num_expr: int = 50,
    iterations: int = 60,
    lr: float = 0.02,
    device: str = "cpu",
) -> AvatarMesh:
    import torch

    model_dir = Path(model_dir or FLAME_DIR)
    if not flame_assets_present(model_dir):
        raise FlameBackendUnavailable(
            f"FLAME assets missing under {model_dir}. Register at "
            "https://flame.is.tue.mpg.de and place generic_model.pkl there."
        )
    model_file = _find_model_file(model_dir)
    flame = _create_flame(model_file, num_betas, num_expr).to(device)
    embedding = _load_landmark_embedding(model_dir)
    if embedding is None:
        logger.warning(
            "no landmark embedding found; fitting identity with neutral landmarks "
            "(expression/shape quality reduced)"
        )
    if isinstance(embedding, np.ndarray) and embedding.ndim == 2 and embedding.shape[1] == 3:
        lmk_verts = None
        lmk_fixed = torch.tensor(embedding, dtype=torch.float32, device=device)
    elif isinstance(embedding, np.ndarray):
        lmk_verts = torch.tensor(embedding, dtype=torch.long, device=device)
        lmk_fixed = None
    else:
        lmk_verts = None
        lmk_fixed = None

    betas_all = []
    expr_all = []
    t0 = time.perf_counter()
    for lms in landmarks_seq:
        target = torch.tensor(np.asarray(lms, dtype=np.float32), device=device)
        if lmk_verts is not None:
            sel = lmk_verts.clamp(max=flame.vertex_tensor.shape[0] - 1)
        elif lmk_fixed is not None:
            sel = None
        else:
            step = max(1, target.shape[0] // 68)
            sel = torch.arange(0, target.shape[0], step, device=device)[:68]
            target = target[::step][:68]

        betas = torch.zeros(num_betas, dtype=torch.float32, device=device, requires_grad=True)
        expr = torch.zeros(num_expr, dtype=torch.float32, device=device, requires_grad=True)
        jaw = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
        translation = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
        scale = torch.ones(1, dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam([betas, expr, jaw, translation, scale], lr=lr)
        for _ in range(iterations):
            out = flame(
                betas=betas.unsqueeze(0),
                expression=expr.unsqueeze(0),
                jaw=jaw.unsqueeze(0),
                return_verts=True,
            )
            verts = out.vertices[0]
            if lmk_verts is not None:
                pred = verts[sel]
            else:
                pred = verts
            if lmk_fixed is not None and sel is None:
                pred = lmk_fixed
                pred = pred * scale + translation
                loss = ((pred - target) ** 2).mean()
            elif sel is not None and lmk_verts is None:
                loss = (((pred * scale + translation) - target) ** 2).mean()
            else:
                loss = (((pred * scale + translation) - target) ** 2).mean()
            loss = loss + 1e-4 * (betas.pow(2).mean() + expr.pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        betas_all.append(betas.detach().cpu().numpy())
        expr_all.append(expr.detach().cpu().numpy())

    if betas_all:
        beta_id = np.median(np.stack(betas_all), axis=0).astype(np.float32)
    else:
        beta_id = np.zeros(num_betas, dtype=np.float32)
    logger.info(
        "FLAME identity fit done in %.1fs (%d frames)",
        time.perf_counter() - t0,
        len(landmarks_seq),
    )
    with torch.no_grad():
        out = flame(
            betas=torch.tensor(beta_id, dtype=torch.float32, device=device).unsqueeze(0),
            expression=torch.zeros(1, num_expr, dtype=torch.float32, device=device),
            return_verts=True,
        )
        verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
        exp_basis = flame.expr_basis.detach().cpu().numpy().astype(np.float32)
        faces = flame.faces.astype(np.uint32)
        uv = None
        if getattr(flame, "uv", None) is not None:
            uv = flame.uv.detach().cpu().numpy().astype(np.float32)
    mesh = AvatarMesh(
        vertices=verts,
        faces=faces,
        uvs=uv if uv is not None else _flame_cylindrical_uv(verts),
        identity={"beta": beta_id.tolist()},
    )
    mesh._exp_basis = exp_basis
    mesh._betas = beta_id
    mesh._flame_vertices = verts
    return mesh


def _flame_cylindrical_uv(verts: np.ndarray) -> np.ndarray:
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    center = np.array(
        [
            0.5 * (x.min() + x.max()),
            0.5 * (y.min() + y.max()),
            0.5 * (z.min() + z.max()),
        ],
        dtype=np.float32,
    )
    p = verts - center
    u = 0.5 + np.arctan2(p[:, 0], p[:, 2]) / (2.0 * np.pi)
    v = 0.5 + np.arctan2(p[:, 1], np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)) / np.pi
    return np.stack([u, v], axis=1).astype(np.float32)
