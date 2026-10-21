from __future__ import annotations

import numpy as np

from core.animation.mapping import ARKIT_51, EYE_LOOK_CHANNELS
from core.common.logging_utils import get_logger
from core.creation.template_head import AvatarMesh

logger = get_logger(__name__)


def bake_flame_arkit_deltas(
    mesh: AvatarMesh,
    w_matrix: np.ndarray,
    num_expr: int = 50,
) -> AvatarMesh:
    exp_basis = getattr(mesh, "_exp_basis", None)
    if exp_basis is None:
        raise ValueError("mesh has no FLAME expression basis; fit FLAME first")
    m = np.asarray(w_matrix, dtype=np.float32)
    if m.shape[0] != len(ARKIT_51) and m.shape[1] == len(ARKIT_51):
        m = m.T
    if m.shape[0] != len(ARKIT_51):
        raise ValueError(f"W matrix shape {m.shape} != ({len(ARKIT_51)}, K)")
    k = min(num_expr, exp_basis.shape[0], m.shape[1])
    m_sub = m[:, :k]
    pinv = np.linalg.pinv(m_sub).astype(np.float32)
    drop = set(EYE_LOOK_CHANNELS)
    for i, name in enumerate(ARKIT_51):
        if name in drop:
            pinv[:, i] = 0.0
    deltas = np.einsum("kv,vi->vi", pinv[:k], exp_basis[:k]).astype(np.float32)
    mesh.blendshape_names = list(ARKIT_51)
    mesh.blendshape_deltas = [deltas[i].copy() for i in range(len(ARKIT_51))]
    logger.info("baked %d ARKit morph targets from FLAME expression basis", len(ARKIT_51))
    return mesh


def mesh_stats(mesh: AvatarMesh) -> dict[str, float | int]:
    return {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "blendshapes": int(len(mesh.blendshape_names)),
        "mean_delta_norm": float(
            np.mean([np.linalg.norm(d) for d in mesh.blendshape_deltas])
            if mesh.blendshape_deltas
            else 0.0
        ),
    }
