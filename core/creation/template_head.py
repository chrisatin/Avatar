from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.animation.mapping import ARKIT_51

HEAD_AXES = np.array([0.82, 1.12, 0.92], dtype=np.float32)

ROLE_ANCHORS: dict[str, np.ndarray] = {
    "forehead_top": np.array([0.0, 0.95, 0.30]),
    "chin": np.array([0.0, -0.85, 0.45]),
    "cheek_l": np.array([-0.85, -0.08, 0.45]),
    "cheek_r": np.array([0.85, -0.08, 0.45]),
    "nose_tip": np.array([0.0, -0.08, 1.0]),
    "nose_bridge": np.array([0.0, 0.22, 0.95]),
    "nostril_l": np.array([-0.14, -0.14, 0.96]),
    "nostril_r": np.array([0.14, -0.14, 0.96]),
    "eye_outer_l": np.array([-0.42, 0.22, 0.86]),
    "eye_inner_l": np.array([-0.14, 0.22, 0.95]),
    "eye_outer_r": np.array([0.42, 0.22, 0.86]),
    "eye_inner_r": np.array([0.14, 0.22, 0.95]),
    "mouth_l": np.array([-0.34, -0.42, 0.84]),
    "mouth_r": np.array([0.34, -0.42, 0.84]),
    "lip_upper": np.array([0.0, -0.34, 0.94]),
    "lip_lower": np.array([0.0, -0.50, 0.90]),
    "brow_l": np.array([-0.32, 0.42, 0.80]),
    "brow_r": np.array([0.32, 0.42, 0.80]),
    "jaw_l": np.array([-0.58, -0.55, 0.55]),
    "jaw_r": np.array([0.58, -0.55, 0.55]),
}

MP_LMK_ROLES: dict[str, int] = {
    "forehead_top": 10,
    "chin": 152,
    "cheek_l": 234,
    "cheek_r": 454,
    "nose_tip": 1,
    "nose_bridge": 168,
    "nostril_l": 48,
    "nostril_r": 278,
    "eye_outer_l": 33,
    "eye_inner_l": 133,
    "eye_outer_r": 263,
    "eye_inner_r": 362,
    "mouth_l": 61,
    "mouth_r": 291,
    "lip_upper": 0,
    "lip_lower": 17,
    "brow_l": 105,
    "brow_r": 334,
    "jaw_l": 172,
    "jaw_r": 397,
}

IDENTITY_PARAMS: list[str] = [
    "face_width",
    "face_length",
    "jaw_width",
    "chin_length",
    "nose_length",
    "nose_width",
    "eye_size",
    "eye_sep",
    "mouth_width",
    "brow_height",
]

_IDENTITY_GAINS = np.array(
    [2.2, 2.0, 2.0, 1.8, 2.0, 2.0, 1.5, 1.8, 1.8, 1.5], dtype=np.float32
)


@dataclass
class AvatarMesh:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    blendshape_names: list[str] = field(default_factory=list)
    blendshape_deltas: list[np.ndarray] = field(default_factory=list)
    identity: dict[str, float] = field(default_factory=dict)
    texture: np.ndarray | None = None


def _unit_sphere(segments: int) -> tuple[np.ndarray, np.ndarray]:
    rings = segments
    sectors = segments * 2
    verts = []
    for i in range(rings + 1):
        phi = np.pi * i / rings
        y = np.cos(phi)
        r = np.sin(phi)
        for j in range(sectors + 1):
            theta = 2.0 * np.pi * j / sectors
            x = r * np.sin(theta)
            z = r * np.cos(theta)
            verts.append([x, y, z])
    v = np.asarray(verts, dtype=np.float32)
    faces = []
    for i in range(rings):
        for j in range(sectors):
            a = i * (sectors + 1) + j
            b = a + sectors + 1
            faces.append([a, b, a + 1])
            faces.append([a + 1, b, b + 1])
    return v, np.asarray(faces, dtype=np.uint32)


def _to_uv(v: np.ndarray) -> np.ndarray:
    u = 0.5 + np.arctan2(v[:, 0], v[:, 2]) / (2.0 * np.pi)
    vv = 0.5 + np.arctan2(v[:, 1], np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2)) / np.pi
    return np.stack([u, vv], axis=1).astype(np.float32)


def _gauss(verts: np.ndarray, anchor: np.ndarray, sigma: float) -> np.ndarray:
    d2 = np.sum((verts - anchor) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)


def _dir_mask_lower(verts: np.ndarray, y0: float = -0.05, k: float = 6.0) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(k * (verts[:, 1] - y0)))).astype(np.float32)


def build_template(subdivisions: int = 32) -> AvatarMesh:
    unit, faces = _unit_sphere(subdivisions)
    verts = unit * HEAD_AXES
    uvs = _to_uv(unit)
    return AvatarMesh(vertices=verts.astype(np.float32), faces=faces, uvs=uvs)


def _semantic_points(unit_verts: np.ndarray) -> dict[str, np.ndarray]:
    pts: dict[str, np.ndarray] = {}
    for role, anchor in ROLE_ANCHORS.items():
        a = anchor / (np.linalg.norm(anchor) or 1.0)
        idx = int(np.argmin(np.sum((unit_verts - a) ** 2, axis=1)))
        pts[role] = unit_verts[idx] * HEAD_AXES
    return pts


def _measure(semantic: dict[str, np.ndarray]) -> dict[str, float]:
    def d(a: str, b: str) -> float:
        return float(np.linalg.norm(semantic[a] - semantic[b]))

    iod = max(d("eye_outer_l", "eye_outer_r"), 1e-6)
    return {
        "face_width": d("cheek_l", "cheek_r") / iod,
        "face_length": d("forehead_top", "chin") / iod,
        "jaw_width": d("jaw_l", "jaw_r") / iod,
        "chin_length": d("lip_lower", "chin") / iod,
        "nose_length": d("nose_bridge", "nose_tip") / iod,
        "nose_width": d("nostril_l", "nostril_r") / iod,
        "eye_size": 0.5
        * (d("eye_outer_l", "eye_inner_l") + d("eye_outer_r", "eye_inner_r"))
        / iod,
        "eye_sep": d("eye_inner_l", "eye_inner_r") / iod,
        "mouth_width": d("mouth_l", "mouth_r") / iod,
        "brow_height": d("forehead_top", "brow_l") / iod,
    }


def default_measurements() -> dict[str, float]:
    unit, _ = _unit_sphere(16)
    return _measure(_semantic_points(unit))


def fit_identity_params(landmarks: np.ndarray) -> dict[str, float]:
    lms = np.asarray(landmarks, dtype=np.float32)
    pts: dict[str, np.ndarray] = {}
    for role, idx in MP_LMK_ROLES.items():
        if idx < len(lms):
            p = lms[idx].copy()
            pts[role] = np.array([p[0], -p[1], -p[2]], dtype=np.float32)
    if "eye_outer_l" not in pts or "eye_outer_r" not in pts:
        return {name: 0.5 for name in IDENTITY_PARAMS}
    iod = max(float(np.linalg.norm(pts["eye_outer_l"] - pts["eye_outer_r"])), 1e-6)
    pts = {k: v / iod for k, v in pts.items()}
    m = _measure(pts)
    m0 = default_measurements()
    out: dict[str, float] = {}
    for i, name in enumerate(IDENTITY_PARAMS):
        ratio = m[name] / max(m0[name], 1e-6)
        out[name] = float(np.clip(0.5 + _IDENTITY_GAINS[i] * (ratio - 1.0), 0.0, 1.0))
    return out


def apply_identity(mesh: AvatarMesh, identity: dict[str, float]) -> AvatarMesh:
    unit = mesh.vertices / HEAD_AXES
    v = mesh.vertices.copy()
    fw = identity.get("face_width", 0.5)
    fl = identity.get("face_length", 0.5)
    jw = identity.get("jaw_width", 0.5)
    cl = identity.get("chin_length", 0.5)
    nl = identity.get("nose_length", 0.5)
    nw = identity.get("nose_width", 0.5)
    es = identity.get("eye_size", 0.5)
    esep = identity.get("eye_sep", 0.5)
    mw = identity.get("mouth_width", 0.5)
    bh = identity.get("brow_height", 0.5)

    mid_mask = _gauss(unit, np.array([0.0, -0.05, 0.9]), 0.7)
    v[:, 0] += v[:, 0] * (fw - 0.5) * 0.45 * mid_mask
    v[:, 1] *= 1.0 + (fl - 0.5) * 0.25
    lower = _dir_mask_lower(unit, y0=-0.15, k=4.0)
    v[:, 0] += v[:, 0] * (jw - 0.5) * 0.4 * lower
    chin_m = _gauss(unit, ROLE_ANCHORS["chin"] / np.linalg.norm(ROLE_ANCHORS["chin"]), 0.35)
    v[:, 1] -= chin_m * (cl - 0.5) * 0.18
    nose_dir = ROLE_ANCHORS["nose_tip"] / np.linalg.norm(ROLE_ANCHORS["nose_tip"])
    nose_m = _gauss(unit, nose_dir, 0.22)
    v += nose_m[:, None] * (nl - 0.5) * 0.12 * nose_dir[None, :]
    nostril = 0.5 * (
        _gauss(unit, ROLE_ANCHORS["nostril_l"] / np.linalg.norm(ROLE_ANCHORS["nostril_l"]), 0.15)
        + _gauss(unit, ROLE_ANCHORS["nostril_r"] / np.linalg.norm(ROLE_ANCHORS["nostril_r"]), 0.15)
    )
    v[:, 0] += np.sign(v[:, 0] + 1e-6) * nostril * (nw - 0.5) * 0.06
    for side, sx in (("l", -1.0), ("r", 1.0)):
        eye = ROLE_ANCHORS[f"eye_outer_{side}"]
        em = _gauss(unit, eye / np.linalg.norm(eye), 0.18)
        v += em[:, None] * (es - 0.5) * 0.05 * (unit[:, [0, 1, 2]] - eye[None, :] / np.linalg.norm(eye))
        inner = ROLE_ANCHORS[f"eye_inner_{side}"] / np.linalg.norm(ROLE_ANCHORS[f"eye_inner_{side}"])
        v[:, 0] += sx * _gauss(unit, inner, 0.25) * (esep - 0.5) * 0.06
    for side, sx in (("l", -1.0), ("r", 1.0)):
        mouth = ROLE_ANCHORS[f"mouth_{side}"] / np.linalg.norm(ROLE_ANCHORS[f"mouth_{side}"])
        mm = _gauss(unit, mouth, 0.2)
        v[:, 0] += sx * mm * (mw - 0.5) * 0.08
        brow = ROLE_ANCHORS[f"brow_{side}"] / np.linalg.norm(ROLE_ANCHORS[f"brow_{side}"])
        bm = _gauss(unit, brow, 0.22)
        v[:, 1] += bm * (bh - 0.5) * 0.12
    out = AvatarMesh(
        vertices=v.astype(np.float32),
        faces=mesh.faces,
        uvs=mesh.uvs,
        identity=dict(identity),
    )
    out._unit = unit
    return out


def _shape_deltas(unit: np.ndarray, verts: np.ndarray) -> dict[str, np.ndarray]:
    def anchor(name: str) -> np.ndarray:
        a = ROLE_ANCHORS[name]
        return a / np.linalg.norm(a)

    def g(name: str, sigma: float) -> np.ndarray:
        return _gauss(unit, anchor(name), sigma)

    deltas: dict[str, np.ndarray] = {}

    def add(name: str, vec: np.ndarray) -> None:
        deltas[name] = vec.astype(np.float32)

    n = len(unit)
    zero = np.zeros((n, 3), dtype=np.float32)
    lower = _dir_mask_lower(unit, y0=-0.08, k=8.0)
    up = np.array([0.0, 1.0, 0.0])
    fwd = np.array([0.0, 0.0, 1.0])

    for side, sx in (("Left", -1.0), ("Right", 1.0)):
        s = "l" if sx < 0 else "r"
        bm = g(f"brow_{s}", 0.2)
        d = zero.copy()
        d[:, 1] = -0.07 * bm
        add(f"browDown{side}", d)
        d = zero.copy()
        d[:, 1] = 0.05 * bm
        add(f"browOuterUp{side}", d)
        em = g(f"eye_outer_{s}", 0.16) + g(f"eye_inner_{s}", 0.16)
        em = np.clip(em, 0.0, 1.0)
        d = zero.copy()
        d[:, 1] = -0.045 * em * lower
        d[:, 2] = 0.01 * em
        add(f"eyeBlink{side}", d)
        d = zero.copy()
        d[:, 1] = -0.02 * em
        d[:, 2] = 0.02 * em
        add(f"eyeSquint{side}", d)
        d = zero.copy()
        d[:, 1] = 0.035 * em
        d[:, 2] = -0.012 * em
        add(f"eyeWide{side}", d)
        look = {
            "Down": np.array([0.0, -0.018, 0.0]),
            "Up": np.array([0.0, 0.018, 0.0]),
            "In": np.array([-0.015 * sx, 0.0, 0.0]),
            "Out": np.array([0.015 * sx, 0.0, 0.0]),
        }
        for dirname, dv in look.items():
            d = zero.copy()
            d += em[:, None] * dv[None, :]
            add(f"eyeLook{dirname}{side}", d)
        ck = g(f"cheek_{s}", 0.22)
        d = zero.copy()
        d[:, 1] = 0.035 * ck
        d[:, 2] = 0.03 * ck
        add(f"cheekSquint{side}", d)
        mm = g(f"mouth_{s}", 0.18)
        d = zero.copy()
        d[:, 0] = sx * 0.045 * mm
        d[:, 1] = 0.04 * mm
        d[:, 2] = 0.012 * mm
        add(f"mouthSmile{side}", d)
        d = zero.copy()
        d[:, 0] = sx * 0.03 * mm
        d[:, 1] = -0.035 * mm
        add(f"mouthFrown{side}", d)
        d = zero.copy()
        d[:, 0] = sx * 0.035 * mm
        d[:, 1] = 0.012 * mm
        add(f"mouthDimple{side}", d)
        d = zero.copy()
        d[:, 0] = sx * 0.055 * mm
        add(f"mouthStretch{side}", d)
        lm = g("lip_lower", 0.16)
        d = zero.copy()
        d[:, 0] = sx * 0.02 * lm
        d[:, 1] = -0.045 * lm
        add(f"mouthLowerDown{side}", d)
        um = g("lip_upper", 0.15)
        d = zero.copy()
        d[:, 0] = sx * 0.018 * um
        d[:, 1] = 0.04 * um
        add(f"mouthUpperUp{side}", d)
        pm = 0.5 * (lm + um)
        d = zero.copy()
        d[:, 1] = 0.02 * pm
        add(f"mouthPress{side}", d)
        nm = g(f"nostril_{s}", 0.14)
        d = zero.copy()
        d[:, 0] = sx * 0.02 * nm
        d[:, 1] = 0.02 * nm
        d[:, 2] = 0.015 * nm
        add(f"noseSneer{side}", d)

    d = zero.copy()
    d[:, 1] = 0.075 * (g("brow_l", 0.2) + g("brow_r", 0.2))
    d[:, 2] = -0.012 * (g("brow_l", 0.2) + g("brow_r", 0.2))
    add("browInnerUp", d)
    d = zero.copy()
    cheek = g("cheek_l", 0.28) + g("cheek_r", 0.28)
    d[:, 0] = np.sign(unit[:, 0] + 1e-6) * 0.07 * cheek
    d[:, 2] = 0.03 * cheek
    add("cheekPuff", d)
    jaw_fwd = lower.copy()
    d = zero.copy()
    d[:, 2] = 0.07 * jaw_fwd
    d[:, 1] = -0.03 * jaw_fwd
    add("jawForward", d)
    for side, sx in (("Left", -1.0), ("Right", 1.0)):
        d = zero.copy()
        d[:, 0] = sx * 0.055 * lower
        add(f"jaw{side}", d)
    d = zero.copy()
    d[:, 1] = -0.18 * lower
    d[:, 2] = -0.03 * lower
    ll = g("lip_lower", 0.2)
    d[:, 1] -= 0.06 * ll
    d[:, 2] += 0.02 * ll
    add("jawOpen", d)
    d = zero.copy()
    d[:, 1] = 0.05 * ll
    add("mouthClose", d)
    ring = np.clip(g("mouth_l", 0.22) + g("mouth_r", 0.22) + g("lip_upper", 0.2) + ll, 0, 1)
    center = anchor("mouth_l") * 0 + np.array([0.0, -0.42, 0.88])
    to_c = (center[None, :] - unit[:, :3] * HEAD_AXES) / (
        np.linalg.norm(center[None, :] - unit[:, :3] * HEAD_AXES, axis=1, keepdims=True) + 1e-6
    )
    d = zero.copy()
    d += ring[:, None] * to_c * 0.035
    d[:, 2] += 0.035 * ring
    add("mouthPucker", d)
    d = zero.copy()
    d += ring[:, None] * to_c * 0.02
    d[:, 2] += 0.045 * ring
    add("mouthFunnel", d)
    d = zero.copy()
    d[:, 1] = 0.025 * ll
    d[:, 2] = 0.018 * ll
    add("mouthRollLower", d)
    d = zero.copy()
    d[:, 1] = -0.018 * um
    d[:, 2] = 0.018 * um
    add("mouthRollUpper", d)
    d = zero.copy()
    d[:, 1] = 0.035 * ll
    add("mouthShrugLower", d)
    d = zero.copy()
    d[:, 1] = 0.035 * um
    add("mouthShrugUpper", d)
    for side, sx in (("Left", -1.0), ("Right", 1.0)):
        s = "l" if sx < 0 else "r"
        mm = g(f"mouth_{s}", 0.2)
        d = zero.copy()
        d[:, 0] = sx * 0.05 * mm
        add(f"mouth{side}", d)
    return deltas


def bake_blendshapes(mesh: AvatarMesh) -> AvatarMesh:
    unit = getattr(mesh, "_unit", None)
    if unit is None:
        unit = mesh.vertices / HEAD_AXES
    raw = _shape_deltas(unit, mesh.vertices)
    names: list[str] = []
    deltas: list[np.ndarray] = []
    for name in ARKIT_51:
        if name in raw:
            names.append(name)
            deltas.append(raw[name])
        else:
            names.append(name)
            deltas.append(np.zeros_like(mesh.vertices))
    mesh.blendshape_names = names
    mesh.blendshape_deltas = deltas
    return mesh


def create_avatar_mesh(
    landmarks: np.ndarray | None = None,
    subdivisions: int = 32,
) -> AvatarMesh:
    base = build_template(subdivisions)
    identity = (
        fit_identity_params(landmarks)
        if landmarks is not None and len(landmarks) >= 400
        else {name: 0.5 for name in IDENTITY_PARAMS}
    )
    mesh = apply_identity(base, identity)
    mesh = bake_blendshapes(mesh)
    return mesh
