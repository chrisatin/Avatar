from __future__ import annotations

from typing import Iterable

ARKIT_52: list[str] = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]

ARKIT_51: list[str] = [n for n in ARKIT_52 if n != "_neutral"]

EYE_LOOK_CHANNELS: list[str] = [
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
]

EYE_TRACKED_CHANNELS: list[str] = [
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
]

JAW_INDEX_51 = ARKIT_51.index("jawOpen")

VRM_PRESET_LUT: dict[str, dict[str, float]] = {
    "happy": {
        "mouthSmileLeft": 0.8,
        "mouthSmileRight": 0.8,
        "cheekSquintLeft": 0.5,
        "cheekSquintRight": 0.5,
        "eyeSquintLeft": 0.3,
        "eyeSquintRight": 0.3,
    },
    "angry": {
        "browDownLeft": 0.8,
        "browDownRight": 0.8,
        "noseSneerLeft": 0.5,
        "noseSneerRight": 0.5,
        "mouthFrownLeft": 0.3,
        "mouthFrownRight": 0.3,
    },
    "sad": {
        "browInnerUp": 0.9,
        "mouthFrownLeft": 0.7,
        "mouthFrownRight": 0.7,
        "eyeSquintLeft": 0.2,
        "eyeSquintRight": 0.2,
    },
    "relaxed": {
        "eyeSquintLeft": 0.4,
        "eyeSquintRight": 0.4,
        "mouthSmileLeft": 0.25,
        "mouthSmileRight": 0.25,
    },
    "surprised": {
        "eyeWideLeft": 0.9,
        "eyeWideRight": 0.9,
        "browOuterUpLeft": 0.8,
        "browOuterUpRight": 0.8,
        "jawOpen": 0.35,
    },
    "aa": {"jawOpen": 1.0, "mouthClose": -0.2},
    "ih": {"mouthStretchLeft": 0.6, "mouthStretchRight": 0.6, "jawOpen": 0.25},
    "ou": {"mouthPucker": 0.9, "jawOpen": 0.15},
    "ee": {"mouthStretchLeft": 0.9, "mouthStretchRight": 0.9, "jawOpen": 0.12},
    "oh": {"mouthFunnel": 0.8, "jawOpen": 0.3},
    "blinkL": {"eyeBlinkLeft": 1.0},
    "blinkR": {"eyeBlinkRight": 1.0},
    "lookLeft": {"eyeLookInRight": 0.8, "eyeLookOutLeft": 0.8},
    "lookRight": {"eyeLookInLeft": 0.8, "eyeLookOutRight": 0.8},
    "lookUp": {"eyeLookUpLeft": 0.8, "eyeLookUpRight": 0.8},
    "lookDown": {"eyeLookDownLeft": 0.8, "eyeLookDownRight": 0.8},
}

VRM_PRESET_NAMES: list[str] = list(VRM_PRESET_LUT.keys())


def arkit_index(name: str) -> int:
    if name not in ARKIT_51:
        raise KeyError(f"unknown ARKit blendshape: {name}")
    return ARKIT_51.index(name)


def weights_to_vector(weights: dict[str, float]) -> list[float]:
    return [float(weights.get(name, 0.0)) for name in ARKIT_51]


def vector_to_weights(vec: Iterable[float]) -> dict[str, float]:
    values = list(vec)
    if len(values) != len(ARKIT_51):
        raise ValueError(f"expected {len(ARKIT_51)} values, got {len(values)}")
    return {name: float(v) for name, v in zip(ARKIT_51, values)}


def arkit_to_vrm(weights: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for preset, contributors in VRM_PRESET_LUT.items():
        acc = 0.0
        for channel, coeff in contributors.items():
            v = weights.get(channel, 0.0)
            if coeff >= 0:
                acc += max(v, 0.0) * coeff
            else:
                acc += min(v, 0.0) * -coeff
        out[preset] = max(0.0, min(1.0, acc))
    return out


def gaze_from_weights(weights: dict[str, float]) -> tuple[float, float]:
    x = 0.5 * (
        weights.get("eyeLookOutLeft", 0.0)
        - weights.get("eyeLookInLeft", 0.0)
        + weights.get("eyeLookInRight", 0.0)
        - weights.get("eyeLookOutRight", 0.0)
    )
    y = 0.5 * (
        weights.get("eyeLookUpLeft", 0.0)
        + weights.get("eyeLookUpRight", 0.0)
        - weights.get("eyeLookDownLeft", 0.0)
        - weights.get("eyeLookDownRight", 0.0)
    )
    return max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))


def flame_expression_from_arkit(
    weights: dict[str, float], w_matrix: "object"
) -> "object":
    import numpy as np

    x = np.array(weights_to_vector(weights), dtype=np.float32)
    m = np.asarray(w_matrix, dtype=np.float32)
    if m.shape[0] != x.shape[0]:
        raise ValueError(f"W matrix shape {m.shape} incompatible with {x.shape}")
    y = x @ m
    if y.shape[0] >= 103:
        y = y.copy()
        y[100:103] *= 100.0
    return y


def arkit_mask_excluding_eyes() -> list[int]:
    drop = set(EYE_LOOK_CHANNELS)
    return [i for i, n in enumerate(ARKIT_51) if n not in drop]


def validate_lut_coverage() -> list[str]:
    missing: list[str] = []
    known = set(ARKIT_51)
    for preset, contributors in VRM_PRESET_LUT.items():
        for channel in contributors:
            if channel not in known:
                missing.append(f"{preset}:{channel}")
    return missing
