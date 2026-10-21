from __future__ import annotations

from typing import Any

from core.animation.mapping import (
    ARKIT_51,
    arkit_to_vrm,
    gaze_from_weights,
    weights_to_vector,
)
from core.animation.smoothing import QuaternionSmoother, VectorSmoother
from core.animation.tracker import TrackResult


class Retargeter:
    def __init__(
        self,
        morph_names: list[str] | None = None,
        smooth: bool = True,
        min_weight: float = 0.01,
    ) -> None:
        self.morph_names = morph_names
        self.smooth = smooth
        self.min_weight = min_weight
        self._vec_smoother = VectorSmoother(51, min_cutoff=1.2, beta=0.25)
        self._quat_smoother = QuaternionSmoother(beta=0.25)

    def process(self, track: TrackResult, t: float | None = None) -> dict[str, Any]:
        if not track.present:
            return {"present": False, "weights": {}, "vrm": {}, "gaze": [0.0, 0.0]}
        vec = weights_to_vector(track.blendshapes)
        if self.smooth:
            vec = self._vec_smoother(vec, t)
        weights = {
            k: v
            for k, v in zip(ARKIT_51, vec)
            if v >= self.min_weight or k.startswith("eye")
        }
        rotation = track.rotation
        if self.smooth:
            rotation = self._quat_smoother(rotation)
        out: dict[str, Any] = {
            "present": True,
            "weights": weights,
            "vrm": arkit_to_vrm(weights),
            "gaze": list(gaze_from_weights(weights)),
            "rotation": rotation,
            "translation": track.translation,
            "inference_ms": track.inference_ms,
        }
        if self.morph_names is not None:
            out["morphs"] = {
                name: weights.get(name, 0.0) for name in self.morph_names
            }
        return out

    def reset(self) -> None:
        self._vec_smoother.reset()
        self._quat_smoother.reset()
