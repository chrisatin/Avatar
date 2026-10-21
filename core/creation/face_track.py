from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.common.logging_utils import get_logger
from core.creation.frames import ScoredFrame, crop_face
from core.animation.tracker import BaseTracker, create_tracker

logger = get_logger(__name__)


@dataclass
class TrackedFrame:
    scored: ScoredFrame
    landmarks: np.ndarray | None
    blendshapes: dict[str, float] = field(default_factory=dict)
    inference_ms: float = 0.0


def track_frames(
    scored: list[ScoredFrame],
    tracker: BaseTracker | None = None,
    crop_margin: float = 0.35,
) -> list[TrackedFrame]:
    tracker = tracker or create_tracker("auto")
    out: list[TrackedFrame] = []
    for f in scored:
        cropped = crop_face(f.image, None, margin=crop_margin)
        result = tracker.process(cropped)
        lms = np.asarray(result.landmarks, dtype=np.float32) if result.present else None
        if lms is not None:
            scale_x = f.image.shape[1] / max(cropped.shape[1], 1)
            scale_y = f.image.shape[0] / max(cropped.shape[0], 1)
            lms[:, 0] *= scale_x
            lms[:, 1] *= scale_y
        out.append(
            TrackedFrame(
                scored=f,
                landmarks=lms,
                blendshapes=result.blendshapes,
                inference_ms=result.inference_ms,
            )
        )
    tracked = sum(1 for t in out if t.landmarks is not None)
    logger.info("tracked %d/%d keyframes (avg %.1f ms/frame)", tracked, len(out),
                float(np.mean([t.inference_ms for t in out])) if out else 0.0)
    return out
