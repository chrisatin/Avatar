from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.common.logging_utils import get_logger
from core.creation.frames import ScoredFrame, crop_face_with_offset
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
        cropped, ox, oy = crop_face_with_offset(f.image, None, margin=crop_margin)
        result = tracker.process(cropped)
        lms = np.asarray(result.landmarks, dtype=np.float32) if result.present else None
        if lms is not None:
            lms[:, 0] += ox
            lms[:, 1] += oy
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
