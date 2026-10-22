from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def draw_face(frame: np.ndarray, t: float, w: int, h: int) -> np.ndarray:
    import cv2

    cx = int(w * 0.5 + math.sin(t * 0.9) * w * 0.04)
    cy = int(h * 0.5 + math.sin(t * 0.6) * h * 0.02)
    rx = int(w * 0.22)
    ry = int(h * 0.30)
    # OpenCV colors are BGR; these values render as the intended RGB palette.
    frame[:] = (40, 34, 32)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (140, 160, 190), -1)
    eye_dx = int(rx * 0.42)
    eye_dy = int(ry * 0.22)
    blink = max(0.0, math.sin(t * 5.0)) ** 8
    for sx in (-1, 1):
        ex = cx + sx * eye_dx
        ey = cy - eye_dy
        eh = max(1, int(10 * (1.0 - blink)))
        cv2.ellipse(frame, (ex, ey), (12, eh), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(frame, (ex + int(math.sin(t) * 3), ey), 5, (30, 40, 60), -1)
        by = ey - 18
        cv2.ellipse(frame, (ex, by), (14, 4), 0, 0, 180, (40, 50, 70), 2)
    nose_y = cy + int(ry * 0.08)
    cv2.line(frame, (cx, cy), (cx - 4, nose_y + 8), (100, 120, 150), 2)
    jaw = 0.5 + 0.5 * math.sin(t * 4.0)
    mouth_h = int(6 + 18 * jaw)
    smile = int(math.sin(t * 1.5) * 8)
    cv2.ellipse(
        frame,
        (cx + smile // 2, cy + int(ry * 0.45)),
        (int(rx * 0.35), mouth_h),
        0,
        0,
        180,
        (140, 60, 80),
        3,
    )
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/samples/sample_face.mp4")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480])
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = args.size
    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out), fourcc, args.fps, (w, h))
        n = int(args.seconds * args.fps)
        for i in range(n):
            t = i / args.fps
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            draw_face(frame, t, w, h)
            vw.write(frame)
        vw.release()
        print(f"wrote {out} ({n} frames)")
        return 0
    except ImportError:
        print("OpenCV required: pip install opencv-python-headless", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
