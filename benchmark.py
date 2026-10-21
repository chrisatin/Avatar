from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.animation.retarget import Retargeter
from core.animation.tracker import create_tracker
from core.common.logging_utils import get_logger

logger = get_logger("benchmark")


def _nvml_sample() -> dict:
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetUtilizationRates,
            nvmlInit,
        )

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        return {
            "vram_used_mb": round(mem.used / 1e6, 1),
            "vram_total_mb": round(mem.total / 1e6, 1),
            "gpu_util_pct": int(util.gpu),
        }
    except Exception:
        return {"vram_used_mb": None, "gpu_util_pct": None}


def bench_creation(video: Path, out_dir: Path, repeats: int = 1) -> dict:
    from core.common.config import load_config
    from core.creation.pipeline import run_creation_pipeline

    config = load_config()
    runs = []
    peak_vram = 0.0
    for i in range(repeats):
        target = out_dir / f"creation_run_{i}"
        t0 = time.perf_counter()
        before = _nvml_sample()
        meta = run_creation_pipeline(video, target, config=config)
        elapsed = time.perf_counter() - t0
        after = _nvml_sample()
        used = after.get("vram_used_mb") or 0.0
        peak_vram = max(peak_vram, used or 0.0)
        runs.append(
            {
                "total_s": round(elapsed, 2),
                "timings": meta["timings_s"],
                "backend": meta["backend"],
                "mesh": meta["mesh"],
                "vram_before_mb": before.get("vram_used_mb"),
                "vram_after_mb": after.get("vram_used_mb"),
                "glb_bytes": (target / "model.glb").stat().st_size
                if (target / "model.glb").exists()
                else 0,
            }
        )
    return {
        "mode": "creation",
        "video": str(video),
        "repeats": repeats,
        "runs": runs,
        "peak_vram_mb": peak_vram or None,
    }


def bench_driving(frames: int = 300, resolution: tuple[int, int] = (640, 480), backend: str = "auto") -> dict:
    tracker = create_tracker(backend)
    retargeter = Retargeter(smooth=True)
    w, h = resolution
    rng = np.random.default_rng(0)
    latency: list[float] = []
    vram_before = _nvml_sample()
    for i in range(frames):
        img = (rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))
        t0 = time.perf_counter()
        track = tracker.process(img)
        out = retargeter.process(track, t=t0)
        latency.append((time.perf_counter() - t0) * 1000.0)
        if i == 0:
            logger.info("sample output: %d weights", len(out.get("weights", {})))
    vram_after = _nvml_sample()
    tracker.close()
    latency_ms = latency[10:]
    return {
        "mode": "driving",
        "backend": type(tracker).__name__,
        "frames": frames,
        "resolution": list(resolution),
        "mean_ms": round(statistics.fmean(latency_ms), 2),
        "p50_ms": round(statistics.median(latency_ms), 2),
        "p95_ms": round(statistics.quantiles(latency_ms, n=20)[18], 2),
        "max_ms": round(max(latency_ms), 2),
        "fps": round(1000.0 / statistics.fmean(latency_ms), 1),
        "vram_before_mb": vram_before.get("vram_used_mb"),
        "vram_after_mb": vram_after.get("vram_used_mb"),
    }


def bench_glb_export(out_dir: Path) -> dict:
    from core.creation.export import read_glb_header, write_glb
    from core.creation.template_head import create_avatar_mesh

    t0 = time.perf_counter()
    mesh = create_avatar_mesh(subdivisions=32)
    t_mesh = time.perf_counter() - t0
    t0 = time.perf_counter()
    glb = write_glb(out_dir / "bench.glb", mesh)
    t_export = time.perf_counter() - t0
    header = read_glb_header(glb)
    return {
        "mode": "export",
        "mesh_ms": round(t_mesh * 1000, 1),
        "export_ms": round(t_export * 1000, 1),
        "glb_header": header,
        "glb_bytes": glb.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Avatar pipeline benchmarks (T4)")
    ap.add_argument("--mode", choices=["creation", "driving", "export", "all"], default="all")
    ap.add_argument("--video", default=None, help="video for creation mode (default: sample)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--tracker", default="auto", choices=["auto", "mediapipe", "synthetic"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="outputs/benchmark")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    if args.mode in ("creation", "all"):
        video = Path(args.video) if args.video else None
        if video is None or not video.exists():
            sample = Path("assets/samples/sample_face.mp4")
            if not sample.exists():
                import subprocess

                subprocess.run(
                    [sys.executable, "scripts/make_sample_video.py", "--out", str(sample)],
                    check=True,
                )
            video = sample
        results.append(bench_creation(video, out_dir, repeats=args.repeats))

    if args.mode in ("driving", "all"):
        results.append(bench_driving(frames=args.frames, backend=args.tracker))

    if args.mode in ("export", "all"):
        results.append(bench_glb_export(out_dir))

    report_path = out_dir / "benchmark.json"
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            prior = [
                r
                for r in existing.get("results", [])
                if not any(
                    r.get("mode") == n.get("mode") and r.get("backend") == n.get("backend")
                    for n in results
                )
            ]
            results = prior + results
        except Exception:
            pass
    report = {"results": results, "updated_at": time.time()}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    logger.info("benchmark written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
