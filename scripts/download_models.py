from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.common.config import (
    FACE_LANDMARKER_PATH,
    FACE_LANDMARKER_URL,
    MAT_FINAL_PATH,
    MAT_FINAL_URL,
    MODELS_DIR,
)
from core.common.logging_utils import get_logger

logger = get_logger("download_models")


def fetch(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and not force:
        logger.info("already present: %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        logger.info("done: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:
        logger.error("download failed for %s: %s", url, exc)
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Download redistributable model assets")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-w-matrix", action="store_true")
    args = ap.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ok = fetch(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH, force=args.force)
    if not args.skip_w_matrix:
        # Keep the face-landmarker result: a successful optional download must
        # not mask a missing/incomplete required model. Fetch it regardless so
        # a required-model failure doesn't skip the optional asset.
        mat_ok = fetch(MAT_FINAL_URL, MAT_FINAL_PATH, force=args.force)
        ok = ok and mat_ok
    flame_dir = MODELS_DIR / "flame"
    logger.info(
        "FLAME model is NOT auto-downloaded (registration required).\n"
        "  1. Register at https://flame.is.tue.mpg.de\n"
        "  2. Download FLAME 2020 (generic_model.pkl)\n"
        "  3. Place it under %s (e.g. %s)\n"
        "Without it the pipeline uses the parametric template fallback.",
        flame_dir,
        flame_dir / "FLAME2020" / "generic_model.pkl",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
