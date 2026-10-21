from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.animation.retarget import Retargeter
from core.animation.tracker import create_tracker, load_w_matrix
from core.common.config import (
    CLIENT_DIR,
    FACE_LANDMARKER_PATH,
    OUTPUTS_DIR,
    load_config,
)
from core.common.logging_utils import get_logger
from core.creation.flame_fit import flame_assets_present
from core.creation.pipeline import run_creation_pipeline
from server.jobs import JobManager

logger = get_logger(__name__)

config = load_config()
app = FastAPI(title="Avatar Pipeline", version="0.1.0")
jobs = JobManager(workers=config.server.job_workers)

_tracker = None
_retargeter = Retargeter(smooth=True)
_tracker_lock = threading.Lock()


def _get_tracker():
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = create_tracker("auto")
        return _tracker


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    tracker_name = None
    try:
        tracker_name = type(_get_tracker()).__name__
    except Exception as exc:
        tracker_name = f"unavailable: {exc}"
    cuda = False
    gpu = None
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        pass
    return {
        "status": "ok",
        "tracker": tracker_name,
        "cuda": cuda,
        "gpu": gpu,
        "flame_assets": flame_assets_present(),
        "face_landmarker_task": FACE_LANDMARKER_PATH.exists(),
        "w_matrix": load_w_matrix() is not None,
    }


@app.post("/api/v1/avatar/create")
async def avatar_create(
    file: UploadFile | None = File(default=None),
    source_path: str | None = None,
):
    if file is None and not source_path:
        raise HTTPException(400, "provide a video upload or source_path")
    if file is not None:
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        incoming = OUTPUTS_DIR / "uploads"
        incoming.mkdir(parents=True, exist_ok=True)
        dest = incoming / f"{uuid.uuid4().hex[:12]}{suffix}"
        data = await file.read()
        max_bytes = config.server.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(413, f"upload exceeds {config.server.max_upload_mb} MB")
        dest.write_bytes(data)
        source = str(dest)
    else:
        source = str(source_path)
        if not Path(source).exists():
            raise HTTPException(404, f"source not found: {source}")

    out_dir = OUTPUTS_DIR / "jobs"

    def run(job) -> dict:
        job_out = out_dir / job.id
        return run_creation_pipeline(
            source,
            job_out,
            config=config,
            progress=lambda stage, frac: jobs.update_progress(job, stage, frac),
        )

    job = jobs.submit("creation", run, source=source)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/v1/avatar/jobs")
def avatar_jobs():
    return [j.to_dict() for j in jobs.list()]


@app.get("/api/v1/avatar/jobs/{job_id}")
def avatar_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@app.get("/api/v1/avatar/jobs/{job_id}/model.glb")
def avatar_model(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    glb = OUTPUTS_DIR / "jobs" / job_id / "model.glb"
    if not glb.exists():
        raise HTTPException(409, f"model not ready (status={job.status})")
    return FileResponse(glb, media_type="model/gltf-binary", filename=f"avatar_{job_id}.glb")


@app.get("/api/v1/avatar/models/{job_id}/meta.json")
def avatar_meta(job_id: str):
    meta = OUTPUTS_DIR / "jobs" / job_id / "meta.json"
    if not meta.exists():
        raise HTTPException(404, "meta not found")
    return JSONResponse(json.loads(meta.read_text(encoding="utf-8")))


@app.websocket("/ws/track")
async def ws_track(ws: WebSocket):
    import time

    await ws.accept()
    tracker = _get_tracker()
    min_interval = 1.0 / max(config.server.tracker_max_fps, 1.0)
    last = 0.0
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            text = message.get("text")
            if text:
                import json as _json

                try:
                    payload = _json.loads(text)
                except Exception:
                    await ws.send_json({"type": "error", "error": "bad json"})
                    continue
                if payload.get("type") == "ping":
                    await ws.send_json({"type": "pong", "t": time.time()})
                    continue
                if payload.get("type") == "reset":
                    _retargeter.reset()
                    await ws.send_json({"type": "reset_ack"})
                    continue
                continue
            if not data:
                continue
            now = time.perf_counter()
            if now - last < min_interval:
                continue
            last = now
            img = _decode_jpeg(data)
            if img is None:
                await ws.send_json({"type": "error", "error": "decode failed"})
                continue
            track = tracker.process(img)
            out = _retargeter.process(track, t=now)
            out["type"] = "blendshapes"
            out["t"] = time.time()
            await ws.send_json(out)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("ws/track closed: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass


def _decode_jpeg(data: bytes):
    import numpy as np

    try:
        import cv2

        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except ImportError:
        pass
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img)[:, :, ::-1].copy()
    except Exception:
        return None


@app.exception_handler(Exception)
async def unhandled(request, exc):
    logger.error("unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


if CLIENT_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(CLIENT_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(CLIENT_DIR / "index.html"))

    if FACE_LANDMARKER_PATH.exists():

        @app.get("/models/face_landmarker.task")
        def face_landmarker_task():
            return FileResponse(
                FACE_LANDMARKER_PATH,
                media_type="application/octet-stream",
                filename="face_landmarker.task",
            )
