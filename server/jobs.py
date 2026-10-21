from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    stage: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": (self.finished_at - self.started_at)
            if self.started_at and self.finished_at
            else None,
            "error": self.error,
            "result": self.result,
            "source": self.source,
        }


class JobManager:
    def __init__(self, workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max(1, workers))

    def submit(self, kind: str, fn: Callable[[Job], dict], source: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, source=source)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, fn), daemon=True).start()
        return job

    def _run(self, job: Job, fn: Callable[[Job], dict]) -> None:
        with self._sem:
            job.status = "running"
            job.started_at = time.time()
            try:
                job.result = fn(job)
                job.status = "done"
                job.progress = 1.0
            except Exception as exc:
                job.status = "error"
                job.error = f"{exc}"
                logger.error("job %s failed: %s\n%s", job.id, exc, traceback.format_exc())
            finally:
                job.finished_at = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update_progress(self, job: Job, stage: str, frac: float) -> None:
        job.stage = stage
        job.progress = float(frac)
