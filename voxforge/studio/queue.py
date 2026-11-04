"""In-process generation queue for Studio projects (VF-005).

Write order guarantees no half state: audio is produced in a temp folder,
moved into ``audio/`` with ``os.replace`` and only then recorded in the
database. A failure (including CUDA OOM, which surfaces as an exception) marks
that job ``failed`` and the worker continues with the next one.
"""

from __future__ import annotations

import os
import shutil
import threading
import wave
from pathlib import Path
from typing import Any, Callable

from voxforge.engines.base import TTSEngine
from voxforge.generation import synthesize_text
from voxforge.studio.project import StudioProject, file_sha256

TEMP_DIRNAME = ".tmp"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _check_engine(project: StudioProject, engine: TTSEngine) -> None:
    expected = project.settings
    caps = engine.capabilities
    if (caps.engine_id, caps.model_id) != (expected.engine_id, expected.model_id):
        raise ValueError(
            f"Motor proje ayarlariyla uyusmuyor: proje {expected.engine_id}/{expected.model_id}, "
            f"verilen {caps.engine_id}/{caps.model_id}"
        )


def process_job(project: StudioProject, engine: TTSEngine, job: dict[str, Any]) -> str:
    """Run one claimed job. Returns the final job status."""
    if project.current_key_for(job["segment_id"]) != job["cache_key"]:
        project.finish_job(job, "cancelled", "Segment kuyruga alindiktan sonra degisti.")
        return "cancelled"

    reusable = project.find_artifact_by_key(job["cache_key"])
    if reusable:
        project.complete_job(job, Path(reusable["path"]), reusable["sha256"], reusable["duration_s"])
        return "done"

    settings = project.settings
    audio_dir = project.store.audio_dir
    temp_dir = audio_dir / TEMP_DIRNAME / f"job{job['id']}"
    final_path = audio_dir / f"seg{job['segment_id']:05d}_{job['cache_key'][:16]}.wav"
    try:
        result = synthesize_text(
            engine,
            job["spoken_text"],
            output_dir=temp_dir,
            output_stem="segment",
            reference_path=Path(settings.reference_path) if settings.reference_path else None,
            preset=settings.preset,
            language=settings.language,
            max_chars=settings.max_chars,
        )
        os.replace(result["final_output_path"], final_path)
        project.complete_job(job, final_path, file_sha256(final_path), wav_duration(final_path))
        return "done"
    except Exception as exc:  # noqa: BLE001 - every engine error must become a failed job
        project.finish_job(job, "failed", f"{type(exc).__name__}: {exc}")
        return "failed"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_pending(
    project: StudioProject,
    engine: TTSEngine,
    *,
    stop_event: threading.Event | None = None,
    max_jobs: int | None = None,
    on_progress: Callable[[dict[str, Any], str], None] | None = None,
) -> dict[str, int]:
    """Process queued jobs until the queue is empty, ``stop_event`` is set or ``max_jobs`` is reached."""
    _check_engine(project, engine)
    counts = {"done": 0, "failed": 0, "cancelled": 0}
    processed = 0
    while not (stop_event and stop_event.is_set()):
        if max_jobs is not None and processed >= max_jobs:
            break
        job = project.claim_next_job()
        if job is None:
            break
        status = process_job(project, engine, job)
        counts[status] += 1
        processed += 1
        if on_progress:
            on_progress(job, status)
    return counts


class StudioWorker:
    """Background thread for the UI. ``stop()`` finishes the current segment, then stops."""

    def __init__(self, project: StudioProject, engine: TTSEngine) -> None:
        self.project = project
        self.engine = engine
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_counts: dict[str, int] | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="voxforge-studio-worker", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.last_counts = run_pending(self.project, self.engine, stop_event=self.stop_event)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def stop(self, timeout: float | None = None) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout)

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())
