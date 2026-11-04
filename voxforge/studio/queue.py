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
import uuid
import wave
from pathlib import Path
from typing import Any, Callable

from voxforge.engines.base import TTSEngine
from voxforge.generation import synthesize_text
from voxforge.studio.project import HEARTBEAT_SECONDS, StudioProject, file_sha256

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


class _Heartbeat:
    """Keeps the job lease alive while a long synthesis call blocks."""

    def __init__(self, project: StudioProject, job: dict[str, Any]) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(project, job), daemon=True)

    def _run(self, project: StudioProject, job: dict[str, Any]) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            if not project.heartbeat(job):
                return

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()


CHANGED_MESSAGE = "Segment veya referans ses kuyruga alindiktan sonra degisti."


def process_job(project: StudioProject, engine: TTSEngine, job: dict[str, Any]) -> str:
    """Run one claimed job. Returns ``done``, ``failed``, ``cancelled`` or ``lost`` (lease taken over)."""
    if project.current_key_for(job["segment_id"]) != job["cache_key"]:
        return "cancelled" if project.finish_job(job, "cancelled", CHANGED_MESSAGE) else "lost"

    reusable = project.find_artifact_by_key(job["cache_key"])
    if reusable:
        owned = project.complete_job(job, Path(reusable["path"]), reusable["sha256"], reusable["duration_s"])
        return "done" if owned else "lost"

    settings = project.settings
    audio_dir = project.store.audio_dir
    # Her deneme kendi gecici klasorune ve kendi son dosyasina yazar: kurtarilan is ayni job id ile
    # yeniden alindigindan, kiralamasi dusmus eski worker ile yeni worker hicbir yolu paylasmaz.
    attempt = uuid.uuid4().hex[:8]
    temp_dir = audio_dir / TEMP_DIRNAME / f"job{job['id']}_{attempt}"
    final_path = audio_dir / f"seg{job['segment_id']:05d}_{job['cache_key'][:16]}_{attempt}.wav"
    try:
        with _Heartbeat(project, job):
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
        # Referans ses veya metin uretim sirasinda degistiyse bu ses eski anahtara yazilmaz.
        if project.current_key_for(job["segment_id"]) != job["cache_key"]:
            return "cancelled" if project.finish_job(job, "cancelled", CHANGED_MESSAGE) else "lost"
        os.replace(result["final_output_path"], final_path)
        owned = project.complete_job(job, final_path, file_sha256(final_path), wav_duration(final_path))
        if not owned:
            final_path.unlink(missing_ok=True)  # sahipsiz deneme dosyasi; kayitli artefakt baska dosyada
            return "lost"
        return "done"
    except Exception as exc:  # noqa: BLE001 - every engine error must become a failed job
        failed = project.finish_job(job, "failed", f"{type(exc).__name__}: {exc}")
        return "failed" if failed else "lost"
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
    counts = {"done": 0, "failed": 0, "cancelled": 0, "lost": 0}
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
