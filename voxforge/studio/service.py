"""Studio service layer used by the Gradio Studio tab (VF-006).

Keeps all project/queue logic out of the UI so it can be tested with the Mock
engine. One process = one local user, so open projects and their background
workers are kept in a small in-memory registry.

Local data safety:
- project slugs are validated and must resolve inside the Studio root;
- the reference voice is *copied* into the project folder, so later changes to
  a profile do not silently alter (or invalidate) an existing project;
- nothing here deletes user audio; previews are written to ``exports/``.
"""

from __future__ import annotations

import re
import shutil
import threading
import unicodedata
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from scripts.audio_concat_utils import concatenate_wavs
from voxforge.engines import ToFileEngineAdapter, engine_spec
from voxforge.studio.project import ProjectSettings, StudioProject
from voxforge.studio.queue import StudioWorker
from voxforge.studio.store import DB_FILENAME
from voxforge.text import LexiconEntry

REFERENCE_FILENAME = "reference.wav"
EXPORTS_DIRNAME = "exports"
STATE_LABELS = {
    "done": "Tamam",
    "running": "Üretiliyor",
    "queued": "Kuyrukta",
    "failed": "Hata",
    "cancelled": "Durduruldu",
    "pending": "Bekliyor",
}
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_TR_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


class StudioError(Exception):
    """User-facing error with a plain Turkish message."""


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.translate(_TR_ASCII)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]
    return slug or "proje"


def parse_lexicon(text: str) -> list[LexiconEntry]:
    """One entry per line: ``kelime = okunuş``. Lines starting with ``#`` are ignored."""
    entries: list[LexiconEntry] = []
    for number, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise StudioError(f"Sözlük satırı {number}: 'kelime = okunuş' biçiminde olmalı.")
        word, spoken = (part.strip() for part in line.split("=", 1))
        if not word or not spoken:
            raise StudioError(f"Sözlük satırı {number}: kelime ve okunuş boş olamaz.")
        entries.append(LexiconEntry(word=word, spoken=spoken, case_sensitive=word != word.lower()))
    return entries


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


class StudioService:
    def __init__(self, root: Path, engine_provider: Callable[[], ToFileEngineAdapter]) -> None:
        self.root = Path(root)
        self.engine_provider = engine_provider
        self._projects: dict[str, StudioProject] = {}
        self._workers: dict[str, StudioWorker] = {}
        self._lock = threading.Lock()

    # ---- projects ----------------------------------------------------
    def project_dir(self, slug: str) -> Path:
        if not slug or not _SLUG_RE.fullmatch(slug):
            raise StudioError("Geçersiz proje adı.")
        folder = (self.root / slug).resolve()
        if folder.parent != self.root.resolve():
            raise StudioError("Proje klasörü Studio kökünün dışında olamaz.")
        return folder

    def list_projects(self) -> list[tuple[str, str]]:
        if not self.root.is_dir():
            return []
        found = []
        for folder in self.root.iterdir():
            if folder.is_dir() and _SLUG_RE.fullmatch(folder.name) and (folder / DB_FILENAME).is_file():
                found.append((folder.stat().st_mtime, folder.name))
        choices = []
        for _, slug in sorted(found, reverse=True):
            try:
                name = self.open(slug).name
            except Exception:  # noqa: BLE001 - a broken project must not hide the others
                name = f"{slug} (açılamadı)"
            choices.append((name, slug))
        return choices

    def open(self, slug: str) -> StudioProject:
        with self._lock:
            if slug not in self._projects:
                folder = self.project_dir(slug)
                if not (folder / DB_FILENAME).is_file():
                    raise StudioError("Proje bulunamadı.")
                self._projects[slug] = StudioProject.open(folder)
            return self._projects[slug]

    def create(
        self,
        name: str,
        text: str,
        reference: Path | None,
        preset: dict[str, Any] | None = None,
        lexicon_text: str = "",
    ) -> str:
        name = (name or "").strip()
        if not name:
            raise StudioError("Proje adı gerekli.")
        if not (text or "").strip():
            raise StudioError("Metin boş olamaz.")
        spec = engine_spec()
        if spec["needs_reference"] and (reference is None or not Path(reference).is_file()):
            raise StudioError(f"{spec['engine_id']} motoru referans ses gerektirir; profil seçin veya ses yükleyin.")
        entries = parse_lexicon(lexicon_text)

        base = slugify(name)
        slug, counter = base, 2
        while (self.root / slug).exists():
            slug, counter = f"{base}-{counter}", counter + 1
        folder = self.project_dir(slug)
        folder.mkdir(parents=True)
        try:
            reference_path = None
            if reference is not None and spec["needs_reference"]:
                reference_path = folder / REFERENCE_FILENAME
                shutil.copy2(reference, reference_path)
            settings = ProjectSettings(
                engine_id=spec["engine_id"],
                model_id=spec["model_id"],
                reference_path=str(reference_path) if reference_path else None,
                preset=dict(preset or {}) if spec["supports_presets"] else {},
            )
            project = StudioProject.create(folder, name, text, settings, entries)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)  # yalnız bu çağrının az önce açtığı boş proje klasörü
            raise
        if not project.segments():
            shutil.rmtree(folder, ignore_errors=True)
            raise StudioError("Metinden segment çıkarılamadı.")
        with self._lock:
            self._projects[slug] = project
        return slug

    # ---- segments ----------------------------------------------------
    def segments(self, slug: str) -> list[dict[str, Any]]:
        return self.open(slug).segments()

    def segment(self, slug: str, segment_id: int) -> dict[str, Any]:
        for item in self.segments(slug):
            if item["id"] == segment_id:
                return item
        raise StudioError("Segment bulunamadı.")

    def update_source(self, slug: str, segment_id: int, text: str) -> None:
        if not (text or "").strip():
            raise StudioError("Orijinal metin boş olamaz.")
        self.open(slug).update_segment_text(segment_id, text)

    def set_spoken(self, slug: str, segment_id: int, spoken: str) -> None:
        try:
            self.open(slug).set_spoken_text(segment_id, spoken)
        except ValueError as exc:
            raise StudioError(str(exc)) from exc

    # ---- generation --------------------------------------------------
    def is_running(self, slug: str) -> bool:
        worker = self._workers.get(slug)
        return bool(worker and worker.running)

    def start(self, slug: str) -> int:
        """Queue segments missing audio for their current key and start the worker.

        Returns the number of jobs waiting in the queue. Jobs queued earlier but never
        processed (e.g. after a crash and reopen) also start the worker, even when
        nothing new was queued.
        """
        project = self.open(slug)
        if self.is_running(slug):
            raise StudioError("Üretim zaten devam ediyor.")
        settings = project.settings
        spec = engine_spec()
        if (spec["engine_id"], spec["model_id"]) != (settings.engine_id, settings.model_id):
            raise StudioError(
                f"Bu proje {settings.engine_id} motoruyla oluşturuldu; uygulama şu an {spec['engine_id']} ile çalışıyor."
            )
        project.enqueue_missing()
        waiting = project.queued_job_count()
        if waiting:
            worker = StudioWorker(project, self.engine_provider())
            self._workers[slug] = worker
            worker.start()
        return waiting

    def stop(self, slug: str) -> int:
        """Stop after the current segment; queued segments are cancelled (restart re-queues them)."""
        worker = self._workers.get(slug)
        if worker:
            worker.stop_event.set()
        return self.open(slug).cancel_queued()

    def wait(self, slug: str, timeout: float | None = None) -> None:
        worker = self._workers.get(slug)
        if worker and worker.thread:
            worker.thread.join(timeout)

    def snapshot(self, slug: str) -> dict[str, Any]:
        """Consistent view for the UI: worker state first, then segments.

        The worker thread exits only after its last job is committed. Reading
        ``running`` *before* the segments therefore guarantees that a snapshot
        with ``running=False`` already contains every finished segment. (Reading it
        last could pair a stale table with ``running=False`` and the UI would stop
        refreshing on that stale state.)
        """
        running = self.is_running(slug)
        worker = self._workers.get(slug)
        segments = self.segments(slug)
        counts: dict[str, int] = {}
        for item in segments:
            counts[item["state"]] = counts.get(item["state"], 0) + 1
        return {
            "segments": segments,
            "total": len(segments),
            "counts": counts,
            "running": running,
            "worker_error": worker.last_error if worker else None,
            "duration_s": sum(item["duration_s"] or 0.0 for item in segments),
        }

    def progress(self, slug: str) -> dict[str, Any]:
        info = self.snapshot(slug)
        info.pop("segments")
        return info

    def build_preview(self, slug: str) -> Path:
        """Concatenate all segments in order into ``exports/``; refuses if any segment is missing."""
        segments = self.segments(slug)
        missing = [str(item["ord"] + 1) for item in segments if item["state"] != "done"]
        if missing:
            shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            raise StudioError(f"Önce tüm segmentler üretilmeli. Eksik segment: {shown}")
        if not segments:
            raise StudioError("Projede segment yok.")
        exports = self.project_dir(slug) / EXPORTS_DIRNAME
        exports.mkdir(exist_ok=True)
        target = exports / f"onizleme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        ok, message = concatenate_wavs([Path(item["audio_path"]) for item in segments], target)
        if not ok:
            raise StudioError(f"Birleştirme başarısız: {message}")
        return target

    @staticmethod
    def preview_seconds(path: Path) -> float:
        return _wav_seconds(path)
