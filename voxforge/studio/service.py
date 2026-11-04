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
from voxforge.text.dialogue import dialogue_hint, speaker_label

REFERENCE_FILENAME = "reference.wav"
EXPORTS_DIRNAME = "exports"
VOICES_DIRNAME = "voices"
NARRATOR_LABEL = "Anlatıcı"
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

    # ---- voices (VF-007) ------------------------------------------------
    def voices(self, slug: str) -> list[dict[str, Any]]:
        return list(self.open(slug).voices().values())

    def add_voice(
        self,
        slug: str,
        name: str,
        reference: Path | None,
        preset: dict[str, Any] | None = None,
    ) -> str:
        """Add a character voice; its reference is copied into ``voices/`` inside the project."""
        project = self.open(slug)
        name = (name or "").strip()
        if not name:
            raise StudioError("Karakter adı gerekli.")
        spec = engine_spec(project.settings.engine_id)
        if spec["needs_reference"] and (reference is None or not Path(reference).is_file()):
            raise StudioError(f"{spec['engine_id']} motorunda her karakter için referans ses gerekir.")
        existing = project.voices()
        if any(voice["name"].lower() == name.lower() for voice in existing.values()):
            raise StudioError(f"Bu adla bir karakter zaten var: {name}")
        base = slugify(name)
        key, counter = base, 2
        while key in existing:
            key, counter = f"{base}-{counter}", counter + 1
        reference_path = None
        if reference is not None and Path(reference).is_file():
            voices_dir = self.project_dir(slug) / VOICES_DIRNAME
            voices_dir.mkdir(exist_ok=True)
            reference_path = voices_dir / f"{key}.wav"
            shutil.copy2(reference, reference_path)
        try:
            project.add_voice(
                key, name, str(reference_path) if reference_path else None,
                dict(preset or {}) if spec["supports_presets"] else {},
            )
        except ValueError as exc:
            raise StudioError(str(exc)) from exc
        return key

    def segment_ids_from_numbers(self, slug: str, numbers: str) -> list[int]:
        """``"3, 5-8"`` (1-based, as shown in the table) -> segment ids."""
        by_number = {item["ord"] + 1: item["id"] for item in self.segments(slug)}
        wanted: list[int] = []
        for part in (numbers or "").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "-" in part:
                    first, last = (int(value) for value in part.split("-", 1))
                    span = range(min(first, last), max(first, last) + 1)
                else:
                    span = [int(part)]
            except ValueError as exc:
                raise StudioError(f"Geçersiz segment numarası: {part}") from exc
            for number in span:
                if number not in by_number:
                    raise StudioError(f"Segment {number} yok (1–{len(by_number)}).")
                if by_number[number] not in wanted:
                    wanted.append(by_number[number])
        if not wanted:
            raise StudioError("Segment numarası girin; örnek: 3, 5-8")
        return wanted

    def assign_voice(self, slug: str, segment_ids: list[int], voice_key: str | None) -> int:
        try:
            return self.open(slug).assign_voice(segment_ids, voice_key or None)
        except KeyError as exc:
            raise StudioError("Seçilen karakter bulunamadı.") from exc

    def dialogue_suggestions(self, slug: str) -> list[dict[str, Any]]:
        """Segments that look like dialogue, with the matching existing voice if the speaker is named."""
        by_name = {voice["name"].lower(): voice["key"] for voice in self.voices(slug)}
        suggestions = []
        for item in self.segments(slug):
            hint = dialogue_hint(item["source_text"])
            if hint is None:
                continue
            suggestions.append(
                {
                    "segment_id": item["id"],
                    "number": item["ord"] + 1,
                    "kind": hint.kind,
                    "speaker": hint.speaker,
                    "line": hint.line,
                    "voice_key": by_name.get(hint.speaker.lower()) if hint.speaker else None,
                    "current_voice": item["voice_id"],
                }
            )
        return suggestions

    def apply_suggestions(self, slug: str, strip_labels: bool = True) -> dict[str, Any]:
        """Assign named speakers that match an existing character; optionally drop ``Ad:`` from the spoken text.

        Unnamed dialogue and speakers without a character are only reported.
        """
        project = self.open(slug)
        spoken_by_id = {item["id"]: item["spoken_text"] for item in self.segments(slug)}
        assigned = relabelled = 0
        unmatched: set[str] = set()
        for suggestion in self.dialogue_suggestions(slug):
            if suggestion["kind"] != "named":
                continue
            if suggestion["voice_key"] is None:
                unmatched.add(suggestion["speaker"])
                continue
            assigned += project.assign_voice([suggestion["segment_id"]], suggestion["voice_key"])
            if strip_labels:
                if speaker_label(spoken_by_id[suggestion["segment_id"]]):
                    project.set_spoken_from_text(suggestion["segment_id"], suggestion["line"])
                    relabelled += 1
        return {"assigned": assigned, "relabelled": relabelled, "unmatched": sorted(unmatched)}

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
