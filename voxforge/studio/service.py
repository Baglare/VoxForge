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

import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import unicodedata
import uuid
import wave
from contextlib import closing
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable
from zipfile import BadZipFile

from scripts.audio_concat_utils import concatenate_wavs
from voxforge.engines import TTSEngine, engine_spec
from voxforge.playground.adapter import WorkerEngineAdapter, worker_engine_spec
from voxforge.playground.audio_join import read_part, render_gain
from voxforge.playground.prosody import ControlError, VoxControls
from voxforge.playground.service import DEFAULT_SEED, VARIANTS
from voxforge.playground.workers import EngineManager, WorkerError
from voxforge.quality.director import QualityDirector
from voxforge.runs import utc_now_iso
from voxforge.studio.backup import backup_project, restore_project
from voxforge.studio.book import read_book
from voxforge.studio.design import DesignLibrary
from voxforge.studio.project import ProjectSettings, StudioProject, file_sha256
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
    seen: set[tuple[str, bool]] = set()
    for number, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise StudioError(f"Sözlük satırı {number}: 'kelime = okunuş' biçiminde olmalı.")
        word, spoken = (part.strip() for part in line.split("=", 1))
        mode = word[:1]
        if mode in ("!", "~"):
            word = word[1:].strip()
        word, spoken = " ".join(word.split()), " ".join(spoken.split())
        if not word or not spoken:
            raise StudioError(f"Sözlük satırı {number}: kelime ve okunuş boş olamaz.")
        exact = mode == "!" or (mode != "~" and word != word.lower())
        key = (word if exact else word.casefold(), exact)
        if key in seen:
            raise StudioError(f"Sözlük satırı {number}: yinelenen giriş: {word}")
        seen.add(key)
        entries.append(LexiconEntry(word=word, spoken=spoken, case_sensitive=exact))
    return entries


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


class StudioService:
    def __init__(
        self, root: Path, engine_provider: Callable[[], TTSEngine] | None = None, *,
        manager: EngineManager | None = None, quality: QualityDirector | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.designs = DesignLibrary(self.root)
        self.engine_provider = engine_provider
        self.manager = manager
        self.quality = quality
        self._projects: dict[str, StudioProject] = {}
        self._workers: dict[str, StudioWorker] = {}
        self._lock = threading.Lock()

    def _engine_spec(self, engine_id: str | None = None) -> dict[str, Any]:
        if self.manager is not None:
            return worker_engine_spec(engine_id or "voxcpm2", self.manager.env)
        spec = engine_spec()
        if engine_id is not None and engine_id != spec["engine_id"]:
            raise StudioError(f"Bu oturum {spec['engine_id']} kullanıyor. {engine_id} için Playground'da Studio'yu açın.")
        return spec

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
                database = self.project_dir(slug) / DB_FILENAME
                with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
                    name = db.execute("SELECT name FROM project WHERE id = 1").fetchone()[0]
            except (sqlite3.Error, StudioError, TypeError):  # a broken project must not hide the others
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
        *,
        engine_id: str | None = None,
        document: Path | None = None,
    ) -> str:
        name = (name or "").strip()
        if not name:
            raise StudioError("Proje adı gerekli.")
        if document is not None and (text or "").strip():
            raise StudioError("Bir kaynak seçin: metin alanı veya dosya.")
        if document is None and not (text or "").strip():
            raise StudioError("Metin boş olamaz.")
        spec = self._engine_spec(engine_id)
        if spec["needs_reference"] and (reference is None or not Path(reference).is_file()):
            raise StudioError(f"{spec['engine_id']} motoru referans ses gerektirir; profil seçin veya ses yükleyin.")
        entries = parse_lexicon(lexicon_text)
        try:
            book = read_book(document) if document is not None else None
        except (OSError, ValueError) as exc:
            raise StudioError(str(exc)) from exc

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
                max_chars=VARIANTS[spec["engine_id"]].max_chars,
            )
            if book:
                (folder / f"source{book.extension}").write_bytes(book.original)
            project = StudioProject.create(
                folder, name, text, settings, entries,
                chapters=book.chapters if book else None, source_format=book.extension if book else None,
            )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)  # yalnız bu çağrının az önce açtığı boş proje klasörü
            raise
        if not project.segments():
            shutil.rmtree(folder, ignore_errors=True)
            raise StudioError("Metinden segment çıkarılamadı.")
        with self._lock:
            self._projects[slug] = project
        return slug

    def backup(self, slug: str) -> Path:
        folder = self.project_dir(slug)
        database = folder / DB_FILENAME
        if database.resolve() != database:
            raise StudioError("Yedek veritabanı proje içinde olmalı.")
        project = self.open(slug)
        exports = folder / EXPORTS_DIRNAME
        try:
            exports.mkdir(exist_ok=True)
            if exports.resolve() != exports:
                raise StudioError("Yedek dizini proje içinde olmalı.")
            return backup_project(project.store, exports / f"yedek_{uuid.uuid4().hex}.zip")
        except (OSError, ValueError, sqlite3.Error) as exc:
            raise StudioError(f"Yedek oluşturulamadı: {exc}") from exc

    def restore(self, bundle: Path, name: str) -> str:
        if not (name or "").strip():
            raise StudioError("Geri yükleme için yeni bir proje adı girin.")
        slug = slugify(name)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            restore_project(bundle, self.project_dir(slug), name.strip())
        except (OSError, ValueError, sqlite3.Error, BadZipFile, RuntimeError, TypeError, AttributeError) as exc:
            raise StudioError(f"Yedek geri yüklenemedi: {exc}") from exc
        return slug

    # ---- segments ----------------------------------------------------
    def segments(self, slug: str) -> list[dict[str, Any]]:
        return self.open(slug).segments()

    def chapters(self, slug: str) -> list[tuple[int, str]]:
        return self.open(slug).chapters()

    def lexicon_text(self, slug: str) -> str:
        return "\n".join(
            f"{'!' if entry.case_sensitive else '~'} {entry.word} = {entry.spoken}"
            for entry in self.open(slug).settings.lexicon_entries()
        )

    def update_lexicon(self, slug: str, text: str) -> int:
        project = self.open(slug)
        entries = parse_lexicon(text)
        with self._lock:
            if self.is_running(slug):
                raise StudioError("Sözlüğü kaydetmek için üretimi durdurun.")
            return project.update_lexicon(entries)

    def segment(self, slug: str, segment_id: int) -> dict[str, Any]:
        items = self.open(slug).segments(segment_id)
        if items:
            return items[0]
        raise StudioError("Segment bulunamadı.")

    def take_view(self, slug: str, segment_id: int) -> dict[str, Any]:
        project = self.open(slug)
        item = self.segment(slug, segment_id)
        return {**item, **project.take_details(item)}

    def change_take(self, slug: str, segment_id: int, cache_key: str, take_id: int | None) -> None:
        try:
            self.open(slug).change_take(segment_id, cache_key, take_id)
        except ValueError as exc:
            raise StudioError(str(exc)) from exc

    def check_take(self, slug: str, segment_id: int, cache_key: str, take_id: int) -> None:
        project = self.open(slug)
        item = self.take_view(slug, segment_id)
        if item["cache_key"] != cache_key:
            raise StudioError("Segment değişti. Yeniden seçin.")
        take = next((take for take in item["takes"] if take["id"] == take_id and take["path"]), None)
        if not take:
            raise StudioError("Denetlenecek take bulunamadı.")
        path = Path(take["path"])
        reference, _ = project.synthesis_voice(item["voice_id"])
        director = self.quality or QualityDirector(None)
        pauses = (take["synthesis"].get("join") or {}).get("pauses")
        try:
            if file_sha256(path) != take["sha256"]:
                raise ValueError("Take'in WAV dosyası değişmiş.")
            report = director.check(path, item["spoken_text"], pauses, Path(reference) if reference else None)
            project.save_quality(item, take, report)
        except (OSError, ValueError) as exc:
            raise StudioError(str(exc)) from exc

    def update_source(self, slug: str, segment_id: int, text: str) -> None:
        if not (text or "").strip():
            raise StudioError("Orijinal metin boş olamaz.")
        try:
            self.open(slug).update_segment_text(segment_id, text)
        except ValueError as exc:
            raise StudioError(str(exc)) from exc

    def set_spoken(self, slug: str, segment_id: int, spoken: str) -> None:
        try:
            self.open(slug).set_spoken_text(segment_id, spoken)
        except ValueError as exc:
            raise StudioError(str(exc)) from exc

    # ---- voices (VF-007) ------------------------------------------------
    def voices(self, slug: str) -> list[dict[str, Any]]:
        return list(self.open(slug).voices().values())

    def preview_design(self, description: str, text: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
        """Audition one short utterance using native, reference-free VoxCPM2 synthesis."""
        if self.manager is None:
            raise StudioError("Ses tasarımı için Voice Playground’da Studio’yu açın.")
        try:
            description = VoxControls(style=description or "").validated().style
        except ControlError as exc:
            raise StudioError(str(exc)) from exc
        text = " ".join((text or "").split())
        limit = VARIANTS["voxcpm2"].max_chars
        if not description or not text or len(text) > limit:
            raise StudioError(f"Ses tanımı ve 1–{limit} karakterlik bir önizleme repliği girin.")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise StudioError("Tohum 0 ile 4294967295 arasında bir tam sayı olmalı.")

        token = uuid.uuid4().hex
        folder = self.designs.preview_dir(token)
        folder.mkdir(parents=True)
        wav = folder / "voice.wav"
        model_id = self._engine_spec("voxcpm2")["model_id"]
        try:
            reply = self.manager.synthesize(
                "voxcpm2", {"chunks": [text], "style": description, "seed": seed,
                           "voice_design": True, "out": str(wav)}, device="cuda",
            )
            source = (reply.get("worker") or {}).get("source") or {}
            actual = f"{source.get('model')}@{source.get('revision')}"
            if actual != model_id or (reply.get("applied_settings") or {}).get("mode") != "voice_design":
                raise ValueError("Sabit modelde referanssız ses tasarımı doğrulanamadı.")
            if _wav_seconds(wav) <= 0:
                raise ValueError("Ses tasarımı WAV dosyası boş.")
            metadata = {
                "id": token, "description": description, "text": text, "seed": seed, "model_id": model_id,
                "created_at": utc_now_iso(), "sha256": file_sha256(wav), "vram_peak_mb": reply.get("vram_peak_mb"),
            }
            with (folder / "voice.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False)
        except (OSError, ValueError, wave.Error, WorkerError) as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise StudioError(f"Ses tasarımı üretilemedi: {exc}") from exc
        return {**metadata, "wav": wav}

    def save_design(self, preview_id: str, name: str) -> dict[str, Any]:
        try:
            return self.designs.save(preview_id, name)
        except (OSError, ValueError, KeyError) as exc:
            raise StudioError(f"Ses tasarımı kaydedilemedi: {exc}") from exc

    def add_designed_voice(
        self, slug: str, name: str, version_id: str, preset: dict[str, Any] | None = None,
    ) -> str:
        project = self.open(slug)
        try:
            version = self.designs.version(version_id)
        except (OSError, ValueError, KeyError) as exc:
            raise StudioError(f"Ses tasarımı açılamadı: {exc}") from exc
        if project.settings.engine_id != "voxcpm2" or project.settings.model_id != version["model_id"]:
            raise StudioError("Tasarım, aynı VoxCPM2 model revizyonunu kullanan bir projeye atanabilir.")
        return self.add_voice(slug, name.strip() or version["name"], version["wav"], preset, design_version=version_id)

    def add_voice(
        self,
        slug: str,
        name: str,
        reference: Path | None,
        preset: dict[str, Any] | None = None,
        *,
        design_version: str | None = None,
    ) -> str:
        """Add a character voice; its reference is copied into ``voices/`` inside the project."""
        project = self.open(slug)
        name = (name or "").strip()
        if not name:
            raise StudioError("Karakter adı gerekli.")
        spec = worker_engine_spec(project.settings.engine_id)
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
        try:
            if reference is not None and Path(reference).is_file():
                voices_dir = self.project_dir(slug) / VOICES_DIRNAME
                voices_dir.mkdir(exist_ok=True)
                filename = f"{key}_{uuid.uuid4().hex[:8]}.wav" if design_version else f"{key}.wav"
                reference_path = voices_dir / filename
                shutil.copy2(reference, reference_path)
            project.add_voice(
                key, name, str(reference_path) if reference_path else None,
                dict(preset or {}) if spec["supports_presets"] else {},
                design_version=design_version,
            )
        except (OSError, ValueError) as exc:
            if design_version and reference_path:
                reference_path.unlink(missing_ok=True)
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

    def start(self, slug: str, *, chapter_id: int | None = None) -> int:
        """Queue segments missing audio for their current key and start the worker.

        Returns the number of jobs waiting in the queue. Jobs queued earlier but never
        processed (e.g. after a crash and reopen) also start the worker, even when
        nothing new was queued.
        """
        project = self.open(slug)
        self._validate_chapter(project, chapter_id)
        with self._lock:
            if self.is_running(slug):
                raise StudioError("Üretim zaten devam ediyor.")
            self._validate_engine(project)
            project.enqueue_missing(chapter_id=chapter_id)
            if self.manager is not None:
                project.resume_takes(chapter_id=chapter_id)
            return self._start_worker(slug, project, chapter_id=chapter_id)

    def start_segment(self, slug: str, segment_id: int) -> int:
        """Queue a missing or changed selected sentence without draining other jobs."""
        project = self.open(slug)
        with self._lock:
            if self.is_running(slug):
                raise StudioError("Tek segment üretmek için kuyruk durana kadar bekleyin.")
            self._validate_engine(project)
            rows = project.segments(segment_id)
            if not rows:
                raise StudioError("Segment bulunamadı.")
            item = rows[0]
            if item["state"] == "done":
                raise StudioError("Segment hazır. Yeni bir okuma için alternatif take üretin.")
            if item["selected_take"]:
                raise StudioError("Seçili take eksik. Önce mevcut bir take seçin.")
            project.enqueue_missing([segment_id])
            job_id = project.unfinished_base_job(segment_id, item["cache_key"])
            return self._start_worker(slug, project, job_id=job_id) if job_id is not None else 0

    def start_take(self, slug: str, segment_id: int, cache_key: str) -> None:
        project = self.open(slug)
        with self._lock:
            if self.manager is None or project.settings.engine_id == "mock":
                raise StudioError("Alternatif üretmek için Playground Studio'da VoxCPM2 veya XTTS kullanın.")
            if self.is_running(slug):
                raise StudioError("Alternatif üretmek için kuyruk durana kadar bekleyin.")
            self._validate_engine(project)
            try:
                job_id = project.enqueue_take(segment_id, cache_key)
            except ValueError as exc:
                raise StudioError(str(exc)) from exc
            self._start_worker(slug, project, job_id=job_id)

    @staticmethod
    def _validate_chapter(project: StudioProject, chapter_id: int | None) -> None:
        if chapter_id is not None and not any(key == chapter_id for key, _ in project.chapters()):
            raise StudioError("Bölüm bulunamadı.")

    def _validate_engine(self, project: StudioProject) -> None:
        settings = project.settings
        spec = self._engine_spec(settings.engine_id)
        if spec["model_id"] != settings.model_id:
            raise StudioError(f"Bu proje {settings.model_id} ile oluşturuldu. Etkin model: {spec['model_id']}.")

    def _start_worker(
        self, slug: str, project: StudioProject, *, chapter_id: int | None = None, job_id: int | None = None,
    ) -> int:
        waiting = project.queued_job_count(
            include_takes=self.manager is not None, chapter_id=chapter_id, job_id=job_id,
        )
        if waiting:
            if self.manager is not None:
                engine = WorkerEngineAdapter(project.settings.engine_id, self.manager)
            else:
                assert self.engine_provider is not None
                engine = self.engine_provider()
            worker = StudioWorker(
                project, engine, self.quality, chapter_id=chapter_id, job_id=job_id,
            )
            self._workers[slug] = worker
            worker.start()
        return waiting

    def stop(self, slug: str) -> int:
        """Stop after the current job; cancelled alternatives retain their seeds for resume."""
        project = self.open(slug)
        with self._lock:
            worker = self._workers.get(slug)
            if worker:
                worker.stop_event.set()
                return project.cancel_queued(chapter_id=worker.chapter_id, job_id=worker.job_id)
            return project.cancel_queued()

    def wait(self, slug: str, timeout: float | None = None) -> None:
        worker = self._workers.get(slug)
        if worker and worker.thread:
            worker.thread.join(timeout)

    def change_mix(self, slug: str, segment_id: int, mix: tuple[float, float | None] | None) -> None:
        try:
            self.open(slug).change_mix(segment_id, mix)
        except ValueError as exc:
            raise StudioError(str(exc)) from exc

    @staticmethod
    def _pause_after(engine_id: str, current: dict[str, Any], following: dict[str, Any]) -> float:
        if current["pause_s"] is not None:
            return current["pause_s"]
        preset = current["preset"]
        if engine_id != "voxcpm2" or preset.get("pacing") != "narration":
            return 0.0
        try:
            controls = VoxControls.from_dict(preset)
        except ControlError as exc:
            raise StudioError(str(exc)) from exc
        if (current["chapter_id"], current["paragraph"]) != (following["chapter_id"], following["paragraph"]):
            return controls.paragraph_pause
        return controls.sentence_pause

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
        project = self.open(slug)
        segments = project.segments()
        engine_id = project.settings.engine_id
        cursor: float | None = 0.0
        counts: dict[str, int] = {}
        for index, item in enumerate(segments):
            item["timeline_start_s"] = cursor
            following = segments[index + 1] if index + 1 < len(segments) else None
            pause = self._pause_after(engine_id, item, following) if following is not None else None
            item["timeline_pause_s"] = pause
            if cursor is not None:
                duration = item["duration_s"]
                cursor = None if duration is None else cursor + duration + (pause or 0.0)
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

    def build_preview(self, slug: str, *, chapter_id: int | None = None) -> Path:
        return self.export_audio(slug, chapter_id=chapter_id)

    def export_audio(self, slug: str, *, chapter_id: int | None = None, audio_format: str = "wav") -> Path:
        path, _ = self.export_audio_with_report(slug, chapter_id=chapter_id, audio_format=audio_format)
        return path

    def export_audio_with_report(
        self, slug: str, *, chapter_id: int | None = None, audio_format: str = "wav",
    ) -> tuple[Path, list[str]]:
        """Export active takes and report any boost limited by peak protection."""
        if audio_format not in ("wav", "mp3"):
            raise StudioError("Çıktı biçimi WAV veya MP3 olmalı.")
        project = self.open(slug)
        self._validate_chapter(project, chapter_id)
        segments = project.segments(chapter_id=chapter_id)
        missing = [str(item["ord"] + 1) for item in segments if item["state"] != "done"]
        if missing:
            shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            raise StudioError(f"Önce tüm segmentler üretilmeli. Eksik segment: {shown}")
        if not segments:
            raise StudioError("Projede segment yok.")
        engine_id = project.settings.engine_id
        pauses = [self._pause_after(engine_id, current, following) for current, following in pairwise(segments)]
        exports = self.project_dir(slug) / EXPORTS_DIRNAME
        exports.mkdir(exist_ok=True)
        prefix = f"bolum_{chapter_id:03d}" if chapter_id is not None else "kitap"
        target = exports / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{audio_format}"
        paths = [Path(item["audio_path"]) for item in segments]
        write = self._write_export_wav if audio_format == "wav" else self._write_export_mp3
        warnings: list[str] = []
        if any(item["gain_db"] for item in segments):
            with tempfile.TemporaryDirectory(prefix=".mix-", dir=exports) as temporary:
                for index, item in enumerate(segments):
                    requested = item["gain_db"]
                    if requested:
                        paths[index], applied = self._render_gain(paths[index], Path(temporary) / f"{index}.wav", requested)
                        if requested > 0 and applied < requested:
                            warnings.append(
                                f"Segment {item['ord'] + 1}: tepe koruması nedeniyle "
                                f"istenen +{requested:.2f} dB artışın +{applied:.2f} dB kadarı uygulandı."
                            )
                write(paths, target, pauses)
        else:
            write(paths, target, pauses)
        return target, warnings

    @staticmethod
    def _render_gain(source: Path, target: Path, gain_db: float) -> tuple[Path, float]:
        try:
            part, applied = render_gain(read_part(source), gain_db, clamp_boost=True)
            if applied == 0.0:
                return source, applied
            with wave.open(str(target), "wb") as destination:
                destination.setparams((part.channels, 2, part.rate, 0, "NONE", "not compressed"))
                destination.writeframes(part.samples.tobytes())
        except (ValueError, wave.Error, EOFError) as exc:
            raise StudioError(f"Segment seviyesi uygulanamadı: {exc}") from exc
        return target, applied

    @staticmethod
    def _write_export_wav(paths: list[Path], target: Path, pauses: list[float] | None) -> None:
        ok, message = concatenate_wavs(paths, target, pauses=pauses)
        if not ok:
            target.unlink(missing_ok=True)
            raise StudioError(f"Birleştirme başarısız: {message}")

    def _write_export_mp3(self, paths: list[Path], target: Path, pauses: list[float] | None) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise StudioError("MP3 dışa aktarmak için FFmpeg gerekli.")
        with tempfile.TemporaryDirectory(prefix=".mp3-", dir=target.parent) as temporary:
            wav, mp3 = Path(temporary) / "book.wav", Path(temporary) / "book.mp3"
            self._write_export_wav(paths, wav, pauses)
            try:
                result = subprocess.run(
                    [ffmpeg, "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "libmp3lame", "-q:a", "2", str(mp3)],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode != 0 or not mp3.is_file() or not mp3.stat().st_size:
                    raise StudioError(f"MP3 kodlanamadı: {result.stderr.strip()}")
                mp3.replace(target)
            except OSError as exc:
                raise StudioError(f"MP3 dışa aktarılamadı: {exc}") from exc

    @staticmethod
    def preview_seconds(path: Path) -> float:
        return _wav_seconds(path)
