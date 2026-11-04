"""Studio project: segments, deterministic cache keys and job bookkeeping (VF-005).

A segment is "done" when an artifact exists for its *current* cache key. The
key covers everything that changes the audio (spoken text, engine, model,
reference file content, preset, language, limits, normaliser and adapter
versions), so editing one sentence invalidates only that segment.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from voxforge.runs import utc_now_iso
from voxforge.studio.store import ProjectStore
from voxforge.text import NORMALIZER_VERSION, LexiconEntry, normalize_text, prepare_segments

ADAPTER_VERSION = "1"
# A running job whose heartbeat is older than this is considered abandoned (crash/kill).
LEASE_SECONDS = 60.0
HEARTBEAT_SECONDS = 15.0


@dataclass
class ProjectSettings:
    engine_id: str
    model_id: str
    reference_path: str | None = None
    preset: dict[str, Any] = field(default_factory=dict)
    language: str = "tr"
    max_chars: int = 220
    lexicon: list[dict[str, Any]] = field(default_factory=list)

    def lexicon_entries(self) -> list[LexiconEntry]:
        return [LexiconEntry(**item) for item in self.lexicon]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_supports_presets(engine_id: str) -> bool:
    from voxforge.engines import ENGINE_SPECS

    return bool(ENGINE_SPECS.get(engine_id, {}).get("supports_presets", False))


def _recover_stale_jobs(db) -> int:
    """Requeue running jobs without a live lease (no heartbeat, or older than LEASE_SECONDS)."""
    return db.execute(
        "UPDATE job SET status = 'queued', started_at = NULL, worker_id = NULL, heartbeat_at = NULL"
        " WHERE status = 'running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)",
        (time.time() - LEASE_SECONDS,),
    ).rowcount


def _transformations_json(result) -> str:
    return json.dumps([asdict(item) for item in result.transformations], ensure_ascii=False)


class StudioProject:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._reference_cache: dict[str, tuple[tuple[int, int], str]] = {}

    # ---- lifecycle -------------------------------------------------
    @classmethod
    def create(
        cls,
        folder: Path,
        name: str,
        text: str,
        settings: ProjectSettings,
        lexicon: Iterable[LexiconEntry] = (),
    ) -> "StudioProject":
        store = ProjectStore(folder)
        entries = list(lexicon)
        settings.lexicon = [asdict(entry) for entry in entries]
        now = utc_now_iso()
        with store.transaction() as db:
            if db.execute("SELECT 1 FROM project").fetchone():
                raise FileExistsError(f"Proje zaten var: {folder}")
            db.execute(
                "INSERT INTO project (id, name, created_at, settings_json) VALUES (1, ?, ?, ?)",
                (name, now, json.dumps(asdict(settings), ensure_ascii=False)),
            )
            for order, item in enumerate(prepare_segments(text, settings.max_chars, entries)):
                db.execute(
                    "INSERT INTO segment (ord, paragraph, source_text, spoken_text, transformations_json, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        order,
                        item.segment.paragraph,
                        item.segment.text,
                        item.spoken,
                        _transformations_json(item.normalization),
                        now,
                    ),
                )
        return cls(store)

    @classmethod
    def open(cls, folder: Path) -> "StudioProject":
        """Open and recover abandoned jobs.

        Only ``running`` jobs whose heartbeat is stale go back to ``queued``; a job
        held by a live worker (same or another process) is left alone.
        """
        store = ProjectStore(folder)
        with store.transaction() as db:
            if not db.execute("SELECT 1 FROM project").fetchone():
                raise FileNotFoundError(f"Studio projesi bulunamadi: {folder}")
            _recover_stale_jobs(db)
        return cls(store)

    # ---- settings & keys -------------------------------------------
    @property
    def name(self) -> str:
        with self.store.connect() as db:
            return db.execute("SELECT name FROM project WHERE id = 1").fetchone()["name"]

    @property
    def settings(self) -> ProjectSettings:
        with self.store.connect() as db:
            row = db.execute("SELECT settings_json FROM project WHERE id = 1").fetchone()
        return ProjectSettings(**json.loads(row["settings_json"]))

    def reference_hash(self, reference: str | None = None, *, use_project_default: bool = True) -> str | None:
        """Content hash of a reference WAV, re-read whenever its size or mtime changes."""
        if reference is None and use_project_default:
            reference = self.settings.reference_path
        if reference is None:
            return None
        path = Path(reference)
        try:
            stat = path.stat()
        except OSError:
            self._reference_cache.pop(str(path), None)
            return f"missing:{reference}"
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._reference_cache.get(str(path))
        if cached is None or cached[0] != signature:
            cached = (signature, file_sha256(path))
            self._reference_cache[str(path)] = cached
        return cached[1]

    # ---- voices (VF-007) -------------------------------------------
    def voices(self) -> dict[str, dict[str, Any]]:
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM voice ORDER BY created_at, key").fetchall()
        return {
            row["key"]: {
                "key": row["key"],
                "name": row["name"],
                "reference_path": row["reference_path"],
                "preset": json.loads(row["preset_json"]),
            }
            for row in rows
        }

    def add_voice(self, key: str, name: str, reference_path: str | None, preset: dict[str, Any]) -> None:
        with self.store.transaction() as db:
            if db.execute("SELECT 1 FROM voice WHERE key = ? OR lower(name) = lower(?)", (key, name)).fetchone():
                raise ValueError(f"Bu adla bir ses zaten var: {name}")
            db.execute(
                "INSERT INTO voice (key, name, reference_path, preset_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (key, name, reference_path, json.dumps(preset, ensure_ascii=False), utc_now_iso()),
            )

    def update_voice_preset(self, key: str, preset: dict[str, Any]) -> None:
        """Changing a voice's preset invalidates only segments using that voice (via the cache key)."""
        with self.store.transaction() as db:
            if not db.execute(
                "UPDATE voice SET preset_json = ? WHERE key = ?", (json.dumps(preset, ensure_ascii=False), key)
            ).rowcount:
                raise KeyError(f"Ses yok: {key}")

    def assign_voice(self, segment_ids: Iterable[int], voice_key: str | None) -> int:
        """Assign a voice (``None`` = narrator); queued jobs of those segments are cancelled."""
        ids = list(segment_ids)
        with self.store.transaction() as db:
            if voice_key is not None and not db.execute("SELECT 1 FROM voice WHERE key = ?", (voice_key,)).fetchone():
                raise KeyError(f"Ses yok: {voice_key}")
            now = utc_now_iso()
            changed = 0
            for segment_id in ids:
                changed += db.execute(
                    "UPDATE segment SET voice_id = ?, updated_at = ? WHERE id = ? AND voice_id IS NOT ?",
                    (voice_key, now, segment_id, voice_key),
                ).rowcount
                db.execute(
                    "UPDATE job SET status = 'cancelled', finished_at = ? WHERE segment_id = ? AND status = 'queued'",
                    (now, segment_id),
                )
        return changed

    def synthesis_voice(self, voice_key: str | None, voices: dict[str, dict[str, Any]] | None = None,
                        settings: ProjectSettings | None = None) -> tuple[str | None, dict[str, Any]]:
        """``(reference_path, preset)`` for a segment's voice; narrator uses the project defaults."""
        settings = settings or self.settings
        voice = (voices if voices is not None else self.voices()).get(voice_key) if voice_key else None
        if voice is None:
            return settings.reference_path, settings.preset
        preset = voice["preset"] if engine_supports_presets(settings.engine_id) else {}
        return voice["reference_path"], preset

    def cache_key(
        self,
        spoken_text: str,
        settings: ProjectSettings | None = None,
        voice_key: str | None = None,
        voices: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        settings = settings or self.settings
        reference, preset = self.synthesis_voice(voice_key, voices, settings)
        payload = {
            "spoken_text": spoken_text,
            "engine_id": settings.engine_id,
            "model_id": settings.model_id,
            "reference_sha256": self.reference_hash(reference, use_project_default=False),
            "preset": preset,
            "language": settings.language,
            "max_chars": settings.max_chars,
            "normalizer_version": NORMALIZER_VERSION,
            "adapter_version": ADAPTER_VERSION,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ---- segments --------------------------------------------------
    def segments(self) -> list[dict[str, Any]]:
        """Segments in order with derived state: done / queued / running / failed / cancelled / pending."""
        settings = self.settings
        voices = self.voices()
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM segment ORDER BY ord").fetchall()
            result = []
            for row in rows:
                key = self.cache_key(row["spoken_text"], settings, row["voice_id"], voices)
                artifact = db.execute(
                    "SELECT path, duration_s FROM artifact WHERE segment_id = ? AND cache_key = ?",
                    (row["id"], key),
                ).fetchone()
                job = db.execute(
                    "SELECT status, error FROM job WHERE segment_id = ? AND cache_key = ? ORDER BY id DESC LIMIT 1",
                    (row["id"], key),
                ).fetchone()
                if artifact and Path(artifact["path"]).is_file():
                    state, error = "done", None
                elif job:
                    state, error = job["status"], job["error"]
                    if state == "done":
                        state = "pending"  # artifact file disappeared
                else:
                    state, error = "pending", None
                result.append(
                    {
                        "id": row["id"],
                        "ord": row["ord"],
                        "paragraph": row["paragraph"],
                        "source_text": row["source_text"],
                        "spoken_text": row["spoken_text"],
                        "transformations": json.loads(row["transformations_json"]),
                        "voice_id": row["voice_id"],
                        "voice_name": voices[row["voice_id"]]["name"] if row["voice_id"] in voices else None,
                        "cache_key": key,
                        "state": state,
                        "error": error,
                        "audio_path": artifact["path"] if state == "done" else None,
                        "duration_s": artifact["duration_s"] if state == "done" else None,
                    }
                )
        return result

    def update_segment_text(self, segment_id: int, source_text: str) -> str:
        """Change one segment's source text; returns its new spoken text. Other segments are untouched."""
        settings = self.settings
        normalization = normalize_text(" ".join(source_text.split()), settings.lexicon_entries())
        with self.store.transaction() as db:
            updated = db.execute(
                "UPDATE segment SET source_text = ?, spoken_text = ?, transformations_json = ?, updated_at = ?"
                " WHERE id = ?",
                (
                    normalization.original,
                    normalization.spoken,
                    _transformations_json(normalization),
                    utc_now_iso(),
                    segment_id,
                ),
            ).rowcount
            if not updated:
                raise KeyError(f"Segment yok: {segment_id}")
            db.execute(
                "UPDATE job SET status = 'cancelled', finished_at = ? WHERE segment_id = ? AND status = 'queued'",
                (utc_now_iso(), segment_id),
            )
        return normalization.spoken

    def set_spoken_text(self, segment_id: int, spoken_text: str) -> None:
        """Manual pronunciation override; the original text stays as written."""
        spoken = " ".join(spoken_text.split())
        if not spoken:
            raise ValueError("Okunacak metin bos olamaz.")
        manual = json.dumps([{"rule": "manual", "original": None, "spoken": spoken}], ensure_ascii=False)
        with self.store.transaction() as db:
            updated = db.execute(
                "UPDATE segment SET spoken_text = ?, transformations_json = ?, updated_at = ? WHERE id = ?",
                (spoken, manual, utc_now_iso(), segment_id),
            ).rowcount
            if not updated:
                raise KeyError(f"Segment yok: {segment_id}")
            db.execute(
                "UPDATE job SET status = 'cancelled', finished_at = ? WHERE segment_id = ? AND status = 'queued'",
                (utc_now_iso(), segment_id),
            )

    # ---- jobs ------------------------------------------------------
    def enqueue_missing(self, segment_ids: Iterable[int] | None = None) -> int:
        """Queue segments whose current key has no artifact and no queued/running job (failed ones are retried)."""
        wanted = set(segment_ids) if segment_ids is not None else None
        now = utc_now_iso()
        rows = [
            (segment["id"], segment["cache_key"], now)
            for segment in self.segments()
            if (wanted is None or segment["id"] in wanted)
            and segment["state"] not in ("done", "queued", "running")
        ]
        with self.store.transaction() as db:
            db.executemany(
                "INSERT INTO job (segment_id, cache_key, status, queued_at) VALUES (?, ?, 'queued', ?)", rows
            )
        return len(rows)

    def queued_job_count(self) -> int:
        """Jobs waiting for a worker, after requeueing abandoned (stale-lease) running jobs."""
        with self.store.transaction() as db:
            _recover_stale_jobs(db)
            return int(db.execute("SELECT COUNT(*) FROM job WHERE status = 'queued'").fetchone()[0])

    def cancel_queued(self) -> int:
        with self.store.transaction() as db:
            return db.execute(
                "UPDATE job SET status = 'cancelled', finished_at = ? WHERE status = 'queued'",
                (utc_now_iso(),),
            ).rowcount

    def claim_next_job(self) -> dict[str, Any] | None:
        """Atomically move the oldest queued job to ``running`` under this worker's lease."""
        with self.store.transaction() as db:
            _recover_stale_jobs(db)
            row = db.execute(
                "SELECT job.id, job.segment_id, job.cache_key, job.attempts, segment.spoken_text, segment.voice_id"
                " FROM job JOIN segment ON segment.id = job.segment_id"
                " WHERE job.status = 'queued' ORDER BY segment.ord, job.id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE job SET status = 'running', attempts = attempts + 1, started_at = ?,"
                " worker_id = ?, heartbeat_at = ? WHERE id = ?",
                (utc_now_iso(), self.worker_id, time.time(), row["id"]),
            )
            return {**dict(row), "worker_id": self.worker_id}

    def heartbeat(self, job: dict[str, Any]) -> bool:
        """Extend the lease; False means the job was taken away (recovered by someone else)."""
        with self.store.transaction() as db:
            return bool(
                db.execute(
                    "UPDATE job SET heartbeat_at = ? WHERE id = ? AND status = 'running' AND worker_id = ?",
                    (time.time(), job["id"], job["worker_id"]),
                ).rowcount
            )

    def find_artifact_by_key(self, cache_key: str) -> dict[str, Any] | None:
        with self.store.connect() as db:
            for row in db.execute(
                "SELECT path, sha256, duration_s FROM artifact WHERE cache_key = ? ORDER BY id DESC", (cache_key,)
            ):
                if Path(row["path"]).is_file():
                    return dict(row)
        return None

    def complete_job(self, job: dict[str, Any], path: Path, sha256: str, duration_s: float) -> bool:
        """Record the artifact only if this worker still owns the running job."""
        now = utc_now_iso()
        with self.store.transaction() as db:
            owned = db.execute(
                "UPDATE job SET status = 'done', error = NULL, finished_at = ?"
                " WHERE id = ? AND status = 'running' AND worker_id = ?",
                (now, job["id"], job["worker_id"]),
            ).rowcount
            if not owned:
                return False
            db.execute(
                "INSERT OR REPLACE INTO artifact (segment_id, cache_key, path, sha256, duration_s, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (job["segment_id"], job["cache_key"], str(path), sha256, duration_s, now),
            )
            return True

    def finish_job(self, job: dict[str, Any], status: str, error: str | None) -> bool:
        with self.store.transaction() as db:
            return bool(
                db.execute(
                    "UPDATE job SET status = ?, error = ?, finished_at = ?"
                    " WHERE id = ? AND status = 'running' AND worker_id = ?",
                    (status, error, utc_now_iso(), job["id"], job["worker_id"]),
                ).rowcount
            )

    def current_key_for(self, segment_id: int) -> str | None:
        with self.store.connect() as db:
            row = db.execute("SELECT spoken_text, voice_id FROM segment WHERE id = ?", (segment_id,)).fetchone()
        return self.cache_key(row["spoken_text"], voice_key=row["voice_id"]) if row else None

    def set_spoken_from_text(self, segment_id: int, text: str) -> str:
        """Normalise ``text`` into the segment's spoken text, keeping the original source (e.g. drop a speaker label)."""
        settings = self.settings
        normalization = normalize_text(" ".join(text.split()), settings.lexicon_entries())
        if not normalization.spoken:
            raise ValueError("Okunacak metin bos olamaz.")
        with self.store.transaction() as db:
            if not db.execute(
                "UPDATE segment SET spoken_text = ?, transformations_json = ?, updated_at = ? WHERE id = ?",
                (normalization.spoken, _transformations_json(normalization), utc_now_iso(), segment_id),
            ).rowcount:
                raise KeyError(f"Segment yok: {segment_id}")
            db.execute(
                "UPDATE job SET status = 'cancelled', finished_at = ? WHERE segment_id = ? AND status = 'queued'",
                (utc_now_iso(), segment_id),
            )
        return normalization.spoken

    def orphan_audio_files(self) -> list[Path]:
        """Audio files not referenced by any artifact. Reported only; never deleted automatically."""
        with self.store.connect() as db:
            known = {Path(row["path"]).resolve() for row in db.execute("SELECT path FROM artifact")}
        return sorted(
            path for path in self.store.audio_dir.glob("*.wav") if path.resolve() not in known
        )
