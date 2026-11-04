"""SQLite storage for a Studio project (VF-005).

One project = one folder: ``project.sqlite`` (metadata only) + ``audio/``.
Schema changes go through ``MIGRATIONS`` keyed by ``PRAGMA user_version``; an
existing database is copied to ``project.sqlite.bak-v<old>`` before migrating.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DB_FILENAME = "project.sqlite"
AUDIO_DIRNAME = "audio"

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE project (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        settings_json TEXT NOT NULL
    );
    CREATE TABLE segment (
        id INTEGER PRIMARY KEY,
        ord INTEGER NOT NULL,
        paragraph INTEGER NOT NULL DEFAULT 0,
        source_text TEXT NOT NULL,
        spoken_text TEXT NOT NULL,
        transformations_json TEXT NOT NULL DEFAULT '[]',
        voice_id TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX segment_ord ON segment(ord);
    CREATE TABLE artifact (
        id INTEGER PRIMARY KEY,
        segment_id INTEGER NOT NULL REFERENCES segment(id) ON DELETE CASCADE,
        cache_key TEXT NOT NULL,
        path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        duration_s REAL NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (segment_id, cache_key)
    );
    CREATE INDEX artifact_key ON artifact(cache_key);
    CREATE TABLE job (
        id INTEGER PRIMARY KEY,
        segment_id INTEGER NOT NULL REFERENCES segment(id) ON DELETE CASCADE,
        cache_key TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('queued','running','done','failed','cancelled')),
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        queued_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE INDEX job_status ON job(status, id);
    """,
    # v2: worker lease. A running job is recovered only when its heartbeat is stale,
    # so reopening a project while a worker is active does not start the job twice.
    2: """
    ALTER TABLE job ADD COLUMN worker_id TEXT;
    ALTER TABLE job ADD COLUMN heartbeat_at REAL;
    """,
    # v3: character voices (VF-007). segment.voice_id holds a voice key; NULL = narrator
    # (the project's default reference and preset).
    3: """
    CREATE TABLE voice (
        key TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        reference_path TEXT,
        preset_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    """,
}
SCHEMA_VERSION = max(MIGRATIONS)


class ProjectStore:
    """Short-lived connections per operation so a worker thread can share the file (WAL)."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.db_path = self.folder / DB_FILENAME
        self.audio_dir = self.folder / AUDIO_DIRNAME
        self.folder.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self._migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` so claims and state changes are atomic across processes/threads."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    def schema_version(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def backup_to(self, target: Path) -> Path:
        """Consistent copy via the SQLite backup API.

        In WAL mode recent commits may live only in ``project.sqlite-wal``; a plain
        file copy of the main database would silently miss them.
        """
        temporary = target.with_name(target.name + ".partial")
        temporary.unlink(missing_ok=True)
        with self.connect() as source:
            destination = sqlite3.connect(temporary)
            try:
                source.backup(destination)
            finally:
                destination.close()
        os.replace(temporary, target)
        return target

    def _migrate(self) -> None:
        existed = self.db_path.is_file()
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"Proje veritabani daha yeni bir surumden ({current}); bu VoxForge surumu {SCHEMA_VERSION} destekliyor."
            )
        if current == SCHEMA_VERSION:
            return
        if existed and current > 0:
            self.backup_to(self.db_path.with_name(f"{DB_FILENAME}.bak-v{current}"))
        for version in range(current + 1, SCHEMA_VERSION + 1):
            with self.connect() as connection:
                connection.executescript(
                    "BEGIN;\n" + MIGRATIONS[version] + f"\nPRAGMA user_version = {version};\nCOMMIT;"
                )
