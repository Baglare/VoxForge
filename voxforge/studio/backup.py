"""Portable, project-local Studio backups. SQLite and referenced assets are one snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import stat
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Callable
from zipfile import ZIP_STORED, ZipFile

from voxforge.studio.project import ProjectSettings, file_sha256
from voxforge.studio.store import DB_FILENAME, SCHEMA_VERSION, ProjectStore

SOURCE_EXTENSIONS = {".txt", ".docx", ".epub"}
AUDIO_MEMBER = re.compile(r"(?:audio|voices)/[a-zA-Z0-9][a-zA-Z0-9_-]*\.wav")
WINDOWS_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])\.wav", re.IGNORECASE)


def _valid_member(name: str) -> bool:
    return name in {DB_FILENAME, "reference.wav", *(f"source{ext}" for ext in SOURCE_EXTENSIONS)} or (
        AUDIO_MEMBER.fullmatch(name) is not None and not WINDOWS_DEVICE.fullmatch(name.rsplit("/", 1)[-1])
    )


def _rewrite_paths(
    db_path: Path, resolve: Callable[[str, str], tuple[str, str]], *, project_name: str | None = None,
) -> dict[str, str | None]:
    """Map stored data paths and collect required archive members and audio hashes."""
    members: dict[str, str | None] = {DB_FILENAME: None}

    def asset(value: str, kind: str, digest: str | None = None) -> str:
        name, rewritten = resolve(value, kind)
        if name in members and members[name] != digest:
            raise ValueError("Yedekte dosya özeti çakışıyor.")
        members[name] = digest
        return rewritten

    with closing(sqlite3.connect(db_path)) as db, db:
        db.execute("PRAGMA trusted_schema = OFF")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if not 7 <= version <= SCHEMA_VERSION:
            raise ValueError("Yedek veritabanı sürümü desteklenmiyor.")
        if db.execute("SELECT 1 FROM sqlite_schema WHERE type IN ('trigger', 'view')").fetchone():
            raise ValueError("Yedek veritabanında beklenmeyen şema öğesi var.")
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)] or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Yedek veritabanı bütünlük denetiminden geçmedi.")
        db.execute("PRAGMA journal_mode = DELETE")
        project = db.execute("SELECT settings_json, source_format FROM project WHERE id = 1").fetchone()
        if not project:
            raise ValueError("Yedekte Studio projesi bulunamadı.")
        if project_name is not None:
            db.execute("UPDATE project SET name = ? WHERE id = 1", (project_name,))
        settings = json.loads(project[0])
        ProjectSettings(**settings)
        if settings.get("reference_path"):
            settings["reference_path"] = asset(settings["reference_path"], "reference")
            db.execute("UPDATE project SET settings_json = ? WHERE id = 1", (json.dumps(settings, ensure_ascii=False),))
        if project[1] is not None:
            if project[1] not in SOURCE_EXTENSIONS:
                raise ValueError("Yedek kaynak belge biçimi desteklenmiyor.")
            members[f"source{project[1]}"] = None
        for key, path in db.execute("SELECT key, reference_path FROM voice WHERE reference_path IS NOT NULL").fetchall():
            db.execute("UPDATE voice SET reference_path = ? WHERE key = ?", (asset(path, "voices"), key))
        for table in ("artifact", "take"):
            rows = db.execute(f"SELECT id, path, sha256, quality_json, synthesis_json FROM {table}").fetchall()
            for row_id, path, digest, quality, synthesis in rows:
                target = asset(path, "audio", digest)
                if quality:
                    report = json.loads(quality)
                    if isinstance(report, dict) and "wav" in report:
                        report["wav"] = target
                        quality = json.dumps(report, ensure_ascii=False)
                if synthesis:
                    metadata = json.loads(synthesis)
                    reference = metadata.get("reference")
                    if reference:
                        # Historical recipes may refer to a previous local character file.
                        for kind in ("reference", "voices"):
                            try:
                                metadata["reference"] = asset(reference, kind)
                                break
                            except ValueError:
                                continue
                        else:
                            metadata["reference"] = None
                        synthesis = json.dumps(metadata, ensure_ascii=False)
                db.execute(
                    f"UPDATE {table} SET path = ?, quality_json = ?, synthesis_json = ? WHERE id = ?",
                    (target, quality, synthesis, row_id),
                )
        db.execute(
            "UPDATE job SET status = 'queued', started_at = NULL, worker_id = NULL, heartbeat_at = NULL"
            " WHERE status = 'running'"
        )
    return members


def _project_member(name: str, kind: str) -> str:
    if not _valid_member(name) or not (
        name == "reference.wav" if kind == "reference" else name.startswith(f"{kind}/")
    ):
        raise ValueError("Geçersiz proje dosya yolu.")
    return name


def _verify_file(path: Path, digest: str | None) -> None:
    if not path.is_file():
        raise ValueError(f"Yedek için gerekli dosya eksik: {path.name}")
    if digest is not None and file_sha256(path) != digest:
        raise ValueError(f"WAV özeti eşleşmiyor: {path.name}")


def backup_project(store: ProjectStore, target: Path) -> Path:
    folder = store.folder.resolve()

    def portable(value: str, kind: str) -> tuple[str, str]:
        name = Path(value).resolve().relative_to(folder).as_posix()
        name = _project_member(name, kind)
        return name, name

    with tempfile.TemporaryDirectory(prefix=".backup-", dir=target.parent) as temporary:
        scratch = Path(temporary)
        database = store.backup_to(scratch / DB_FILENAME)
        members = _rewrite_paths(database, portable)
        if len({name.lower() for name in members}) != len(members):
            raise ValueError("Proje dosya adları yalnız büyük/küçük harf farkıyla çakışıyor.")
        bundle = scratch / "project.zip"
        with ZipFile(bundle, "w", compression=ZIP_STORED) as archive:
            for name, digest in members.items():
                source = database if name == DB_FILENAME else (folder / name).resolve()
                if name != DB_FILENAME and source != folder / name:
                    raise ValueError("Yedek yalnız proje içindeki düzenli dosyaları kabul eder.")
                if digest is None:
                    archive.write(source, name)
                    continue
                checksum = hashlib.sha256()
                with source.open("rb") as handle, archive.open(name, "w", force_zip64=True) as output:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        checksum.update(block)
                        output.write(block)
                if checksum.hexdigest() != digest:
                    raise ValueError(f"WAV özeti eşleşmiyor: {name}")
        bundle.replace(target)
    return target


def restore_project(bundle: Path, target: Path, project_name: str) -> None:
    if target.exists():
        raise FileExistsError("Geri yükleme adı zaten kullanılıyor.")

    def absolute(value: str, kind: str) -> tuple[str, str]:
        name = _project_member(value, kind)
        return name, str(target / name)

    with tempfile.TemporaryDirectory(prefix=".restore-", dir=target.parent) as temporary:
        staging = Path(temporary) / "project"
        staging.mkdir()
        with ZipFile(bundle) as archive:
            names: set[str] = set()
            entries = archive.infolist()
            for entry in entries:
                name = entry.orig_filename
                if (
                    not _valid_member(name) or name.lower() in names
                    or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG)
                ):
                    raise ValueError("Yedekte geçersiz veya yinelenen dosya var.")
                names.add(name.lower())
            if DB_FILENAME not in names:
                raise ValueError("Yedekte Studio veritabanı yok.")
            if sum(entry.file_size for entry in entries) > shutil.disk_usage(staging).free:
                raise ValueError("Yedeği açmak için yeterli disk alanı yok.")
            for entry in entries:
                path = staging / entry.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        members = _rewrite_paths(staging / DB_FILENAME, absolute, project_name=project_name)
        if names != {name.lower() for name in members}:
            raise ValueError("Yedek dosyaları proje kayıtlarıyla eşleşmiyor.")
        for name, digest in members.items():
            _verify_file(staging / name, digest)
        try:
            target.mkdir()
        except FileExistsError as exc:
            raise FileExistsError("Geri yükleme adı zaten kullanılıyor.") from exc
        try:
            for asset in staging.iterdir():
                if asset.name != DB_FILENAME:
                    asset.rename(target / asset.name)
            # Studio lists a project only once the validated database is present.
            (staging / DB_FILENAME).rename(target / DB_FILENAME)
        except OSError:
            shutil.rmtree(target)  # Only the directory reserved by this restore.
            raise
