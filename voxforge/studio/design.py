"""Local, immutable versions of native VoxCPM2 voice design previews."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from voxforge.runs import utc_now_iso
from voxforge.studio.project import file_sha256

_ID = re.compile(r"[0-9a-f]{32}")


class DesignLibrary:
    def __init__(self, studio_root: Path) -> None:
        self.root = studio_root / ".voice-design"

    def preview_dir(self, token: str) -> Path:
        return self._folder("previews", token)

    def _folder(self, group: str, token: str) -> Path:
        if not _ID.fullmatch(token):
            raise ValueError("Geçersiz ses tasarımı kimliği.")
        parent = (self.root / group).resolve()
        folder = parent / token
        if folder.resolve().parent != parent:
            raise ValueError("Ses tasarımı klasörü kütüphane dışında.")
        return folder

    @staticmethod
    def _read(folder: Path, *, verify_audio: bool = False) -> dict[str, Any]:
        metadata = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("id") != folder.name:
            raise ValueError("Ses tasarımı kaydı geçersiz.")
        keys = ("created_at", "sha256", "model_id")
        if folder.parent.name == "versions":
            keys += ("name",)
        if any(not isinstance(metadata.get(key), str) or not metadata[key] for key in keys):
            raise ValueError("Ses tasarımı bilgileri eksik.")
        wav = folder / "voice.wav"
        if not wav.is_file() or (verify_audio and file_sha256(wav) != metadata["sha256"]):
            raise ValueError("Ses tasarımı WAV dosyası eksik veya değişmiş.")
        return {**metadata, "wav": wav}

    def versions(self) -> list[dict[str, Any]]:
        versions = []
        for folder in (self.root / "versions").glob("*"):
            if _ID.fullmatch(folder.name):
                try:
                    versions.append(self._read(self._folder("versions", folder.name)))
                except (OSError, ValueError, KeyError):
                    continue  # one damaged version cannot hide the library
        return sorted(versions, key=lambda item: item["created_at"], reverse=True)

    def version(self, token: str) -> dict[str, Any]:
        return self._read(self._folder("versions", token), verify_audio=True)

    def save(self, preview_id: str, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("Tasarım adı 1–80 karakter olmalı.")
        preview = self._read(self.preview_dir(preview_id), verify_audio=True)
        source = preview.pop("wav")
        version_id = uuid.uuid4().hex
        target = self._folder("versions", version_id)
        temporary = target.with_name(f".{version_id}.partial")
        temporary.mkdir(parents=True)
        try:
            metadata = {**preview, "preview_id": preview_id, "id": version_id, "name": name, "created_at": utc_now_iso()}
            shutil.copy2(source, temporary / "voice.wav")
            if file_sha256(temporary / "voice.wav") != metadata["sha256"]:
                raise ValueError("Örnek ses kaydedilirken değişti.")
            (temporary / "voice.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            temporary.rename(target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return self.version(version_id)

    def rename(self, token: str, name: str) -> dict[str, Any]:
        """Change only the display name; the WAV, its hash and the version id stay the same."""
        name = " ".join((name or "").split())
        if not name or len(name) > 80:
            raise ValueError("Tasarım adı 1–80 karakter olmalı.")
        folder = self._folder("versions", token)
        metadata = self._read(folder)
        metadata.pop("wav")
        metadata["name"] = name
        temporary = folder / "voice.json.partial"
        temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        temporary.replace(folder / "voice.json")
        return self.version(token)

    def archive(self, token: str) -> Path:
        """Move a saved version to ``trash/`` instead of deleting it.

        Studio projects keep their own copy of the sample, so archived versions do not
        break existing characters; they only leave the library listings.
        """
        folder = self._folder("versions", token)
        self._read(folder)
        trash = self.root / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / f"{token}-{uuid.uuid4().hex[:8]}"
        shutil.move(str(folder), str(target))
        return target
