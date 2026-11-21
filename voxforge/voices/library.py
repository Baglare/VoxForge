"""One library for every reusable voice (UI independent).

Two kinds live side by side and are told apart by their key:

* ``profile`` — a saved reference recording under ``profiles/<slug>`` (key = slug, exactly
  as before, so Studio, scenes and Playground transcripts keep working unchanged);
* ``design`` — an immutable VoxCPM2 voice-design version under the Studio root
  (key = ``design:<id>``); its sample WAV is used as the cloning reference.

Nothing here adds storage: profiles go through ``scripts.voice_profile_utils`` and
designs through ``voxforge.studio.design.DesignLibrary``. Deletions are moves to a trash
folder, never ``rmtree``.
"""

from __future__ import annotations

import json
import re
import shutil
import wave
from pathlib import Path
from typing import Any

from voxforge.studio.design import DesignLibrary

DESIGN_PREFIX = "design:"
KIND_LABELS = {"profile": "Kayıtlı ses profili", "design": "VoxCPM2 tasarım sesi"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRASH = REPO_ROOT / "outputs" / "voice_library_trash"
DEFAULT_PLAYGROUND = REPO_ROOT / "outputs" / "playground"


class VoiceLibraryError(ValueError):
    """User-facing library problem."""


def is_design_key(key: str | None) -> bool:
    return bool(key) and str(key).startswith(DESIGN_PREFIX)


def _design_id(key: str) -> str:
    return str(key)[len(DESIGN_PREFIX):]


def _wav_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 2)
    except (OSError, wave.Error, ZeroDivisionError):
        return None


def _quality(report: Any) -> str:
    if not isinstance(report, dict):
        return "—"
    return str(report.get("quality") or "—")


class VoiceLibrary:
    def __init__(
        self,
        studio_root: Path | None = None,
        *,
        designs: DesignLibrary | None = None,
        trash_root: Path | None = None,
        playground_root: Path | None = None,
    ) -> None:
        if designs is None:
            if studio_root is None:
                from voxforge.studio import studio_root as default_root

                studio_root = default_root()
            designs = DesignLibrary(Path(studio_root))
        self.designs = designs
        self.trash_root = Path(trash_root) if trash_root is not None else DEFAULT_TRASH
        self.playground_root = Path(playground_root) if playground_root is not None else DEFAULT_PLAYGROUND

    # -- reading -------------------------------------------------------------------------
    @staticmethod
    def _utils():
        from scripts import voice_profile_utils

        return voice_profile_utils

    def entries(self) -> list[dict[str, Any]]:
        """Every usable voice: profiles (alphabetical) first, then designs (newest first)."""
        utils = self._utils()
        items: list[dict[str, Any]] = []
        for profile in utils.list_voice_profiles().get("profiles", []):
            reference = Path(profile["preprocessed_reference"])
            if not reference.is_file():
                continue
            items.append({
                "key": profile["profile_slug"],
                "kind": "profile",
                "name": profile.get("profile_name") or profile["profile_slug"],
                "reference": reference,
                "original": Path(profile["original_reference"]),
                "created_at": profile.get("created_at") or "",
                "seconds": _wav_seconds(reference),
                "quality": _quality(profile.get("preprocessed_quality")),
                "original_quality": profile.get("original_quality") or {},
                "preprocessed_quality": profile.get("preprocessed_quality") or {},
                "variant": profile.get("selected_preprocessing_variant"),
                "warning": profile.get("preprocessing_warning"),
            })
        for version in self.designs.versions():
            items.append({
                "key": DESIGN_PREFIX + version["id"],
                "kind": "design",
                "name": version["name"],
                "reference": Path(version["wav"]),
                "created_at": version.get("created_at") or "",
                "seconds": _wav_seconds(Path(version["wav"])),
                "quality": "—",
                "description": version.get("description") or "",
                "text": version.get("text") or "",
                "seed": version.get("seed"),
                "model_id": version.get("model_id"),
            })
        return items

    def broken_profiles(self) -> list[dict[str, Any]]:
        return list(self._utils().list_voice_profiles().get("broken_profiles", []))

    def choices(self) -> list[tuple[str, str]]:
        """Dropdown choices; designs are marked so the two kinds stay distinguishable."""
        result = []
        for item in self.entries():
            if item["kind"] == "design":
                result.append((f"{item['name']} · tasarım ({_design_id(item['key'])[:8]})", item["key"]))
            else:
                result.append((f"{item['name']} ({item['key']})", item["key"]))
        return result

    def keys(self) -> set[str]:
        return {key for _label, key in self.choices()}

    def get(self, key: str) -> dict[str, Any]:
        if not key:
            raise VoiceLibraryError("Bir ses seçin.")
        if is_design_key(key):
            try:
                version = self.designs.version(_design_id(key))
            except (OSError, ValueError, KeyError) as exc:
                raise VoiceLibraryError(f"Tasarım sesi açılamadı: {exc}") from exc
            return next((item for item in self.entries() if item["key"] == key), None) or {
                "key": key, "kind": "design", "name": version["name"], "reference": Path(version["wav"]),
                "text": version.get("text") or "", "description": version.get("description") or "",
            }
        utils = self._utils()
        try:
            utils.get_voice_profile(key)
        except utils.VoiceProfileError as exc:
            raise VoiceLibraryError(str(exc)) from exc
        for item in self.entries():
            if item["key"] == key:
                return item
        raise VoiceLibraryError(f"Profil kullanılamıyor: {key}")

    def reference(self, key: str) -> Path:
        """Cloning reference for any library key (profile: preprocessed WAV, design: sample)."""
        if is_design_key(key):
            try:
                return Path(self.designs.version(_design_id(key))["wav"])
            except (OSError, ValueError, KeyError) as exc:
                raise VoiceLibraryError(f"Tasarım sesi açılamadı: {exc}") from exc
        utils = self._utils()
        try:
            return Path(utils.get_voice_profile(key).preprocessed_reference)
        except utils.VoiceProfileError as exc:
            raise VoiceLibraryError(str(exc)) from exc

    def design_transcript(self, key: str) -> str:
        """Exact text spoken in a design sample (usable as an Ultimate Cloning transcript)."""
        if not is_design_key(key):
            return ""
        try:
            return str(self.designs.version(_design_id(key)).get("text") or "")
        except (OSError, ValueError, KeyError):
            return ""

    # -- dependencies --------------------------------------------------------------------
    def usages(self, key: str) -> dict[str, list[str]]:
        """Where a voice is still referenced by key: Sahneli Anlatım projects (read only).

        Studio projects keep their own copy of the reference and Playground history records
        keep their WAV, so those do not depend on the library entry; scene projects do (they
        resolve the voice again for every regeneration and for the merge check).
        """
        scenes = []
        for path in sorted((self.playground_root / "scenes").glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("profile") == key:
                scenes.append(str(data.get("name") or path.stem))
        return {"scenes": scenes} if scenes else {}

    @staticmethod
    def describe_usages(usages: dict[str, list[str]]) -> str:
        parts = [f"sahne projeleri: {', '.join(usages['scenes'])}"] if usages.get("scenes") else []
        return "; ".join(parts)

    def _transcript_path(self, key: str) -> Path | None:
        if is_design_key(key):
            return self.playground_root / "transcripts" / f"design-{_design_id(key)}.txt"
        if re.fullmatch(r"[A-Za-z0-9_.-]+", key or ""):
            return self.playground_root / "transcripts" / f"{key}.txt"
        return None

    def _park_transcript(self, key: str) -> Path | None:
        """Move a voice's saved Ultimate Cloning transcript to the trash.

        The transcript describes one specific recording. A deleted voice whose slug is reused,
        or a replaced reference, must not silently inherit it (a wrong transcript degrades
        Ultimate Cloning without any error).
        """
        path = self._transcript_path(key)
        if path is None or not path.is_file():
            return None
        target = self._trash_slot("transcripts", key).with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        return target

    # -- changes ---------------------------------------------------------------------------
    def _profile_call(self, function: str, *args: Any, **kwargs: Any) -> Any:
        utils = self._utils()
        try:
            return getattr(utils, function)(*args, **kwargs)
        except utils.VoiceProfileError as exc:
            raise VoiceLibraryError(str(exc)) from exc

    @staticmethod
    def _require_permission(permission: bool) -> None:
        if not permission:
            raise VoiceLibraryError("Sesin size ait olduğunu veya kullanma izniniz olduğunu onaylayın.")

    @staticmethod
    def _audio_path(audio: Any) -> Path:
        if not audio:
            raise VoiceLibraryError("Bir referans ses dosyası yükleyin.")
        path = Path(str(audio))
        if not path.is_file():
            raise VoiceLibraryError(f"Ses dosyası bulunamadı: {path.name}")
        return path

    def create_profile(self, name: str, audio: Any, permission: bool) -> str:
        self._require_permission(permission)
        if not (name or "").strip():
            raise VoiceLibraryError("Profil adı girin.")
        result = self._profile_call("create_voice_profile", name, self._audio_path(audio))
        return result.profile_slug

    def rename(self, key: str, name: str) -> None:
        if is_design_key(key):
            try:
                self.designs.rename(_design_id(key), name)
            except (OSError, ValueError, KeyError) as exc:
                raise VoiceLibraryError(str(exc)) from exc
            return
        self._profile_call("rename_voice_profile", key, name)

    def _trash_slot(self, group: str, key: str) -> Path:
        from voxforge.runs import utc_now_iso

        stamp = utc_now_iso().replace(":", "").replace("-", "").replace(".", "")
        return self.trash_root / group / f"{str(key).replace(':', '-')}-{stamp}"

    def replace_reference(self, key: str, audio: Any, permission: bool) -> str | None:
        """Replace a profile reference; returns a warning if the old files stayed in the profile."""
        if is_design_key(key):
            raise VoiceLibraryError("Tasarım sesleri değiştirilemez; yeni sürüm kaydedin.")
        self._require_permission(permission)
        result = self._profile_call(
            "replace_voice_profile_reference", key, self._audio_path(audio),
            keep_previous_dir=self._trash_slot("replaced-references", key),
        )
        self._park_transcript(key)  # eski kaydın transkripti yeni kayda ait değil
        return result.archive_warning

    def pending_replacements(self) -> dict[str, int]:
        """Profiles holding an unfinished or unarchived reference-replacement backup."""
        utils = self._utils()
        pending = {}
        for item in self.entries():
            if item["kind"] == "profile":
                count = len(utils.pending_profile_replacements(item["key"]))
                if count:
                    pending[item["key"]] = count
        return pending

    def recover_replacements(self, key: str) -> list[str]:
        if is_design_key(key):
            return []
        return self._profile_call(
            "recover_profile_replacement", key, self._trash_slot("replaced-references", key),
        )

    def reprocess(self, key: str) -> None:
        if is_design_key(key):
            raise VoiceLibraryError("Tasarım sesleri yeniden işlenmez.")
        self.recover_replacements(key)  # yarıda kalan değişim önce tamamlanır veya geri alınır
        self._profile_call("recreate_voice_profile", key)

    def delete(self, key: str, confirmation: str, *, force: bool = False) -> Path:
        """Move a voice to the trash; ``confirmation`` must repeat its confirm token.

        A voice that scene projects still use is only removed with ``force`` (the scenes
        would otherwise stop being regenerable or mergeable without warning).
        """
        if (confirmation or "").strip() != self.confirm_token(key):
            raise VoiceLibraryError(f"Silmek için onay kutusuna `{self.confirm_token(key)}` yazın.")
        used = self.describe_usages(self.usages(key))
        if used and not force:
            raise VoiceLibraryError(
                f"Bu ses hâlâ kullanılıyor ({used}). Yine de kaldırmak için «Bağlı projelere rağmen kaldır» kutusunu "
                "işaretleyin; sahneler yeni bir ses seçilene kadar yeniden üretilemez veya birleştirilemez."
            )
        if is_design_key(key):
            try:
                target = self.designs.archive(_design_id(key))
            except (OSError, ValueError, KeyError) as exc:
                raise VoiceLibraryError(str(exc)) from exc
            self._park_transcript(key)
            return target
        result = self._profile_call("archive_voice_profile", key, self.trash_root / "profiles")
        self._park_transcript(key)
        return result.profile_dir

    @staticmethod
    def confirm_token(key: str) -> str:
        return _design_id(key)[:8] if is_design_key(key) else str(key or "")
