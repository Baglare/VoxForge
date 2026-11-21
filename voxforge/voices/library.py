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

import wave
from pathlib import Path
from typing import Any

from voxforge.studio.design import DesignLibrary

DESIGN_PREFIX = "design:"
KIND_LABELS = {"profile": "Kayıtlı ses profili", "design": "VoxCPM2 tasarım sesi"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRASH = REPO_ROOT / "outputs" / "voice_library_trash"


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
    def __init__(self, studio_root: Path | None = None, *, designs: DesignLibrary | None = None, trash_root: Path | None = None) -> None:
        if designs is None:
            if studio_root is None:
                from voxforge.studio import studio_root as default_root

                studio_root = default_root()
            designs = DesignLibrary(Path(studio_root))
        self.designs = designs
        self.trash_root = Path(trash_root) if trash_root is not None else DEFAULT_TRASH

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
        return self.trash_root / group / f"{key}-{stamp}"

    def replace_reference(self, key: str, audio: Any, permission: bool) -> None:
        if is_design_key(key):
            raise VoiceLibraryError("Tasarım sesleri değiştirilemez; yeni sürüm kaydedin.")
        self._require_permission(permission)
        self._profile_call(
            "replace_voice_profile_reference", key, self._audio_path(audio),
            keep_previous_dir=self._trash_slot("replaced-references", key),
        )

    def reprocess(self, key: str) -> None:
        if is_design_key(key):
            raise VoiceLibraryError("Tasarım sesleri yeniden işlenmez.")
        self._profile_call("recreate_voice_profile", key)

    def delete(self, key: str, confirmation: str) -> Path:
        """Move a voice to the trash; ``confirmation`` must repeat its confirm token."""
        if (confirmation or "").strip() != self.confirm_token(key):
            raise VoiceLibraryError(f"Silmek için onay kutusuna `{self.confirm_token(key)}` yazın.")
        if is_design_key(key):
            try:
                return self.designs.archive(_design_id(key))
            except (OSError, ValueError, KeyError) as exc:
                raise VoiceLibraryError(str(exc)) from exc
        result = self._profile_call("archive_voice_profile", key, self.trash_root / "profiles")
        return result.profile_dir

    @staticmethod
    def confirm_token(key: str) -> str:
        return _design_id(key)[:8] if is_design_key(key) else str(key or "")
