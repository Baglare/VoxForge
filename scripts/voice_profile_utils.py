# -*- coding: utf-8 -*-
"""VoxForge yerel voice profile olusturma yardimcilari."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import unicodedata
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "profiles"

ORIGINAL_REFERENCE_NAME = "original_reference.wav"
PREPROCESSED_REFERENCE_NAME = "preprocessed_reference.wav"
PROFILE_JSON_NAME = "profile.json"

TURKISH_CHAR_MAP = str.maketrans(
    {
        "\u00e7": "c",
        "\u00c7": "c",
        "\u011f": "g",
        "\u011e": "g",
        "\u0131": "i",
        "I": "i",
        "\u0130": "i",
        "\u00f6": "o",
        "\u00d6": "o",
        "\u015f": "s",
        "\u015e": "s",
        "\u00fc": "u",
        "\u00dc": "u",
    }
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_preprocessing_utils import PreprocessingError, preprocess_reference_audio


class VoiceProfileError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


@dataclass(frozen=True)
class VoiceProfileResult:
    """Olusturulan profilin Gradio ve CLI tarafinda kullanacagi sonuc."""

    profile_name: str
    profile_slug: str
    source_audio: Path
    profile_dir: Path
    original_reference: Path
    preprocessed_reference: Path
    profile_json_path: Path
    original_quality: dict[str, Any]
    preprocessed_quality: dict[str, Any]
    selected_preprocessing_variant: str
    preprocessing_warning: str | None
    preprocessing_candidate_reports: dict[str, Any]


def slugify_profile_name(profile_name: str) -> str:
    """Profil adini guvenli klasor adina cevirir."""
    normalized = profile_name.strip().translate(TURKISH_CHAR_MAP).lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")

    if not normalized:
        raise VoiceProfileError(
            "Profil adi guvenli klasor adina cevrilemedi. "
            "Lutfen harf veya rakam iceren bir ad kullanin."
        )

    return normalized


def resolve_input_path(input_audio_path: str | Path) -> Path:
    """Giris dosyasini proje kokune gore cozer."""
    candidate = Path(input_audio_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    return candidate.resolve()


def ensure_profile_path_is_safe(profile_dir: Path) -> None:
    """Profil yolunun profiles klasoru disina cikmadigini dogrular."""
    profiles_root = PROFILES_DIR.resolve()
    resolved_profile_dir = profile_dir.resolve()

    try:
        resolved_profile_dir.relative_to(profiles_root)
    except ValueError as exc:
        raise VoiceProfileError("Profil yolu profiles klasoru disina cikamaz.") from exc


def project_relative_path(path: Path) -> str:
    """profile.json icin proje kokune gore tasinabilir yol dondurur."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_profile_payload(
    profile_name: str,
    profile_slug: str,
    original_reference: Path,
    preprocessed_reference: Path,
    original_quality: dict[str, Any],
    preprocessed_quality: dict[str, Any],
    selected_preprocessing_variant: str,
    preprocessing_warning: str | None,
    preprocessing_candidate_reports: dict[str, Any],
) -> dict[str, Any]:
    """profile.json icin yazilacak metadata verisini hazirlar."""
    return {
        "profile_name": profile_name,
        "profile_slug": profile_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_reference_path": project_relative_path(original_reference),
        "preprocessed_reference_path": project_relative_path(preprocessed_reference),
        "original_quality": original_quality,
        "preprocessed_quality": preprocessed_quality,
        "selected_preprocessing_variant": selected_preprocessing_variant,
        "preprocessing_warning": preprocessing_warning,
        "preprocessing_candidate_reports": preprocessing_candidate_reports,
        "notes": [
            "Bu profil yerel kullanim icindir; profiles/* GitHub'a yuklenmez.",
            "original_reference.wav giris sesinin kopyasidir.",
            "preprocessed_reference.wav guvenli on isleme ile mono 24000 Hz "
            "pcm_s16le WAV olarak olusturulur.",
        ],
    }


def create_voice_profile(
    profile_name: str,
    input_audio_path: str | Path,
) -> VoiceProfileResult:
    """Referans sesten yerel voice profile klasoru olusturur."""
    cleaned_profile_name = profile_name.strip()
    profile_slug = slugify_profile_name(cleaned_profile_name)
    source_audio = resolve_input_path(input_audio_path)
    profile_dir = PROFILES_DIR / profile_slug
    ensure_profile_path_is_safe(profile_dir)

    if not source_audio.is_file():
        raise VoiceProfileError(f"Giris ses dosyasi bulunamadi: {source_audio}")

    if profile_dir.exists():
        raise VoiceProfileError(
            "Bu profil zaten var; uzerine yazilmadi.\n"
            f"Mevcut profil klasoru: {profile_dir}"
        )

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir()
    profile_created = False

    try:
        original_reference = profile_dir / ORIGINAL_REFERENCE_NAME
        preprocessed_reference = profile_dir / PREPROCESSED_REFERENCE_NAME
        profile_json = profile_dir / PROFILE_JSON_NAME

        shutil.copy2(source_audio, original_reference)
        try:
            preprocessing_result = preprocess_reference_audio(
                original_reference,
                preprocessed_reference,
                mode="safe",
            )
        except PreprocessingError as exc:
            raise VoiceProfileError(str(exc)) from exc

        selected_variant = preprocessing_result["selected_variant"]
        candidate_reports = preprocessing_result["candidate_reports"]
        original_quality = candidate_reports.get("original")
        preprocessed_quality = candidate_reports.get(selected_variant)
        if original_quality is None or preprocessed_quality is None:
            raise VoiceProfileError("On isleme kalite raporu eksik olustu.")

        selected_preprocessed_reference = Path(
            preprocessing_result["selected_output_path"]
        )

        payload = build_profile_payload(
            profile_name=cleaned_profile_name,
            profile_slug=profile_slug,
            original_reference=original_reference,
            preprocessed_reference=selected_preprocessed_reference,
            original_quality=original_quality,
            preprocessed_quality=preprocessed_quality,
            selected_preprocessing_variant=selected_variant,
            preprocessing_warning=preprocessing_result.get("preprocessing_warning"),
            preprocessing_candidate_reports=candidate_reports,
        )
        profile_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        profile_created = True
    finally:
        if not profile_created and profile_dir.exists():
            shutil.rmtree(profile_dir, ignore_errors=True)

    return VoiceProfileResult(
        profile_name=cleaned_profile_name,
        profile_slug=profile_slug,
        source_audio=source_audio,
        profile_dir=profile_dir,
        original_reference=original_reference,
        preprocessed_reference=selected_preprocessed_reference,
        profile_json_path=profile_json,
        original_quality=original_quality,
        preprocessed_quality=preprocessed_quality,
        selected_preprocessing_variant=selected_variant,
        preprocessing_warning=preprocessing_result.get("preprocessing_warning"),
        preprocessing_candidate_reports=candidate_reports,
    )
