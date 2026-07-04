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
PROFILE_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

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


@dataclass(frozen=True)
class VoiceProfileRecord:
    """Mevcut bir profilin okunmus ve dogrulanmis hali."""

    profile_name: str
    profile_slug: str
    profile_dir: Path
    original_reference: Path
    preprocessed_reference: Path
    profile_json_path: Path
    metadata: dict[str, Any]
    original_quality: dict[str, Any]
    preprocessed_quality: dict[str, Any]
    selected_preprocessing_variant: str
    preprocessing_warning: str | None


@dataclass(frozen=True)
class VoiceProfileDeleteResult:
    """Silme islemi sonunda CLI ve Gradio tarafina donecek sade sonuc."""

    profile_slug: str
    profile_dir: Path
    deleted: bool
    message: str


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


def validate_profile_slug(profile_slug: str) -> str:
    """Kullanici girdisi slug degerinin yalnizca profil slug'i oldugunu dogrular."""
    cleaned_profile_slug = (profile_slug or "").strip()
    if not cleaned_profile_slug:
        raise VoiceProfileError("Profil slug bos olamaz.")

    if not PROFILE_SLUG_PATTERN.fullmatch(cleaned_profile_slug):
        raise VoiceProfileError(
            "Gecersiz profil slug. Sadece kucuk harf, rakam ve tek tire "
            "kullanilabilir."
        )

    return cleaned_profile_slug


def resolve_profile_dir(profile_slug: str) -> Path:
    """Slug degerinden guvenli profil klasoru yolu uretir."""
    cleaned_profile_slug = validate_profile_slug(profile_slug)
    profile_dir = PROFILES_DIR / cleaned_profile_slug
    ensure_profile_path_is_safe(profile_dir)

    if profile_dir.resolve() == PROFILES_DIR.resolve():
        raise VoiceProfileError("Profil klasoru profiles kok dizini olamaz.")

    return profile_dir


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


def read_profile_metadata(profile_json_path: Path) -> dict[str, Any]:
    """profile.json dosyasini okur ve kullaniciya sade hata dondurur."""
    try:
        metadata = json.loads(profile_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VoiceProfileError(
            "profile.json bozuk: "
            f"{profile_json_path} (satir {exc.lineno}, kolon {exc.colno})"
        ) from exc
    except OSError as exc:
        raise VoiceProfileError(f"profile.json okunamadi: {profile_json_path}") from exc

    if not isinstance(metadata, dict):
        raise VoiceProfileError("profile.json beklenen JSON obje formatinda degil.")

    return metadata


def quality_label(report: Any) -> str:
    """Terminal ve Gradio icin kalite raporundan kisa ozet uretir."""
    if not isinstance(report, dict):
        return "UNKNOWN"

    quality = report.get("quality") or "UNKNOWN"
    duration = report.get("duration_seconds")
    sample_rate = report.get("sample_rate")

    details = []
    if duration is not None:
        details.append(f"sure={duration} sn")
    if sample_rate is not None:
        details.append(f"sample_rate={sample_rate} Hz")

    if details:
        return f"{quality} ({', '.join(details)})"
    return str(quality)


def voice_profile_record_to_dict(record: VoiceProfileRecord) -> dict[str, Any]:
    """VoiceProfileRecord nesnesini JSON uyumlu sozluge cevirir."""
    return {
        "profile_name": record.profile_name,
        "profile_slug": record.profile_slug,
        "created_at": record.metadata.get("created_at"),
        "profile_dir": str(record.profile_dir.resolve()),
        "profile_dir_relative": project_relative_path(record.profile_dir),
        "profile_json_path": str(record.profile_json_path.resolve()),
        "profile_json_exists": record.profile_json_path.is_file(),
        "original_reference": str(record.original_reference.resolve()),
        "original_reference_exists": record.original_reference.is_file(),
        "preprocessed_reference": str(record.preprocessed_reference.resolve()),
        "preprocessed_reference_exists": record.preprocessed_reference.is_file(),
        "original_quality": record.original_quality,
        "preprocessed_quality": record.preprocessed_quality,
        "selected_preprocessing_variant": record.selected_preprocessing_variant,
        "preprocessing_warning": record.preprocessing_warning,
    }


def build_broken_profile_record(profile_dir: Path, issues: list[str]) -> dict[str, Any]:
    """Eksik veya bozuk profil icin sade rapor kaydi hazirlar."""
    profile_json_path = profile_dir / PROFILE_JSON_NAME
    original_reference = profile_dir / ORIGINAL_REFERENCE_NAME
    preprocessed_reference = profile_dir / PREPROCESSED_REFERENCE_NAME
    return {
        "profile_slug": profile_dir.name,
        "profile_dir": str(profile_dir.resolve()),
        "profile_dir_relative": project_relative_path(profile_dir),
        "profile_json_exists": profile_json_path.is_file(),
        "original_reference_exists": original_reference.is_file(),
        "preprocessed_reference_exists": preprocessed_reference.is_file(),
        "issues": issues,
    }


def build_profile_record(profile_dir: Path) -> tuple[VoiceProfileRecord | None, dict[str, Any] | None]:
    """Tek profil klasorunu gecerli veya bozuk kayit olarak cozer."""
    profile_slug = profile_dir.name
    issues: list[str] = []

    try:
        validate_profile_slug(profile_slug)
        ensure_profile_path_is_safe(profile_dir)
    except VoiceProfileError as exc:
        issues.append(str(exc))

    profile_json_path = profile_dir / PROFILE_JSON_NAME
    original_reference = profile_dir / ORIGINAL_REFERENCE_NAME
    preprocessed_reference = profile_dir / PREPROCESSED_REFERENCE_NAME

    if not profile_json_path.is_file():
        issues.append("profile.json bulunamadi.")
    if not original_reference.is_file():
        issues.append("original_reference.wav bulunamadi.")
    if not preprocessed_reference.is_file():
        issues.append("preprocessed_reference.wav bulunamadi.")

    metadata: dict[str, Any] | None = None
    if profile_json_path.is_file():
        try:
            metadata = read_profile_metadata(profile_json_path)
        except VoiceProfileError as exc:
            issues.append(str(exc))

    if issues:
        return None, build_broken_profile_record(profile_dir, issues)

    if metadata is None:
        return None, build_broken_profile_record(
            profile_dir,
            ["profile.json okunamadi."],
        )

    profile_name = str(metadata.get("profile_name") or profile_slug).strip() or profile_slug
    original_quality = metadata.get("original_quality")
    if not isinstance(original_quality, dict):
        original_quality = {"quality": "UNKNOWN", "warnings": ["original_quality yok."]}

    preprocessed_quality = metadata.get("preprocessed_quality")
    if not isinstance(preprocessed_quality, dict):
        preprocessed_quality = {
            "quality": "UNKNOWN",
            "warnings": ["preprocessed_quality yok."],
        }

    selected_variant = str(
        metadata.get("selected_preprocessing_variant") or "bilinmiyor"
    )

    return (
        VoiceProfileRecord(
            profile_name=profile_name,
            profile_slug=profile_slug,
            profile_dir=profile_dir,
            original_reference=original_reference,
            preprocessed_reference=preprocessed_reference,
            profile_json_path=profile_json_path,
            metadata=metadata,
            original_quality=original_quality,
            preprocessed_quality=preprocessed_quality,
            selected_preprocessing_variant=selected_variant,
            preprocessing_warning=metadata.get("preprocessing_warning"),
        ),
        None,
    )


def list_voice_profiles() -> dict[str, Any]:
    """profiles/ klasorundeki gecerli ve bozuk profilleri listeler."""
    valid_profiles: list[dict[str, Any]] = []
    broken_profiles: list[dict[str, Any]] = []

    if PROFILES_DIR.is_dir():
        for profile_dir in sorted(PROFILES_DIR.iterdir(), key=lambda item: item.name.lower()):
            if not profile_dir.is_dir():
                continue

            valid_record, broken_record = build_profile_record(profile_dir)
            if valid_record is not None:
                valid_profiles.append(voice_profile_record_to_dict(valid_record))
            elif broken_record is not None:
                broken_profiles.append(broken_record)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profiles_dir": str(PROFILES_DIR.resolve()),
        "valid_profile_count": len(valid_profiles),
        "broken_profile_count": len(broken_profiles),
        "profiles": valid_profiles,
        "broken_profiles": broken_profiles,
    }


def get_voice_profile(profile_slug: str) -> VoiceProfileRecord:
    """Slug ile tek bir gecerli profili okur."""
    profile_dir = resolve_profile_dir(profile_slug)
    if not profile_dir.is_dir():
        raise VoiceProfileError(f"Profil bulunamadi: {profile_slug}")

    valid_record, broken_record = build_profile_record(profile_dir)
    if valid_record is None:
        issues = "; ".join((broken_record or {}).get("issues") or ["bilinmeyen sorun"])
        raise VoiceProfileError(f"Profil eksik veya bozuk: {issues}")

    return valid_record


def delete_voice_profile(profile_slug: str) -> VoiceProfileDeleteResult:
    """Tek bir profil klasorunu guvenli sekilde siler."""
    profile_dir = resolve_profile_dir(profile_slug)
    if not profile_dir.exists():
        raise VoiceProfileError(f"Profil bulunamadi: {profile_slug}")
    if not profile_dir.is_dir():
        raise VoiceProfileError(f"Profil yolu klasor degil: {profile_slug}")

    shutil.rmtree(profile_dir)
    return VoiceProfileDeleteResult(
        profile_slug=profile_dir.name,
        profile_dir=profile_dir,
        deleted=True,
        message=f"Profil silindi: {profile_dir.name}",
    )


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


def recreate_voice_profile(profile_slug: str) -> VoiceProfileResult:
    """Mevcut profilin original_reference.wav dosyasindan profili yeniler."""
    profile = get_voice_profile(profile_slug)
    temp_preprocessed_reference = (
        profile.profile_dir / ".preprocessed_reference.tmp.wav"
    )
    temp_preprocessed_reference.unlink(missing_ok=True)

    try:
        preprocessing_result = preprocess_reference_audio(
            profile.original_reference,
            temp_preprocessed_reference,
            mode="safe",
        )
    except PreprocessingError as exc:
        temp_preprocessed_reference.unlink(missing_ok=True)
        raise VoiceProfileError(str(exc)) from exc

    selected_variant = preprocessing_result["selected_variant"]
    candidate_reports = preprocessing_result["candidate_reports"]
    original_quality = candidate_reports.get("original")
    preprocessed_quality = candidate_reports.get(selected_variant)
    if original_quality is None or preprocessed_quality is None:
        temp_preprocessed_reference.unlink(missing_ok=True)
        raise VoiceProfileError("Yenileme kalite raporu eksik olustu.")

    selected_temp_reference = Path(preprocessing_result["selected_output_path"])
    if not selected_temp_reference.is_file():
        raise VoiceProfileError("Yenilenen on islenmis referans dosyasi olusmadi.")

    backup_preprocessed_reference = (
        profile.profile_dir / ".preprocessed_reference.backup.wav"
    )
    backup_preprocessed_reference.unlink(missing_ok=True)
    if profile.preprocessed_reference.is_file():
        shutil.copy2(profile.preprocessed_reference, backup_preprocessed_reference)

    try:
        profile.preprocessed_reference.unlink(missing_ok=True)
        shutil.move(str(selected_temp_reference), str(profile.preprocessed_reference))
    except OSError as exc:
        if backup_preprocessed_reference.is_file():
            shutil.move(
                str(backup_preprocessed_reference),
                str(profile.preprocessed_reference),
            )
        raise VoiceProfileError("Yenilenen referans dosyasi kaydedilemedi.") from exc
    finally:
        backup_preprocessed_reference.unlink(missing_ok=True)

    payload = build_profile_payload(
        profile_name=profile.profile_name,
        profile_slug=profile.profile_slug,
        original_reference=profile.original_reference,
        preprocessed_reference=profile.preprocessed_reference,
        original_quality=original_quality,
        preprocessed_quality=preprocessed_quality,
        selected_preprocessing_variant=selected_variant,
        preprocessing_warning=preprocessing_result.get("preprocessing_warning"),
        preprocessing_candidate_reports=candidate_reports,
    )
    profile.profile_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return VoiceProfileResult(
        profile_name=profile.profile_name,
        profile_slug=profile.profile_slug,
        source_audio=profile.original_reference,
        profile_dir=profile.profile_dir,
        original_reference=profile.original_reference,
        preprocessed_reference=profile.preprocessed_reference,
        profile_json_path=profile.profile_json_path,
        original_quality=original_quality,
        preprocessed_quality=preprocessed_quality,
        selected_preprocessing_variant=selected_variant,
        preprocessing_warning=preprocessing_result.get("preprocessing_warning"),
        preprocessing_candidate_reports=candidate_reports,
    )


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
