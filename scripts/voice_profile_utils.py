# -*- coding: utf-8 -*-
"""VoxForge yerel voice profile olusturma yardimcilari."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
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
    archive_warning: str | None = None


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


MAX_PROFILE_NAME_CHARS = 80


def _write_profile_json(profile_json_path: Path, payload: dict[str, Any]) -> None:
    """profile.json'u gecici dosya + os.replace ile atomik yazar."""
    temporary = profile_json_path.with_name(profile_json_path.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(profile_json_path)


def rename_voice_profile(profile_slug: str, new_name: str) -> VoiceProfileRecord:
    """Yalniz gorunen adi degistirir; slug, klasor ve referanslar ayni kalir.

    Slug degismedigi icin Playground transkriptleri ve sahne projelerindeki profil
    baglantilari bozulmaz.
    """
    cleaned = " ".join((new_name or "").split())
    if not cleaned or len(cleaned) > MAX_PROFILE_NAME_CHARS:
        raise VoiceProfileError(f"Profil adi 1-{MAX_PROFILE_NAME_CHARS} karakter olmali.")
    profile = get_voice_profile(profile_slug)
    metadata = dict(profile.metadata)
    metadata["profile_name"] = cleaned
    metadata["renamed_at"] = datetime.now(timezone.utc).isoformat()
    _write_profile_json(profile.profile_json_path, metadata)
    return get_voice_profile(profile_slug)


REPLACEMENT_BACKUP_PREFIX = ".replace.backup-"
REPLACEMENT_STAGING_NAME = ".replace.tmp"
REPLACEMENT_JOURNAL_NAME = "journal.json"
PROFILE_FILE_NAMES = (ORIGINAL_REFERENCE_NAME, PREPROCESSED_REFERENCE_NAME, PROFILE_JSON_NAME)

# Dosya islemleri tek noktadan gecer; hata enjeksiyonlu testler bunlari degistirir.
_replace_file = os.replace
_copy_file = shutil.copy2
_move_tree = shutil.move


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_journal(backup_dir: Path, journal: dict[str, Any]) -> None:
    temporary = backup_dir / (REPLACEMENT_JOURNAL_NAME + ".partial")
    temporary.write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8")
    _replace_file(temporary, backup_dir / REPLACEMENT_JOURNAL_NAME)


def _read_journal(backup_dir: Path) -> dict[str, Any] | None:
    try:
        journal = json.loads((backup_dir / REPLACEMENT_JOURNAL_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return journal if isinstance(journal, dict) and isinstance(journal.get("files"), dict) else None


def pending_profile_replacements(profile_slug: str) -> list[Path]:
    """Profil icinde tamamlanmamis veya arsivlenmemis referans degisimi yedekleri."""
    profile_dir = resolve_profile_dir(profile_slug)
    if not profile_dir.is_dir():
        return []
    return sorted(path for path in profile_dir.glob(REPLACEMENT_BACKUP_PREFIX + "*") if path.is_dir())


def _restore_from_backup(profile_dir: Path, backup_dir: Path, journal: dict[str, Any]) -> None:
    """Yedekteki eski dosyalari hash dogrulayarak geri koyar; yedek yalniz basarida silinir."""
    for name, expected in journal["files"].items():
        source = backup_dir / name
        if not source.is_file() or _file_sha256(source) != expected:
            raise VoiceProfileError(
                f"Yedekteki {name} eksik veya degismis; yedek klasoru korundu: {backup_dir}"
            )
    for name, expected in journal["files"].items():
        target = profile_dir / name
        if target.is_file() and _file_sha256(target) == expected:
            continue
        temporary = profile_dir / f".{name}.restore"
        _copy_file(backup_dir / name, temporary)
        _replace_file(temporary, target)
        if _file_sha256(target) != expected:
            raise VoiceProfileError(f"{name} geri yuklenemedi; yedek klasoru korundu: {backup_dir}")
    journal["state"] = "restored"
    _write_journal(backup_dir, journal)
    shutil.rmtree(backup_dir)


def _archive_backup(backup_dir: Path, keep_previous_dir: Path | None) -> str | None:
    """Basarili degisimden sonra eski dosyalari cop kutusuna tasir.

    Tasima basarisiz olursa yedek profil klasorunde ``committed`` durumunda kalir ve
    hicbir dosya silinmez; uyari metni dondurulur.
    """
    if keep_previous_dir is None:
        shutil.rmtree(backup_dir)
        return None
    try:
        keep_previous_dir.parent.mkdir(parents=True, exist_ok=True)
        if keep_previous_dir.exists():
            raise FileExistsError(str(keep_previous_dir))
        _move_tree(str(backup_dir), str(keep_previous_dir))
    except OSError as exc:
        if not backup_dir.is_dir():
            # Tasima kaynagi silip hedefi tamamlamadiysa durum belirsizdir; bunu gizleme.
            raise VoiceProfileError(f"Eski referans arsivi dogrulanamadi: {exc}") from exc
        return (
            "Eski referans cop kutusuna tasinamadi; profil klasorunde korundu "
            f"({backup_dir.name}). Ayrinti: {exc}"
        )
    return None


def recover_profile_replacement(profile_slug: str, keep_previous_dir: Path | None = None) -> list[str]:
    """Yarida kalmis referans degisimlerini guvenli duruma getirir.

    ``prepared``/``swapping`` durumundaki yedek eski dosyalari geri koyar (degisim hic
    olmamis gibi). ``committed`` durumundaki yedek yeni profil gecerli oldugu icin yalniz
    arsivlenir. Gunlugu okunamayan yedek dokunulmadan birakilir.
    """
    profile_dir = resolve_profile_dir(profile_slug)
    messages: list[str] = []
    for backup_dir in pending_profile_replacements(profile_slug):
        journal = _read_journal(backup_dir)
        if journal is None:
            messages.append(f"Gunlugu okunamayan yedek korundu: {backup_dir.name}")
            continue
        state = journal.get("state")
        if state == "committed":
            # Her yedek kendi zaman damgalı adına gider; aynı çağrıdaki yeni arşivle çakışmaz.
            stamp = backup_dir.name.removeprefix(REPLACEMENT_BACKUP_PREFIX)
            target = keep_previous_dir.with_name(f"{keep_previous_dir.name}-recovered-{stamp}") if keep_previous_dir else None
            warning = _archive_backup(backup_dir, target)
            messages.append(warning or f"Tamamlanmis degisimin eski referansi arsivlendi: {backup_dir.name}")
        else:
            _restore_from_backup(profile_dir, backup_dir, journal)
            messages.append(f"Yarida kalan degisim geri alindi, eski referans korundu: {backup_dir.name}")
    return messages


def replace_voice_profile_reference(
    profile_slug: str,
    input_audio_path: str | Path,
    keep_previous_dir: Path | None = None,
) -> VoiceProfileResult:
    """Profilin referans sesini yenisiyle degistirir ve yeniden on isler.

    Akis (her adim kurtarilabilir):
    1. Yeni dosyalar ``.replace.tmp`` icinde hazirlanir; basarisizsa profil hic degismez.
    2. Eski uc dosya ``.replace.backup-<zaman>`` klasorune KOPYALANIR, hash'leri gunluge
       yazilir (``prepared``). Profil dosyalari bu noktada hala eskidir.
    3. Gunluk ``swapping`` olur, yeni dosyalar ``os.replace`` ile yerine konur, sonra
       ``committed`` yazilir. Hata olursa eski dosyalar yedekten hash dogrulanarak geri konur;
       geri koyma da basarisizsa yedek ve gunluk korunur.
    4. Yedek ``keep_previous_dir`` altina tasinir. Tasima basarisiz olursa yedek profil
       klasorunde kalir ve uyari dondurulur; eski kayit hicbir yolda silinmez.
    Kesilen bir islem ``recover_profile_replacement`` ile (bir sonraki degisimde otomatik)
    tamamlanir veya geri alinir.
    """
    recover_profile_replacement(profile_slug, keep_previous_dir)
    profile = get_voice_profile(profile_slug)
    source_audio = resolve_input_path(input_audio_path)
    if not source_audio.is_file():
        raise VoiceProfileError(f"Giris ses dosyasi bulunamadi: {source_audio}")

    staging = profile.profile_dir / REPLACEMENT_STAGING_NAME
    shutil.rmtree(staging, ignore_errors=True)  # yalniz yeni (henuz kullanilmamis) dosyalar
    staging.mkdir()
    try:
        staged_original = staging / ORIGINAL_REFERENCE_NAME
        _copy_file(source_audio, staged_original)
        try:
            preprocessing_result = preprocess_reference_audio(
                staged_original, staging / PREPROCESSED_REFERENCE_NAME, mode="safe"
            )
        except PreprocessingError as exc:
            raise VoiceProfileError(str(exc)) from exc
        selected_variant = preprocessing_result["selected_variant"]
        candidate_reports = preprocessing_result["candidate_reports"]
        original_quality = candidate_reports.get("original")
        preprocessed_quality = candidate_reports.get(selected_variant)
        staged_preprocessed = Path(preprocessing_result["selected_output_path"])
        if original_quality is None or preprocessed_quality is None or not staged_preprocessed.is_file():
            raise VoiceProfileError("On isleme kalite raporu veya referans dosyasi eksik olustu.")
        if staged_preprocessed.name != PREPROCESSED_REFERENCE_NAME:
            _replace_file(staged_preprocessed, staging / PREPROCESSED_REFERENCE_NAME)
            staged_preprocessed = staging / PREPROCESSED_REFERENCE_NAME

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
        payload["created_at"] = profile.metadata.get("created_at") or payload["created_at"]
        payload["reference_replaced_at"] = datetime.now(timezone.utc).isoformat()
        (staging / PROFILE_JSON_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir = profile.profile_dir / f"{REPLACEMENT_BACKUP_PREFIX}{stamp}"
        backup_dir.mkdir()
        journal: dict[str, Any] = {"state": "copying", "created_at": stamp, "files": {}}
        _write_journal(backup_dir, journal)
        for name in PROFILE_FILE_NAMES:
            _copy_file(profile.profile_dir / name, backup_dir / name)
            journal["files"][name] = _file_sha256(profile.profile_dir / name)
            if _file_sha256(backup_dir / name) != journal["files"][name]:
                raise VoiceProfileError(f"{name} yedegi dogrulanamadi; profil degistirilmedi.")
        journal["state"] = "prepared"
        _write_journal(backup_dir, journal)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        # Profil dosyalari henuz degismedi; tamamlanmamis kopya yalniz yedek olarak kalabilir.
        if "backup_dir" in locals() and backup_dir.is_dir() and (_read_journal(backup_dir) or {}).get("state") != "prepared":
            shutil.rmtree(backup_dir, ignore_errors=True)
        raise

    try:
        journal["state"] = "swapping"
        _write_journal(backup_dir, journal)
        for name in PROFILE_FILE_NAMES:
            _replace_file(staging / name, profile.profile_dir / name)
        journal["state"] = "committed"
        _write_journal(backup_dir, journal)
    except BaseException as exc:
        try:
            _restore_from_backup(profile.profile_dir, backup_dir, journal)
        except BaseException as restore_exc:
            raise VoiceProfileError(
                "Referans degisimi yarida kaldi ve eski dosyalar geri konamadi. Hicbir sey silinmedi; "
                f"yedek: {backup_dir}. Bir sonraki islemde otomatik kurtarilir. Ayrinti: {restore_exc}"
            ) from exc
        if isinstance(exc, (OSError, VoiceProfileError)):
            raise VoiceProfileError(f"Referans degistirilemedi, eski referans geri kondu: {exc}") from exc
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    archive_warning = _archive_backup(backup_dir, keep_previous_dir)
    return VoiceProfileResult(
        profile_name=profile.profile_name,
        profile_slug=profile.profile_slug,
        source_audio=source_audio,
        profile_dir=profile.profile_dir,
        original_reference=profile.original_reference,
        preprocessed_reference=profile.preprocessed_reference,
        profile_json_path=profile.profile_json_path,
        original_quality=original_quality,
        preprocessed_quality=preprocessed_quality,
        selected_preprocessing_variant=selected_variant,
        preprocessing_warning=preprocessing_result.get("preprocessing_warning"),
        preprocessing_candidate_reports=candidate_reports,
        archive_warning=archive_warning,
    )


def archive_voice_profile(profile_slug: str, trash_root: Path) -> VoiceProfileDeleteResult:
    """Profili kalici silmek yerine ``trash_root`` altina tasir (geri alinabilir silme)."""
    profile_dir = resolve_profile_dir(profile_slug)
    if not profile_dir.is_dir():
        raise VoiceProfileError(f"Profil bulunamadi: {profile_slug}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = Path(trash_root) / f"{profile_dir.name}-{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(profile_dir), str(target))
    return VoiceProfileDeleteResult(
        profile_slug=profile_dir.name,
        profile_dir=target,
        deleted=True,
        message=f"Profil cop kutusuna tasindi: {profile_dir.name}",
    )
