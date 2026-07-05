# -*- coding: utf-8 -*-
"""VoxForge icin basit lokal Gradio XTTS-v2 demosu."""

from datetime import datetime
from pathlib import Path
import json
import shutil
import sys
import subprocess
from threading import Lock
import traceback
from typing import Any, Optional

import gradio as gr
import torch
from TTS.api import TTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradio_outputs"
PREPROCESSED_REFERENCE_DIR = PROJECT_ROOT / "outputs" / "preprocessed_references"
GRADIO_QUALITY_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "gradio_quality_reports"
PROFILES_DIR = PROJECT_ROOT / "profiles"
PROFILE_JSON_NAME = "profile.json"
PROFILE_ORIGINAL_REFERENCE_NAME = "original_reference.wav"
PROFILE_PREPROCESSED_REFERENCE_NAME = "preprocessed_reference.wav"
NO_PROFILE_VALUE = ""
MAX_GRADIO_TTS_CHARS = 220
INFERENCE_PRESET_DEFAULT = "Dengeli"
INFERENCE_PRESET_STABLE = "Daha stabil"
INFERENCE_PRESET_NATURAL = "Daha doğal deneme"
INFERENCE_PRESET_LONGER = "Daha uzun çıktı denemesi"
INFERENCE_PRESET_CHOICES = [
    INFERENCE_PRESET_DEFAULT,
    INFERENCE_PRESET_STABLE,
    INFERENCE_PRESET_NATURAL,
    INFERENCE_PRESET_LONGER,
]
INFERENCE_PRESET_KWARGS: dict[str, dict[str, Any]] = {
    INFERENCE_PRESET_DEFAULT: {},
    INFERENCE_PRESET_STABLE: {
        "temperature": 0.7,
        "top_p": 0.85,
        "top_k": 50,
        "repetition_penalty": 5.0,
    },
    INFERENCE_PRESET_NATURAL: {
        "temperature": 0.75,
        "top_p": 0.9,
        "top_k": 80,
    },
    INFERENCE_PRESET_LONGER: {
        "temperature": 0.75,
        "top_p": 0.9,
        "top_k": 80,
        "length_penalty": 1.0,
    },
}
OUTPUT_NORMALIZE_FILTER = "loudnorm=I=-18:TP=-2:LRA=11"
NO_PROFILE_INFO_TEXT = (
    "### Seçili profil bilgisi\n"
    "Profil seçilmedi. Ses yüklenirse yüklenen ses, yüklenmezse "
    "varsayılan referans ses kullanılacak."
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_preprocessing_utils import PreprocessingError, preprocess_reference_audio
from scripts.audio_concat_utils import concatenate_wavs
from scripts.audio_quality_utils import analyze_audio_file, format_value
from scripts.text_chunking_utils import split_text_for_tts, summarize_chunks
from scripts.voice_profile_utils import (
    VoiceProfileError,
    VoiceProfileResult,
    create_voice_profile,
    delete_voice_profile,
    recreate_voice_profile,
)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"

_tts_model: Optional[TTS] = None
_tts_device: Optional[str] = None
_model_lock = Lock()


class ProfileSelectionError(Exception):
    """Profil secimi arayuze sade hata mesaji dondursun diye kullanilir."""


def get_tts_model() -> TTS:
    """XTTS modelini bir kez yukler, sonraki isteklerde ayni modeli kullanir."""
    global _tts_model, _tts_device

    with _model_lock:
        if _tts_model is None:
            _tts_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Model yukleniyor: {MODEL_NAME}")
            print(f"Kullanilacak cihaz: {_tts_device}")

            _tts_model = TTS(model_name=MODEL_NAME)
            _tts_model.to(_tts_device)

        return _tts_model


def resolve_reference_audio(uploaded_audio) -> Path:
    """Yuklenen referans sesi kullanir; yoksa varsayilan dosyaya doner."""
    if uploaded_audio:
        if isinstance(uploaded_audio, dict) and uploaded_audio.get("path"):
            return Path(uploaded_audio["path"])
        return Path(uploaded_audio)

    return DEFAULT_REFERENCE_AUDIO


def profile_slug_to_dir(profile_slug: str) -> Path:
    """Dropdown degerinin profiles klasoru disina cikmadigini dogrular."""
    profile_dir = PROFILES_DIR / profile_slug
    profiles_root = PROFILES_DIR.resolve()
    resolved_profile_dir = profile_dir.resolve()

    try:
        resolved_profile_dir.relative_to(profiles_root)
    except ValueError as exc:
        raise ProfileSelectionError("Gecersiz profil yolu secildi.") from exc

    return profile_dir


def read_profile_json(profile_json_path: Path) -> dict:
    """profile.json dosyasini okur; bozuk metadata icin temiz hata verir."""
    try:
        metadata = json.loads(profile_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileSelectionError(
            "Profil metadata dosyasi bozuk: "
            f"{profile_json_path} (satir {exc.lineno}, kolon {exc.colno})"
        ) from exc
    except OSError as exc:
        raise ProfileSelectionError(
            f"Profil metadata dosyasi okunamadi: {profile_json_path}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ProfileSelectionError(
            f"Profil metadata dosyasi beklenen JSON obje formatinda degil: {profile_json_path}"
        )

    return metadata


def profile_name_from_metadata(metadata: dict[str, Any], profile_slug: str) -> str:
    """Profil adini metadata icinden okur, yoksa slug'a duser."""
    profile_name = str(metadata.get("profile_name") or profile_slug).strip()
    return profile_name or profile_slug


def quality_summary(report: Any) -> str:
    """Profil bilgi paneli icin kalite raporunu tek satira indirir."""
    if not isinstance(report, dict):
        return "UNKNOWN"

    quality = report.get("quality") or "UNKNOWN"
    duration = format_value(report.get("duration_seconds"), " saniye")
    sample_rate = format_value(report.get("sample_rate"), " Hz")
    channels = format_value(report.get("channels"))
    return (
        f"{quality} "
        f"(süre: {duration}, sample rate: {sample_rate}, kanal: {channels})"
    )


def scan_local_profiles() -> dict[str, list[dict[str, Any]]]:
    """profiles/ klasorunu tarar ve gecerli/gecersiz profilleri ayirir."""
    valid_profiles: list[dict[str, Any]] = []
    invalid_profiles: list[dict[str, Any]] = []

    if not PROFILES_DIR.is_dir():
        return {
            "valid_profiles": valid_profiles,
            "invalid_profiles": invalid_profiles,
        }

    for profile_dir in sorted(PROFILES_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not profile_dir.is_dir():
            continue

        profile_slug = profile_dir.name
        profile_json_path = profile_dir / PROFILE_JSON_NAME
        original_reference = profile_dir / PROFILE_ORIGINAL_REFERENCE_NAME
        preprocessed_reference = profile_dir / PROFILE_PREPROCESSED_REFERENCE_NAME

        profile_record: dict[str, Any] = {
            "profile_slug": profile_slug,
            "profile_dir": profile_dir,
            "profile_json_path": profile_json_path,
            "original_reference": original_reference,
            "preprocessed_reference": preprocessed_reference,
            "original_reference_exists": original_reference.is_file(),
            "preprocessed_reference_exists": preprocessed_reference.is_file(),
            "issues": [],
        }

        if not profile_json_path.is_file():
            profile_record["issues"].append("profile.json bulunamadi.")
        if not preprocessed_reference.is_file():
            profile_record["issues"].append("preprocessed_reference.wav bulunamadi.")

        metadata = None
        if profile_json_path.is_file():
            try:
                metadata = read_profile_json(profile_json_path)
            except ProfileSelectionError as exc:
                profile_record["issues"].append(str(exc))

        if profile_record["issues"]:
            invalid_profiles.append(profile_record)
            continue

        if metadata is None:
            profile_record["issues"].append("profile.json okunamadi.")
            invalid_profiles.append(profile_record)
            continue

        profile_record.update(
            {
                "profile_name": profile_name_from_metadata(metadata, profile_slug),
                "created_at": metadata.get("created_at"),
                "original_quality": metadata.get("original_quality"),
                "preprocessed_quality": metadata.get("preprocessed_quality"),
                "selected_preprocessing_variant": metadata.get(
                    "selected_preprocessing_variant"
                ),
                "preprocessing_warning": metadata.get("preprocessing_warning"),
            }
        )
        valid_profiles.append(profile_record)

    return {
        "valid_profiles": valid_profiles,
        "invalid_profiles": invalid_profiles,
    }


def build_profile_dropdown_choices(
    profile_scan: dict[str, list[dict[str, Any]]] | None = None,
) -> list[tuple[str, str]]:
    """Guncel profil taramasindan dropdown secenekleri olusturur."""
    if profile_scan is None:
        profile_scan = scan_local_profiles()

    choices: list[tuple[str, str]] = [("Profil yok", NO_PROFILE_VALUE)]
    for profile in profile_scan["valid_profiles"]:
        profile_name = profile.get("profile_name") or profile["profile_slug"]
        choices.append(
            (f"{profile_name} ({profile['profile_slug']})", profile["profile_slug"])
        )
    return choices


def load_selected_profile(profile_slug: str | None) -> dict | None:
    """Secilen profil metadata ve referans yolunu yukler."""
    if not profile_slug or profile_slug == NO_PROFILE_VALUE:
        return None

    profile_dir = profile_slug_to_dir(str(profile_slug))
    profile_json_path = profile_dir / PROFILE_JSON_NAME
    preprocessed_reference = profile_dir / PROFILE_PREPROCESSED_REFERENCE_NAME

    if not profile_dir.is_dir():
        raise ProfileSelectionError(f"Profil klasoru bulunamadi: {profile_slug}")
    if not profile_json_path.is_file():
        raise ProfileSelectionError(
            f"Profil metadata dosyasi bulunamadi: {profile_json_path}"
        )
    if not preprocessed_reference.is_file():
        raise ProfileSelectionError(
            f"Profil on islenmis referans sesi bulunamadi: {preprocessed_reference}"
        )

    metadata = read_profile_json(profile_json_path)
    profile_name = (
        str(metadata.get("profile_name") or profile_slug).strip()
        or str(profile_slug)
    )
    selected_variant = (
        metadata.get("selected_preprocessing_variant")
        or "profile_preprocessed_reference"
    )
    original_quality = metadata.get("original_quality")
    if not isinstance(original_quality, dict):
        original_quality = {
            "quality": "UNKNOWN",
            "warnings": ["profile.json icinde original_quality yok."],
        }

    preprocessed_quality = metadata.get("preprocessed_quality")
    if not isinstance(preprocessed_quality, dict):
        preprocessed_quality = analyze_audio_file(preprocessed_reference)

    candidate_reports = metadata.get("preprocessing_candidate_reports")
    if not isinstance(candidate_reports, dict):
        candidate_reports = {
            "original": original_quality,
            selected_variant: preprocessed_quality,
        }

    preprocessing_result = {
        "selected_output_path": str(preprocessed_reference),
        "selected_variant": selected_variant,
        "original_duration_seconds": original_quality.get("duration_seconds"),
        "selected_duration_seconds": preprocessed_quality.get("duration_seconds"),
        "preprocessing_warning": metadata.get("preprocessing_warning"),
        "candidate_reports": candidate_reports,
    }

    return {
        "profile_name": profile_name,
        "profile_slug": str(profile_slug),
        "profile_dir": profile_dir,
        "profile_json_path": profile_json_path,
        "preprocessed_reference": preprocessed_reference,
        "metadata": metadata,
        "original_quality": original_quality,
        "preprocessed_quality": preprocessed_quality,
        "preprocessing_result": preprocessing_result,
    }


def build_profile_refresh_message(profile_scan: dict[str, list[dict[str, Any]]]) -> str:
    """Profil yenileme sonucunu kullaniciya sade metinle gosterir."""
    valid_count = len(profile_scan["valid_profiles"])
    invalid_profiles = profile_scan["invalid_profiles"]

    if valid_count == 0:
        lines = ["Henüz yerel ses profili bulunamadı."]
    else:
        lines = [f"Geçerli yerel ses profili sayısı: {valid_count}."]

    if invalid_profiles:
        lines.append(
            f"Atlanan eksik veya bozuk profil sayısı: {len(invalid_profiles)}."
        )
        for profile in invalid_profiles:
            issue_text = "; ".join(profile.get("issues") or ["bilinmeyen sorun"])
            lines.append(f"- `{profile['profile_slug']}`: {issue_text}")

    return "\n".join(lines)


def find_scanned_profile(
    profile_scan: dict[str, list[dict[str, Any]]],
    profile_slug: str,
) -> dict[str, Any] | None:
    """Taranmis gecerli profiller icinde slug arar."""
    for profile in profile_scan["valid_profiles"]:
        if profile["profile_slug"] == profile_slug:
            return profile
    return None


def build_selected_profile_info(selected_profile_slug: str | None) -> str:
    """Secili profil bilgi panelini olusturur."""
    if not selected_profile_slug or selected_profile_slug == NO_PROFILE_VALUE:
        return NO_PROFILE_INFO_TEXT

    profile_scan = scan_local_profiles()
    profile = find_scanned_profile(profile_scan, str(selected_profile_slug))
    if profile is None:
        for invalid_profile in profile_scan["invalid_profiles"]:
            if invalid_profile["profile_slug"] == selected_profile_slug:
                issues = "; ".join(
                    invalid_profile.get("issues") or ["bilinmeyen sorun"]
                )
                return (
                    "### Seçili profil bilgisi\n"
                    f"Uyarı: `{selected_profile_slug}` profili geçersiz görünüyor.\n\n"
                    f"Sorun: {issues}"
                )

        return (
            "### Seçili profil bilgisi\n"
            f"Uyarı: `{selected_profile_slug}` profili artık bulunamadı. "
            "Profilleri yenileyin veya başka bir profil seçin."
        )

    warning = profile.get("preprocessing_warning") or "yok"
    selected_variant = profile.get("selected_preprocessing_variant") or "bilinmiyor"
    preprocessed_exists = (
        "evet" if profile.get("preprocessed_reference_exists") else "hayır"
    )

    return "\n".join(
        [
            "### Seçili profil bilgisi",
            f"- Profil adı: {profile.get('profile_name', 'bilinmiyor')}",
            f"- Profil slug: `{profile.get('profile_slug', 'bilinmiyor')}`",
            f"- Oluşturulma tarihi: {profile.get('created_at') or 'bilinmiyor'}",
            f"- original_quality: {quality_summary(profile.get('original_quality'))}",
            "- preprocessed_quality: "
            f"{quality_summary(profile.get('preprocessed_quality'))}",
            f"- selected_preprocessing_variant: `{selected_variant}`",
            f"- preprocessing_warning: {warning}",
            f"- preprocessed_reference.wav var mı: {preprocessed_exists}",
        ]
    )


def refresh_profile_choices(selected_profile_slug: str | None):
    """Butona basildiginda profilleri yeniden tarar ve secimi korumaya calisir."""
    profile_scan = scan_local_profiles()
    choices = build_profile_dropdown_choices(profile_scan)
    valid_values = {value for _, value in choices if value != NO_PROFILE_VALUE}

    if selected_profile_slug in valid_values:
        next_value = selected_profile_slug
    else:
        next_value = NO_PROFILE_VALUE

    return (
        gr.update(choices=choices, value=next_value),
        build_profile_refresh_message(profile_scan),
        build_selected_profile_info(next_value),
    )


def resolve_uploaded_audio_path(uploaded_audio) -> Path | None:
    """Profil olusturma input'undaki yuklenen sesi Path olarak cozer."""
    if not uploaded_audio:
        return None
    if isinstance(uploaded_audio, dict) and uploaded_audio.get("path"):
        return Path(uploaded_audio["path"])
    return Path(uploaded_audio)


def build_created_profile_status(result: VoiceProfileResult) -> str:
    """Basarili profil olusturma sonucunu Gradio icin ozetler."""
    warning = result.preprocessing_warning or "yok"
    return "\n".join(
        [
            "Profil oluşturuldu.",
            f"- Profil adı: {result.profile_name}",
            f"- Profil slug: `{result.profile_slug}`",
            f"- original_quality: {quality_summary(result.original_quality)}",
            f"- preprocessed_quality: {quality_summary(result.preprocessed_quality)}",
            "- selected_preprocessing_variant: "
            f"`{result.selected_preprocessing_variant}`",
            f"- preprocessing_warning: {warning}",
        ]
    )


def build_recreated_profile_status(result: VoiceProfileResult) -> str:
    """Basarili profil yenileme sonucunu Gradio icin ozetler."""
    warning = result.preprocessing_warning or "yok"
    return "\n".join(
        [
            "Profil yenilendi.",
            f"- Profil adı: {result.profile_name}",
            f"- Profil slug: `{result.profile_slug}`",
            f"- Korunan orijinal referans: `{result.original_reference}`",
            f"- Yenilenen ön işlenmiş referans: `{result.preprocessed_reference}`",
            f"- original_quality: {quality_summary(result.original_quality)}",
            f"- preprocessed_quality: {quality_summary(result.preprocessed_quality)}",
            "- selected_preprocessing_variant: "
            f"`{result.selected_preprocessing_variant}`",
            f"- preprocessing_warning: {warning}",
        ]
    )


def build_profile_create_error_outputs(
    current_profile_slug: str | None,
    message: str,
):
    """Profil olusturma hatasinda mevcut dropdown durumunu korur."""
    profile_scan = scan_local_profiles()
    return (
        gr.update(),
        build_profile_refresh_message(profile_scan),
        build_selected_profile_info(current_profile_slug),
        message,
    )


def create_profile_from_gradio(
    profile_name: str,
    uploaded_audio,
    has_permission: bool,
    current_profile_slug: str | None,
):
    """Gradio arayuzunden yeni yerel voice profile olusturur."""
    if not has_permission:
        return build_profile_create_error_outputs(
            current_profile_slug,
            "Uyarı: Profil oluşturmak için sesin size ait olduğunu veya "
            "kullanma izniniz olduğunu onaylamalısınız.",
        )

    cleaned_profile_name = (profile_name or "").strip()
    if not cleaned_profile_name:
        return build_profile_create_error_outputs(
            current_profile_slug,
            "Uyarı: Profil oluşturmak için profil adı girmelisiniz.",
        )

    input_audio_path = resolve_uploaded_audio_path(uploaded_audio)
    if input_audio_path is None:
        return build_profile_create_error_outputs(
            current_profile_slug,
            "Uyarı: Profil oluşturmak için referans ses dosyası yüklemelisiniz.",
        )

    try:
        result = create_voice_profile(cleaned_profile_name, input_audio_path)
    except VoiceProfileError as exc:
        print(f"Profil olusturma hatasi: {exc}")
        return build_profile_create_error_outputs(
            current_profile_slug,
            f"HATA: {exc}",
        )
    except Exception:
        print("Profil olusturma sirasinda beklenmeyen teknik hata olustu.")
        traceback.print_exc()
        return build_profile_create_error_outputs(
            current_profile_slug,
            "HATA: Profil oluşturma sırasında teknik bir sorun oluştu. "
            "Ayrıntı terminale yazıldı.",
        )

    profile_scan = scan_local_profiles()
    choices = build_profile_dropdown_choices(profile_scan)
    valid_values = {value for _, value in choices if value != NO_PROFILE_VALUE}
    next_value = (
        result.profile_slug
        if result.profile_slug in valid_values
        else NO_PROFILE_VALUE
    )
    created_status = build_created_profile_status(result)
    if next_value == NO_PROFILE_VALUE:
        created_status += (
            "\n\nUyarı: Profil oluşturuldu ancak geçerli profil listesinde "
            "görünmedi. Lütfen dosyaları ve kalite raporunu kontrol edin."
        )
    return (
        gr.update(choices=choices, value=next_value),
        build_profile_refresh_message(profile_scan),
        build_selected_profile_info(next_value),
        created_status,
    )


def build_profile_management_error_outputs(
    current_profile_slug: str | None,
    message: str,
):
    """Profil silme/yenileme hatasinda secili durumu korur."""
    profile_scan = scan_local_profiles()
    choices = build_profile_dropdown_choices(profile_scan)
    valid_values = {value for _, value in choices if value != NO_PROFILE_VALUE}
    next_value = (
        current_profile_slug
        if current_profile_slug in valid_values
        else NO_PROFILE_VALUE
    )
    return (
        gr.update(choices=choices, value=next_value),
        build_profile_refresh_message(profile_scan),
        build_selected_profile_info(next_value),
        message,
    )


def recreate_selected_profile_from_gradio(selected_profile_slug: str | None):
    """Secili profili model yuklemeden original_reference.wav uzerinden yeniler."""
    if not selected_profile_slug or selected_profile_slug == NO_PROFILE_VALUE:
        return build_profile_management_error_outputs(
            selected_profile_slug,
            "Uyarı: Yenilemek için önce bir profil seçmelisiniz.",
        )

    try:
        result = recreate_voice_profile(str(selected_profile_slug))
    except VoiceProfileError as exc:
        print(f"Profil yenileme hatasi: {exc}")
        return build_profile_management_error_outputs(
            selected_profile_slug,
            f"HATA: {exc}",
        )
    except Exception:
        print("Profil yenileme sirasinda beklenmeyen teknik hata olustu.")
        traceback.print_exc()
        return build_profile_management_error_outputs(
            selected_profile_slug,
            "HATA: Profil yenileme sırasında teknik bir sorun oluştu. "
            "Ayrıntı terminale yazıldı.",
        )

    profile_scan = scan_local_profiles()
    choices = build_profile_dropdown_choices(profile_scan)
    valid_values = {value for _, value in choices if value != NO_PROFILE_VALUE}
    next_value = (
        result.profile_slug
        if result.profile_slug in valid_values
        else NO_PROFILE_VALUE
    )
    status = build_recreated_profile_status(result)
    if next_value == NO_PROFILE_VALUE:
        status += (
            "\n\nUyarı: Profil yenilendi ancak geçerli profil listesinde "
            "görünmedi. Lütfen profil dosyalarını kontrol edin."
        )

    return (
        gr.update(choices=choices, value=next_value),
        build_profile_refresh_message(profile_scan),
        build_selected_profile_info(next_value),
        status,
    )


def delete_selected_profile_from_gradio(
    selected_profile_slug: str | None,
    delete_confirmed: bool,
):
    """Onaylanan secili profili siler ve dropdown secimini temizler."""
    if not selected_profile_slug or selected_profile_slug == NO_PROFILE_VALUE:
        return (
            *build_profile_management_error_outputs(
                selected_profile_slug,
                "Uyarı: Silmek için önce bir profil seçmelisiniz.",
            ),
            gr.update(value=False),
        )

    if not delete_confirmed:
        return (
            *build_profile_management_error_outputs(
                selected_profile_slug,
                "Uyarı: Profil silinmedi. Önce silme onay checkbox'ını işaretleyin.",
            ),
            gr.update(value=False),
        )

    try:
        result = delete_voice_profile(str(selected_profile_slug))
    except VoiceProfileError as exc:
        print(f"Profil silme hatasi: {exc}")
        return (
            *build_profile_management_error_outputs(
                selected_profile_slug,
                f"HATA: {exc}",
            ),
            gr.update(value=False),
        )
    except Exception:
        print("Profil silme sirasinda beklenmeyen teknik hata olustu.")
        traceback.print_exc()
        return (
            *build_profile_management_error_outputs(
                selected_profile_slug,
                "HATA: Profil silme sırasında teknik bir sorun oluştu. "
                "Ayrıntı terminale yazıldı.",
            ),
            gr.update(value=False),
        )

    profile_scan = scan_local_profiles()
    status = "\n".join(
        [
            result.message,
            f"- Profil slug: `{result.profile_slug}`",
            f"- Silinen klasör: `{result.profile_dir}`",
            "- Seçim temizlendi.",
        ]
    )

    return (
        gr.update(
            choices=build_profile_dropdown_choices(profile_scan),
            value=NO_PROFILE_VALUE,
        ),
        build_profile_refresh_message(profile_scan),
        build_selected_profile_info(NO_PROFILE_VALUE),
        status,
        gr.update(value=False),
    )


def format_quality_report_section(title: str, report: dict, is_raw: bool) -> str:
    """Tek bir ses raporunu Gradio icin sade Markdown metnine cevirir."""
    lines = [
        f"### {title}",
        f"- Kalite sonucu: **{report.get('quality', 'BAD')}**",
        f"- Sure: {format_value(report.get('duration_seconds'), ' saniye')}",
        f"- Sample rate: {format_value(report.get('sample_rate'), ' Hz')}",
        f"- Kanal sayisi: {format_value(report.get('channels'))}",
        f"- Mean volume: {format_value(report.get('mean_volume_db'), ' dB')}",
        f"- Max volume: {format_value(report.get('max_volume_db'), ' dB')}",
    ]

    if is_raw and report.get("channels") and report["channels"] > 1:
        lines.append("- Not: On isleme adiminda ses mono WAV formatina cevrilir.")

    if is_raw and report.get("sample_rate") and report["sample_rate"] != 24000:
        lines.append("- Not: On isleme adiminda sample rate 24000 Hz'e cevrilir.")

    if report.get("quality") == "BAD" and report.get("exists"):
        lines.append("- Dikkat: Kalite BAD. Uretim engellenmedi, ama sonucu mutlaka dinleyerek kontrol et.")

    warnings = report.get("warnings") or ["Belirgin uyari yok."]
    lines.append("")
    lines.append("Uyarilar:")
    lines.extend(f"- {warning}" for warning in warnings)

    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.append("")
        lines.append("Oneriler:")
        lines.extend(f"- {recommendation}" for recommendation in recommendations)

    return "\n".join(lines)


def build_preprocessing_section(preprocessing_result: dict | None) -> str:
    """On isleme guvenlik sonucunu kalite raporunda gorunur yapar."""
    if preprocessing_result is None:
        return "### On isleme guvenlik notu\n- preprocessing_warning: rapor olusturulamadi."

    warning = preprocessing_result.get("preprocessing_warning") or "yok"
    selected_variant = preprocessing_result.get("selected_variant", "bilinmiyor")
    original_duration = format_value(
        preprocessing_result.get("original_duration_seconds"),
        " saniye",
    )
    selected_duration = format_value(
        preprocessing_result.get("selected_duration_seconds"),
        " saniye",
    )

    return "\n".join(
        [
            "### On isleme guvenlik notu",
            f"- Secilen varyant: `{selected_variant}`",
            f"- Ham sure: {original_duration}",
            f"- Secilen sure: {selected_duration}",
            f"- preprocessing_warning: {warning}",
        ]
    )


def build_profile_report_section(profile_info: dict | None) -> str:
    """Secilen yerel profil bilgisini kalite raporuna ekler."""
    if not profile_info:
        return ""

    preprocessing_result = profile_info.get("preprocessing_result") or {}
    warning = preprocessing_result.get("preprocessing_warning") or "yok"
    selected_variant = preprocessing_result.get("selected_variant") or "bilinmiyor"

    return "\n".join(
        [
            "### Yerel ses profili",
            f"- Profil adi: {profile_info.get('profile_name', 'bilinmiyor')}",
            f"- Profil slug: `{profile_info.get('profile_slug', 'bilinmiyor')}`",
            f"- selected_preprocessing_variant: `{selected_variant}`",
            f"- preprocessing_warning: {warning}",
            "- original_quality: asagidaki ham referans ses raporunda gosterildi.",
            "- preprocessed_quality: asagidaki on islenmis referans ses raporunda gosterildi.",
        ]
    )


def build_quality_report_text(
    raw_report: dict,
    preprocessed_report: dict | None,
    preprocessing_result: dict | None = None,
    profile_info: dict | None = None,
) -> str:
    """Ham ve on islenmis referans raporlarini tek metinde birlestirir."""
    sections = []
    profile_section = build_profile_report_section(profile_info)
    if profile_section:
        sections.append(profile_section)

    sections.append(format_quality_report_section("Ham referans ses", raw_report, is_raw=True))

    if preprocessed_report is None:
        sections.append("### On islenmis referans ses\n- Rapor olusturulamadi.")
    else:
        sections.append(
            format_quality_report_section(
                "On islenmis referans ses",
                preprocessed_report,
                is_raw=False,
            )
        )

    sections.append(build_preprocessing_section(preprocessing_result))
    return "\n\n".join(sections)


def write_gradio_quality_report(
    timestamp: str,
    raw_report: dict,
    preprocessed_report: dict | None,
    reference_audio: Path,
    preprocessed_audio: Path | None,
    output_audio: Path | None,
    status: str,
    preprocessing_result: dict | None = None,
    source_info: dict | None = None,
    profile_info: dict | None = None,
    generation_report: dict | None = None,
    error_message: str | None = None,
) -> Path:
    """Her Gradio uretim denemesi icin kalite raporunu JSON olarak kaydeder."""
    GRADIO_QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = GRADIO_QUALITY_REPORT_DIR / f"quality_report_{timestamp}.json"
    generation_report = generation_report or {}
    payload = {
        "created_at": timestamp,
        "status": status,
        "reference_audio": str(reference_audio),
        "preprocessed_audio": str(preprocessed_audio) if preprocessed_audio else None,
        "output_audio": str(output_audio) if output_audio else None,
        "inference_preset": generation_report.get("inference_preset"),
        "ab_test_enabled": generation_report.get("ab_test_enabled"),
        "post_processing_enabled": generation_report.get("post_processing_enabled"),
        "chunking_used": generation_report.get("chunking_used"),
        "chunk_count": generation_report.get("chunk_count"),
        "chunks": generation_report.get("chunks"),
        "primary_output_path": generation_report.get("primary_output_path"),
        "comparison_output_path": generation_report.get("comparison_output_path"),
        "source_type": source_info.get("source_type") if source_info else None,
        "source_label": source_info.get("source_label") if source_info else None,
        "device": _tts_device,
        "error_message": error_message,
        "raw_reference_report": raw_report,
        "preprocessed_reference_report": preprocessed_report,
        "profile_name": profile_info.get("profile_name") if profile_info else None,
        "profile_slug": profile_info.get("profile_slug") if profile_info else None,
        "profile_json_path": (
            str(profile_info.get("profile_json_path")) if profile_info else None
        ),
        "selected_preprocessing_variant": (
            preprocessing_result.get("selected_variant") if preprocessing_result else None
        ),
        "preprocessing_warning": (
            preprocessing_result.get("preprocessing_warning") if preprocessing_result else None
        ),
        "preprocessing_candidate_reports": (
            preprocessing_result.get("candidate_reports") if preprocessing_result else None
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def build_quality_status_warning(raw_report: dict, preprocessed_report: dict | None) -> str:
    """BAD kalite varsa durum mesajina kisa uyari ekler."""
    bad_labels = []
    if raw_report.get("quality") == "BAD":
        bad_labels.append("ham referans")
    if preprocessed_report and preprocessed_report.get("quality") == "BAD":
        bad_labels.append("on islenmis referans")

    if not bad_labels:
        return ""

    return f"\nKalite uyarisi: {', '.join(bad_labels)} icin sonuc BAD. Ciktiyi dinleyerek kontrol et."


def build_preprocessing_status_warning(preprocessing_result: dict | None) -> str:
    """Durum mesajina on isleme guvenlik uyarisini ekler."""
    if not preprocessing_result:
        return ""

    warning = preprocessing_result.get("preprocessing_warning")
    if not warning:
        return ""

    return f"\nOn isleme uyarisi: {warning}"


def normalize_inference_preset(preset_label: str | None) -> str:
    """Bilinmeyen preset degerini guvenli varsayilana indirger."""
    if preset_label in INFERENCE_PRESET_KWARGS:
        return preset_label
    return INFERENCE_PRESET_DEFAULT


def run_tts_to_file_with_preset(
    tts: TTS,
    text: str,
    speaker_wav: Path,
    output_path: Path,
    preset_label: str,
) -> str | None:
    """XTTS cagrısını preset kwargs ile dener; TypeError olursa default'a duser."""
    kwargs = dict(INFERENCE_PRESET_KWARGS.get(preset_label, {}))
    base_call = {
        "text": text,
        "speaker_wav": str(speaker_wav),
        "language": LANGUAGE,
        "file_path": str(output_path),
    }

    try:
        tts.tts_to_file(**base_call, **kwargs)
        return None
    except TypeError as exc:
        if not kwargs:
            raise

        print(f"Preset parametreleri desteklenmedi, default uretime dusuluyor: {exc}")
        tts.tts_to_file(**base_call)
        return (
            "Seçilen üretim modu bu XTTS API sürümünde tamamen desteklenmedi; "
            "default üretime geri düşüldü."
        )


def normalize_output_wav(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Final WAV icin hafif FFmpeg loudnorm post-processing uygular."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False, "FFmpeg bulunamadı; ham çıktı korundu."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        OUTPUT_NORMALIZE_FILTER,
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    print("Cikti post-processing FFmpeg komutu:")
    print(subprocess.list2cmdline(command))

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "bilinmeyen FFmpeg hatasi"
        return False, f"Çıktı normalize edilemedi; ham çıktı korundu. Ayrıntı: {detail}"

    if not output_path.is_file():
        return False, "Normalize komutu tamamlandı ancak işlenmiş çıktı oluşmadı; ham çıktı korundu."

    return True, "Çıktı sesi normalize edildi."


def synthesize_gradio_output(
    tts: TTS,
    text: str,
    speaker_wav: Path,
    output_stem: str,
    preset_label: str,
    post_processing_enabled: bool,
) -> dict[str, Any]:
    """Tek veya chunked XTTS uretimini yapar ve final cikti yolunu dondurur."""
    chunks = (
        split_text_for_tts(text, max_chars=MAX_GRADIO_TTS_CHARS)
        if len(text) > MAX_GRADIO_TTS_CHARS
        else [text]
    )
    chunk_summary = summarize_chunks(chunks)
    raw_output_path = OUTPUT_DIR / f"{output_stem}_raw.wav"
    warnings: list[str] = []
    chunk_paths: list[Path] = []

    if chunk_summary["chunking_used"]:
        chunk_dir = OUTPUT_DIR / "chunks" / output_stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = chunk_dir / f"chunk_{index:02d}.wav"
            warning = run_tts_to_file_with_preset(
                tts,
                chunk,
                speaker_wav,
                chunk_path,
                preset_label,
            )
            if warning:
                warnings.append(warning)
            chunk_paths.append(chunk_path)

        concat_ok, concat_message = concatenate_wavs(chunk_paths, raw_output_path)
        if not concat_ok:
            raise RuntimeError(f"WAV parçaları birleştirilemedi: {concat_message}")
    else:
        warning = run_tts_to_file_with_preset(
            tts,
            chunks[0],
            speaker_wav,
            raw_output_path,
            preset_label,
        )
        if warning:
            warnings.append(warning)

    final_output_path = raw_output_path
    post_processing_applied = False
    if post_processing_enabled:
        normalized_output_path = OUTPUT_DIR / f"{output_stem}_normalized.wav"
        normalized_ok, normalized_message = normalize_output_wav(
            raw_output_path,
            normalized_output_path,
        )
        if normalized_ok:
            final_output_path = normalized_output_path
            post_processing_applied = True
        else:
            warnings.append(normalized_message)

    return {
        "preset_label": preset_label,
        "raw_output_path": raw_output_path,
        "final_output_path": final_output_path,
        "post_processing_enabled": post_processing_enabled,
        "post_processing_applied": post_processing_applied,
        "chunking_used": chunk_summary["chunking_used"],
        "chunk_count": chunk_summary["chunk_count"],
        "chunks": chunk_summary["chunks"],
        "warnings": warnings,
    }


def build_generation_report(
    inference_preset: str,
    ab_test_enabled: bool,
    post_processing_enabled: bool,
    primary_result: dict[str, Any] | None = None,
    comparison_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gradio kalite raporuna eklenecek uretim ayarlarini olusturur."""
    chunking_used = primary_result.get("chunking_used") if primary_result else False
    chunk_count = primary_result.get("chunk_count") if primary_result else 0
    chunks = primary_result.get("chunks") if primary_result else []
    primary_output_path = (
        str(primary_result.get("final_output_path")) if primary_result else None
    )
    comparison_output_path = (
        str(comparison_result.get("final_output_path")) if comparison_result else None
    )

    return {
        "inference_preset": inference_preset,
        "ab_test_enabled": ab_test_enabled,
        "post_processing_enabled": post_processing_enabled,
        "chunking_used": chunking_used,
        "chunk_count": chunk_count,
        "chunks": chunks,
        "primary_output_path": primary_output_path,
        "comparison_output_path": comparison_output_path,
    }


def build_generation_status_lines(
    primary_result: dict[str, Any],
    comparison_result: dict[str, Any] | None,
    ab_test_enabled: bool,
) -> list[str]:
    """Uretim ayarlarini sade durum satirlarina cevirir."""
    chunking_text = (
        f"kullanıldı ({primary_result['chunk_count']} parça)"
        if primary_result["chunking_used"]
        else "kullanılmadı"
    )
    post_processing_text = (
        "uygulandı"
        if primary_result["post_processing_applied"]
        else (
            "denendi, ham çıktı korundu"
            if primary_result["post_processing_enabled"]
            else "kapalı"
        )
    )
    lines = [
        f"Üretim modu: {primary_result['preset_label']}",
        f"Chunking: {chunking_text}",
        f"Post-processing: {post_processing_text}",
        f"Final çıktı yolu: {primary_result['final_output_path'].resolve()}",
        f"A/B karşılaştırma: {'açık' if ab_test_enabled else 'kapalı'}",
    ]

    if comparison_result:
        comparison_post = (
            "uygulandı"
            if comparison_result["post_processing_applied"]
            else (
                "denendi, ham çıktı korundu"
                if comparison_result["post_processing_enabled"]
                else "kapalı"
            )
        )
        lines.extend(
            [
                f"Karşılaştırma modu: {comparison_result['preset_label']}",
                f"Karşılaştırma post-processing: {comparison_post}",
                "Karşılaştırma çıktı yolu: "
                f"{comparison_result['final_output_path'].resolve()}",
            ]
        )

    warnings = list(primary_result.get("warnings") or [])
    if comparison_result:
        warnings.extend(comparison_result.get("warnings") or [])
    for warning in dict.fromkeys(warnings):
        lines.append(f"Uyarı: {warning}")

    return lines


def generate_voice(
    text: str,
    uploaded_audio,
    selected_profile_slug,
    inference_preset: str,
    post_processing_enabled: bool,
    ab_test_enabled: bool,
    has_permission: bool,
):
    """Gradio butonuna basilinca Turkce ses uretimi yapar."""
    if not has_permission:
        return (
            None,
            None,
            "Uyari: Ses uretimi icin sesin size ait oldugunu veya kullanma izniniz oldugunu onaylamalisiniz.",
            "",
        )

    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return None, None, "Uyari: Ses uretmek icin Turkce metin girmelisiniz.", ""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    selected_preset = normalize_inference_preset(inference_preset)
    post_processing_enabled = bool(post_processing_enabled)
    ab_test_enabled = bool(ab_test_enabled)
    output_stem = f"xtts_tr_{timestamp}"
    planned_output_audio = OUTPUT_DIR / f"{output_stem}_raw.wav"
    generation_report = build_generation_report(
        selected_preset,
        ab_test_enabled,
        post_processing_enabled,
    )
    profile_info = None

    try:
        profile_info = load_selected_profile(selected_profile_slug)
    except ProfileSelectionError as exc:
        print(f"Profil secimi hatasi: {exc}")
        return None, None, f"HATA: {exc}", ""

    if profile_info:
        source_info = {
            "source_type": "local_profile",
            "source_label": "Kaynak: yerel profil",
        }
        reference_audio = profile_info["preprocessed_reference"]
        preprocessed_audio = profile_info["preprocessed_reference"]
        raw_report = profile_info["original_quality"]
        preprocessed_report = profile_info["preprocessed_quality"]
        preprocessing_result = profile_info["preprocessing_result"]

        print("Kaynak: yerel profil")
        print(f"Profil adi: {profile_info['profile_name']}")
        print(f"Profil slug: {profile_info['profile_slug']}")
        print(f"Profil on islenmis referans: {preprocessed_audio}")
        print(f"Planlanan cikti ses dosyasi: {planned_output_audio}")

        quality_report_text = build_quality_report_text(
            raw_report,
            preprocessed_report,
            preprocessing_result,
            profile_info,
        )
    else:
        source_info = {
            "source_type": "uploaded_reference" if uploaded_audio else "default_reference",
            "source_label": (
                "Kaynak: yüklenen referans ses"
                if uploaded_audio
                else "Kaynak: varsayılan referans ses"
            ),
        }
        reference_audio = resolve_reference_audio(uploaded_audio)
        raw_report = analyze_audio_file(reference_audio)
        if not raw_report.get("exists"):
            quality_report_text = build_quality_report_text(raw_report, None, None)
            return (
                None,
                None,
                f"HATA: Referans ses dosyasi bulunamadi: {reference_audio}\n"
                f"{source_info['source_label']}",
                quality_report_text,
            )

        print(source_info["source_label"])
        print(f"Referans ses dosyasi: {reference_audio}")
        PREPROCESSED_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        preprocessed_audio = PREPROCESSED_REFERENCE_DIR / f"reference_{timestamp}.wav"
        try:
            preprocessing_result = preprocess_reference_audio(
                reference_audio,
                preprocessed_audio,
                mode="safe",
            )
        except PreprocessingError as exc:
            preprocess_error = str(exc)
            quality_report_path = write_gradio_quality_report(
                timestamp,
                raw_report,
                None,
                reference_audio,
                None,
                None,
                "preprocess_failed",
                source_info=source_info,
                generation_report=generation_report,
                error_message=preprocess_error,
            )
            quality_report_text = build_quality_report_text(raw_report, None, None)
            return (
                None,
                None,
                "HATA: Referans ses on isleme basarisiz oldu.\n"
                f"{source_info['source_label']}\n"
                f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
                f"Ayrinti: {preprocess_error}\n"
                f"Kalite raporu JSON: {quality_report_path.resolve()}",
                quality_report_text,
            )

        preprocessed_audio = Path(preprocessing_result["selected_output_path"])

        print(f"On islenmis referans ses dosyasi: {preprocessed_audio}")
        print(f"On isleme varyanti: {preprocessing_result.get('selected_variant')}")
        if preprocessing_result.get("preprocessing_warning"):
            print(f"On isleme uyarisi: {preprocessing_result['preprocessing_warning']}")
        print(f"Planlanan cikti ses dosyasi: {planned_output_audio}")
        preprocessed_report = (
            preprocessing_result.get("candidate_reports", {})
            .get(preprocessing_result.get("selected_variant"))
        )
        if preprocessed_report is None:
            preprocessed_report = analyze_audio_file(preprocessed_audio)

        quality_report_text = build_quality_report_text(
            raw_report,
            preprocessed_report,
            preprocessing_result,
        )

    primary_result = None
    comparison_result = None
    try:
        tts = get_tts_model()
        device = _tts_device or "bilinmiyor"
        primary_result = synthesize_gradio_output(
            tts,
            cleaned_text,
            preprocessed_audio,
            output_stem,
            selected_preset,
            post_processing_enabled,
        )
        if ab_test_enabled:
            comparison_result = synthesize_gradio_output(
                tts,
                cleaned_text,
                preprocessed_audio,
                f"{output_stem}_ab_stable",
                INFERENCE_PRESET_STABLE,
                post_processing_enabled,
            )
        generation_report = build_generation_report(
            selected_preset,
            ab_test_enabled,
            post_processing_enabled,
            primary_result,
            comparison_result,
        )
    except Exception as exc:
        device = _tts_device or "bilinmiyor"
        print(f"Ses uretimi hatasi: {exc}")
        generation_report = build_generation_report(
            selected_preset,
            ab_test_enabled,
            post_processing_enabled,
            primary_result,
            comparison_result,
        )
        report_output_audio = (
            primary_result.get("final_output_path")
            if primary_result
            else planned_output_audio
        )
        quality_report_path = write_gradio_quality_report(
            timestamp,
            raw_report,
            preprocessed_report,
            reference_audio,
            preprocessed_audio,
            report_output_audio,
            "tts_failed",
            preprocessing_result=preprocessing_result,
            source_info=source_info,
            profile_info=profile_info,
            generation_report=generation_report,
            error_message="Ses uretimi sirasinda sorun olustu.",
        )
        return (
            None,
            None,
            "HATA: Ses uretimi sirasinda sorun olustu.\n"
            f"{source_info['source_label']}\n"
            f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
            f"Kullanilan speaker_wav yolu: {preprocessed_audio.resolve()}\n"
            f"Uretilmesi planlanan cikti ses yolu: {planned_output_audio.resolve()}\n"
            f"Uretim modu: {selected_preset}\n"
            f"A/B karsilastirma: {'acik' if ab_test_enabled else 'kapali'}\n"
            f"Kullanilan cihaz: {device}\n"
            f"Kalite raporu JSON: {quality_report_path.resolve()}\n"
            f"{build_preprocessing_status_warning(preprocessing_result)}\n"
            "Ayrinti: Teknik hata terminale yazildi.",
            quality_report_text,
        )

    primary_output_path = primary_result["final_output_path"]
    comparison_output_path = (
        comparison_result["final_output_path"] if comparison_result else None
    )
    quality_report_path = write_gradio_quality_report(
        timestamp,
        raw_report,
        preprocessed_report,
        reference_audio,
        preprocessed_audio,
        primary_output_path,
        "success",
        preprocessing_result=preprocessing_result,
        source_info=source_info,
        profile_info=profile_info,
        generation_report=generation_report,
    )
    status_lines = [
        "Ses uretildi.",
        source_info["source_label"],
        f"Kullanilan referans ses yolu: {reference_audio.resolve()}",
        f"Kullanilan speaker_wav yolu: {preprocessed_audio.resolve()}",
        *build_generation_status_lines(
            primary_result,
            comparison_result,
            ab_test_enabled,
        ),
        f"Kullanilan cihaz: {device}",
        f"Kalite raporu JSON: {quality_report_path.resolve()}",
    ]
    preprocessing_warning = build_preprocessing_status_warning(preprocessing_result)
    if preprocessing_warning:
        status_lines.append(preprocessing_warning.strip())
    quality_warning = build_quality_status_warning(raw_report, preprocessed_report)
    if quality_warning:
        status_lines.append(quality_warning.strip())

    status_message = "\n".join(status_lines)
    return (
        str(primary_output_path),
        str(comparison_output_path) if comparison_output_path else None,
        status_message,
        quality_report_text,
    )


def build_demo() -> gr.Blocks:
    """Basit lokal Gradio arayuzunu olusturur."""
    initial_profile_scan = scan_local_profiles()

    with gr.Blocks(title="VoxForge XTTS Demo") as demo:
        gr.Markdown("# VoxForge XTTS Türkçe Demo")

        profile_dropdown = gr.Dropdown(
            label="Yerel ses profili",
            choices=build_profile_dropdown_choices(initial_profile_scan),
            value=NO_PROFILE_VALUE,
            interactive=True,
        )
        refresh_profiles_button = gr.Button("Profilleri yenile")
        profile_refresh_status = gr.Markdown(
            value=build_profile_refresh_message(initial_profile_scan)
        )
        selected_profile_info = gr.Markdown(value=NO_PROFILE_INFO_TEXT)
        gr.Markdown(
            "Profil seçerseniz yüklenen ses dosyası yerine profilin ön işlenmiş referansı kullanılır."
        )
        gr.Markdown("## Seçili profil yönetimi")
        recreate_selected_profile_button = gr.Button("Seçili profili yenile")
        delete_profile_confirm_checkbox = gr.Checkbox(
            label="Seçili profili silmeyi onaylıyorum",
            value=False,
        )
        delete_selected_profile_button = gr.Button("Seçili profili sil")
        profile_management_status = gr.Markdown(
            value="Profil yönetimi sonucu burada görünecek."
        )

        gr.Markdown("## Yeni yerel ses profili oluştur")
        new_profile_name_input = gr.Textbox(
            label="Profil adı",
            placeholder="Örnek: baglare",
        )
        new_profile_audio_input = gr.Audio(
            label="Profil referans ses dosyası",
            type="filepath",
        )
        new_profile_permission_checkbox = gr.Checkbox(
            label="Bu ses bana ait veya kullanma iznim var",
            value=False,
        )
        create_profile_button = gr.Button("Profil oluştur")
        create_profile_status = gr.Markdown(
            value="Profil oluşturma sonucu burada görünecek."
        )

        text_input = gr.Textbox(
            label="Türkçe metin",
            lines=4,
            placeholder="Seslendirmek istediğiniz Türkçe metni yazın.",
        )
        reference_audio_input = gr.Audio(
            label="Referans ses dosyası",
            type="filepath",
        )
        inference_preset_dropdown = gr.Dropdown(
            label="Üretim modu",
            choices=INFERENCE_PRESET_CHOICES,
            value=INFERENCE_PRESET_DEFAULT,
            interactive=True,
        )
        post_processing_checkbox = gr.Checkbox(
            label="Çıktı sesini normalize et",
            value=True,
        )
        ab_test_checkbox = gr.Checkbox(
            label="A/B karşılaştırma üret",
            value=False,
        )
        permission_checkbox = gr.Checkbox(
            label="Bu ses bana ait veya kullanma iznim var",
            value=False,
        )
        generate_button = gr.Button("Ses üret")

        audio_output = gr.Audio(
            label="Üretilen ses",
            type="filepath",
        )
        comparison_audio_output = gr.Audio(
            label="Karşılaştırma çıktısı",
            type="filepath",
        )
        status_output = gr.Textbox(
            label="Durum",
            interactive=False,
        )
        quality_report_output = gr.Markdown(
            value="Kalite raporu ses uretiminden sonra burada gorunecek."
        )

        refresh_profiles_button.click(
            fn=refresh_profile_choices,
            inputs=[profile_dropdown],
            outputs=[
                profile_dropdown,
                profile_refresh_status,
                selected_profile_info,
            ],
        )

        profile_dropdown.change(
            fn=build_selected_profile_info,
            inputs=[profile_dropdown],
            outputs=[selected_profile_info],
        )

        recreate_selected_profile_button.click(
            fn=recreate_selected_profile_from_gradio,
            inputs=[profile_dropdown],
            outputs=[
                profile_dropdown,
                profile_refresh_status,
                selected_profile_info,
                profile_management_status,
            ],
        )

        delete_selected_profile_button.click(
            fn=delete_selected_profile_from_gradio,
            inputs=[
                profile_dropdown,
                delete_profile_confirm_checkbox,
            ],
            outputs=[
                profile_dropdown,
                profile_refresh_status,
                selected_profile_info,
                profile_management_status,
                delete_profile_confirm_checkbox,
            ],
        )

        create_profile_button.click(
            fn=create_profile_from_gradio,
            inputs=[
                new_profile_name_input,
                new_profile_audio_input,
                new_profile_permission_checkbox,
                profile_dropdown,
            ],
            outputs=[
                profile_dropdown,
                profile_refresh_status,
                selected_profile_info,
                create_profile_status,
            ],
        )

        generate_button.click(
            fn=generate_voice,
            inputs=[
                text_input,
                reference_audio_input,
                profile_dropdown,
                inference_preset_dropdown,
                post_processing_checkbox,
                ab_test_checkbox,
                permission_checkbox,
            ],
            outputs=[
                audio_output,
                comparison_audio_output,
                status_output,
                quality_report_output,
            ],
        )

    return demo


if __name__ == "__main__":
    print(f"Proje kok dizini: {PROJECT_ROOT}")
    print(f"Varsayilan referans ses: {DEFAULT_REFERENCE_AUDIO}")
    print(f"Gradio cikti klasoru: {OUTPUT_DIR}")
    print(f"On islenmis referans klasoru: {PREPROCESSED_REFERENCE_DIR}")
    print(f"Gradio kalite raporu klasoru: {GRADIO_QUALITY_REPORT_DIR}")
    print(f"Model: {MODEL_NAME}")

    app = build_demo()
    app.launch(server_name="127.0.0.1", share=False)
