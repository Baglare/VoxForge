# -*- coding: utf-8 -*-
"""VoxForge icin basit lokal Gradio XTTS-v2 demosu."""

from datetime import datetime
from pathlib import Path
import json
import sys
from threading import Lock
from typing import Optional

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
PROFILE_PREPROCESSED_REFERENCE_NAME = "preprocessed_reference.wav"
NO_PROFILE_VALUE = ""

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_preprocessing_utils import PreprocessingError, preprocess_reference_audio
from scripts.audio_quality_utils import analyze_audio_file, format_value

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
            f"Profil metadata dosyasi bozuk: {profile_json_path}"
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


def build_profile_label(profile_slug: str, profile_json_path: Path) -> str:
    """Dropdown etiketini mumkunse profile.json icindeki adla olusturur."""
    try:
        metadata = read_profile_json(profile_json_path)
    except ProfileSelectionError:
        return f"{profile_slug} (profile.json okunamadi)"

    profile_name = str(metadata.get("profile_name") or profile_slug).strip()
    if not profile_name:
        profile_name = profile_slug

    return f"{profile_name} ({profile_slug})"


def build_profile_dropdown_choices() -> list[tuple[str, str]]:
    """profiles/ altindaki kullanilabilir yerel profilleri dropdown'a hazirlar."""
    choices: list[tuple[str, str]] = [("Profil yok", NO_PROFILE_VALUE)]

    if not PROFILES_DIR.is_dir():
        return choices

    for profile_dir in sorted(PROFILES_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not profile_dir.is_dir():
            continue

        profile_json_path = profile_dir / PROFILE_JSON_NAME
        preprocessed_reference = profile_dir / PROFILE_PREPROCESSED_REFERENCE_NAME
        if not profile_json_path.is_file() or not preprocessed_reference.is_file():
            continue

        choices.append(
            (
                build_profile_label(profile_dir.name, profile_json_path),
                profile_dir.name,
            )
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
        raise ProfileSelectionError(f"Profil metadata dosyasi bulunamadi: {profile_json_path}")
    if not preprocessed_reference.is_file():
        raise ProfileSelectionError(
            f"Profil on islenmis referans sesi bulunamadi: {preprocessed_reference}"
        )

    metadata = read_profile_json(profile_json_path)
    profile_name = str(metadata.get("profile_name") or profile_slug).strip() or str(profile_slug)
    selected_variant = (
        metadata.get("selected_preprocessing_variant")
        or "profile_preprocessed_reference"
    )
    original_quality = metadata.get("original_quality")
    if not isinstance(original_quality, dict):
        original_quality = {"quality": "UNKNOWN", "warnings": ["profile.json icinde original_quality yok."]}

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
    error_message: str | None = None,
) -> Path:
    """Her Gradio uretim denemesi icin kalite raporunu JSON olarak kaydeder."""
    GRADIO_QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = GRADIO_QUALITY_REPORT_DIR / f"quality_report_{timestamp}.json"
    payload = {
        "created_at": timestamp,
        "status": status,
        "reference_audio": str(reference_audio),
        "preprocessed_audio": str(preprocessed_audio) if preprocessed_audio else None,
        "output_audio": str(output_audio) if output_audio else None,
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


def generate_voice(text: str, uploaded_audio, selected_profile_slug, has_permission: bool):
    """Gradio butonuna basilinca Turkce ses uretimi yapar."""
    if not has_permission:
        return None, "Uyari: Ses uretimi icin sesin size ait oldugunu veya kullanma izniniz oldugunu onaylamalisiniz.", ""

    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return None, "Uyari: Ses uretmek icin Turkce metin girmelisiniz.", ""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_audio = OUTPUT_DIR / f"xtts_tr_{timestamp}.wav"
    profile_info = None

    try:
        profile_info = load_selected_profile(selected_profile_slug)
    except ProfileSelectionError as exc:
        print(f"Profil secimi hatasi: {exc}")
        return None, f"HATA: {exc}", ""

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
        print(f"Cikti ses dosyasi: {output_audio}")

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
                error_message=preprocess_error,
            )
            quality_report_text = build_quality_report_text(raw_report, None, None)
            return (
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
        print(f"Cikti ses dosyasi: {output_audio}")
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

    try:
        tts = get_tts_model()
        device = _tts_device or "bilinmiyor"
        tts.tts_to_file(
            text=cleaned_text,
            speaker_wav=str(preprocessed_audio),
            language=LANGUAGE,
            file_path=str(output_audio),
        )
    except Exception as exc:
        device = _tts_device or "bilinmiyor"
        print(f"Ses uretimi hatasi: {exc}")
        quality_report_path = write_gradio_quality_report(
            timestamp,
            raw_report,
            preprocessed_report,
            reference_audio,
            preprocessed_audio,
            output_audio,
            "tts_failed",
            preprocessing_result=preprocessing_result,
            source_info=source_info,
            profile_info=profile_info,
            error_message="Ses uretimi sirasinda sorun olustu.",
        )
        return (
            None,
            "HATA: Ses uretimi sirasinda sorun olustu.\n"
            f"{source_info['source_label']}\n"
            f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
            f"Kullanilan speaker_wav yolu: {preprocessed_audio.resolve()}\n"
            f"Uretilmesi planlanan cikti ses yolu: {output_audio.resolve()}\n"
            f"Kullanilan cihaz: {device}\n"
            f"Kalite raporu JSON: {quality_report_path.resolve()}\n"
            f"{build_preprocessing_status_warning(preprocessing_result)}\n"
            "Ayrinti: Teknik hata terminale yazildi.",
            quality_report_text,
        )

    quality_report_path = write_gradio_quality_report(
        timestamp,
        raw_report,
        preprocessed_report,
        reference_audio,
        preprocessed_audio,
        output_audio,
        "success",
        preprocessing_result=preprocessing_result,
        source_info=source_info,
        profile_info=profile_info,
    )
    status_message = (
        "Ses uretildi.\n"
        f"{source_info['source_label']}\n"
        f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
        f"Kullanilan speaker_wav yolu: {preprocessed_audio.resolve()}\n"
        f"Uretilen cikti ses yolu: {output_audio.resolve()}\n"
        f"Kullanilan cihaz: {device}\n"
        f"Kalite raporu JSON: {quality_report_path.resolve()}"
        f"{build_preprocessing_status_warning(preprocessing_result)}"
        f"{build_quality_status_warning(raw_report, preprocessed_report)}"
    )
    return str(output_audio), status_message, quality_report_text


def build_demo() -> gr.Blocks:
    """Basit lokal Gradio arayuzunu olusturur."""
    with gr.Blocks(title="VoxForge XTTS Demo") as demo:
        gr.Markdown("# VoxForge XTTS Türkçe Demo")

        profile_dropdown = gr.Dropdown(
            label="Yerel ses profili",
            choices=build_profile_dropdown_choices(),
            value=NO_PROFILE_VALUE,
            interactive=True,
        )
        gr.Markdown(
            "Profil seçerseniz yüklenen ses dosyası yerine profilin ön işlenmiş referansı kullanılır."
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
        permission_checkbox = gr.Checkbox(
            label="Bu ses bana ait veya kullanma iznim var",
            value=False,
        )
        generate_button = gr.Button("Ses üret")

        audio_output = gr.Audio(
            label="Üretilen ses",
            type="filepath",
        )
        status_output = gr.Textbox(
            label="Durum",
            interactive=False,
        )
        quality_report_output = gr.Markdown(
            value="Kalite raporu ses uretiminden sonra burada gorunecek."
        )

        generate_button.click(
            fn=generate_voice,
            inputs=[
                text_input,
                reference_audio_input,
                profile_dropdown,
                permission_checkbox,
            ],
            outputs=[audio_output, status_output, quality_report_output],
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
