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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_preprocessing_utils import PreprocessingError, preprocess_reference_audio
from scripts.audio_quality_utils import analyze_audio_file, format_value

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"

_tts_model: Optional[TTS] = None
_tts_device: Optional[str] = None
_model_lock = Lock()


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


def build_quality_report_text(
    raw_report: dict,
    preprocessed_report: dict | None,
    preprocessing_result: dict | None = None,
) -> str:
    """Ham ve on islenmis referans raporlarini tek metinde birlestirir."""
    sections = [format_quality_report_section("Ham referans ses", raw_report, is_raw=True)]

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
        "device": _tts_device,
        "error_message": error_message,
        "raw_reference_report": raw_report,
        "preprocessed_reference_report": preprocessed_report,
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


def generate_voice(text: str, uploaded_audio, has_permission: bool):
    """Gradio butonuna basilinca Turkce ses uretimi yapar."""
    if not has_permission:
        return None, "Uyari: Ses uretimi icin sesin size ait oldugunu veya kullanma izniniz oldugunu onaylamalisiniz.", ""

    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return None, "Uyari: Ses uretmek icin Turkce metin girmelisiniz.", ""

    reference_audio = resolve_reference_audio(uploaded_audio)
    raw_report = analyze_audio_file(reference_audio)
    if not raw_report.get("exists"):
        quality_report_text = format_quality_report_section("Ham referans ses", raw_report, is_raw=True)
        return None, f"HATA: Referans ses dosyasi bulunamadi: {reference_audio}", quality_report_text

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_audio = OUTPUT_DIR / f"xtts_tr_{timestamp}.wav"

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
            error_message=preprocess_error,
        )
        quality_report_text = build_quality_report_text(raw_report, None, None)
        return (
            None,
            "HATA: Referans ses on isleme basarisiz oldu.\n"
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
            preprocessing_result,
            "Ses uretimi sirasinda sorun olustu.",
        )
        return (
            None,
            "HATA: Ses uretimi sirasinda sorun olustu.\n"
            f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
            f"Olusturulan on islenmis referans ses yolu: {preprocessed_audio.resolve()}\n"
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
        preprocessing_result,
    )
    status_message = (
        "Ses uretildi.\n"
        f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
        f"Olusturulan on islenmis referans ses yolu: {preprocessed_audio.resolve()}\n"
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
            inputs=[text_input, reference_audio_input, permission_checkbox],
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
