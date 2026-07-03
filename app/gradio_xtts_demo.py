# -*- coding: utf-8 -*-
"""VoxForge icin basit lokal Gradio XTTS-v2 demosu."""

from datetime import datetime
from pathlib import Path
import json
import shutil
import subprocess
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

from scripts.audio_quality_utils import analyze_audio_file, format_value

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"

# Referans sesleri XTTS icin daha tutarli hale getiren FFmpeg filtresi.
AUDIO_FILTER = (
    "silenceremove="
    "start_periods=1:start_duration=0.20:start_threshold=-45dB:"
    "stop_periods=1:stop_duration=0.20:stop_threshold=-45dB,"
    "loudnorm=I=-20:TP=-2:LRA=11"
)

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


def build_preprocess_command(
    ffmpeg_path: str,
    reference_audio: Path,
    preprocessed_audio: Path,
) -> list[str]:
    """Referans sesi standart XTTS girisine ceviren FFmpeg komutunu hazirlar."""
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-i",
        str(reference_audio),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-af",
        AUDIO_FILTER,
        "-c:a",
        "pcm_s16le",
        str(preprocessed_audio),
    ]


def preprocess_reference_audio(reference_audio: Path) -> tuple[Path | None, str | None]:
    """Ham referans sesi on isleyip yeni bir WAV dosyasi olarak kaydeder."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return None, "FFmpeg bulunamadi. Lutfen FFmpeg kurulumunu ve PATH ayarini kontrol edin."

    PREPROCESSED_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    preprocessed_audio = PREPROCESSED_REFERENCE_DIR / f"reference_{timestamp}.wav"

    command = build_preprocess_command(ffmpeg_path, reference_audio, preprocessed_audio)
    print("Referans ses on isleme FFmpeg komutu:")
    print(subprocess.list2cmdline(command))

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        error_detail = result.stderr.strip() or "FFmpeg hata ayrintisi dondurmedi."
        print(f"FFmpeg on isleme hatasi: {error_detail}")
        return None, "FFmpeg referans ses on isleme adimi basarisiz oldu."

    if not preprocessed_audio.is_file():
        return None, "FFmpeg tamamlandi ancak on islenmis referans ses dosyasi olusmadi."

    return preprocessed_audio, None


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


def build_quality_report_text(raw_report: dict, preprocessed_report: dict | None) -> str:
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

    return "\n\n".join(sections)


def write_gradio_quality_report(
    timestamp: str,
    raw_report: dict,
    preprocessed_report: dict | None,
    reference_audio: Path,
    preprocessed_audio: Path | None,
    output_audio: Path | None,
    status: str,
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
    preprocessed_audio, preprocess_error = preprocess_reference_audio(reference_audio)
    if preprocess_error:
        quality_report_path = write_gradio_quality_report(
            timestamp,
            raw_report,
            None,
            reference_audio,
            None,
            None,
            "preprocess_failed",
            preprocess_error,
        )
        quality_report_text = build_quality_report_text(raw_report, None)
        return (
            None,
            "HATA: Referans ses on isleme basarisiz oldu.\n"
            f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
            f"Ayrinti: {preprocess_error}\n"
            f"Kalite raporu JSON: {quality_report_path.resolve()}",
            quality_report_text,
        )
    if preprocessed_audio is None:
        quality_report_path = write_gradio_quality_report(
            timestamp,
            raw_report,
            None,
            reference_audio,
            None,
            None,
            "preprocess_failed",
            "On islenmis referans ses yolu olusturulamadi.",
        )
        return (
            None,
            "HATA: On islenmis referans ses yolu olusturulamadi.\n"
            f"Kalite raporu JSON: {quality_report_path.resolve()}",
            build_quality_report_text(raw_report, None),
        )

    print(f"On islenmis referans ses dosyasi: {preprocessed_audio}")
    print(f"Cikti ses dosyasi: {output_audio}")
    preprocessed_report = analyze_audio_file(preprocessed_audio)
    quality_report_text = build_quality_report_text(raw_report, preprocessed_report)

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
    )
    status_message = (
        "Ses uretildi.\n"
        f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
        f"Olusturulan on islenmis referans ses yolu: {preprocessed_audio.resolve()}\n"
        f"Uretilen cikti ses yolu: {output_audio.resolve()}\n"
        f"Kullanilan cihaz: {device}\n"
        f"Kalite raporu JSON: {quality_report_path.resolve()}"
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
