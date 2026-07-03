# -*- coding: utf-8 -*-
"""VoxForge icin basit lokal Gradio XTTS-v2 demosu."""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from threading import Lock
from typing import Optional

import gradio as gr
import torch
from TTS.api import TTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradio_outputs"
PREPROCESSED_REFERENCE_DIR = PROJECT_ROOT / "outputs" / "preprocessed_references"

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
        return None, "FFmpeg bulunamadi. Lutfen FFmpeg'in PATH icinde oldugunu kontrol edin."

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
        return None, f"FFmpeg referans ses on isleme adimi basarisiz oldu: {error_detail}"

    if not preprocessed_audio.is_file():
        return None, "FFmpeg tamamlandi ancak on islenmis referans ses dosyasi olusmadi."

    return preprocessed_audio, None


def generate_voice(text: str, uploaded_audio, has_permission: bool):
    """Gradio butonuna basilinca Turkce ses uretimi yapar."""
    if not has_permission:
        return None, "Uyari: Ses uretimi icin sesin size ait oldugunu veya kullanma izniniz oldugunu onaylamalisiniz."

    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return None, "Uyari: Ses uretmek icin Turkce metin girmelisiniz."

    reference_audio = resolve_reference_audio(uploaded_audio)
    if not reference_audio.exists():
        return None, f"HATA: Referans ses dosyasi bulunamadi: {reference_audio}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_audio = OUTPUT_DIR / f"xtts_tr_{timestamp}.wav"

    print(f"Referans ses dosyasi: {reference_audio}")
    preprocessed_audio, preprocess_error = preprocess_reference_audio(reference_audio)
    if preprocess_error:
        return (
            None,
            "HATA: Referans ses on isleme basarisiz oldu.\n"
            f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
            f"Ayrinti: {preprocess_error}",
        )
    if preprocessed_audio is None:
        return None, "HATA: On islenmis referans ses yolu olusturulamadi."

    print(f"On islenmis referans ses dosyasi: {preprocessed_audio}")
    print(f"Cikti ses dosyasi: {output_audio}")

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
        return (
            None,
            "HATA: Ses uretimi sirasinda sorun olustu.\n"
            f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
            f"Olusturulan on islenmis referans ses yolu: {preprocessed_audio.resolve()}\n"
            f"Uretilmesi planlanan cikti ses yolu: {output_audio.resolve()}\n"
            f"Kullanilan cihaz: {device}\n"
            f"Ayrinti: {exc}",
        )

    status_message = (
        "Ses uretildi.\n"
        f"Kullanilan referans ses yolu: {reference_audio.resolve()}\n"
        f"Olusturulan on islenmis referans ses yolu: {preprocessed_audio.resolve()}\n"
        f"Uretilen cikti ses yolu: {output_audio.resolve()}\n"
        f"Kullanilan cihaz: {device}"
    )
    return str(output_audio), status_message


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

        generate_button.click(
            fn=generate_voice,
            inputs=[text_input, reference_audio_input, permission_checkbox],
            outputs=[audio_output, status_output],
        )

    return demo


if __name__ == "__main__":
    print(f"Proje kok dizini: {PROJECT_ROOT}")
    print(f"Varsayilan referans ses: {DEFAULT_REFERENCE_AUDIO}")
    print(f"Gradio cikti klasoru: {OUTPUT_DIR}")
    print(f"On islenmis referans klasoru: {PREPROCESSED_REFERENCE_DIR}")
    print(f"Model: {MODEL_NAME}")

    app = build_demo()
    app.launch(server_name="127.0.0.1", share=False)
