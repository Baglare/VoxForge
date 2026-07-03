# -*- coding: utf-8 -*-
"""VoxForge icin basit lokal Gradio XTTS-v2 demosu."""

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import gradio as gr
import torch
from TTS.api import TTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradio_outputs"

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
    print(f"Cikti ses dosyasi: {output_audio}")

    try:
        tts = get_tts_model()
        tts.tts_to_file(
            text=cleaned_text,
            speaker_wav=str(reference_audio),
            language=LANGUAGE,
            file_path=str(output_audio),
        )
    except Exception as exc:
        return None, f"HATA: Ses uretimi sirasinda sorun olustu: {exc}"

    return str(output_audio), f"Ses uretildi: {output_audio.resolve()}"


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
    print(f"Model: {MODEL_NAME}")

    app = build_demo()
    app.launch(server_name="127.0.0.1", share=False)
