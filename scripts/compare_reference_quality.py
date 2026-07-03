# -*- coding: utf-8 -*-
"""Ham ve on islenmis referans seslerle XTTS A/B testi yapar."""

from pathlib import Path
import sys

import torch
from TTS.api import TTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_REFERENCE_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
PREPROCESSED_REFERENCE_AUDIO = PROJECT_ROOT / "outputs" / "preprocessed_reference.wav"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ab_tests"

RAW_OUTPUT_AUDIO = OUTPUT_DIR / "raw_reference_output.wav"
PREPROCESSED_OUTPUT_AUDIO = OUTPUT_DIR / "preprocessed_reference_output.wav"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"
TEST_TEXT = (
    "Bu kayıt, VoxForge projesinde ham ve temizlenmiş referans seslerin "
    "karşılaştırılması için oluşturulmuştur."
)


def validate_reference_files() -> bool:
    """Gerekli referans ses dosyalarinin var oldugunu kontrol eder."""
    required_files = [
        ("Ham referans ses", RAW_REFERENCE_AUDIO),
        ("On islenmis referans ses", PREPROCESSED_REFERENCE_AUDIO),
    ]

    missing_files = [
        (label, path) for label, path in required_files if not path.is_file()
    ]

    if not missing_files:
        return True

    print("HATA: A/B testi icin gerekli referans dosyalari eksik.", file=sys.stderr)
    for label, path in missing_files:
        print(f"- {label} bulunamadi: {path}", file=sys.stderr)
    return False


def generate_output(tts: TTS, reference_audio: Path, output_audio: Path) -> None:
    """Tek bir referans ses dosyasindan XTTS ciktisi uretir."""
    print(f"Referans: {reference_audio}")
    print(f"Uretilecek cikti: {output_audio}")

    tts.tts_to_file(
        text=TEST_TEXT,
        speaker_wav=str(reference_audio),
        language=LANGUAGE,
        file_path=str(output_audio),
    )

    print(f"Tamamlandi: {output_audio}")


def main() -> int:
    print("VoxForge XTTS referans kalite A/B testi baslatiliyor.")
    print(f"Model: {MODEL_NAME}")
    print(f"Test metni: {TEST_TEXT}")

    if not validate_reference_files():
        return 1

    # Cikti klasoru yoksa otomatik olusturulur.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CUDA varsa GPU denenir, yoksa CPU ile devam edilir.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanilacak cihaz: {device}")

    print("Model yukleniyor...")
    tts = TTS(model_name=MODEL_NAME)
    tts.to(device)
    print("Model yuklendi. A/B ciktisi uretimi basliyor.")

    generate_output(tts, RAW_REFERENCE_AUDIO, RAW_OUTPUT_AUDIO)
    generate_output(tts, PREPROCESSED_REFERENCE_AUDIO, PREPROCESSED_OUTPUT_AUDIO)

    print("A/B testi tamamlandi.")
    print(f"Ham referans ciktisi: {RAW_OUTPUT_AUDIO}")
    print(f"On islenmis referans ciktisi: {PREPROCESSED_OUTPUT_AUDIO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
