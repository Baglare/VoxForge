# -*- coding: utf-8 -*-
"""VoxForge icin ilk minimum XTTS-v2 ses uretim denemesi."""

from pathlib import Path
import sys

import torch
from TTS.api import TTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
OUTPUT_AUDIO = PROJECT_ROOT / "outputs" / "first_xtts_test.wav"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"
TEST_TEXT = "Merhaba, bu VoxForge'un ilk Türkçe ses üretim denemesidir."


def main() -> int:
    print("VoxForge XTTS-v2 ilk deneme scripti baslatiliyor.")
    print(f"Model: {MODEL_NAME}")
    print(f"Referans ses dosyasi: {REFERENCE_AUDIO}")
    print(f"Cikti ses dosyasi: {OUTPUT_AUDIO}")

    if not REFERENCE_AUDIO.exists():
        print(
            "HATA: Referans ses dosyasi bulunamadi. "
            f"Lutfen dosyanin burada oldugunu kontrol edin: {REFERENCE_AUDIO}",
            file=sys.stderr,
        )
        return 1

    # Cikti klasoru yoksa otomatik olusturulur.
    OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)

    # CUDA varsa GPU denenir, yoksa CPU ile devam edilir.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanilacak cihaz: {device}")

    tts = TTS(model_name=MODEL_NAME)
    tts.to(device)

    print("Ses uretimi baslatiliyor...")
    tts.tts_to_file(
        text=TEST_TEXT,
        speaker_wav=str(REFERENCE_AUDIO),
        language=LANGUAGE,
        file_path=str(OUTPUT_AUDIO),
    )

    print("Ses uretimi tamamlandi.")
    print(f"Cikti dosyasi: {OUTPUT_AUDIO.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
