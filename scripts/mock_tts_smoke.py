# -*- coding: utf-8 -*-
"""Run an actual chunking / WAV concatenation smoke test without any TTS model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_concat_utils import concatenate_wavs  # noqa: E402
from scripts.text_chunking_utils import split_text_for_tts  # noqa: E402
from scripts.tts_engine_factory import create_tts_engine  # noqa: E402

DEFAULT_TEXT = (
    "Bu testte gerçek bir ses kaydı ya da konuşma modeli kullanılmaz. "
    "Türkçe metin önce cümlelere göre bölünür. "
    "Her bölüm için yapay, konuşma içermeyen bir WAV dosyası üretilir. "
    "Son olarak bütün dosyalar birleştirilir ve oluşan WAV kontrol edilir. "
    "Bu testin başarılı olması ses kalitesinin kanıtı değildir."
)


def run_mock_smoke(text: str, output_path: Path) -> tuple[int, int]:
    chunks = split_text_for_tts(text, max_chars=100)
    if not chunks:
        raise ValueError("Test text must not be empty")
    engine, device = create_tts_engine("mock")
    assert device == "mock"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vf001_mock_") as temporary:
        temporary_dir = Path(temporary)
        parts: list[Path] = []
        for index, chunk in enumerate(chunks):
            part = temporary_dir / f"part_{index:03d}.wav"
            engine.tts_to_file(text=chunk, speaker_wav=None, language="tr", file_path=str(part))
            parts.append(part)
        ok, message = concatenate_wavs(parts, output_path)
        if not ok:
            raise RuntimeError(message)
    with wave.open(str(output_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24_000:
            raise RuntimeError("Output WAV format does not match expected Mock format")
        frames = wav.getnframes()
        if frames <= 0:
            raise RuntimeError("Output WAV has no audio frames")
    return len(chunks), frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "mock_smoke" / "sample.wav")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()
    try:
        chunks, frames = run_mock_smoke(args.text, args.output)
    except (ValueError, RuntimeError, OSError, wave.Error) as exc:
        print(f"MOCK SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"MOCK SMOKE PASSED: chunks={chunks}, frames={frames}, output={args.output}")
    print("This is a synthetic file; it does not validate real voice quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
