# -*- coding: utf-8 -*-
"""Dependency-free, deterministic synthetic WAV generator for VoxForge tests.

This is not a voice model. It never uses the provided speaker reference and
cannot evaluate speech intelligibility, pronunciation or voice similarity.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import wave
from typing import Any


SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
CHANNELS = 1


class MockTTSEngine:
    """Small drop-in replacement for the Coqui ``tts_to_file`` interface.

    ``fail_on_substring`` supports deterministic fault-injection tests. It is
    never populated by the Gradio application in normal Mock mode.
    """

    def __init__(self, *, fail_on_substring: str | None = None) -> None:
        self.fail_on_substring = fail_on_substring
        self.calls: list[dict[str, Any]] = []

    def tts_to_file(
        self,
        *,
        text: str,
        speaker_wav: str | None = None,
        language: str = "tr",
        file_path: str,
        **kwargs: Any,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("Mock TTS cannot synthesize empty text.")
        if language != "tr":
            raise ValueError("Mock TTS only accepts 'tr' in this VoxForge fixture.")
        if self.fail_on_substring and self.fail_on_substring in text:
            raise RuntimeError("Injected Mock TTS failure")

        # No recordings are opened, copied or analyzed. A reference filename is
        # accepted only to preserve compatibility with the existing API.
        self.calls.append(
            {
                "text": text,
                "speaker_wav": str(speaker_wav) if speaker_wav is not None else None,
                "language": language,
                "file_path": str(file_path),
                "kwargs": dict(kwargs),
            }
        )

        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = min(3.0, max(0.24, len(text.strip()) * 0.022))
        nframes = round(duration * SAMPLE_RATE)
        # The generated tone is a test marker, emphatically not human speech.
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        frequency = 200 + (int.from_bytes(seed[:2], "big") % 250)
        amplitude = 5000
        pcm = bytearray(nframes * SAMPLE_WIDTH)
        for i in range(nframes):
            phase = 2 * math.pi * frequency * i / SAMPLE_RATE
            fade = min(1.0, i / 240.0, (nframes - i - 1) / 240.0)
            sample = round(amplitude * fade * math.sin(phase))
            struct.pack_into("<h", pcm, i * SAMPLE_WIDTH, sample)

        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        return str(output_path)
