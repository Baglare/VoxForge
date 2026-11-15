# -*- coding: utf-8 -*-
"""WAV concatenation, with FFmpeg and a safe stdlib PCM-only fallback."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import wave


def ffmpeg_concat_path(path: Path) -> str:
    """Escape a path in FFmpeg's concat file list."""
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def _concatenate_pcm_wavs(
    input_paths: list[Path], output_path: Path, pauses: list[float],
) -> tuple[bool, str]:
    """Stream same-format PCM WAV parts and optional gaps without external packages."""
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    try:
        with wave.open(str(input_paths[0]), "rb") as first:
            params = first.getparams()
            expected = (params.nchannels, params.sampwidth, params.framerate, params.comptype)
            if params.comptype != "NONE":
                return False, "FFmpeg olmadan sadece PCM WAV birlestirilebilir."

        with wave.open(str(temporary_path), "wb") as combined:
            combined.setnchannels(params.nchannels)
            combined.setsampwidth(params.sampwidth)
            combined.setframerate(params.framerate)
            silence_frame = (b"\x80" if params.sampwidth == 1 else bytes(params.sampwidth)) * params.nchannels
            for index, path in enumerate(input_paths):
                with wave.open(str(path), "rb") as source:
                    current = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                    if current != expected:
                        return False, f"WAV parcalari uyumsuz: {path.name}"
                    while True:
                        frames = source.readframes(16_384)
                        if not frames:
                            break
                        combined.writeframesraw(frames)
                if index < len(pauses):
                    combined.writeframesraw(silence_frame * round(params.framerate * pauses[index]))
        os.replace(temporary_path, output_path)
        return True, "PCM WAV parcalari Python ile birlestirildi."
    except (OSError, EOFError, wave.Error) as exc:
        return False, f"WAV birlestirme basarisiz: {exc}"
    finally:
        temporary_path.unlink(missing_ok=True)


def concatenate_wavs(
    input_paths: list[Path], output_path: Path, *, pauses: list[float] | None = None,
) -> tuple[bool, str]:
    """Concatenate WAVs with optional silence in seconds at each boundary."""
    if not input_paths:
        return False, "Birlestirilecek WAV parcasi bulunamadi."
    if pauses is not None and (
        len(pauses) != len(input_paths) - 1 or any(not 0 <= pause <= 5 for pause in pauses)
    ):
        return False, "Her WAV siniri icin 0-5 saniye araliginda bir duraklama gerekir."
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        return False, "Eksik WAV parcasi var: " + ", ".join(str(path) for path in missing)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        if input_paths[0].resolve() != output_path.resolve():
            shutil.copy2(input_paths[0], output_path)
        return True, "Tek WAV parcasi dogrudan kopyalandi."

    if pauses and any(pauses):
        return _concatenate_pcm_wavs(input_paths, output_path, pauses)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return _concatenate_pcm_wavs(input_paths, output_path, [])

    concat_list_path = output_path.parent / f"{output_path.stem}_concat_list.txt"
    try:
        concat_lines = [f"file '{ffmpeg_concat_path(path)}'" for path in input_paths]
        concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        command = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "bilinmeyen ffmpeg hatasi"
            return False, f"ffmpeg concat basarisiz: {message}"
        return True, "WAV parcalari birlestirildi."
    finally:
        concat_list_path.unlink(missing_ok=True)
