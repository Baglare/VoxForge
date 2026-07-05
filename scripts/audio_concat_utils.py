# -*- coding: utf-8 -*-
"""WAV parcalarini FFmpeg ile guvenli sekilde birlestirme yardimcilari."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_concat_path(path: Path) -> str:
    """FFmpeg concat listesi icin Windows yollarini sade ve guvenli yazar."""
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def concatenate_wavs(input_paths: list[Path], output_path: Path) -> tuple[bool, str]:
    """WAV dosyalarini tek output dosyasinda birlestirir."""
    existing_inputs = [path for path in input_paths if path.is_file()]
    if not existing_inputs:
        return False, "Birlestirilecek WAV parcasi bulunamadi."

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(existing_inputs) == 1:
        shutil.copy2(existing_inputs[0], output_path)
        return True, "Tek WAV parcasi dogrudan kopyalandi."

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg bulunamadi; WAV parcalari birlestirilemedi."

    concat_list_path = output_path.parent / f"{output_path.stem}_concat_list.txt"
    try:
        concat_lines = [
            f"file '{ffmpeg_concat_path(path)}'"
            for path in existing_inputs
        ]
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
        try:
            concat_list_path.unlink()
        except FileNotFoundError:
            pass
