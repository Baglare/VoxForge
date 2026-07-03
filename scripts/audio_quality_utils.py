# -*- coding: utf-8 -*-
"""Referans ses kalite analizi icin ortak yardimci fonksiyonlar."""

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
from typing import Any


MIN_RECOMMENDED_SECONDS = 30.0
MAX_RECOMMENDED_SECONDS = 90.0
BAD_SHORT_SECONDS = 10.0
BAD_LONG_SECONDS = 180.0
TARGET_SAMPLE_RATE = 24000

BASE_RECOMMENDATIONS = [
    "Daha sessiz ortamda kayit al.",
    "Mikrofona cok yakin konusma.",
    "30-90 saniye arasi dogal konus.",
    "Arka plan muzigi, klavye sesi veya ortam gurultusu olmasin.",
    "Tek kisi konussun.",
]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Dis komutu standart ciktilari yakalayarak calistirir."""
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_float(value: Any) -> float | None:
    """FFprobe string alanlarini guvenli sekilde float degere cevirir."""
    if value in (None, "N/A"):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    """FFprobe string alanlarini guvenli sekilde int degere cevirir."""
    if value in (None, "N/A"):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_probe_data(audio_path: Path, ffprobe_path: str) -> tuple[dict[str, Any] | None, str | None]:
    """FFprobe ile dosya teknik bilgilerini JSON olarak okur."""
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(audio_path),
    ]
    result = run_command(command)

    if result.returncode != 0:
        detail = result.stderr.strip() or "FFprobe hata ayrintisi dondurmedi."
        return None, detail

    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"FFprobe JSON ciktisi okunamadi: {exc}"


def extract_audio_metadata(probe_data: dict[str, Any]) -> dict[str, Any]:
    """FFprobe ciktisinden ilk ses akisi bilgilerini ayiklar."""
    streams = probe_data.get("streams", [])
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        {},
    )
    format_data = probe_data.get("format", {})

    duration = parse_float(audio_stream.get("duration"))
    if duration is None:
        duration = parse_float(format_data.get("duration"))

    return {
        "duration_seconds": duration,
        "sample_rate": parse_int(audio_stream.get("sample_rate")),
        "channels": parse_int(audio_stream.get("channels")),
        "codec": audio_stream.get("codec_name"),
        "format": format_data.get("format_name"),
    }


def read_volume_data(audio_path: Path, ffmpeg_path: str) -> tuple[float | None, float | None, str | None]:
    """FFmpeg volumedetect ile ortalama ve maksimum ses seviyesini okur."""
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        "volumedetect",
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "null",
        os.devnull,
    ]
    result = run_command(command)

    if result.returncode != 0:
        detail = result.stderr.strip() or "FFmpeg hata ayrintisi dondurmedi."
        return None, None, detail

    output = f"{result.stdout}\n{result.stderr}"
    mean_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", output)
    max_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", output)

    mean_volume = float(mean_match.group(1)) if mean_match else None
    max_volume = float(max_match.group(1)) if max_match else None
    return mean_volume, max_volume, None


def classify_quality(
    exists: bool,
    duration_seconds: float | None,
    mean_volume_db: float | None,
    max_volume_db: float | None,
    hard_errors: list[str],
    warnings: list[str],
) -> str:
    """Uyari listesini sade GOOD / WARNING / BAD sonucuna cevirir."""
    if not exists or hard_errors:
        return "BAD"

    if duration_seconds is not None:
        if duration_seconds < BAD_SHORT_SECONDS or duration_seconds > BAD_LONG_SECONDS:
            return "BAD"

    if mean_volume_db is not None and mean_volume_db < -45.0:
        return "BAD"

    if max_volume_db is not None and max_volume_db >= -0.1:
        return "BAD"

    if warnings:
        return "WARNING"

    return "GOOD"


def analyze_audio_file(audio_path: str | Path) -> dict[str, Any]:
    """Ses dosyasini analiz eder ve Gradio/CLI tarafinda kullanilacak dict'i dondurur."""
    resolved_path = Path(audio_path)
    exists = resolved_path.is_file()
    warnings: list[str] = []
    recommendations = list(BASE_RECOMMENDATIONS)
    hard_errors: list[str] = []

    report: dict[str, Any] = {
        "input_path": str(resolved_path),
        "exists": exists,
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
        "codec": None,
        "format": None,
        "file_size_mb": None,
        "mean_volume_db": None,
        "max_volume_db": None,
        "clipping_risk": None,
        "too_short": None,
        "too_long": None,
        "quality": "BAD",
        "warnings": warnings,
        "recommendations": recommendations,
    }

    if not exists:
        hard_errors.append(f"Dosya bulunamadi: {resolved_path}")
        warnings.append("Analiz edilecek referans ses dosyasi yok.")
        warnings.extend(hard_errors)
        report["quality"] = classify_quality(exists, None, None, None, hard_errors, warnings)
        return report

    report["file_size_mb"] = round(resolved_path.stat().st_size / (1024 * 1024), 3)

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        hard_errors.append("FFprobe bulunamadi. FFmpeg paketinin PATH icinde oldugunu kontrol et.")
    else:
        probe_data, probe_error = read_probe_data(resolved_path, ffprobe_path)
        if probe_error:
            hard_errors.append(f"FFprobe analizi basarisiz oldu: {probe_error}")
        elif probe_data is not None:
            report.update(extract_audio_metadata(probe_data))

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        warnings.append("FFmpeg bulunamadi; ses seviyesi olcumu yapilamadi.")
    else:
        mean_volume, max_volume, volume_error = read_volume_data(resolved_path, ffmpeg_path)
        if volume_error:
            warnings.append(f"Ses seviyesi olcumu yapilamadi: {volume_error}")
        else:
            report["mean_volume_db"] = mean_volume
            report["max_volume_db"] = max_volume

    duration = report["duration_seconds"]
    if duration is not None:
        report["too_short"] = duration < MIN_RECOMMENDED_SECONDS
        report["too_long"] = duration > MAX_RECOMMENDED_SECONDS

        if duration < MIN_RECOMMENDED_SECONDS:
            warnings.append("Kayit 30 saniyeden kisa; XTTS icin daha uzun dogal konusma onerilir.")
        if duration > MAX_RECOMMENDED_SECONDS:
            warnings.append("Kayit 90 saniyeden uzun; daha kisa ve temiz bir referans tercih edilebilir.")
    else:
        warnings.append("Ses suresi okunamadi.")

    sample_rate = report["sample_rate"]
    if sample_rate is not None and sample_rate != TARGET_SAMPLE_RATE:
        warnings.append("Sample rate 24000 Hz degil; XTTS oncesi 24000 Hz'e donusturme onerilir.")

    channels = report["channels"]
    if channels is not None and channels > 1:
        warnings.append("Ses stereo veya cok kanalli; XTTS icin mono referans onerilir.")

    mean_volume_db = report["mean_volume_db"]
    if mean_volume_db is not None:
        if mean_volume_db < -35.0:
            warnings.append("Ortalama ses seviyesi dusuk; daha net ve dengeli kayit onerilir.")
        elif mean_volume_db > -12.0:
            warnings.append("Ortalama ses seviyesi yuksek; mikrofona biraz daha uzak konusmak daha iyi olabilir.")

    max_volume_db = report["max_volume_db"]
    if max_volume_db is not None:
        clipping_risk = max_volume_db >= -1.0
        report["clipping_risk"] = clipping_risk
        if clipping_risk:
            warnings.append("Maksimum ses seviyesi 0 dB'ye cok yakin; clipping riski var.")
    else:
        warnings.append("Maksimum ses seviyesi okunamadi; clipping riski degerlendirilemedi.")

    report["quality"] = classify_quality(
        exists,
        duration,
        mean_volume_db,
        max_volume_db,
        hard_errors,
        warnings,
    )

    if hard_errors:
        report["warnings"].extend(hard_errors)

    return report


def format_value(value: Any, suffix: str = "") -> str:
    """Terminal ve arayuz icin None/bool/float degerleri okunur hale getirir."""
    if value is None:
        return "bilinmiyor"
    if isinstance(value, bool):
        return "evet" if value else "hayir"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"
