# -*- coding: utf-8 -*-
"""XTTS referans sesi icin basit kalite raporu olusturur."""

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_JSON = REPORT_DIR / "reference_audio_report.json"

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


def resolve_input_path(argv: list[str]) -> Path:
    """Komut satiri yolu verilirse onu, yoksa varsayilan referansi kullanir."""
    if len(argv) < 2:
        return DEFAULT_INPUT_AUDIO

    input_path = Path(argv[1])
    if input_path.is_absolute():
        return input_path

    return PROJECT_ROOT / input_path


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


def build_report(audio_path: Path) -> dict[str, Any]:
    """Ses dosyasi icin terminal ve JSON raporunda kullanilacak veriyi toplar."""
    exists = audio_path.is_file()
    warnings: list[str] = []
    recommendations = list(BASE_RECOMMENDATIONS)
    hard_errors: list[str] = []

    report: dict[str, Any] = {
        "input_path": str(audio_path),
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
        hard_errors.append(f"Dosya bulunamadi: {audio_path}")
        warnings.append("Analiz edilecek referans ses dosyasi yok.")
        warnings.extend(hard_errors)
        report["quality"] = classify_quality(exists, None, None, None, hard_errors, warnings)
        return report

    report["file_size_mb"] = round(audio_path.stat().st_size / (1024 * 1024), 3)

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        hard_errors.append("FFprobe bulunamadi. FFmpeg paketinin PATH icinde oldugunu kontrol et.")
    else:
        probe_data, probe_error = read_probe_data(audio_path, ffprobe_path)
        if probe_error:
            hard_errors.append(f"FFprobe analizi basarisiz oldu: {probe_error}")
        elif probe_data is not None:
            report.update(extract_audio_metadata(probe_data))

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        warnings.append("FFmpeg bulunamadi; ses seviyesi olcumu yapilamadi.")
    else:
        mean_volume, max_volume, volume_error = read_volume_data(audio_path, ffmpeg_path)
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
    """Terminalde None degerleri okunur hale getirir."""
    if value is None:
        return "bilinmiyor"
    if isinstance(value, bool):
        return "evet" if value else "hayir"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def print_report(report: dict[str, Any]) -> None:
    """Analiz sonucunu terminale teknik ama okunabilir sekilde yazar."""
    print("VoxForge referans ses kalite raporu")
    print("-----------------------------------")
    print(f"Giris dosyasi: {report['input_path']}")
    print(f"Dosya var mi: {'evet' if report['exists'] else 'hayir'}")
    print(f"Dosya boyutu: {format_value(report['file_size_mb'], ' MB')}")
    print(f"Sure: {format_value(report['duration_seconds'], ' saniye')}")
    print(f"Sample rate: {format_value(report['sample_rate'], ' Hz')}")
    print(f"Kanal sayisi: {format_value(report['channels'])}")
    print(f"Format: {format_value(report.get('format'))}")
    print(f"Codec: {format_value(report['codec'])}")
    print(f"Mean volume: {format_value(report['mean_volume_db'], ' dB')}")
    print(f"Max volume: {format_value(report['max_volume_db'], ' dB')}")
    print(f"Clipping riski: {format_value(report['clipping_risk'])}")
    print(f"Cok kisa mi: {format_value(report['too_short'])}")
    print(f"Cok uzun mu: {format_value(report['too_long'])}")
    print(f"Kalite sonucu: {report['quality']}")

    print("\nUyarilar:")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"- {warning}")
    else:
        print("- Belirgin teknik uyari yok.")

    print("\nKisa oneriler:")
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")


def write_json_report(report: dict[str, Any]) -> None:
    """Raporu outputs/reports klasorune JSON olarak kaydeder."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    input_audio = resolve_input_path(argv)
    report = build_report(input_audio)
    write_json_report(report)
    print_report(report)
    print(f"\nJSON raporu: {REPORT_JSON}")

    return 0 if report["exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
