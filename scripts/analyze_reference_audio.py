# -*- coding: utf-8 -*-
"""XTTS referans sesi icin basit kalite raporu olusturur."""

from pathlib import Path
import json
import sys
from typing import Any

try:
    from audio_quality_utils import analyze_audio_file, format_value
except ModuleNotFoundError:
    from scripts.audio_quality_utils import analyze_audio_file, format_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_AUDIO = PROJECT_ROOT / "samples" / "my_voice.wav"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_JSON = REPORT_DIR / "reference_audio_report.json"


def resolve_input_path(argv: list[str]) -> Path:
    """Komut satiri yolu verilirse onu, yoksa varsayilan referansi kullanir."""
    if len(argv) < 2:
        return DEFAULT_INPUT_AUDIO

    input_path = Path(argv[1])
    if input_path.is_absolute():
        return input_path

    return PROJECT_ROOT / input_path


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
    print(f"Clipping riski: {format_value(report.get('clipping_risk'))}")
    print(f"Cok kisa mi: {format_value(report.get('too_short'))}")
    print(f"Cok uzun mu: {format_value(report.get('too_long'))}")
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
    report = analyze_audio_file(input_audio)
    write_json_report(report)
    print_report(report)
    print(f"\nJSON raporu: {REPORT_JSON}")

    return 0 if report["exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
