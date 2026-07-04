# -*- coding: utf-8 -*-
"""VoxForge yerel voice profile klasorlerini listeler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "local_profiles_report.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.voice_profile_utils import list_voice_profiles, quality_label


def print_profile(record: dict[str, Any]) -> None:
    """Tek bir gecerli profili terminalde okunabilir bicimde yazar."""
    print(f"- Profil adi: {record.get('profile_name')}")
    print(f"  Profil slug: {record.get('profile_slug')}")
    print(f"  Profil klasoru: {record.get('profile_dir')}")
    print(f"  Olusturulma tarihi: {record.get('created_at') or 'bilinmiyor'}")
    print(f"  original_quality: {quality_label(record.get('original_quality'))}")
    print(f"  preprocessed_quality: {quality_label(record.get('preprocessed_quality'))}")
    print(
        "  selected_preprocessing_variant: "
        f"{record.get('selected_preprocessing_variant') or 'bilinmiyor'}"
    )
    print(f"  preprocessing_warning: {record.get('preprocessing_warning') or 'yok'}")
    print(
        "  preprocessed_reference.wav var mi: "
        f"{'evet' if record.get('preprocessed_reference_exists') else 'hayir'}"
    )
    print(
        "  original_reference.wav var mi: "
        f"{'evet' if record.get('original_reference_exists') else 'hayir'}"
    )


def print_broken_profile(record: dict[str, Any]) -> None:
    """Eksik veya bozuk profil kaydini sade uyari olarak yazar."""
    print(f"- Profil slug: {record.get('profile_slug')}")
    print(f"  Profil klasoru: {record.get('profile_dir')}")
    print(
        "  preprocessed_reference.wav var mi: "
        f"{'evet' if record.get('preprocessed_reference_exists') else 'hayir'}"
    )
    print(
        "  original_reference.wav var mi: "
        f"{'evet' if record.get('original_reference_exists') else 'hayir'}"
    )
    print("  Sorunlar:")
    for issue in record.get("issues", []):
        print(f"  - {issue}")


def print_report(report: dict[str, Any]) -> None:
    """Toplanan profil bilgisini terminale yazar."""
    profiles = report["profiles"]
    broken_profiles = report["broken_profiles"]

    if not profiles and not broken_profiles:
        print("Henüz yerel voice profile oluşturulmamış.")
        return

    if profiles:
        print(f"Geçerli yerel voice profile sayisi: {len(profiles)}")
        for profile in profiles:
            print_profile(profile)
    else:
        print("Geçerli yerel voice profile bulunamadi.")

    if broken_profiles:
        print("")
        print(f"Eksik veya bozuk profil sayisi: {len(broken_profiles)}")
        for profile in broken_profiles:
            print_broken_profile(profile)


def write_json_report(report: dict[str, Any]) -> Path:
    """JSON raporunu outputs/reports altina yazar."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return JSON_REPORT_PATH


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge yerel voice profile klasorlerini listeler."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Raporu outputs/reports/local_profiles_report.json dosyasina yazar.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = list_voice_profiles()

    print_report(report)

    if args.json:
        report_path = write_json_report(report)
        print("")
        print(f"JSON raporu yazildi: {report_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
