# -*- coding: utf-8 -*-
"""VoxForge yerel voice profile klasorlerini listeler."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "profiles"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "local_profiles_report.json"

ORIGINAL_REFERENCE_NAME = "original_reference.wav"
PREPROCESSED_REFERENCE_NAME = "preprocessed_reference.wav"
PROFILE_JSON_NAME = "profile.json"


def project_relative_path(path: Path) -> str:
    """Proje kokune gore okunabilir yol dondurur."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_profile_json(profile_json_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """profile.json dosyasini okur, bozuk JSON icin scripti dusurmez."""
    try:
        metadata = json.loads(profile_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"profile.json bozuk: {exc.msg} (satir {exc.lineno}, kolon {exc.colno})"
    except OSError as exc:
        return None, f"profile.json okunamadi: {exc}"

    if not isinstance(metadata, dict):
        return None, "profile.json beklenen JSON obje formatinda degil."

    return metadata, None


def quality_label(report: Any) -> str:
    """Kalite raporundan terminal icin kisa sonuc uretir."""
    if not isinstance(report, dict):
        return "UNKNOWN"

    quality = report.get("quality") or "UNKNOWN"
    duration = report.get("duration_seconds")
    sample_rate = report.get("sample_rate")

    details = []
    if duration is not None:
        details.append(f"sure={duration} sn")
    if sample_rate is not None:
        details.append(f"sample_rate={sample_rate} Hz")

    if not details:
        return str(quality)

    return f"{quality} ({', '.join(details)})"


def build_profile_record(profile_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Tek profil klasorunu gecerlilik ve temel durum bilgileriyle cozer."""
    profile_slug = profile_dir.name
    profile_json_path = profile_dir / PROFILE_JSON_NAME
    preprocessed_reference = profile_dir / PREPROCESSED_REFERENCE_NAME
    original_reference = profile_dir / ORIGINAL_REFERENCE_NAME

    issues = []
    if not profile_json_path.is_file():
        issues.append("profile.json bulunamadi.")
    if not preprocessed_reference.is_file():
        issues.append("preprocessed_reference.wav bulunamadi.")

    metadata: dict[str, Any] | None = None
    if profile_json_path.is_file():
        metadata, json_error = read_profile_json(profile_json_path)
        if json_error:
            issues.append(json_error)

    base_record = {
        "profile_slug": profile_slug,
        "profile_dir": str(profile_dir.resolve()),
        "profile_dir_relative": project_relative_path(profile_dir),
        "profile_json_exists": profile_json_path.is_file(),
        "preprocessed_reference_exists": preprocessed_reference.is_file(),
        "original_reference_exists": original_reference.is_file(),
    }

    if issues:
        broken_record = {
            **base_record,
            "issues": issues,
        }
        return None, broken_record

    if metadata is None:
        broken_record = {
            **base_record,
            "issues": ["profile.json okunamadi."],
        }
        return None, broken_record

    profile_record = {
        **base_record,
        "profile_name": metadata.get("profile_name") or profile_slug,
        "created_at": metadata.get("created_at"),
        "original_quality": metadata.get("original_quality"),
        "preprocessed_quality": metadata.get("preprocessed_quality"),
        "selected_preprocessing_variant": metadata.get("selected_preprocessing_variant"),
        "preprocessing_warning": metadata.get("preprocessing_warning"),
    }
    return profile_record, None


def collect_profiles() -> dict[str, Any]:
    """profiles/ klasorundeki gecerli ve bozuk profil kayitlarini toplar."""
    valid_profiles = []
    broken_profiles = []

    if PROFILES_DIR.is_dir():
        for profile_dir in sorted(PROFILES_DIR.iterdir(), key=lambda item: item.name.lower()):
            if not profile_dir.is_dir():
                continue

            valid_record, broken_record = build_profile_record(profile_dir)
            if valid_record is not None:
                valid_profiles.append(valid_record)
            elif broken_record is not None:
                broken_profiles.append(broken_record)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profiles_dir": str(PROFILES_DIR.resolve()),
        "valid_profile_count": len(valid_profiles),
        "broken_profile_count": len(broken_profiles),
        "profiles": valid_profiles,
        "broken_profiles": broken_profiles,
    }


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
    report = collect_profiles()

    print_report(report)

    if args.json:
        report_path = write_json_report(report)
        print("")
        print(f"JSON raporu yazildi: {report_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
