# -*- coding: utf-8 -*-
"""VoxForge fine-tuning dataset metadata ve ses dosyalarini dogrular."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = REPORTS_DIR / "finetune_dataset_report.json"
METADATA_FILE_NAME = "metadata.csv"
EXPECTED_HEADER = "audio_path|text"

# Fine-tuning klipleri referans ses profillerinden farklidir. Referans ses
# analizindeki 30-90 sn onerisi burada dogrudan karar olarak kullanilmaz.
MIN_ACCEPTABLE_SAMPLE_SECONDS = 1.5
SHORT_WARNING_SAMPLE_SECONDS = 2.5
IDEAL_MAX_SAMPLE_SECONDS = 15.0
LONG_WARNING_SAMPLE_SECONDS = 25.0
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_quality_utils import analyze_audio_file


class DatasetValidationError(Exception):
    """Dataset seviyesindeki kritik sorunlar icin sade hata sinifi."""


def resolve_dataset_path(dataset_arg: str) -> Path:
    """Dataset yolunu proje kokune gore cozer."""
    dataset_path = Path(dataset_arg)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return dataset_path.resolve()


def project_relative_path(path: Path) -> str:
    """Rapor icin proje kokune gore okunabilir yol dondurur."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_metadata_rows(metadata_path: Path) -> list[tuple[int, str, str]]:
    """metadata.csv dosyasini audio_path|text formatinda okur."""
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"metadata.csv okunamadi: {metadata_path}") from exc

    if not lines:
        raise DatasetValidationError("metadata.csv bos. Beklenen baslik: audio_path|text")

    header = lines[0].strip()
    if header != EXPECTED_HEADER:
        raise DatasetValidationError(
            "metadata.csv basligi hatali. Beklenen baslik: audio_path|text"
        )

    rows: list[tuple[int, str, str]] = []
    for line_number, raw_line in enumerate(lines[1:], start=2):
        if not raw_line.strip():
            continue

        parts = raw_line.split("|", 1)
        audio_path = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ""
        rows.append((line_number, audio_path, text))

    return rows


def resolve_audio_path(dataset_path: Path, audio_path_text: str) -> Path:
    """Metadata icindeki ses yolunu dataset klasorune gore cozer."""
    audio_path = Path(audio_path_text)
    if not audio_path.is_absolute():
        audio_path = dataset_path / audio_path
    return audio_path.resolve()


def check_duration(duration: float | None, errors: list[str], warnings: list[str]) -> None:
    """Fine-tuning klip suresini referans ses esiklerinden bagimsiz degerlendirir."""
    if duration is None:
        errors.append("Ses suresi okunamadi.")
        return

    if duration < MIN_ACCEPTABLE_SAMPLE_SECONDS:
        errors.append(
            f"Ses suresi cok kisa: {duration:.2f} sn "
            f"(< {MIN_ACCEPTABLE_SAMPLE_SECONDS:.2f} sn)."
        )
    elif duration < SHORT_WARNING_SAMPLE_SECONDS:
        warnings.append(
            f"Ses suresi kisa: {duration:.2f} sn "
            f"({MIN_ACCEPTABLE_SAMPLE_SECONDS:.2f}-{SHORT_WARNING_SAMPLE_SECONDS:.2f} sn arasi)."
        )
    elif duration > LONG_WARNING_SAMPLE_SECONDS:
        warnings.append(
            f"Ses suresi cok uzun olabilir: {duration:.2f} sn "
            f"(> {LONG_WARNING_SAMPLE_SECONDS:.2f} sn)."
        )
    elif duration > IDEAL_MAX_SAMPLE_SECONDS:
        warnings.append(
            f"Ses suresi ideal araligin uzerinde: {duration:.2f} sn "
            f"(ideal: {SHORT_WARNING_SAMPLE_SECONDS:.2f}-{IDEAL_MAX_SAMPLE_SECONDS:.2f} sn)."
        )


def check_audio_levels(
    mean_volume_db: float | None,
    max_volume_db: float | None,
    warnings: list[str],
) -> None:
    """Ses seviyesi ve clipping riskini dataset klibi icin yorumlar."""
    if mean_volume_db is None:
        warnings.append("Ortalama ses seviyesi okunamadi.")
    elif mean_volume_db < -35.0:
        warnings.append("Ortalama ses seviyesi dusuk; kaydi dinleyerek kontrol edin.")
    elif mean_volume_db > -12.0:
        warnings.append("Ortalama ses seviyesi yuksek; clipping riskini kontrol edin.")

    if max_volume_db is None:
        warnings.append("Maksimum ses seviyesi okunamadi; clipping riski degerlendirilemedi.")
    elif max_volume_db >= -1.0:
        warnings.append("Maksimum ses seviyesi 0 dB'ye cok yakin; clipping riski var.")


def sample_status(errors: list[str], warnings: list[str]) -> str:
    """Satir sonucunu raporda kullanilan buyuk harfli statuye cevirir."""
    if errors:
        return "ERROR"
    if warnings:
        return "WARNING"
    return "VALID"


def validate_row(
    dataset_path: Path,
    line_number: int,
    audio_path_text: str,
    text: str,
) -> dict[str, Any]:
    """Tek metadata satirini dogrular ve rapor kaydi dondurur."""
    errors: list[str] = []
    warnings: list[str] = []
    audio_path: Path | None = None
    quality_report: dict[str, Any] | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    mean_volume_db: float | None = None
    max_volume_db: float | None = None

    if not audio_path_text:
        errors.append("audio_path bos.")
    else:
        audio_path = resolve_audio_path(dataset_path, audio_path_text)
        if not audio_path.is_file():
            errors.append(f"Ses dosyasi bulunamadi: {audio_path}")
        if audio_path.suffix.lower() != ".wav":
            errors.append("Ses dosyasi .wav uzantili olmali.")

    if not text:
        errors.append("text bos.")

    if audio_path is not None and audio_path.is_file() and audio_path.suffix.lower() == ".wav":
        try:
            quality_report = analyze_audio_file(audio_path)
        except Exception as exc:
            errors.append(f"Kalite analizi yapilamadi: {type(exc).__name__}: {exc}")
        else:
            duration_seconds = quality_report.get("duration_seconds")
            sample_rate = quality_report.get("sample_rate")
            channels = quality_report.get("channels")
            mean_volume_db = quality_report.get("mean_volume_db")
            max_volume_db = quality_report.get("max_volume_db")

            check_duration(duration_seconds, errors, warnings)

            if sample_rate is None:
                errors.append("Sample rate okunamadi.")
            elif sample_rate != TARGET_SAMPLE_RATE:
                warnings.append(
                    f"Sample rate {TARGET_SAMPLE_RATE} Hz degil: {sample_rate} Hz."
                )

            if channels is None:
                errors.append("Kanal sayisi okunamadi.")
            elif channels != TARGET_CHANNELS:
                warnings.append(f"Ses mono degil; kanal sayisi: {channels}.")

            check_audio_levels(mean_volume_db, max_volume_db, warnings)

    return {
        "row_number": line_number,
        "audio_path": audio_path_text,
        "text": text,
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "channels": channels,
        "mean_volume_db": mean_volume_db,
        "max_volume_db": max_volume_db,
        "sample_status": sample_status(errors, warnings),
        "warnings": warnings,
        "errors": errors,
    }


def build_report(dataset_path: Path) -> dict[str, Any]:
    """Dataset klasorunu ve metadata satirlarini dogrular."""
    if not dataset_path.is_dir():
        raise DatasetValidationError(f"Dataset klasoru bulunamadi: {dataset_path}")

    metadata_path = dataset_path / METADATA_FILE_NAME
    if not metadata_path.is_file():
        raise DatasetValidationError(f"metadata.csv bulunamadi: {metadata_path}")

    metadata_rows = read_metadata_rows(metadata_path)
    rows = [
        validate_row(dataset_path, line_number, audio_path, text)
        for line_number, audio_path, text in metadata_rows
    ]

    valid_samples = sum(1 for row in rows if row["sample_status"] == "VALID")
    warning_samples = sum(1 for row in rows if row["sample_status"] == "WARNING")
    error_samples = sum(1 for row in rows if row["sample_status"] == "ERROR")
    durations = [
        float(row["duration_seconds"])
        for row in rows
        if row.get("duration_seconds") is not None
    ]
    total_duration = round(sum(durations), 3)
    average_duration = round(total_duration / len(durations), 3) if durations else 0.0
    estimated_minutes = round(total_duration / 60.0, 3)
    warnings = [
        f"satir {row['row_number']}: {warning}"
        for row in rows
        for warning in row["warnings"]
    ]
    if not rows:
        warnings.append("metadata.csv icinde dogrulanacak ornek yok.")

    errors = [
        f"satir {row['row_number']}: {error}"
        for row in rows
        for error in row["errors"]
    ]
    result = (
        "FAILED"
        if error_samples
        else "PASSED_WITH_WARNINGS"
        if warnings
        else "PASSED"
    )

    return {
        "dataset_path": str(dataset_path),
        "dataset_path_relative": project_relative_path(dataset_path),
        "total_rows": len(rows),
        "valid_samples": valid_samples,
        "warning_samples": warning_samples,
        "error_samples": error_samples,
        "total_duration_seconds": total_duration,
        "average_duration_seconds": average_duration,
        "estimated_minutes": estimated_minutes,
        "rows": rows,
        "warnings": warnings,
        "errors": errors,
        "summary": {
            "result": result,
            "total_rows": len(rows),
            "valid_samples": valid_samples,
            "warning_samples": warning_samples,
            "error_samples": error_samples,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": average_duration,
            "estimated_minutes": estimated_minutes,
            "metadata_path": str(metadata_path),
            "report_path": str(REPORT_PATH),
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "target_channels": TARGET_CHANNELS,
            "min_acceptable_sample_seconds": MIN_ACCEPTABLE_SAMPLE_SECONDS,
            "short_warning_sample_seconds": SHORT_WARNING_SAMPLE_SECONDS,
            "ideal_max_sample_seconds": IDEAL_MAX_SAMPLE_SECONDS,
            "long_warning_sample_seconds": LONG_WARNING_SAMPLE_SECONDS,
        },
    }


def write_report(report: dict[str, Any]) -> Path:
    """JSON raporunu outputs/reports altina yazar."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return REPORT_PATH


def print_report(report: dict[str, Any], report_path: Path) -> None:
    """Terminal icin kisa ve okunabilir ozet basar."""
    print("Fine-tuning dataset dogrulama raporu")
    print(f"Dataset: {report['dataset_path']}")
    print(f"Toplam satir: {report['total_rows']}")
    print(f"Gecerli ornek: {report['valid_samples']}")
    print(f"Uyarili ornek: {report['warning_samples']}")
    print(f"Hatali ornek: {report['error_samples']}")
    print(f"Toplam yaklasik sure: {report['total_duration_seconds']} sn")
    print(f"Ortalama ornek suresi: {report['average_duration_seconds']} sn")
    print(f"Tahmini toplam sure: {report['estimated_minutes']} dk")
    print(f"JSON raporu: {report_path}")

    if report["errors"]:
        print("")
        print("Hatalar:")
        for error in report["errors"]:
            print(f"- {error}")

    if report["warnings"]:
        print("")
        print("Uyarilar:")
        for warning in report["warnings"]:
            print(f"- {warning}")

    print("")
    print(f"Sonuc: {report['summary']['result']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge fine-tuning dataset metadata ve seslerini dogrular."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dogrulanacak dataset klasoru. Ornek: datasets/baglare-finetune-v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_path = resolve_dataset_path(args.dataset)

    try:
        report = build_report(dataset_path)
    except DatasetValidationError as exc:
        report = {
            "dataset_path": str(dataset_path),
            "total_rows": 0,
            "valid_samples": 0,
            "warning_samples": 0,
            "error_samples": 1,
            "total_duration_seconds": 0.0,
            "average_duration_seconds": 0.0,
            "estimated_minutes": 0.0,
            "rows": [],
            "warnings": [],
            "errors": [str(exc)],
            "summary": {
                "result": "FAILED",
                "total_rows": 0,
                "valid_samples": 0,
                "warning_samples": 0,
                "error_samples": 1,
                "total_duration_seconds": 0.0,
                "average_duration_seconds": 0.0,
                "estimated_minutes": 0.0,
                "report_path": str(REPORT_PATH),
            },
        }
        report_path = write_report(report)
        print(f"HATA: {exc}", file=sys.stderr)
        print(f"JSON raporu: {report_path}", file=sys.stderr)
        return 1

    report_path = write_report(report)
    print_report(report, report_path)
    return 1 if report["error_samples"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
