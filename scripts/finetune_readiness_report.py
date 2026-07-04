# -*- coding: utf-8 -*-
"""Fine-tuning dataset icin teknik hazirlik raporu uretir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "finetune_readiness_report.json"
MARKDOWN_REPORT_PATH = REPORTS_DIR / "finetune_readiness_report.md"

NOT_READY = "NOT_READY"
STRUCTURE_READY = "STRUCTURE_READY"
DATASET_VALID_BUT_SMALL = "DATASET_VALID_BUT_SMALL"
READY_FOR_EXPERIMENTAL_FINETUNING = "READY_FOR_EXPERIMENTAL_FINETUNING"
READY_FOR_STRONGER_FINETUNING = "READY_FOR_STRONGER_FINETUNING"

MIN_STRUCTURE_READY_MINUTES = 1.0
MIN_EXPERIMENTAL_MINUTES = 10.0
MIN_STRONGER_MINUTES = 30.0
MIN_STRUCTURE_READY_VALID_SAMPLES = 10

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_finetune_dataset import (  # noqa: E402
    DatasetValidationError,
    build_report as build_validation_report,
    resolve_dataset_path,
)


def duration_values(rows: list[dict[str, Any]]) -> list[float]:
    """Dogrulama satirlarindan okunabilen sureleri toplar."""
    durations: list[float] = []
    for row in rows:
        duration = row.get("duration_seconds")
        if duration is None:
            continue
        durations.append(float(duration))
    return durations


def base_readiness_level(
    total_rows: int,
    valid_samples: int,
    total_duration_minutes: float,
) -> str:
    """Sure ve gecerli ornek sayisina gore ilk hazirlik seviyesini secer."""
    if total_rows == 0 or valid_samples == 0:
        return NOT_READY

    if (
        valid_samples < MIN_STRUCTURE_READY_VALID_SAMPLES
        or total_duration_minutes < MIN_STRUCTURE_READY_MINUTES
    ):
        return STRUCTURE_READY

    if total_duration_minutes < MIN_EXPERIMENTAL_MINUTES:
        return DATASET_VALID_BUT_SMALL

    if total_duration_minutes < MIN_STRONGER_MINUTES:
        return READY_FOR_EXPERIMENTAL_FINETUNING

    return READY_FOR_STRONGER_FINETUNING


def downgrade_for_errors(readiness_level: str, error_samples: int) -> str:
    """Hatali ornek varsa seviyeyi iyimser gostermemek icin dusurur."""
    if error_samples == 0:
        return readiness_level

    if readiness_level in {
        READY_FOR_EXPERIMENTAL_FINETUNING,
        READY_FOR_STRONGER_FINETUNING,
    }:
        return DATASET_VALID_BUT_SMALL

    if readiness_level == DATASET_VALID_BUT_SMALL:
        return STRUCTURE_READY

    return readiness_level


def make_summary(
    readiness_level: str,
    total_rows: int,
    valid_samples: int,
    warning_samples: int,
    error_samples: int,
    total_duration_minutes: float,
) -> str:
    """Terminal, JSON ve Markdown icin kisa insan okunur ozet dondurur."""
    if readiness_level == NOT_READY:
        return "Dataset fine-tuning hazırlığı için hazır değil; metadata veya geçerli örnek eksik."

    if error_samples:
        return (
            "Dataset yapısı okunabiliyor, ancak hatalı örnekler düzeltilmeden "
            "fine-tuning denemesi önerilmez."
        )

    if readiness_level == STRUCTURE_READY:
        return (
            "Dataset yapısı hazır, ancak geçerli kayıt sayısı veya toplam süre "
            "teknik değerlendirme için hâlâ çok düşük."
        )

    if readiness_level == DATASET_VALID_BUT_SMALL:
        return (
            "Dataset teknik olarak geçerli, ancak gerçek fine-tuning kalitesi "
            "için daha fazla veri önerilir."
        )

    if readiness_level == READY_FOR_EXPERIMENTAL_FINETUNING:
        return (
            "Dataset deneysel fine-tuning denemesi için teknik olarak anlamlı "
            "bir seviyeye yaklaşmış görünüyor."
        )

    return (
        "Dataset daha güçlü fine-tuning denemeleri için teknik olarak daha "
        "olgun bir süreye sahip görünüyor."
    )


def make_recommendations(
    readiness_level: str,
    warning_samples: int,
    error_samples: int,
    total_duration_minutes: float,
) -> list[str]:
    """Hazirlik seviyesine gore uygulanabilir oneriler uretir."""
    recommendations: list[str] = []

    if readiness_level == NOT_READY:
        recommendations.append("metadata.csv dosyasını ve wavs/ klasöründeki kayıtları tamamla.")
        recommendations.append("Önce dataset doğrulama scriptini hatasız çalıştır.")

    if error_samples:
        recommendations.append("Hatalı örnekleri düzeltmeden fine-tuning denemesi başlatma.")

    if warning_samples:
        recommendations.append("Uyarılı örnekleri dinleyerek sample rate, mono kanal ve clipping riskini kontrol et.")

    if total_duration_minutes < MIN_EXPERIMENTAL_MINUTES:
        recommendations.append("Daha gerçekçi fine-tuning denemesi için toplam süreyi en az 10 dakikaya yaklaştır.")

    if total_duration_minutes < MIN_STRONGER_MINUTES:
        recommendations.append("Daha güçlü sonuçlar için uzun vadede 30 dakika ve üzeri temiz veri hedefle.")

    recommendations.append("Gerçek ses kayıtlarını ve rapor çıktılarını GitHub'a ekleme.")
    return recommendations


def make_next_steps(readiness_level: str, error_samples: int) -> list[str]:
    """Rapor sonuna konacak kisa sonraki adim listesini dondurur."""
    if readiness_level == NOT_READY:
        return [
            "Dataset iskeletini oluştur.",
            "recording_plan.csv üzerinden kayıtları tamamla.",
            "DONE satırlardan metadata.csv üret.",
            "Dataset doğrulamasını tekrar çalıştır.",
        ]

    if error_samples:
        return [
            "Hatalı metadata satırlarını veya eksik WAV dosyalarını düzelt.",
            "Dataset doğrulamasını tekrar çalıştır.",
            "Readiness raporunu yeniden üret.",
        ]

    return [
        "Uyarı varsa ilgili kayıtları dinleyerek kontrol et.",
        "Toplam süreyi artırmak için yeni temiz kayıtlar ekle.",
        "Her veri eklemesinden sonra validate ve readiness raporlarını yeniden çalıştır.",
    ]


def build_not_ready_report(dataset_path: Path, message: str) -> dict[str, Any]:
    """Metadata veya dataset okunamadiginda yine de rapor dosyasi uretir."""
    summary = f"Dataset fine-tuning hazırlığı için hazır değil: {message}"
    return {
        "dataset_path": str(dataset_path),
        "total_rows": 0,
        "valid_samples": 0,
        "warning_samples": 0,
        "error_samples": 1,
        "total_duration_seconds": 0.0,
        "total_duration_minutes": 0.0,
        "average_duration_seconds": 0.0,
        "min_duration_seconds": 0.0,
        "max_duration_seconds": 0.0,
        "readiness_level": NOT_READY,
        "summary": summary,
        "recommendations": make_recommendations(NOT_READY, 0, 1, 0.0),
        "next_steps": make_next_steps(NOT_READY, 1),
    }


def build_readiness_report(dataset_path: Path) -> dict[str, Any]:
    """Validate raporunu kullanarak fine-tuning hazirlik ozetini olusturur."""
    try:
        validation_report = build_validation_report(dataset_path)
    except DatasetValidationError as exc:
        return build_not_ready_report(dataset_path, str(exc))

    rows = validation_report.get("rows", [])
    durations = duration_values(rows)
    total_rows = int(validation_report.get("total_rows", 0))
    valid_samples = int(validation_report.get("valid_samples", 0))
    warning_samples = int(validation_report.get("warning_samples", 0))
    error_samples = int(validation_report.get("error_samples", 0))
    total_duration_seconds = round(float(validation_report.get("total_duration_seconds", 0.0)), 3)
    total_duration_minutes = round(total_duration_seconds / 60.0, 3)
    average_duration_seconds = round(
        float(validation_report.get("average_duration_seconds", 0.0)),
        3,
    )
    min_duration_seconds = round(min(durations), 3) if durations else 0.0
    max_duration_seconds = round(max(durations), 3) if durations else 0.0

    readiness_level = base_readiness_level(
        total_rows,
        valid_samples,
        total_duration_minutes,
    )
    readiness_level = downgrade_for_errors(readiness_level, error_samples)
    summary = make_summary(
        readiness_level,
        total_rows,
        valid_samples,
        warning_samples,
        error_samples,
        total_duration_minutes,
    )

    return {
        "dataset_path": str(dataset_path),
        "total_rows": total_rows,
        "valid_samples": valid_samples,
        "warning_samples": warning_samples,
        "error_samples": error_samples,
        "total_duration_seconds": total_duration_seconds,
        "total_duration_minutes": total_duration_minutes,
        "average_duration_seconds": average_duration_seconds,
        "min_duration_seconds": min_duration_seconds,
        "max_duration_seconds": max_duration_seconds,
        "readiness_level": readiness_level,
        "summary": summary,
        "recommendations": make_recommendations(
            readiness_level,
            warning_samples,
            error_samples,
            total_duration_minutes,
        ),
        "next_steps": make_next_steps(readiness_level, error_samples),
    }


def write_json_report(report: dict[str, Any]) -> Path:
    """Readiness raporunu JSON olarak yazar."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return JSON_REPORT_PATH


def markdown_list(items: list[str]) -> str:
    """Markdown liste metni uretir."""
    if not items:
        return "- Yok"
    return "\n".join(f"- {item}" for item in items)


def write_markdown_report(report: dict[str, Any]) -> Path:
    """Readiness raporunu kisa Markdown olarak yazar."""
    markdown = f"""# Fine-Tuning Readiness Report

## Özet

- Dataset: `{report['dataset_path']}`
- Hazırlık seviyesi: `{report['readiness_level']}`
- Toplam metadata satırı: {report['total_rows']}
- Geçerli örnek: {report['valid_samples']}
- Uyarılı örnek: {report['warning_samples']}
- Hatalı örnek: {report['error_samples']}
- Toplam süre: {report['total_duration_seconds']} sn / {report['total_duration_minutes']} dk
- Ortalama örnek süresi: {report['average_duration_seconds']} sn
- Minimum örnek süresi: {report['min_duration_seconds']} sn
- Maksimum örnek süresi: {report['max_duration_seconds']} sn

## Yorum

{report['summary']}

## Öneriler

{markdown_list(report['recommendations'])}

## Sonraki Adımlar

{markdown_list(report['next_steps'])}
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_REPORT_PATH.write_text(markdown, encoding="utf-8")
    return MARKDOWN_REPORT_PATH


def print_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Terminale okunabilir Turkce rapor basar."""
    print("Fine-tuning readiness report")
    print(f"Dataset: {report['dataset_path']}")
    print(f"Hazırlık seviyesi: {report['readiness_level']}")
    print(f"Toplam metadata satırı: {report['total_rows']}")
    print(f"Geçerli örnek: {report['valid_samples']}")
    print(f"Uyarılı örnek: {report['warning_samples']}")
    print(f"Hatalı örnek: {report['error_samples']}")
    print(f"Toplam süre: {report['total_duration_seconds']} sn")
    print(f"Toplam süre: {report['total_duration_minutes']} dk")
    print(f"Ortalama örnek süresi: {report['average_duration_seconds']} sn")
    print(f"Minimum örnek süresi: {report['min_duration_seconds']} sn")
    print(f"Maksimum örnek süresi: {report['max_duration_seconds']} sn")
    print("")
    print(report["summary"])

    if report["recommendations"]:
        print("")
        print("Öneriler:")
        for recommendation in report["recommendations"]:
            print(f"- {recommendation}")

    if report["next_steps"]:
        print("")
        print("Sonraki adımlar:")
        for next_step in report["next_steps"]:
            print(f"- {next_step}")

    print("")
    print(f"JSON raporu: {json_path}")
    print(f"Markdown raporu: {markdown_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge fine-tuning dataset hazırlık raporu üretir."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Raporlanacak dataset klasoru. Ornek: datasets/baglare-finetune-v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_path = resolve_dataset_path(args.dataset)
    report = build_readiness_report(dataset_path)
    json_path = write_json_report(report)
    markdown_path = write_markdown_report(report)
    print_report(report, json_path, markdown_path)
    return 1 if report["readiness_level"] == NOT_READY else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
