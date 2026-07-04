# -*- coding: utf-8 -*-
"""recording_plan.csv içindeki DONE satırlardan metadata.csv üretir."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_FILE_NAME = "recording_plan.csv"
METADATA_FILE_NAME = "metadata.csv"
PLAN_FIELDS = ["clip_id", "target_audio_path", "text", "status", "notes"]
PIPE_PLAN_HEADER = "|".join(PLAN_FIELDS)
SEMICOLON_PLAN_HEADER = ";".join(PLAN_FIELDS)
PLAN_ENCODING = "utf-8-sig"
METADATA_HEADER = "audio_path|text"


class MetadataBuildError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


def resolve_dataset_path(dataset_arg: str) -> Path:
    """Dataset yolunu proje kokune gore cozer."""
    dataset_path = Path(dataset_arg)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return dataset_path.resolve()


def detect_plan_delimiter(header_line: str) -> str:
    """recording_plan.csv başlığına göre eski veya yeni ayırıcıyı seçer."""
    header = header_line.strip()
    if header == SEMICOLON_PLAN_HEADER:
        return ";"
    if header == PIPE_PLAN_HEADER:
        return "|"

    raise MetadataBuildError(
        "recording_plan.csv başlığı hatalı. Beklenen başlık: "
        f"{SEMICOLON_PLAN_HEADER} veya {PIPE_PLAN_HEADER}"
    )


def read_plan_rows(plan_path: Path) -> list[dict[str, str]]:
    """recording_plan.csv dosyasını eski pipe veya yeni semicolon formatıyla okur."""
    try:
        with plan_path.open("r", encoding=PLAN_ENCODING, newline="") as plan_file:
            first_line = plan_file.readline()
            if not first_line:
                raise MetadataBuildError("recording_plan.csv boş.")

            delimiter = detect_plan_delimiter(first_line)
            plan_file.seek(0)
            reader = csv.reader(plan_file, delimiter=delimiter)
            header = next(reader, None)
            if header != PLAN_FIELDS:
                raise MetadataBuildError(
                    "recording_plan.csv başlığı hatalı. Beklenen alanlar: "
                    f"{', '.join(PLAN_FIELDS)}"
                )

            rows: list[dict[str, str]] = []
            for row in reader:
                line_number = reader.line_num
                if not row or not any(cell.strip() for cell in row):
                    continue

                if len(row) != len(PLAN_FIELDS):
                    raise MetadataBuildError(
                        f"recording_plan.csv satır {line_number} hatalı: "
                        f"{len(PLAN_FIELDS)} alan bekleniyor."
                    )

                cleaned_row = [cell.strip() for cell in row]
                rows.append(
                    {
                        "line_number": str(line_number),
                        "clip_id": cleaned_row[0],
                        "target_audio_path": cleaned_row[1],
                        "text": cleaned_row[2],
                        "status": cleaned_row[3],
                        "notes": cleaned_row[4],
                    }
                )
    except OSError as exc:
        raise MetadataBuildError(f"recording_plan.csv okunamadı: {plan_path}") from exc

    return rows


def build_metadata(dataset_path: Path) -> tuple[Path, int, list[str]]:
    """DONE satırlarını metadata.csv dosyasına yazar."""
    if not dataset_path.is_dir():
        raise MetadataBuildError(f"Dataset klasörü bulunamadı: {dataset_path}")

    plan_path = dataset_path / PLAN_FILE_NAME
    if not plan_path.is_file():
        raise MetadataBuildError(f"recording_plan.csv bulunamadı: {plan_path}")

    rows = read_plan_rows(plan_path)
    metadata_path = dataset_path / METADATA_FILE_NAME
    metadata_lines = [METADATA_HEADER]
    warnings: list[str] = []

    for row in rows:
        if row["status"].strip().upper() != "DONE":
            continue

        target_audio_path = row["target_audio_path"]
        text = row["text"]
        resolved_audio_path = dataset_path / target_audio_path

        if not target_audio_path:
            warnings.append(f"satır {row['line_number']}: target_audio_path boş, atlandı.")
            continue
        if not text:
            warnings.append(f"satır {row['line_number']}: text boş, atlandı.")
            continue
        if not resolved_audio_path.is_file():
            warnings.append(
                f"satır {row['line_number']}: ses dosyası bulunamadı, atlandı: "
                f"{resolved_audio_path}"
            )
            continue

        metadata_lines.append(f"{target_audio_path}|{text}")

    metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    return metadata_path, len(metadata_lines) - 1, warnings


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="recording_plan.csv DONE satırlarından metadata.csv oluşturur."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset klasörü. Örnek: datasets/baglare-finetune-v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_path = resolve_dataset_path(args.dataset)

    try:
        metadata_path, written_rows, warnings = build_metadata(dataset_path)
    except MetadataBuildError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print("metadata.csv oluşturuldu.")
    print(f"Dataset: {dataset_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Yazılan DONE satırı: {written_rows}")
    if warnings:
        print("")
        print("Uyarılar:")
        for warning in warnings:
            print(f"- {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
