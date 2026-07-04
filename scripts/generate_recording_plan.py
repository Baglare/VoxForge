# -*- coding: utf-8 -*-
"""VoxForge kayıt metinlerinden dataset recording_plan.csv üretir."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SET_PATH = PROJECT_ROOT / "docs" / "RECORDING_TEXT_SET_TR.md"
PLAN_FILE_NAME = "recording_plan.csv"
PLAN_HEADER = ["clip_id", "target_audio_path", "text", "status", "notes"]
PLAN_ENCODING = "utf-8-sig"
PLAN_DELIMITER = ";"
RECORD_PATTERN = re.compile(r"^\s*-\s*(VF_TR_\d{3})\s*\|\s*(.+?)\s*$")


class RecordingPlanError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


@dataclass(frozen=True)
class RecordingText:
    """Dokumandan okunan tek kayıt metni."""

    clip_id: str
    text: str


def resolve_dataset_path(dataset_arg: str) -> Path:
    """Dataset yolunu proje kokune gore cozer."""
    dataset_path = Path(dataset_arg)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return dataset_path.resolve()


def read_recording_texts() -> list[RecordingText]:
    """docs/RECORDING_TEXT_SET_TR.md icindeki VF_TR_XXX satirlarini okur."""
    if not TEXT_SET_PATH.is_file():
        raise RecordingPlanError(f"Kayıt metinleri dokümanı bulunamadı: {TEXT_SET_PATH}")

    records: list[RecordingText] = []
    for raw_line in TEXT_SET_PATH.read_text(encoding="utf-8").splitlines():
        match = RECORD_PATTERN.match(raw_line)
        if not match:
            continue

        clip_id, text = match.groups()
        cleaned_text = text.strip()
        if cleaned_text:
            records.append(RecordingText(clip_id=clip_id, text=cleaned_text))

    if not records:
        raise RecordingPlanError("Kayıt metinleri dokümanında VF_TR_XXX satırı bulunamadı.")

    return records


def write_recording_plan(dataset_path: Path, count: int, overwrite: bool) -> Path:
    """İlk N kayıt metninden recording_plan.csv dosyasını oluşturur."""
    if count <= 0:
        raise RecordingPlanError("--count pozitif bir sayı olmalıdır.")

    if not dataset_path.is_dir():
        raise RecordingPlanError(f"Dataset klasörü bulunamadı: {dataset_path}")

    plan_path = dataset_path / PLAN_FILE_NAME
    if plan_path.exists() and not overwrite:
        raise RecordingPlanError(
            "recording_plan.csv zaten var; üzerine yazılmadı.\n"
            f"Mevcut dosya: {plan_path}\n"
            "Yeniden oluşturmak için --overwrite parametresini kullanın."
        )

    records = read_recording_texts()
    if count > len(records):
        raise RecordingPlanError(
            f"İstenen kayıt sayısı ({count}) mevcut metin sayısından fazla ({len(records)})."
        )

    # utf-8-sig, Excel'in dosyayı çift tıklamayla açarken Türkçe karakterleri
    # doğru algılamasına yardımcı olur. Semicolon ayırıcı da TR Excel için daha uygundur.
    with plan_path.open("w", encoding=PLAN_ENCODING, newline="") as plan_file:
        writer = csv.writer(
            plan_file,
            delimiter=PLAN_DELIMITER,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(PLAN_HEADER)
        for record in records[:count]:
            target_audio_path = f"wavs/{record.clip_id}.wav"
            writer.writerow([record.clip_id, target_audio_path, record.text, "TODO", ""])

    return plan_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge fine-tuning dataset kayıt planı oluşturur."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset klasörü. Örnek: datasets/baglare-finetune-v1",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=80,
        help="Planlanacak ilk kayıt sayısı. Varsayılan: 80",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan recording_plan.csv dosyasını yeniden oluştur.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_path = resolve_dataset_path(args.dataset)

    try:
        plan_path = write_recording_plan(dataset_path, args.count, args.overwrite)
    except RecordingPlanError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print("Kayıt planı oluşturuldu.")
    print(f"Dataset: {dataset_path}")
    print(f"Kayıt planı: {plan_path}")
    print(f"Planlanan kayıt sayısı: {args.count}")
    print(f"encoding: {PLAN_ENCODING}")
    print(f"delimiter: {PLAN_DELIMITER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
