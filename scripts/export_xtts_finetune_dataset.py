# -*- coding: utf-8 -*-
"""VoxForge datasetini XTTS GPT fine-tuning deneyi icin export eder."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
METADATA_TRAIN_FILE = "metadata_train.csv"
METADATA_EVAL_FILE = "metadata_eval.csv"
MANIFEST_FILE = "experiment_manifest.json"
LANGUAGE = "tr"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_readiness_report import (  # noqa: E402
    base_readiness_level,
    downgrade_for_errors,
)
from scripts.validate_finetune_dataset import (  # noqa: E402
    DatasetValidationError,
    build_report as build_validation_report,
    read_metadata_rows,
    resolve_audio_path,
    resolve_dataset_path,
)


class ExportError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


def slugify(value: str) -> str:
    """Run adini dosya sistemi icin guvenli slug'a cevirir."""
    translation = str.maketrans({
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "İ": "i",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
    })
    value = value.translate(translation)
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower().strip()
    ascii_text = re.sub(r"[\s_]+", "-", ascii_text)
    ascii_text = re.sub(r"[^a-z0-9-]", "", ascii_text)
    ascii_text = re.sub(r"-{2,}", "-", ascii_text).strip("-")
    if not ascii_text:
        raise ExportError("Run name guvenli bir slug'a cevrilemedi.")
    return ascii_text


def project_relative_path(path: Path) -> str:
    """Manifest icin proje kokune gore okunabilir yol dondurur."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def clean_metadata_text(text: str) -> str:
    """LJSpeech pipe formatini bozmamak icin metni tek satira indirir."""
    return " ".join(text.replace("|", ",").split())


def audio_id_from_path(audio_path_text: str) -> str:
    """LJSpeech ilk kolonu icin .wav uzantisiz audio id dondurur."""
    return Path(audio_path_text).stem


def validate_dataset_or_raise(dataset_path: Path) -> dict[str, Any]:
    """Mevcut dataset dogrulamasini calistirir ve hata varsa exportu durdurur."""
    try:
        report = build_validation_report(dataset_path)
    except DatasetValidationError as exc:
        raise ExportError(str(exc)) from exc

    if report["error_samples"]:
        raise ExportError(
            "Dataset dogrulamasinda hatali ornek var; export yapilmadi. "
            "Once validate_finetune_dataset.py ciktisini duzeltin."
        )

    if report["valid_samples"] <= 0:
        raise ExportError("Dataset icinde export edilecek gecerli ornek bulunamadi.")

    return report


def split_rows(rows: list[tuple[int, str, str]]) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Deterministic sekilde son yuzde 10'u eval olarak ayirir."""
    if len(rows) < 10:
        return rows, []

    eval_count = max(1, math.ceil(len(rows) * 0.10))
    train_rows = rows[:-eval_count]
    eval_rows = rows[-eval_count:]
    return train_rows, eval_rows


def copy_audio_files(
    dataset_path: Path,
    rows: list[tuple[int, str, str]],
    export_wavs_dir: Path,
) -> None:
    """Metadata satirlarindaki WAV dosyalarini deney datasetine kopyalar."""
    export_wavs_dir.mkdir(parents=True, exist_ok=True)

    seen_audio_ids: set[str] = set()
    for line_number, audio_path_text, _text in rows:
        audio_id = audio_id_from_path(audio_path_text)
        if audio_id in seen_audio_ids:
            raise ExportError(f"Tekrarlanan audio id bulundu: {audio_id}")
        seen_audio_ids.add(audio_id)

        source_audio_path = resolve_audio_path(dataset_path, audio_path_text)
        if not source_audio_path.is_file():
            raise ExportError(f"Satir {line_number}: ses dosyasi bulunamadi: {source_audio_path}")

        target_audio_path = export_wavs_dir / f"{audio_id}.wav"
        shutil.copy2(source_audio_path, target_audio_path)


def write_ljspeech_metadata(metadata_path: Path, rows: list[tuple[int, str, str]]) -> None:
    """Basliksiz LJSpeech metadata dosyasi yazar: audio_id|text|text."""
    lines: list[str] = []
    for _line_number, audio_path_text, text in rows:
        audio_id = audio_id_from_path(audio_path_text)
        cleaned_text = clean_metadata_text(text)
        lines.append(f"{audio_id}|{cleaned_text}|{cleaned_text}")

    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    run_name: str,
    run_slug: str,
    dataset_path: Path,
    export_dataset_path: Path,
    train_metadata_path: Path,
    eval_metadata_path: Path | None,
    validation_report: dict[str, Any],
    train_samples: int,
    eval_samples: int,
) -> dict[str, Any]:
    """Deney dataset exportu icin manifest sozlugu olusturur."""
    total_duration_seconds = round(float(validation_report["total_duration_seconds"]), 3)
    total_duration_minutes = round(total_duration_seconds / 60.0, 3)
    readiness_level = base_readiness_level(
        int(validation_report["total_rows"]),
        int(validation_report["valid_samples"]) + int(validation_report["warning_samples"]),
        total_duration_minutes,
    )
    readiness_level = downgrade_for_errors(
        readiness_level,
        int(validation_report["error_samples"]),
    )

    return {
        "run_name": run_name,
        "run_slug": run_slug,
        "source_dataset": project_relative_path(dataset_path),
        "exported_dataset_path": project_relative_path(export_dataset_path),
        "train_metadata_path": project_relative_path(train_metadata_path),
        "eval_metadata_path": (
            project_relative_path(eval_metadata_path) if eval_metadata_path else None
        ),
        "total_samples": train_samples + eval_samples,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "total_duration_seconds": total_duration_seconds,
        "total_duration_minutes": total_duration_minutes,
        "readiness_level": readiness_level,
        "language": LANGUAGE,
    }


def export_dataset(dataset_arg: str, run_name: str) -> dict[str, Any]:
    """Dataseti experiments/<run_slug>/ altina XTTS egitimi icin export eder."""
    dataset_path = resolve_dataset_path(dataset_arg)
    run_slug = slugify(run_name)
    experiment_path = EXPERIMENTS_DIR / run_slug
    export_dataset_path = experiment_path / "dataset"
    export_wavs_dir = export_dataset_path / "wavs"
    train_metadata_path = export_dataset_path / METADATA_TRAIN_FILE
    eval_metadata_path = export_dataset_path / METADATA_EVAL_FILE
    manifest_path = experiment_path / MANIFEST_FILE

    if experiment_path.exists():
        raise ExportError(f"Experiment klasoru zaten var; uzerine yazilmadi: {experiment_path}")

    validation_report = validate_dataset_or_raise(dataset_path)
    metadata_rows = read_metadata_rows(dataset_path / "metadata.csv")
    validated_rows = [
        row for row in metadata_rows
        if resolve_audio_path(dataset_path, row[1]).is_file()
    ]

    exportable_samples = validation_report["valid_samples"] + validation_report["warning_samples"]
    if len(validated_rows) != exportable_samples:
        raise ExportError(
            "Metadata satirlari ile dogrulama raporu ayni sayida gecerli ornek vermedi; "
            "once dataset dogrulamasini kontrol edin."
        )

    train_rows, eval_rows = split_rows(validated_rows)

    try:
        copy_audio_files(dataset_path, validated_rows, export_wavs_dir)
        write_ljspeech_metadata(train_metadata_path, train_rows)
        if eval_rows:
            write_ljspeech_metadata(eval_metadata_path, eval_rows)
        else:
            eval_metadata_path = None

        manifest = build_manifest(
            run_name=run_name,
            run_slug=run_slug,
            dataset_path=dataset_path,
            export_dataset_path=export_dataset_path,
            train_metadata_path=train_metadata_path,
            eval_metadata_path=eval_metadata_path,
            validation_report=validation_report,
            train_samples=len(train_rows),
            eval_samples=len(eval_rows),
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        if experiment_path.exists():
            shutil.rmtree(experiment_path)
        raise

    return {
        "experiment_path": experiment_path,
        "export_dataset_path": export_dataset_path,
        "train_metadata_path": train_metadata_path,
        "eval_metadata_path": eval_metadata_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge datasetini XTTS fine-tuning deneyi icin export eder."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaynak dataset klasoru. Ornek: datasets/baglare-finetune-v1",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Deney adi. Ornek: baglare_xtts_exp01",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = export_dataset(args.dataset, args.run_name)
    except ExportError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    manifest = result["manifest"]
    print("XTTS fine-tuning dataset export tamamlandi.")
    print(f"Experiment: {result['experiment_path']}")
    print(f"Export dataset: {result['export_dataset_path']}")
    print(f"Train metadata: {result['train_metadata_path']}")
    print(f"Eval metadata: {result['eval_metadata_path'] or 'olusturulmadi'}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Toplam ornek: {manifest['total_samples']}")
    print(f"Train ornek: {manifest['train_samples']}")
    print(f"Eval ornek: {manifest['eval_samples']}")
    print(f"Toplam sure: {manifest['total_duration_seconds']} sn / {manifest['total_duration_minutes']} dk")
    print(f"Readiness: {manifest['readiness_level']}")
    print("Not: Export edilen dataset ve deney ciktisi GitHub'a yuklenmemelidir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
