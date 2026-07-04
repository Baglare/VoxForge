# -*- coding: utf-8 -*-
"""VoxForge icin yerel fine-tuning dataset klasoru baslatir."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
METADATA_HEADER = "audio_path|text"

TURKISH_CHAR_MAP = str.maketrans(
    {
        "\u00e7": "c",
        "\u00c7": "c",
        "\u011f": "g",
        "\u011e": "g",
        "\u0131": "i",
        "I": "i",
        "\u0130": "i",
        "\u00f6": "o",
        "\u00d6": "o",
        "\u015f": "s",
        "\u015e": "s",
        "\u00fc": "u",
        "\u00dc": "u",
    }
)


class DatasetInitError(Exception):
    """Kullaniciya sade hata mesaji gostermek icin kullanilir."""


def slugify_dataset_name(dataset_name: str) -> str:
    """Dataset adini guvenli klasor adina cevirir."""
    normalized = dataset_name.strip().translate(TURKISH_CHAR_MAP).lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")

    if not normalized:
        raise DatasetInitError(
            "Dataset adi guvenli klasor adina cevrilemedi. "
            "Lutfen harf veya rakam iceren bir ad kullanin."
        )

    return normalized


def ensure_dataset_path_is_safe(dataset_dir: Path) -> None:
    """Dataset yolunun datasets klasoru disina cikmadigini dogrular."""
    datasets_root = DATASETS_DIR.resolve()
    resolved_dataset_dir = dataset_dir.resolve()

    try:
        resolved_dataset_dir.relative_to(datasets_root)
    except ValueError as exc:
        raise DatasetInitError("Dataset yolu datasets klasoru disina cikamaz.") from exc


def init_dataset(dataset_name: str) -> Path:
    """Bos metadata.csv ve wavs klasoru olan dataset iskeleti olusturur."""
    dataset_slug = slugify_dataset_name(dataset_name)
    dataset_dir = DATASETS_DIR / dataset_slug
    wavs_dir = dataset_dir / "wavs"
    metadata_path = dataset_dir / "metadata.csv"

    ensure_dataset_path_is_safe(dataset_dir)

    if dataset_dir.exists():
        raise DatasetInitError(
            "Bu dataset zaten var; uzerine yazilmadi.\n"
            f"Mevcut dataset klasoru: {dataset_dir}"
        )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    (DATASETS_DIR / ".gitkeep").touch(exist_ok=True)
    wavs_dir.mkdir(parents=True)
    metadata_path.write_text(f"{METADATA_HEADER}\n", encoding="utf-8")

    return dataset_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge icin yerel fine-tuning dataset iskeleti olusturur."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset adi. Guvenli slug'a cevrilir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        dataset_dir = init_dataset(args.name)
    except DatasetInitError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print("Fine-tuning dataset iskeleti olusturuldu.")
    print(f"Dataset klasoru: {dataset_dir}")
    print(f"WAV klasoru: {dataset_dir / 'wavs'}")
    print(f"Metadata: {dataset_dir / 'metadata.csv'}")
    print("metadata.csv basligi: audio_path|text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
