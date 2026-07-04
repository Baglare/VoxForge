# -*- coding: utf-8 -*-
"""VoxForge yerel voice profile on isleme ve kalite bilgisini yeniler."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.voice_profile_utils import (
    VoiceProfileError,
    quality_label,
    recreate_voice_profile,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge yerel voice profile dosyalarini yeniden olusturur."
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Yenilenecek profil slug degeri.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        result = recreate_voice_profile(args.slug)
    except VoiceProfileError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print("Voice profile yenilendi.")
    print(f"Profil adi: {result.profile_name}")
    print(f"Profil slug: {result.profile_slug}")
    print(f"Korunan orijinal referans: {result.original_reference}")
    print(f"Yenilenen on islenmis referans: {result.preprocessed_reference}")
    print(f"Secilen on isleme varyanti: {result.selected_preprocessing_variant}")
    if result.preprocessing_warning:
        print(f"On isleme uyarisi: {result.preprocessing_warning}")
    print(f"Orijinal kalite: {quality_label(result.original_quality)}")
    print(f"On islenmis kalite: {quality_label(result.preprocessed_quality)}")
    print(f"Metadata: {result.profile_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
