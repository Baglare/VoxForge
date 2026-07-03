# -*- coding: utf-8 -*-
"""VoxForge icin yerel voice profile klasoru olusturur."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.voice_profile_utils import VoiceProfileError, create_voice_profile


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge icin yerel voice profile olusturur."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Profil adi. Guvenli klasor adina cevrilir.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Kullanilacak referans ses dosyasi yolu.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        result = create_voice_profile(args.name, args.input)
    except VoiceProfileError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print("Voice profile olusturuldu.")
    print(f"Profil adi: {result.profile_name}")
    print(f"Profil slug: {result.profile_slug}")
    print(f"Giris ses dosyasi: {result.source_audio}")
    print(f"Profil klasoru: {result.profile_dir}")
    print(f"Secilen on isleme varyanti: {result.selected_preprocessing_variant}")
    if result.preprocessing_warning:
        print(f"On isleme uyarisi: {result.preprocessing_warning}")
    print(f"Orijinal kalite: {result.original_quality.get('quality', 'UNKNOWN')}")
    print(
        "On islenmis kalite: "
        f"{result.preprocessed_quality.get('quality', 'UNKNOWN')}"
    )
    print(f"Metadata: {result.profile_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
