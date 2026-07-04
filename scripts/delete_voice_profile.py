# -*- coding: utf-8 -*-
"""VoxForge yerel voice profile klasorunu siler."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.voice_profile_utils import VoiceProfileError, delete_voice_profile


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxForge yerel voice profile klasorunu siler."
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Silinecek profil slug degeri.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Silme islemini onaylar. Bu parametre olmadan silme yapilmaz.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.yes:
        print(
            "UYARI: Profil silinmedi. Silmek icin --yes parametresini ekleyin.",
            file=sys.stderr,
        )
        return 1

    try:
        result = delete_voice_profile(args.slug)
    except VoiceProfileError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print(result.message)
    print(f"Profil slug: {result.profile_slug}")
    print(f"Silinen klasor: {result.profile_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
