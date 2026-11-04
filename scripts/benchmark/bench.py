# -*- coding: utf-8 -*-
"""Benchmark steps that need no GPU and no extra packages (VF-Q1).

    python scripts/benchmark/bench.py prepare (--profile <slug> | --reference <wav>) [--reference-text "..."]
    python scripts/benchmark/bench.py blind  --run <run_dir> [--pilot]
    python scripts/benchmark/bench.py report --run <run_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.benchmark.experiment import (  # noqa: E402
    BenchmarkError,
    build_report,
    create_blind_set,
    prepare_run,
)

DEFAULT_ROOT = REPO_ROOT / "outputs" / "voice_benchmark"
PROFILES_DIR = REPO_ROOT / "profiles"


def profile_reference(slug: str) -> Path:
    """Read-only lookup of a profile's preprocessed reference, contained in profiles/."""
    root = PROFILES_DIR.resolve()
    folder = (PROFILES_DIR / slug).resolve()
    if folder.parent != root:
        raise BenchmarkError("Geçersiz profil adı.")
    reference = folder / "preprocessed_reference.wav"
    if not reference.is_file():
        raise BenchmarkError(f"Profil referansı bulunamadı: {reference}")
    return reference


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Yeni koşu klasörü ve referans kopyası oluştur")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", type=Path)
    source.add_argument("--profile", help="profiles/<slug>/preprocessed_reference.wav kullan")
    prepare.add_argument("--reference-text", default="", help="Referans sesin birebir metni (VoxCPM2 ultimate için)")
    prepare.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    prepare.add_argument("--seed", type=int, default=1234)
    blind = sub.add_parser("blind", help="Kör dinleme seti (tam koşu); --pilot ile ayrı deneme seti")
    blind.add_argument("--run", required=True, type=Path)
    blind.add_argument("--pilot", action="store_true", help="listening_pilot/ altına; karara katılmaz")
    sub.add_parser("report").add_argument("--run", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare":
            reference = args.reference or profile_reference(args.profile)
            run_dir = prepare_run(args.root, reference, args.reference_text, args.seed)
            print(f"Koşu klasörü: {run_dir}")
        elif args.command == "blind":
            print(f"Dinleme formu: {create_blind_set(args.run, pilot=args.pilot)}")
        else:
            report = build_report(args.run)
            print(f"Rapor: {Path(args.run) / 'report.md'}")
            for name, result in report["verdicts"].items():
                print(f"  {name}: kalite={result['quality']['decision']} · performans={result['performance']['label']}")
    except BenchmarkError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
