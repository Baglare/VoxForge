# -*- coding: utf-8 -*-
"""Birden fazla XTTS checkpoint ve test cumlesi icin matrix inference denemesi."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_xtts_finetuned_checkpoint import (
    CHUNK_MAX_CHARS,
    PROJECT_ROOT,
    TRAINING_OUTPUT_DIR_NAME,
    EvaluationError,
    import_xtts_api,
    resolve_base_files,
    resolve_project_path,
    resolve_speaker_wav,
    run_xtts_generation,
)

try:
    from text_chunking_utils import split_text_for_tts, summarize_chunks
except ImportError:
    from scripts.text_chunking_utils import split_text_for_tts, summarize_chunks


REPORT_JSON_PATH = PROJECT_ROOT / "outputs" / "reports" / "finetuned_matrix_report.json"
REPORT_MD_PATH = PROJECT_ROOT / "outputs" / "reports" / "finetuned_matrix_report.md"
MATRIX_ROOT = PROJECT_ROOT / "outputs" / "finetuned_eval" / "matrix"

TEST_SENTENCES = [
    "Merhaba, bugün kısa bir ses karşılaştırması yapıyoruz.",
    "Bu kayıt, VoxForge fine-tuning denemesinin nötr açıklama testidir.",
    "Sence bu cümledeki tonlama doğal ve anlaşılır geliyor mu?",
    "4 Temmuz 2026 tarihinde saat 18.30 için küçük bir deneme planlandı.",
    (
        "Bu biraz daha uzun test paragrafında konuşmanın akışı, nefes hissi ve "
        "cümle sonlarındaki yumuşama daha dikkatli dinlenmelidir."
    ),
    "Çığ, öğüt, şüphe, İğdır ve küçük ünlü uyumu gibi Türkçe karakterler burada özellikle korunmalıdır.",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base ve fine-tuned XTTS checkpointlerini coklu cumlelerle karsilastirir."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment klasoru. Ornek: experiments/baglare-xtts-exp01",
    )
    parser.add_argument(
        "--speaker-wav",
        default=None,
        help="Opsiyonel referans ses yolu.",
    )
    return parser.parse_args(argv)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def find_best_model(training_output: Path) -> Path | None:
    return latest_file([path for path in training_output.rglob("best_model.pth") if path.is_file()])


def find_named_checkpoint(training_output: Path, filename: str) -> Path | None:
    return latest_file([path for path in training_output.rglob(filename) if path.is_file()])


def find_highest_numbered_checkpoint(training_output: Path) -> tuple[str, Path] | None:
    candidates: list[tuple[int, Path]] = []
    for path in training_output.rglob("checkpoint_*.pth"):
        if not path.is_file():
            continue
        match = re.fullmatch(r"checkpoint_(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return None

    number, path = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return f"checkpoint_{number}", path


def discover_variants(
    experiment_path: Path,
    base_checkpoint: Path,
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    training_output = experiment_path / TRAINING_OUTPUT_DIR_NAME
    variants: list[dict[str, Any]] = [
        {
            "name": "base",
            "checkpoint_path": base_checkpoint.resolve(),
            "kind": "base",
        }
    ]
    selected_paths = {base_checkpoint.resolve()}

    if not training_output.is_dir():
        errors.append(
            {
                "variant": "training_output",
                "test": "",
                "message": f"training_output klasoru bulunamadi: {training_output}",
            }
        )
        return variants

    requested: list[tuple[str, Path | None]] = [
        ("best_model", find_best_model(training_output)),
        ("best_model_72", find_named_checkpoint(training_output, "best_model_72.pth")),
    ]

    highest_checkpoint = find_highest_numbered_checkpoint(training_output)
    if highest_checkpoint is None:
        requested.append(
            (
                "checkpoint_*.pth",
                None,
            )
        )
    else:
        requested.append(highest_checkpoint)

    for variant_name, checkpoint_path in requested:
        if checkpoint_path is None:
            errors.append(
                {
                    "variant": variant_name,
                    "test": "",
                    "message": f"{variant_name} bulunamadi; varyant atlandi.",
                }
            )
            continue

        resolved_path = checkpoint_path.resolve()
        if resolved_path in selected_paths:
            errors.append(
                {
                    "variant": variant_name,
                    "test": "",
                    "message": f"Ayni checkpoint tekrar secilmedi: {resolved_path}",
                }
            )
            continue

        selected_paths.add(resolved_path)
        variants.append(
            {
                "name": variant_name,
                "checkpoint_path": resolved_path,
                "kind": "fine_tuned",
            }
        )

    return variants


def build_report(
    experiment_path: Path,
    speaker_wav: Path | None,
    output_root: Path,
    variants: list[dict[str, Any]],
    success_count: int,
    errors: list[dict[str, str]],
    chunking_results: list[dict[str, Any]],
) -> dict[str, Any]:
    tested_variants = []
    for variant in variants:
        tested_variants.append(
            {
                "name": variant["name"],
                "checkpoint_path": str(variant["checkpoint_path"]),
                "kind": variant["kind"],
                "output_dir": str(output_root / variant["name"]),
            }
        )

    text_chunking = []
    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        chunk_summary = summarize_chunks(split_text_for_tts(sentence, max_chars=CHUNK_MAX_CHARS))
        text_chunking.append(
            {
                "test": f"test_{index:02d}",
                "text": sentence,
                **chunk_summary,
            }
        )

    return {
        "experiment": str(experiment_path),
        "speaker_wav": str(speaker_wav) if speaker_wav else None,
        "tested_variants": tested_variants,
        "test_sentences": TEST_SENTENCES,
        "chunking_used": any(item["chunking_used"] for item in text_chunking),
        "chunked_test_count": sum(1 for item in text_chunking if item["chunking_used"]),
        "chunks": text_chunking,
        "chunking_results": chunking_results,
        "output_root": str(output_root),
        "success_count": success_count,
        "error_count": len(errors),
        "errors": errors,
        "notes": [
            "Bu matrix kaliteyi otomatik ölçmez.",
            "Base ve fine-tuned çıktılar insan kulağıyla karşılaştırılmalıdır.",
            "Robotiklik varsa daha fazla training basmadan önce veri, referans ve checkpoint seçimi incelenmelidir.",
            "Küçük dataset nedeniyle ses benzerliği ve doğallık sınırlı olabilir.",
        ],
    }


def write_json_report(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON rapor yazildi: {REPORT_JSON_PATH}")


def write_markdown_report(report: dict[str, Any]) -> None:
    lines = [
        "# Fine-tuned XTTS checkpoint matrix raporu",
        "",
        f"- Experiment: `{report['experiment']}`",
        f"- Speaker wav: `{report['speaker_wav']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Basarili cikti sayisi: {report['success_count']}",
        f"- Hata sayisi: {report['error_count']}",
        "",
        "## Dinleme yönergesi",
        "",
        "1. Önce `base` klasörünü dinle.",
        "2. Sonra fine-tuned varyantları sırayla dinle.",
        "3. Benzerlik, doğallık, telaffuz ve robotiklik açısından 1-5 puan ver.",
        "4. Fine-tuned çıktı base'den kötüyse daha fazla training basmadan veri/ayar tarafı incelenmeli.",
        "",
        "Bu rapor kaliteyi otomatik ölçmez; karar insan kulağıyla verilir.",
        "",
        "## Varyantlar",
        "",
    ]

    for variant in report["tested_variants"]:
        lines.extend(
            [
                f"- `{variant['name']}`",
                f"  - Checkpoint: `{variant['checkpoint_path']}`",
                f"  - Output: `{variant['output_dir']}`",
            ]
        )

    lines.extend(["", "## Test cümleleri", ""])
    for index, sentence in enumerate(report["test_sentences"], start=1):
        lines.append(f"{index}. {sentence}")

    if report["errors"]:
        lines.extend(["", "## Hatalar", ""])
        for error in report["errors"]:
            lines.append(
                f"- `{error.get('variant', '')}` `{error.get('test', '')}`: {error.get('message', '')}"
            )

    lines.extend(["", "## Chunking", ""])
    lines.append(f"- Chunking kullanildi mi: {report['chunking_used']}")
    lines.append(f"- Chunking gereken test sayisi: {report['chunked_test_count']}")
    for item in report["chunks"]:
        if item["chunking_used"]:
            lines.append(f"- `{item['test']}`: {item['chunk_count']} parca")

    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown rapor yazildi: {REPORT_MD_PATH}")


def write_reports(report: dict[str, Any]) -> None:
    write_json_report(report)
    write_markdown_report(report)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    experiment_path = resolve_project_path(args.experiment)
    output_root = MATRIX_ROOT / timestamp_slug()
    errors: list[dict[str, str]] = []
    success_count = 0
    variants: list[dict[str, Any]] = []
    speaker_wav: Path | None = None
    chunking_results: list[dict[str, Any]] = []

    try:
        print("VoxForge XTTS checkpoint matrix evaluation")
        print(f"Experiment: {experiment_path}")
        print(f"Output root: {output_root}")

        base_files = resolve_base_files(experiment_path)
        speaker_wav = resolve_speaker_wav(experiment_path, args.speaker_wav)
        print(f"Kullanilan speaker_wav: {speaker_wav}")

        variants = discover_variants(
            experiment_path=experiment_path,
            base_checkpoint=base_files["model.pth"],
            errors=errors,
        )
        print("Test edilecek varyantlar:")
        for variant in variants:
            print(f"- {variant['name']}: {variant['checkpoint_path']}")

        config_class, model_class = import_xtts_api()

        for variant in variants:
            variant_name = variant["name"]
            checkpoint_path = variant["checkpoint_path"]
            variant_dir = output_root / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)

            for index, sentence in enumerate(TEST_SENTENCES, start=1):
                output_path = variant_dir / f"test_{index:02d}.wav"
                try:
                    print(f"[{variant_name}] test_{index:02d} uretiliyor...")
                    chunk_summary = run_xtts_generation(
                        variant_name,
                        config_class,
                        model_class,
                        base_files,
                        checkpoint_path,
                        speaker_wav,
                        sentence,
                        output_path,
                    )
                    chunking_results.append(
                        {
                            "variant": variant_name,
                            "test": f"test_{index:02d}",
                            **chunk_summary,
                        }
                    )
                    print(f"Ses dosyasi: {output_path}")
                    success_count += 1
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    print(
                        f"UYARI: {variant_name} test_{index:02d} basarisiz: {message}",
                        file=sys.stderr,
                    )
                    errors.append(
                        {
                            "variant": variant_name,
                            "test": f"test_{index:02d}",
                            "message": message,
                        }
                    )
                    continue

        report = build_report(
            experiment_path=experiment_path,
            speaker_wav=speaker_wav,
            output_root=output_root,
            variants=variants,
            success_count=success_count,
            errors=errors,
            chunking_results=chunking_results,
        )
        write_reports(report)

        if success_count == 0:
            print("HATA: Hic ses ciktisi uretilemedi.", file=sys.stderr)
            return 1

        print("Checkpoint matrix evaluation tamamlandi.")
        return 0
    except EvaluationError as exc:
        errors.append({"variant": "", "test": "", "message": str(exc)})
        report = build_report(
            experiment_path=experiment_path,
            speaker_wav=speaker_wav,
            output_root=output_root,
            variants=variants,
            success_count=success_count,
            errors=errors,
            chunking_results=chunking_results,
        )
        write_reports(report)
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append({"variant": "", "test": "", "message": message})
        report = build_report(
            experiment_path=experiment_path,
            speaker_wav=speaker_wav,
            output_root=output_root,
            variants=variants,
            success_count=success_count,
            errors=errors,
            chunking_results=chunking_results,
        )
        write_reports(report)
        print(f"HATA: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
