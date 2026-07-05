# -*- coding: utf-8 -*-
"""Exp01/Exp02 gibi fine-tuning deneyleri icin blind A/B karsilastirma araci."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_quality_utils import analyze_audio_file  # noqa: E402
from evaluate_xtts_finetuned_checkpoint import (  # noqa: E402
    CHUNK_MAX_CHARS,
    EvaluationError,
    find_selected_checkpoint,
    import_xtts_api,
    init_model,
    load_config,
    load_model_checkpoint,
    move_model_to_device,
    patch_config_paths,
    resolve_base_files,
    resolve_project_path,
    resolve_speaker_wav,
    synthesize_text_with_chunking,
)
from text_chunking_utils import split_text_for_tts, summarize_chunks  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "finetuned_eval" / "experiment_compare"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_JSON_PATH = REPORTS_DIR / "experiment_compare_report.json"
REPORT_MD_PATH = REPORTS_DIR / "experiment_compare_report.md"
BLIND_KEY_PATH = REPORTS_DIR / "experiment_blind_key.json"
BLIND_SCORECARD_PATH = REPORTS_DIR / "experiment_blind_scorecard.csv"

TEST_TEXTS = [
    {
        "test_id": "test_01",
        "category": "kısa cümle",
        "text": "Bugün ses karşılaştırması için kısa bir deneme yapıyoruz.",
    },
    {
        "test_id": "test_02",
        "category": "orta uzunlukta açıklama",
        "text": (
            "Bu kayıt, iki farklı fine-tuning denemesinin ton, telaffuz ve akıcılık "
            "açısından nasıl ayrıldığını anlamak için hazırlanmıştır."
        ),
    },
    {
        "test_id": "test_03",
        "category": "soru cümlesi",
        "text": "Sence bu cümledeki vurgu doğal mı, yoksa ses biraz mekanik mi geliyor?",
    },
    {
        "test_id": "test_04",
        "category": "sayı ve tarih",
        "text": "Toplantı 14 Temmuz 2026 saat 09.30'da başlayacak ve toplam 45 dakika sürecek.",
    },
    {
        "test_id": "test_05",
        "category": "Türkçe karakter",
        "text": "Çözüm için ğ, ş, ı, ö, ü ve ç harflerini doğal biçimde söylemek önemlidir.",
    },
    {
        "test_id": "test_06",
        "category": "doğal konuşma",
        "text": "Aslında sonucu hemen yorumlamak istemiyorum; önce iki örneği de sakin şekilde dinleyelim.",
    },
    {
        "test_id": "test_07",
        "category": "uzun metin",
        "text": (
            "Bu uzun metin, XTTS tarafında karakter sınırına yaklaşan ya da sınırı aşan "
            "durumlarda sistemin sesi kesmeden üretip üretmediğini kontrol etmek için "
            "kullanılır. Cümlelerin doğal akışını koruması, parçalar birleştirildiğinde "
            "geçişlerin fazla belirgin olmaması ve son cümlenin duyulur kalması özellikle "
            "dinleme sırasında not edilmelidir."
        ),
    },
    {
        "test_id": "test_08",
        "category": "nötr teknik açıklama",
        "text": (
            "Model çıktıları otomatik olarak kalite hükmü üretmez; süre, ses seviyesi ve "
            "kırpılma sinyalleri yalnızca değerlendirmeyi destekleyen teknik göstergelerdir."
        ),
    },
]

VARIANT_ORDER = ("base", "exp_a", "exp_b")


class CompareError(Exception):
    """Kullaniciya sade hata mesaji dondurmek icin kullanilir."""


@dataclass
class LoadedVariant:
    name: str
    checkpoint_path: Path
    model: Any
    config: Any


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_experiment_path(path_arg: str) -> Path:
    path = resolve_project_path(path_arg)
    if not path.is_dir():
        raise CompareError(f"Experiment klasoru bulunamadi: {path}")
    return path.resolve()


def output_sample_rate(config: Any) -> int:
    for field_name in ("output_sample_rate", "sample_rate"):
        value = getattr(getattr(config, "audio", None), field_name, None)
        if isinstance(value, int) and value > 0:
            return value
        value = getattr(config, field_name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 24000


def load_variant_model(
    variant_name: str,
    checkpoint_path: Path,
    base_files: dict[str, Path],
    config_class: Any,
    model_class: Any,
) -> LoadedVariant:
    """Bir checkpointi inference icin hazirlar."""
    print(f"{variant_name} checkpoint: {checkpoint_path}")
    config = load_config(config_class, base_files["config.json"])
    patch_config_paths(config, base_files, checkpoint_path)
    model = init_model(model_class, config)
    load_model_checkpoint(model, config, base_files, checkpoint_path, variant_name)
    move_model_to_device(model)
    return LoadedVariant(
        name=variant_name,
        checkpoint_path=checkpoint_path,
        model=model,
        config=config,
    )


def test_text_entries() -> list[dict[str, Any]]:
    entries = []
    for item in TEST_TEXTS:
        chunks = split_text_for_tts(item["text"], max_chars=CHUNK_MAX_CHARS)
        entries.append(
            {
                **item,
                "char_count": len(item["text"]),
                **summarize_chunks(chunks),
            }
        )
    return entries


def analyze_output_metrics(output_path: Path, text: str, base_duration: float | None) -> dict[str, Any]:
    """FFprobe/FFmpeg tabanli yardimci metrikleri uretir."""
    report = analyze_audio_file(output_path)
    duration = report.get("duration_seconds")
    flags: list[str] = []

    if len(text) >= 120 and isinstance(duration, (int, float)) and duration < 2.0:
        flags.append("possible_cutoff")

    if (
        base_duration is not None
        and isinstance(duration, (int, float))
        and base_duration > 0
        and duration < (base_duration * 0.65)
        and len(text) >= 120
    ):
        flags.append("possible_cutoff_vs_base")

    return {
        "exists": report.get("exists"),
        "duration_seconds": duration,
        "file_size_mb": report.get("file_size_mb"),
        "mean_volume_db": report.get("mean_volume_db"),
        "max_volume_db": report.get("max_volume_db"),
        "clipping_risk": report.get("clipping_risk"),
        "flags": flags,
        "warnings": report.get("warnings", []),
    }


def generate_variant_outputs(
    variant: LoadedVariant,
    output_root: Path,
    speaker_wav: Path,
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bir varyant icin tum sabit test metinlerini uretir."""
    variant_dir = output_root / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = output_sample_rate(variant.config)
    outputs: list[dict[str, Any]] = []

    for item in TEST_TEXTS:
        output_path = variant_dir / f"{item['test_id']}.wav"
        label = f"{variant.name}/{item['test_id']}"
        try:
            chunk_summary = synthesize_text_with_chunking(
                model=variant.model,
                config=variant.config,
                text=item["text"],
                speaker_wav=speaker_wav,
                output_path=output_path,
                sample_rate=sample_rate,
                label=label,
            )
            outputs.append(
                {
                    "variant": variant.name,
                    "test_id": item["test_id"],
                    "category": item["category"],
                    "text": item["text"],
                    "output_path": str(output_path),
                    "success": True,
                    "chunking_used": chunk_summary["chunking_used"],
                    "chunk_count": chunk_summary["chunk_count"],
                    "chunks": chunk_summary["chunks"],
                }
            )
        except Exception as exc:
            message = str(exc)
            print(f"UYARI: {label} uretilemedi: {message}", file=sys.stderr)
            errors.append(
                {
                    "variant": variant.name,
                    "test_id": item["test_id"],
                    "message": message,
                }
            )
            outputs.append(
                {
                    "variant": variant.name,
                    "test_id": item["test_id"],
                    "category": item["category"],
                    "text": item["text"],
                    "output_path": str(output_path),
                    "success": False,
                    "error": message,
                    "chunking_used": False,
                    "chunk_count": 0,
                    "chunks": [],
                }
            )

    return outputs


def add_metrics_and_warnings(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Uretilen ciktilara teknik metrikleri ve sure farki uyarilarini ekler."""
    by_key = {
        (item["variant"], item["test_id"]): item
        for item in outputs
    }
    warnings: list[dict[str, Any]] = []

    for item in outputs:
        output_path = Path(item["output_path"])
        if item.get("success") and output_path.is_file():
            base_item = by_key.get(("base", item["test_id"]))
            base_duration = None
            if base_item and base_item.get("metrics"):
                base_duration = base_item["metrics"].get("duration_seconds")
            item["metrics"] = analyze_output_metrics(output_path, item["text"], base_duration)
        else:
            item["metrics"] = {
                "exists": False,
                "duration_seconds": None,
                "file_size_mb": None,
                "mean_volume_db": None,
                "max_volume_db": None,
                "clipping_risk": None,
                "flags": [],
                "warnings": ["Cikti dosyasi olusmadi."],
            }

    for item in outputs:
        if item["variant"] == "base":
            continue
        if item.get("success") and item["metrics"].get("duration_seconds") is not None:
            base_item = by_key.get(("base", item["test_id"]))
            base_duration = (
                base_item.get("metrics", {}).get("duration_seconds")
                if base_item
                else None
            )
            if base_duration is not None:
                item["metrics"] = analyze_output_metrics(
                    Path(item["output_path"]),
                    item["text"],
                    base_duration,
                )

    for test in TEST_TEXTS:
        exp_a = by_key.get(("exp_a", test["test_id"]))
        exp_b = by_key.get(("exp_b", test["test_id"]))
        if not exp_a or not exp_b:
            continue
        dur_a = exp_a.get("metrics", {}).get("duration_seconds")
        dur_b = exp_b.get("metrics", {}).get("duration_seconds")
        if not isinstance(dur_a, (int, float)) or not isinstance(dur_b, (int, float)):
            continue
        diff = abs(dur_a - dur_b)
        max_duration = max(dur_a, dur_b)
        ratio = diff / max_duration if max_duration > 0 else 0.0
        if diff >= 1.0 and ratio >= 0.35:
            warnings.append(
                {
                    "test_id": test["test_id"],
                    "type": "duration_difference",
                    "message": (
                        "exp_a ve exp_b sureleri belirgin farkli; "
                        f"exp_a={dur_a:.3f}s, exp_b={dur_b:.3f}s"
                    ),
                }
            )

    return warnings


def create_blind_set(
    output_root: Path,
    outputs: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exp A/B dosyalarini rastgele A/B adlariyla blind klasorune kopyalar."""
    blind_dir = output_root / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    by_key = {
        (item["variant"], item["test_id"]): item
        for item in outputs
    }
    blind_entries: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for test in TEST_TEXTS:
        exp_a = by_key.get(("exp_a", test["test_id"]))
        exp_b = by_key.get(("exp_b", test["test_id"]))
        if not exp_a or not exp_b or not exp_a.get("success") or not exp_b.get("success"):
            warnings.append(
                {
                    "test_id": test["test_id"],
                    "type": "blind_pair_missing",
                    "message": "exp_a veya exp_b ciktisi eksik; blind A/B kopyasi olusturulmadi.",
                }
            )
            continue

        first_is_a = rng.choice([True, False])
        mapping = {
            "A": exp_a if first_is_a else exp_b,
            "B": exp_b if first_is_a else exp_a,
        }
        entry = {
            "test_id": test["test_id"],
            "A": {
                "variant": mapping["A"]["variant"],
                "source_path": mapping["A"]["output_path"],
                "blind_path": str(blind_dir / f"{test['test_id']}_A.wav"),
            },
            "B": {
                "variant": mapping["B"]["variant"],
                "source_path": mapping["B"]["output_path"],
                "blind_path": str(blind_dir / f"{test['test_id']}_B.wav"),
            },
        }

        shutil.copy2(mapping["A"]["output_path"], entry["A"]["blind_path"])
        shutil.copy2(mapping["B"]["output_path"], entry["B"]["blind_path"])
        blind_entries.append(entry)

    return blind_entries, warnings


def write_blind_key(timestamp: str, blind_entries: list[dict[str, Any]]) -> None:
    BLIND_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": timestamp,
        "warning": "Bu dosya dinleme puanlari girilmeden once acilmamalidir.",
        "entries": blind_entries,
    }
    BLIND_KEY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Blind key yazildi: {BLIND_KEY_PATH}")


def write_blind_scorecard(blind_entries: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    BLIND_SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing_by_test = {
        warning["test_id"]: warning["message"]
        for warning in warnings
        if warning.get("type") == "blind_pair_missing"
    }
    fieldnames = [
        "test_id",
        "A_naturalness",
        "A_similarity",
        "A_pronunciation",
        "A_human_likeness",
        "A_text_accuracy",
        "B_naturalness",
        "B_similarity",
        "B_pronunciation",
        "B_human_likeness",
        "B_text_accuracy",
        "preferred",
        "notes",
    ]
    blind_test_ids = {entry["test_id"] for entry in blind_entries}
    with BLIND_SCORECARD_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for test in TEST_TEXTS:
            writer.writerow(
                {
                    "test_id": test["test_id"],
                    "A_naturalness": "",
                    "A_similarity": "",
                    "A_pronunciation": "",
                    "A_human_likeness": "",
                    "A_text_accuracy": "",
                    "B_naturalness": "",
                    "B_similarity": "",
                    "B_pronunciation": "",
                    "B_human_likeness": "",
                    "B_text_accuracy": "",
                    "preferred": "",
                    "notes": "" if test["test_id"] in blind_test_ids else missing_by_test.get(test["test_id"], ""),
                }
            )
    print(f"Blind scorecard yazildi: {BLIND_SCORECARD_PATH}")


def build_report(
    timestamp: str,
    exp_a: Path,
    exp_b: Path,
    selected_checkpoints: dict[str, str | None],
    speaker_wav: Path | None,
    output_root: Path,
    outputs: list[dict[str, Any]],
    blind_entries: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": timestamp,
        "exp_a": str(exp_a),
        "exp_b": str(exp_b),
        "selected_checkpoints": selected_checkpoints,
        "speaker_wav": str(speaker_wav) if speaker_wav else None,
        "output_root": str(output_root),
        "test_texts": test_text_entries(),
        "generated_outputs": outputs,
        "objective_metrics": [
            {
                "variant": item["variant"],
                "test_id": item["test_id"],
                "output_path": item["output_path"],
                "metrics": item.get("metrics"),
            }
            for item in outputs
        ],
        "blind_entries": blind_entries,
        "blind_scorecard_path": str(BLIND_SCORECARD_PATH),
        "blind_key_path": str(BLIND_KEY_PATH),
        "warnings": warnings,
        "errors": errors,
        "notes": [
            "Bu script otomatik kalite karari vermez.",
            "Objektif metrikler yalnizca yardimci teknik sinyaldir.",
            "Nihai karar blind scorecard doldurulduktan sonra verilmelidir.",
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
        "# XTTS blind experiment comparison raporu",
        "",
        f"- Exp A: `{report['exp_a']}`",
        f"- Exp B: `{report['exp_b']}`",
        f"- Speaker wav: `{report['speaker_wav']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Blind scorecard: `{report['blind_scorecard_path']}`",
        f"- Blind key: `{report['blind_key_path']}`",
        "",
        "## Kullanım yönergesi",
        "",
        "1. Önce `blind/` klasöründeki A/B dosyalarını dinle.",
        "2. Dosya adlarına göre hangi experiment olduğunu bilmeden puanla.",
        "3. `experiment_blind_scorecard.csv` dosyasını doldur.",
        "4. Sonra `experiment_blind_key.json` ile hangi dosyanın hangi experiment olduğu açılabilir.",
        "",
        "Bu rapor otomatik kalite hükmü vermez. Objektif metrikler yalnızca yardımcı teknik sinyaldir; nihai karar blind dinleme puanlarıyla verilmelidir.",
        "",
        "## Seçilen checkpointler",
        "",
    ]
    for name, checkpoint_path in report["selected_checkpoints"].items():
        lines.append(f"- `{name}`: `{checkpoint_path}`")

    lines.extend(["", "## Test metinleri", ""])
    for item in report["test_texts"]:
        chunk_note = f", chunks={item['chunk_count']}" if item["chunking_used"] else ""
        lines.append(f"- `{item['test_id']}` ({item['category']}, {item['char_count']} karakter{chunk_note}): {item['text']}")

    lines.extend(["", "## Üretilen çıktılar", ""])
    lines.append("| variant | test | success | duration | flags | output |")
    lines.append("|---|---|---:|---:|---|---|")
    for item in report["generated_outputs"]:
        metrics = item.get("metrics") or {}
        duration = metrics.get("duration_seconds")
        duration_text = "-" if duration is None else f"{duration:.3f}"
        flags = ", ".join(metrics.get("flags") or [])
        lines.append(
            f"| {item['variant']} | {item['test_id']} | {item.get('success')} | "
            f"{duration_text} | {flags or '-'} | `{item['output_path']}` |"
        )

    if report["warnings"]:
        lines.extend(["", "## Uyarılar", ""])
        for warning in report["warnings"]:
            lines.append(f"- `{warning.get('test_id', '-')}` {warning.get('type', '')}: {warning.get('message', '')}")

    if report["errors"]:
        lines.extend(["", "## Hatalar", ""])
        for error in report["errors"]:
            lines.append(f"- `{error.get('variant', '-')}` `{error.get('test_id', '-')}`: {error.get('message', '')}")

    lines.extend(["", "## Notlar", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")

    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown rapor yazildi: {REPORT_MD_PATH}")


def write_reports(report: dict[str, Any]) -> None:
    write_json_report(report)
    write_markdown_report(report)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iki fine-tuning experiment icin blind A/B karsilastirma seti olusturur."
    )
    parser.add_argument("--exp-a", required=True, help="Birinci experiment klasoru.")
    parser.add_argument("--exp-b", required=True, help="Ikinci experiment klasoru.")
    parser.add_argument(
        "--speaker-wav",
        default=None,
        help="Opsiyonel speaker wav. Verilmezse mevcut fallback sirasi kullanilir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    timestamp = timestamp_slug()
    output_root = OUTPUT_ROOT / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    blind_entries: list[dict[str, Any]] = []
    selected_checkpoints: dict[str, str | None] = {
        "base": None,
        "exp_a": None,
        "exp_b": None,
    }
    exp_a = Path(args.exp_a)
    exp_b = Path(args.exp_b)
    speaker_wav: Path | None = None

    try:
        exp_a = resolve_experiment_path(args.exp_a)
        exp_b = resolve_experiment_path(args.exp_b)
        base_files_a = resolve_base_files(exp_a)
        base_files_b = resolve_base_files(exp_b)
        checkpoint_a = find_selected_checkpoint(exp_a).resolve()
        checkpoint_b = find_selected_checkpoint(exp_b).resolve()
        speaker_wav = resolve_speaker_wav(exp_a, args.speaker_wav)

        selected_checkpoints = {
            "base": str(base_files_a["model.pth"].resolve()),
            "exp_a": str(checkpoint_a),
            "exp_b": str(checkpoint_b),
        }

        print("Kullanilan checkpoint yolları:")
        for name, checkpoint_path in selected_checkpoints.items():
            print(f"- {name}: {checkpoint_path}")
        print(f"Kullanilan speaker_wav: {speaker_wav}")
        print(f"Output root: {output_root}")

        config_class, model_class = import_xtts_api()
        variant_specs = [
            ("base", base_files_a["model.pth"].resolve(), base_files_a),
            ("exp_a", checkpoint_a, base_files_a),
            ("exp_b", checkpoint_b, base_files_b),
        ]

        for variant_name, checkpoint_path, base_files in variant_specs:
            try:
                loaded_variant = load_variant_model(
                    variant_name=variant_name,
                    checkpoint_path=checkpoint_path,
                    base_files=base_files,
                    config_class=config_class,
                    model_class=model_class,
                )
                outputs.extend(
                    generate_variant_outputs(
                        variant=loaded_variant,
                        output_root=output_root,
                        speaker_wav=speaker_wav,
                        errors=errors,
                    )
                )
            except Exception as exc:
                message = str(exc)
                print(f"UYARI: {variant_name} modeli hazirlanamadi: {message}", file=sys.stderr)
                errors.append(
                    {
                        "variant": variant_name,
                        "test_id": "",
                        "message": message,
                    }
                )

        warnings.extend(add_metrics_and_warnings(outputs))
        blind_entries, blind_warnings = create_blind_set(
            output_root=output_root,
            outputs=outputs,
            rng=random.SystemRandom(),
        )
        warnings.extend(blind_warnings)
        write_blind_key(timestamp, blind_entries)
        write_blind_scorecard(blind_entries, warnings)

        report = build_report(
            timestamp=timestamp,
            exp_a=exp_a,
            exp_b=exp_b,
            selected_checkpoints=selected_checkpoints,
            speaker_wav=speaker_wav,
            output_root=output_root,
            outputs=outputs,
            blind_entries=blind_entries,
            warnings=warnings,
            errors=errors,
        )
        write_reports(report)
        print("Blind experiment comparison tamamlandi.")
        return 0
    except (CompareError, EvaluationError) as exc:
        errors.append({"variant": "", "test_id": "", "message": str(exc)})
        report = build_report(
            timestamp=timestamp,
            exp_a=exp_a,
            exp_b=exp_b,
            selected_checkpoints=selected_checkpoints,
            speaker_wav=speaker_wav,
            output_root=output_root,
            outputs=outputs,
            blind_entries=blind_entries,
            warnings=warnings,
            errors=errors,
        )
        write_reports(report)
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        errors.append(
            {
                "variant": "",
                "test_id": "",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        report = build_report(
            timestamp=timestamp,
            exp_a=exp_a,
            exp_b=exp_b,
            selected_checkpoints=selected_checkpoints,
            speaker_wav=speaker_wav,
            output_root=output_root,
            outputs=outputs,
            blind_entries=blind_entries,
            warnings=warnings,
            errors=errors,
        )
        write_reports(report)
        print(f"HATA: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
