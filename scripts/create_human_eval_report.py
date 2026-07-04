# -*- coding: utf-8 -*-
"""Matrix evaluation ciktilari icin manuel dinleme scorecard raporu olusturur."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
SCORECARD_CSV_PATH = REPORT_DIR / "human_eval_scorecard.csv"
SUMMARY_JSON_PATH = REPORT_DIR / "human_eval_summary.json"
SUMMARY_MD_PATH = REPORT_DIR / "human_eval_summary.md"

EXPECTED_VARIANTS = ["base", "best_model", "best_model_72", "checkpoint_71"]
METRIC_COLUMNS = [
    "naturalness",
    "similarity",
    "pronunciation",
    "human_likeness",
    "text_accuracy",
]
CSV_COLUMNS = [
    "variant",
    *METRIC_COLUMNS,
    "total_score",
    "notes",
]

DEFAULT_SCORES: list[dict[str, Any]] = [
    {
        "variant": "base",
        "naturalness": 2,
        "similarity": 3,
        "pronunciation": 2,
        "human_likeness": 3,
        "text_accuracy": 3,
        "notes": "",
    },
    {
        "variant": "best_model",
        "naturalness": 2,
        "similarity": 3,
        "pronunciation": 3,
        "human_likeness": 3,
        "text_accuracy": 3,
        "notes": "",
    },
    {
        "variant": "best_model_72",
        "naturalness": 2,
        "similarity": 2,
        "pronunciation": 2,
        "human_likeness": 2,
        "text_accuracy": 2,
        "notes": "",
    },
    {
        "variant": "checkpoint_71",
        "naturalness": 3,
        "similarity": 3,
        "pronunciation": 3,
        "human_likeness": 3,
        "text_accuracy": 4,
        "notes": "Bazı kayıtlarda cümle erken kesiliyor gibi.",
    },
]

GENERAL_CONCLUSION = (
    "Fine-tuning pipeline teknik olarak başarılı. Kalite artışı sınırlı. "
    "best_model base'e göre küçük iyileşme gösteriyor. "
    "checkpoint_71 daha yüksek puan alsa da erken kesilme sorunu nedeniyle güvenilir değil. "
    "Daha fazla veri, daha iyi referans kayıt ve inference ayarı önerilir."
)


class HumanEvalError(RuntimeError):
    """Kullaniciya sade hata yazmak icin beklenen script hatasi."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuned XTTS matrix ciktilari icin manuel puan raporu olusturur."
    )
    parser.add_argument(
        "--matrix-root",
        required=True,
        help="Matrix output klasoru. Ornek: outputs/finetuned_eval/matrix/<timestamp>",
    )
    parser.add_argument(
        "--use-default-scores",
        action="store_true",
        help="Ilk manuel puanlari kullanarak CSV, JSON ve Markdown raporu olustur.",
    )
    return parser.parse_args(argv)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_matrix_root(matrix_root: Path) -> None:
    if not matrix_root.exists():
        raise HumanEvalError(f"Matrix root bulunamadi: {matrix_root}")
    if not matrix_root.is_dir():
        raise HumanEvalError(f"Matrix root klasor degil: {matrix_root}")


def discover_variants(matrix_root: Path) -> list[str]:
    discovered = [path.name for path in matrix_root.iterdir() if path.is_dir()]
    ordered = [variant for variant in EXPECTED_VARIANTS if variant in discovered]
    extras = sorted(variant for variant in discovered if variant not in EXPECTED_VARIANTS)
    return ordered + extras if ordered or extras else EXPECTED_VARIANTS


def blank_rows_for_variants(variants: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        row = {column: "" for column in CSV_COLUMNS}
        row["variant"] = variant
        rows.append(row)
    return rows


def read_existing_scorecard() -> list[dict[str, Any]]:
    if not SCORECARD_CSV_PATH.is_file():
        return []

    with SCORECARD_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        if reader.fieldnames != CSV_COLUMNS:
            raise HumanEvalError(
                f"Mevcut scorecard basligi beklenen formatta degil: {SCORECARD_CSV_PATH}"
            )
        return [dict(row) for row in reader]


def parse_score(value: Any, variant: str, metric: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None

    try:
        score = int(str(value).strip())
    except ValueError as exc:
        raise HumanEvalError(f"{variant} icin {metric} puani sayi degil: {value}") from exc

    if score < 1 or score > 5:
        raise HumanEvalError(f"{variant} icin {metric} puani 1-5 araliginda olmali: {score}")
    return score


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        normalized = {column: row.get(column, "") for column in CSV_COLUMNS}
        variant = str(normalized["variant"]).strip()
        if not variant:
            raise HumanEvalError("Scorecard icinde bos variant satiri var.")
        normalized["variant"] = variant

        total = 0
        score_count = 0
        for metric in METRIC_COLUMNS:
            score = parse_score(normalized.get(metric), variant, metric)
            if score is None:
                normalized[metric] = ""
                continue
            normalized[metric] = score
            total += score
            score_count += 1

        normalized["total_score"] = total if score_count == len(METRIC_COLUMNS) else ""
        normalized["notes"] = str(normalized.get("notes", "")).strip()
        normalized_rows.append(normalized)

    return normalized_rows


def write_scorecard_csv(rows: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with SCORECARD_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def scored_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row.get("total_score"), int)]


def variant_dirs_missing(matrix_root: Path, variants: list[str]) -> list[str]:
    return [variant for variant in variants if not (matrix_root / variant).is_dir()]


def build_base_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = {row["variant"]: row for row in scored_rows(rows)}
    base = scored.get("base")
    if base is None:
        return []

    base_total = base["total_score"]
    comparisons: list[dict[str, Any]] = []
    for variant in EXPECTED_VARIANTS:
        if variant == "base" or variant not in scored:
            continue
        total = scored[variant]["total_score"]
        comparisons.append(
            {
                "variant": variant,
                "total_score": total,
                "difference_from_base": total - base_total,
            }
        )
    return comparisons


def pick_total_extremes(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    scored = scored_rows(rows)
    if not scored:
        return None, None

    highest = max(scored, key=lambda row: int(row["total_score"]))
    lowest = min(scored, key=lambda row: int(row["total_score"]))
    return (
        {"variant": highest["variant"], "total_score": highest["total_score"]},
        {"variant": lowest["variant"], "total_score": lowest["total_score"]},
    )


def checkpoint_71_warning(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        if row["variant"] == "checkpoint_71":
            note = str(row.get("notes", "")).strip()
            if note:
                return note
            return "checkpoint_71 için erken kesilme notu yok; dinleme sırasında özellikle kontrol edin."
    return "checkpoint_71 varyantı scorecard içinde bulunmuyor."


def build_summary(matrix_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [row["variant"] for row in rows]
    highest, lowest = pick_total_extremes(rows)

    return {
        "matrix_root": str(matrix_root),
        "scorecard_csv": str(SCORECARD_CSV_PATH),
        "highest_total_score": highest,
        "lowest_total_score": lowest,
        "base_comparison": build_base_comparison(rows),
        "checkpoint_71_warning": checkpoint_71_warning(rows),
        "overall_result": GENERAL_CONCLUSION,
        "rows": rows,
        "missing_variant_dirs": variant_dirs_missing(matrix_root, variants),
        "notes": [
            "Bu otomatik kalite ölçümü değildir.",
            "Ses benzerliği insan kulağıyla değerlendirilir.",
            "Sonuçlar küçük dataset ve deneysel fine-tuning bağlamında yorumlanmalıdır.",
            "Robotiklik alanında 1 robotik, 5 daha insan gibi kabul edilir.",
        ],
    }


def write_json_summary(summary: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def markdown_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value) if str(value) else "-"


def markdown_extreme(value: dict[str, Any] | None) -> str:
    if value is None:
        return "-"
    return f"`{value['variant']}` ({value['total_score']})"


def write_markdown_summary(summary: dict[str, Any]) -> None:
    highest = summary["highest_total_score"]
    lowest = summary["lowest_total_score"]

    lines = [
        "# Human evaluation scorecard",
        "",
        f"- Matrix root: `{summary['matrix_root']}`",
        f"- Scorecard CSV: `{summary['scorecard_csv']}`",
        "",
        "## Özet",
        "",
        f"- En yüksek toplam puan: {markdown_extreme(highest)}",
        f"- En düşük toplam puan: {markdown_extreme(lowest)}",
        f"- checkpoint_71 uyarısı: {summary['checkpoint_71_warning']}",
        "",
        "## Base farkları",
        "",
    ]

    if summary["base_comparison"]:
        for item in summary["base_comparison"]:
            diff = item["difference_from_base"]
            sign = "+" if diff > 0 else ""
            lines.append(
                f"- `{item['variant']}`: {item['total_score']} toplam puan, base farkı {sign}{diff}"
            )
    else:
        lines.append("- Base farkı hesaplanamadı; base veya fine-tuned puanları eksik.")

    lines.extend(
        [
            "",
            "## Puan tablosu",
            "",
            "| variant | naturalness | similarity | pronunciation | human_likeness | text_accuracy | total_score | notes |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in summary["rows"]:
        lines.append(
            "| {variant} | {naturalness} | {similarity} | {pronunciation} | "
            "{human_likeness} | {text_accuracy} | {total_score} | {notes} |".format(
                variant=markdown_value(row["variant"]),
                naturalness=markdown_value(row["naturalness"]),
                similarity=markdown_value(row["similarity"]),
                pronunciation=markdown_value(row["pronunciation"]),
                human_likeness=markdown_value(row["human_likeness"]),
                text_accuracy=markdown_value(row["text_accuracy"]),
                total_score=markdown_value(row["total_score"]),
                notes=markdown_value(row["notes"]).replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Genel sonuc",
            "",
            summary["overall_result"],
            "",
            "Bu rapor otomatik kalite ölçümü değildir. Ses benzerliği insan kulağıyla "
            "değerlendirilir ve sonuçlar küçük dataset ile deneysel fine-tuning bağlamında yorumlanmalıdır.",
        ]
    )

    if summary["missing_variant_dirs"]:
        lines.extend(["", "## Eksik varyant klasorleri", ""])
        for variant in summary["missing_variant_dirs"]:
            lines.append(f"- `{variant}`")

    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_rows(matrix_root: Path, use_default_scores: bool) -> list[dict[str, Any]]:
    if use_default_scores:
        return normalize_rows(DEFAULT_SCORES)

    existing_rows = read_existing_scorecard()
    if existing_rows:
        return normalize_rows(existing_rows)

    return normalize_rows(blank_rows_for_variants(discover_variants(matrix_root)))


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        matrix_root = resolve_project_path(args.matrix_root)
        validate_matrix_root(matrix_root)
        rows = load_rows(matrix_root, args.use_default_scores)
        summary = build_summary(matrix_root, rows)

        write_scorecard_csv(rows)
        write_json_summary(summary)
        write_markdown_summary(summary)

        print(f"Matrix root: {matrix_root}")
        print(f"Scorecard CSV yazildi: {SCORECARD_CSV_PATH}")
        print(f"JSON ozet yazildi: {SUMMARY_JSON_PATH}")
        print(f"Markdown ozet yazildi: {SUMMARY_MD_PATH}")
        print(f"Genel sonuc: {GENERAL_CONCLUSION}")
    except HumanEvalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
