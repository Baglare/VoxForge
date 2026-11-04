"""Voice quality benchmark run: preparation, blind listening set and report (VF-Q1).

Layout of one run (under ``outputs/voice_benchmark/<run_id>/``, Git-ignored)::

    manifest.json            run id, test set version, items + spoken text, reference info
    reference.wav            copy of the chosen reference (the profile itself is never touched)
    <engine>/tNN.wav         generated audio
    <engine>/results.json    written by scripts/benchmark/run_engine.py (timing, VRAM, versions, errors)
    evaluation/<engine>.json written by scripts/benchmark/evaluate.py (ASR transcript, speaker similarity)
    listening/               blind copies A/B/C... + blind_sheet.csv (to fill) + blind_key.json
    report.json / report.md  aggregated comparison

Decision thresholds are fixed here *before* any result exists, so the outcome
cannot be tuned after listening.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import statistics
import wave
from pathlib import Path
from typing import Any

from voxforge.benchmark.metrics import duration_check, score_transcript
from voxforge.benchmark.testset import TEST_ITEMS, TESTSET_VERSION
from voxforge.runs import SCHEMA_VERSION, new_run_id, utc_now_iso, write_json_atomic
from voxforge.studio.project import file_sha256

BASELINE = "xtts"
ENGINES: dict[str, dict[str, Any]] = {
    "xtts": {"label": "XTTS-v2 (mevcut)", "env": "main"},
    "chatterbox-v3": {"label": "Chatterbox Multilingual V3", "env": "chatterbox"},
    "voxcpm2": {"label": "VoxCPM2 (yalnız referans)", "env": "voxcpm"},
    "voxcpm2-ultimate": {"label": "VoxCPM2 (referans + transkript)", "env": "voxcpm", "needs_reference_text": True},
    "mock": {"label": "Mock (yalnız boru hattı denemesi)", "env": "main"},
}
HUMAN_FIELDS = ("naturalness", "similarity", "pronunciation")
SHEET_COLUMNS = ["item_id", "label", "naturalness", "similarity", "pronunciation", "extra_sound", "notes"]

# Önceden sabitlenen karar ölçütleri (docs/VOICE_QUALITY_BENCHMARK.md ile aynı).
MIN_HUMAN_GAIN = 0.3        # doğallık ve benzerlikte XTTS'e göre en az +0.3 (1-5 ölçek)
MAX_CER_LOSS = 0.02         # CER, XTTS'ten en fazla 2 puan kötü olabilir
MAX_RTF = 1.0               # RTX 4060 Laptop üzerinde gerçek zamandan yavaş olmamalı
MAX_EXTRA_SOUND_SHARE = 0.10  # dinleyicinin "metin dışı ses" işaretlediği çıktı oranı


class BenchmarkError(Exception):
    pass


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def prepare_run(root: Path, reference: Path, reference_text: str | None = None, seed: int = 1234) -> Path:
    """Create a run folder with a *copy* of the reference and the fixed test set."""
    reference = Path(reference)
    if not reference.is_file():
        raise BenchmarkError(f"Referans ses bulunamadı: {reference}")
    run_id = new_run_id()
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    copied = run_dir / "reference.wav"
    shutil.copy2(reference, copied)
    try:
        reference_seconds = round(wav_seconds(copied), 2)
    except (wave.Error, EOFError):
        reference_seconds = None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "voice_benchmark",
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "testset_version": TESTSET_VERSION,
        "seed": seed,
        "reference": {
            "source_name": reference.name,
            "path": str(copied),
            "sha256": file_sha256(copied),
            "seconds": reference_seconds,
            "transcript": (reference_text or "").strip() or None,
        },
        "items": [
            {"item_id": item.item_id, "category": item.category, "text": item.text, "spoken": item.spoken, "focus": item.focus}
            for item in TEST_ITEMS
        ],
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    return run_dir


def read_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(Path(run_dir) / "manifest.json")
    if not manifest or manifest.get("kind") != "voice_benchmark":
        raise BenchmarkError(f"Geçerli bir benchmark klasörü değil: {run_dir}")
    return manifest


def engine_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    found = {}
    for engine in ENGINES:
        data = load_json(Path(run_dir) / engine / "results.json")
        if data:
            found[engine] = data
    return found


# ---- blind listening -------------------------------------------------------
def create_blind_set(run_dir: Path, seed: int | None = None) -> Path:
    """Copy every successful output per item under shuffled labels; never overwrites a filled sheet."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    listening = run_dir / "listening"
    sheet = listening / "blind_sheet.csv"
    if sheet.exists():
        raise BenchmarkError(f"Dinleme formu zaten var, üzerine yazılmadı: {sheet}")
    results = {name: data for name, data in engine_results(run_dir).items() if name != "mock"} or engine_results(run_dir)
    if not results:
        raise BenchmarkError("Henüz hiçbir motor sonucu yok.")
    rng = random.Random(manifest["seed"] if seed is None else seed)
    listening.mkdir(exist_ok=True)
    key: dict[str, dict[str, str]] = {}
    rows = []
    for item in manifest["items"]:
        outputs = []
        for engine, data in results.items():
            for entry in data.get("items", []):
                if entry["item_id"] == item["item_id"] and entry.get("ok"):
                    outputs.append((engine, Path(entry["path"])))
        rng.shuffle(outputs)
        key[item["item_id"]] = {}
        for index, (engine, path) in enumerate(outputs):
            label = chr(ord("A") + index)
            shutil.copy2(path, listening / f"{item['item_id']}_{label}.wav")
            key[item["item_id"]][label] = engine
            rows.append({"item_id": item["item_id"], "label": label})
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SHEET_COLUMNS})
    write_json_atomic(
        listening / "blind_key.json",
        {"run_id": manifest["run_id"], "warning": "Puanlama bitmeden açmayın.", "items": key},
    )
    return sheet


def read_human_scores(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Unblind the filled sheet: engine -> field -> scores (1-5); extra_sound as 0/1."""
    listening = Path(run_dir) / "listening"
    key = load_json(listening / "blind_key.json")
    sheet = listening / "blind_sheet.csv"
    if not key or not sheet.is_file():
        return {}
    scores: dict[str, dict[str, list[float]]] = {}
    with sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            engine = key["items"].get(row["item_id"], {}).get(row["label"])
            if engine is None:
                raise BenchmarkError(f"Formda anahtarla eşleşmeyen satır: {row['item_id']} {row['label']}")
            bucket = scores.setdefault(engine, {field: [] for field in (*HUMAN_FIELDS, "extra_sound")})
            for field in HUMAN_FIELDS:
                value = (row.get(field) or "").strip()
                if value:
                    number = float(value.replace(",", "."))
                    if not 1 <= number <= 5:
                        raise BenchmarkError(f"{row['item_id']} {row['label']} {field}: 1-5 arası olmalı")
                    bucket[field].append(number)
            extra = (row.get("extra_sound") or "").strip().lower()
            if extra:
                bucket["extra_sound"].append(1.0 if extra in {"1", "evet", "e", "x", "var"} else 0.0)
    return scores


# ---- report ----------------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def build_report(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    spoken = {item["item_id"]: item["spoken"] for item in manifest["items"]}
    human = read_human_scores(run_dir)
    engines: dict[str, Any] = {}
    for engine, data in engine_results(run_dir).items():
        evaluation = load_json(run_dir / "evaluation" / f"{engine}.json") or {}
        eval_items = {entry["item_id"]: entry for entry in evaluation.get("items", [])}
        cers, insertions, sims, rtfs = [], [], [], []
        extra_auto = duration_flags = failures = 0
        per_item = []
        for entry in data.get("items", []):
            if not entry.get("ok"):
                failures += 1
                per_item.append({"item_id": entry["item_id"], "ok": False, "error": entry.get("error")})
                continue
            row: dict[str, Any] = {"item_id": entry["item_id"], "ok": True, "rtf": entry.get("rtf")}
            if entry.get("rtf") is not None:
                rtfs.append(entry["rtf"])
            check = duration_check(spoken[entry["item_id"]], entry.get("audio_seconds"))
            row.update(check)
            duration_flags += int(bool(check["too_short"] or check["too_long"]))
            evaluated = eval_items.get(entry["item_id"])
            if evaluated and evaluated.get("transcript") is not None:
                score = score_transcript(spoken[entry["item_id"]], evaluated["transcript"])
                row.update(score, transcript=evaluated["transcript"])
                cers.append(score["cer"])
                insertions.append(score["insertion_rate"])
                extra_auto += int(score["extra_speech_suspect"])
            if evaluated and evaluated.get("similarity") is not None:
                row["similarity"] = evaluated["similarity"]
                sims.append(evaluated["similarity"])
            per_item.append(row)
        scored = human.get(engine, {})
        engines[engine] = {
            "label": ENGINES.get(engine, {}).get("label", engine),
            "status": data.get("status"),
            "items_total": len(data.get("items", [])),
            "failures": failures,
            "load_seconds": data.get("load_seconds"),
            "vram_peak_mb": data.get("vram_peak_mb"),
            "versions": data.get("versions"),
            "rtf_mean": _mean(rtfs),
            "cer_mean": _mean(cers),
            "cer_median": round(statistics.median(cers), 3) if cers else None,
            "insertion_rate_mean": _mean(insertions),
            "extra_speech_suspects": extra_auto,
            "duration_flags": duration_flags,
            "similarity_mean": _mean(sims),
            "human": {field: _mean(scored.get(field, [])) for field in HUMAN_FIELDS}
            | {"extra_sound_share": _mean(scored.get("extra_sound", [])), "ratings": len(scored.get("naturalness", []))},
            "evaluated": bool(evaluation),
            "items": per_item,
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "testset_version": manifest["testset_version"],
        "generated_at": utc_now_iso(),
        "baseline": BASELINE,
        "engines": engines,
        "verdicts": {name: verdict(stats, engines.get(BASELINE)) for name, stats in engines.items() if name != BASELINE},
    }
    write_json_atomic(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def verdict(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the pre-registered thresholds. Missing evidence is reported, never assumed."""
    reasons, missing = [], []
    if baseline is None:
        return {"decision": "eksik", "reasons": ["XTTS referans sonucu yok."]}
    if candidate["failures"] or candidate["status"] != "completed":
        reasons.append(f"{candidate['failures']} çıktı başarısız veya koşu tamamlanmadı")
    for field in ("naturalness", "similarity"):
        cand, base = candidate["human"][field], baseline["human"][field]
        if cand is None or base is None:
            missing.append(f"insan puanı ({field})")
        elif cand - base < MIN_HUMAN_GAIN:
            reasons.append(f"{field} kazancı {cand - base:+.2f} < +{MIN_HUMAN_GAIN}")
    if candidate["cer_mean"] is None or baseline["cer_mean"] is None:
        missing.append("ASR/CER")
    elif candidate["cer_mean"] - baseline["cer_mean"] > MAX_CER_LOSS:
        reasons.append(f"CER {candidate['cer_mean'] - baseline['cer_mean']:+.3f} kötüleşti")
    if candidate["rtf_mean"] is None:
        missing.append("RTF")
    elif candidate["rtf_mean"] > MAX_RTF:
        reasons.append(f"RTF {candidate['rtf_mean']:.2f} > {MAX_RTF}")
    share = candidate["human"]["extra_sound_share"]
    if share is not None and share > MAX_EXTRA_SOUND_SHARE:
        reasons.append(f"metin dışı ses oranı {share:.0%}")
    if reasons:
        return {"decision": "uygun değil", "reasons": reasons, "missing": missing}
    if missing:
        return {"decision": "eksik", "reasons": [], "missing": missing}
    return {"decision": "aday (VF-009 entegrasyonu için)", "reasons": [], "missing": []}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Ses kalitesi karşılaştırması — {report['run_id']}",
        "",
        f"Test seti `{report['testset_version']}`, referans motor `{report['baseline']}`. "
        "Otomatik ölçümler yardımcıdır; doğallık ve benzerlik kararı kör dinleme puanlarına dayanır.",
        "",
        "| Motor | Başarısız | RTF | VRAM (MB) | CER | Ekleme oranı | Metin dışı şüphe | Süre uyarısı | Benzerlik (ECAPA) "
        "| Doğallık | Benzerlik (insan) | Telaffuz | Metin dışı ses (insan) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stats in report["engines"].values():
        human = stats["human"]
        extra = human["extra_sound_share"]
        lines.append(
            f"| {stats['label']} | {stats['failures']}/{stats['items_total']} | {_fmt(stats['rtf_mean'], 2)} "
            f"| {_fmt(stats['vram_peak_mb'])} | {_fmt(stats['cer_mean'])} | {_fmt(stats['insertion_rate_mean'])} "
            f"| {stats['extra_speech_suspects']} | {stats['duration_flags']} | {_fmt(stats['similarity_mean'])} "
            f"| {_fmt(human['naturalness'], 2)} | {_fmt(human['similarity'], 2)} | {_fmt(human['pronunciation'], 2)} "
            f"| {'—' if extra is None else f'{extra:.0%}'} |"
        )
    lines += ["", "## Karar (önceden sabitlenen ölçütler)", ""]
    lines.append(
        f"Aday olmak için: kör dinlemede doğallık ve benzerlikte XTTS'e göre en az +{MIN_HUMAN_GAIN}, "
        f"CER'de en fazla +{MAX_CER_LOSS:.2f}, RTF ≤ {MAX_RTF}, metin dışı ses ≤ %{int(MAX_EXTRA_SOUND_SHARE * 100)}, hatasız koşu."
    )
    lines.append("")
    for name, result in report["verdicts"].items():
        detail = "; ".join(result.get("reasons", []))
        missing = ", ".join(result.get("missing", []))
        text = f"- **{report['engines'][name]['label']}**: {result['decision']}"
        if detail:
            text += f" — {detail}"
        if missing:
            text += f" (eksik: {missing})"
        lines.append(text)
    return "\n".join(lines) + "\n"
