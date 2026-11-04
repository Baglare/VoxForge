"""Voice quality benchmark run: preparation, blind listening set and report (VF-Q1).

Layout of one run (under ``outputs/voice_benchmark/<run_id>/``, Git-ignored)::

    manifest.json            run id, test set version, items + spoken text, reference info
    reference.wav            copy of the chosen reference (the profile itself is never touched)
    <engine>/tNN.wav         generated audio
    <engine>/results.json    written by scripts/benchmark/run_engine.py (timing, VRAM, versions, coverage)
    evaluation/<engine>.json written by scripts/benchmark/evaluate.py (ASR transcript, speaker similarity)
    listening_pilot/         optional blind set from a pilot (never used for the decision)
    listening/               blind set of the full run + blind_sheet.csv (to fill) + blind_key.json
    report.json / report.md  aggregated comparison

Decision rules are fixed here *before* any result exists:
- No final verdict unless the candidate and the XTTS baseline cover all 12 items.
- Missing evidence (human scores, extra-sound marks, ASR, similarity) is listed and
  yields "eksik kanıt", never a pass.
- Voice quality and generation performance are judged separately: a slower model
  is not rejected on speed alone; speed is reported next to the quality verdict.
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
LISTENING_DIR = "listening"
PILOT_LISTENING_DIR = "listening_pilot"

# Önceden sabitlenen KALİTE ölçütleri (docs/VOICE_QUALITY_BENCHMARK.md ile aynı).
MIN_HUMAN_GAIN = 0.3          # doğallık ve benzerlikte XTTS'e göre en az +0.3 (1-5 ölçek)
MAX_CER_LOSS = 0.02           # CER, XTTS'ten en fazla 0.02 kötü olabilir
MAX_EXTRA_SOUND_SHARE = 0.10  # dinleyicinin "metin dışı ses" işaretlediği çıktı oranı
# PERFORMANS yalnız raporlanır, kaliteyi elemez. Bu çalışmadaki RTF tanımı:
#   RTF = madde başına üretim süresi (duvar saati, model yükleme hariç, CUDA senkronlu) / üretilen sesin süresi.
#   Küçük = hızlı; 1'in altı gerçek zamandan hızlı. Kaynaklardaki RTF/hız rakamları farklı donanım ve
#   tanımlarla verildiği için bu ölçümle doğrudan karşılaştırılmaz.
REALTIME_RTF = 1.0


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


def coverage(manifest: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """Computed from the manifest (not trusted from results.json): complete only if every item was attempted."""
    expected = [item["item_id"] for item in manifest["items"]]
    items = results.get("items", [])
    attempted = {entry["item_id"] for entry in items}
    ok = sorted(entry["item_id"] for entry in items if entry.get("ok"))
    return {
        "expected": len(expected),
        "attempted": len(attempted & set(expected)),
        "ok": len(ok),
        "ok_items": ok,
        "missing": [item_id for item_id in expected if item_id not in attempted],
        "mode": "complete" if set(expected) <= attempted else "pilot",
    }


# ---- blind listening -------------------------------------------------------
def create_blind_set(run_dir: Path, seed: int | None = None, pilot: bool = False) -> Path:
    """Copy every successful output per item under shuffled labels; never overwrites a filled sheet.

    The decision set (``listening/``) requires complete runs. A pilot set goes to
    ``listening_pilot/`` so it never blocks or mixes with the full blind set.
    """
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    all_results = engine_results(run_dir)
    results = {name: data for name, data in all_results.items() if name != "mock"} or all_results
    if not results:
        raise BenchmarkError("Henüz hiçbir motor sonucu yok.")
    partial = sorted(name for name, data in results.items() if coverage(manifest, data)["mode"] == "pilot")
    if partial and not pilot:
        raise BenchmarkError(
            "Şu motorların koşusu tamamlanmadı (pilot): " + ", ".join(partial)
            + ". Tam koşuyu bitirin ya da yalnız deneme için --pilot kullanın (karara katılmaz)."
        )
    listening = run_dir / (PILOT_LISTENING_DIR if pilot else LISTENING_DIR)
    sheet = listening / "blind_sheet.csv"
    if sheet.exists():
        raise BenchmarkError(f"Dinleme formu zaten var, üzerine yazılmadı: {sheet}")
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
        {"run_id": manifest["run_id"], "pilot": pilot, "warning": "Puanlama bitmeden açmayın.", "items": key},
    )
    return sheet


def read_human_scores(run_dir: Path, folder: str = LISTENING_DIR) -> dict[str, dict[str, dict[str, float]]]:
    """Unblind a filled sheet: engine -> field -> {item_id: value}. extra_sound is 0/1."""
    listening = Path(run_dir) / folder
    key = load_json(listening / "blind_key.json")
    sheet = listening / "blind_sheet.csv"
    if not key or not sheet.is_file():
        return {}
    scores: dict[str, dict[str, dict[str, float]]] = {}
    with sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            engine = key["items"].get(row["item_id"], {}).get(row["label"])
            if engine is None:
                raise BenchmarkError(f"Formda anahtarla eşleşmeyen satır: {row['item_id']} {row['label']}")
            bucket = scores.setdefault(engine, {field: {} for field in (*HUMAN_FIELDS, "extra_sound")})
            for field in HUMAN_FIELDS:
                value = (row.get(field) or "").strip()
                if value:
                    number = float(value.replace(",", "."))
                    if not 1 <= number <= 5:
                        raise BenchmarkError(f"{row['item_id']} {row['label']} {field}: 1-5 arası olmalı")
                    bucket[field][row["item_id"]] = number
            extra = (row.get("extra_sound") or "").strip().lower()
            if extra:
                bucket["extra_sound"][row["item_id"]] = 1.0 if extra in {"1", "evet", "e", "x", "var"} else 0.0
    return scores


# ---- report ----------------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _engine_stats(run_dir: Path, manifest: dict[str, Any], engine: str, data: dict[str, Any],
                  human: dict[str, dict[str, float]]) -> dict[str, Any]:
    spoken = {item["item_id"]: item["spoken"] for item in manifest["items"]}
    cov = coverage(manifest, data)
    ok_items = set(cov["ok_items"])
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
        evaluated = eval_items.get(entry["item_id"], {})
        if evaluated.get("transcript") is not None:
            score = score_transcript(spoken[entry["item_id"]], evaluated["transcript"])
            row.update(score, transcript=evaluated["transcript"])
            cers.append(score["cer"])
            insertions.append(score["insertion_rate"])
            extra_auto += int(score["extra_speech_suspect"])
        if evaluated.get("similarity") is not None:
            row["similarity"] = evaluated["similarity"]
            sims.append(evaluated["similarity"])
        per_item.append(row)

    def rated(field: str) -> dict[str, float]:
        return {item: value for item, value in human.get(field, {}).items() if item in ok_items}

    evidence = {
        "outputs_ok": len(ok_items),
        "asr_transcripts": len(cers),
        "similarity": len(sims),
        **{f"human_{field}": len(rated(field)) for field in HUMAN_FIELDS},
        "human_extra_sound": len(rated("extra_sound")),
    }
    return {
        "label": ENGINES.get(engine, {}).get("label", engine),
        "coverage": {k: v for k, v in cov.items() if k != "ok_items"},
        "failures": failures,
        "evidence": evidence,
        "quality": {
            "cer_mean": _mean(cers),
            "cer_median": round(statistics.median(cers), 3) if cers else None,
            "insertion_rate_mean": _mean(insertions),
            "extra_speech_suspects": extra_auto,
            "duration_flags": duration_flags,
            "similarity_mean": _mean(sims),
            "human": {field: _mean(list(rated(field).values())) for field in HUMAN_FIELDS}
            | {"extra_sound_share": _mean(list(rated("extra_sound").values()))},
        },
        "performance": {
            "rtf_mean": _mean(rtfs),
            "rtf_median": round(statistics.median(rtfs), 3) if rtfs else None,
            "load_seconds": data.get("load_seconds"),
            "vram_peak_mb": data.get("vram_peak_mb"),
            "device": data.get("device"),
        },
        "versions": data.get("versions"),
        "items": per_item,
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    human = read_human_scores(run_dir)
    engines = {
        engine: _engine_stats(run_dir, manifest, engine, data, human.get(engine, {}))
        for engine, data in engine_results(run_dir).items()
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "testset_version": manifest["testset_version"],
        "generated_at": utc_now_iso(),
        "baseline": BASELINE,
        "pilot_listening_ignored": (run_dir / PILOT_LISTENING_DIR / "blind_sheet.csv").is_file(),
        "engines": engines,
        "verdicts": {
            name: {"quality": quality_verdict(stats, engines.get(BASELINE)), "performance": performance_note(stats)}
            for name, stats in engines.items()
            if name not in (BASELINE, "mock")
        },
    }
    write_json_atomic(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _missing_evidence(stats: dict[str, Any], who: str) -> list[str]:
    evidence, needed = stats["evidence"], stats["evidence"]["outputs_ok"]
    labels = {
        "asr_transcripts": "ASR transkripti",
        "similarity": "ECAPA benzerliği",
        "human_naturalness": "insan puanı: doğallık",
        "human_similarity": "insan puanı: benzerlik",
        "human_pronunciation": "insan puanı: telaffuz",
        "human_extra_sound": "metin dışı ses işareti",
    }
    return [f"{who} {label} {evidence[key]}/{needed}" for key, label in labels.items() if evidence[key] < needed]


def quality_verdict(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Voice quality only. Speed never decides here; missing evidence is never a pass."""
    if baseline is None:
        return {"decision": "eksik kanıt", "reasons": [], "missing": ["XTTS referans koşusu yok"]}
    partial = [
        f"{who} {stats['coverage']['attempted']}/{stats['coverage']['expected']} madde (eksik: {', '.join(stats['coverage']['missing'])})"
        for who, stats in (("aday", candidate), ("XTTS", baseline))
        if stats["coverage"]["mode"] != "complete"
    ]
    if partial:
        return {"decision": "pilot (nihai karar yok)", "reasons": [], "missing": partial}

    reasons: list[str] = []
    if candidate["failures"]:
        reasons.append(f"{candidate['failures']} çıktı başarısız (ör. bellek yetersizliği)")
    missing = _missing_evidence(candidate, "aday") + _missing_evidence(baseline, "XTTS")
    cq, bq = candidate["quality"], baseline["quality"]
    human_complete = not [m for m in missing if "insan puanı" in m]
    if human_complete:
        for field in ("naturalness", "similarity"):
            gain = cq["human"][field] - bq["human"][field]
            if gain < MIN_HUMAN_GAIN:
                reasons.append(f"{field} kazancı {gain:+.2f} < +{MIN_HUMAN_GAIN}")
    if not [m for m in missing if "ASR" in m]:
        loss = cq["cer_mean"] - bq["cer_mean"]
        if loss > MAX_CER_LOSS:
            reasons.append(f"CER {loss:+.3f} kötüleşti (sınır +{MAX_CER_LOSS})")
    if not [m for m in missing if "metin dışı ses" in m] and cq["human"]["extra_sound_share"] > MAX_EXTRA_SOUND_SHARE:
        reasons.append(f"metin dışı ses oranı {cq['human']['extra_sound_share']:.0%} > %{int(MAX_EXTRA_SOUND_SHARE * 100)}")
    if reasons:
        return {"decision": "uygun değil (kalite)", "reasons": reasons, "missing": missing}
    if missing:
        return {"decision": "eksik kanıt", "reasons": [], "missing": missing}
    return {"decision": "kalite adayı (VF-009 için)", "reasons": [], "missing": []}


def performance_note(stats: dict[str, Any]) -> dict[str, Any]:
    perf = stats["performance"]
    rtf = perf["rtf_mean"]
    if rtf is None:
        label = "ölçüm yok"
    elif rtf < REALTIME_RTF:
        label = "gerçek zamandan hızlı"
    else:
        label = "gerçek zamandan yavaş: canlı kullanım için yavaş, toplu üretim (Studio kuyruğu) için kabul edilebilir olabilir"
    return {"label": label, "rtf_mean": rtf, "vram_peak_mb": perf["vram_peak_mb"], "load_seconds": perf["load_seconds"]}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _count(stats: dict[str, Any], key: str) -> str:
    return f"{stats['evidence'][key]}/{stats['evidence']['outputs_ok']}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Ses kalitesi karşılaştırması — {report['run_id']}",
        "",
        f"Test seti `{report['testset_version']}`, referans motor `{report['baseline']}`. "
        "Otomatik ölçümler yardımcıdır; doğallık ve benzerlik kararı kör dinleme puanlarına dayanır. "
        "Eksik koşu veya eksik kanıt başarı sayılmaz.",
        "",
        "## Kapsam ve kanıt",
        "",
        "| Motor | Koşu | Başarılı | ASR | ECAPA | Doğallık | Benzerlik | Telaffuz | Metin dışı ses işareti |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stats in report["engines"].values():
        cov = stats["coverage"]
        mode = "tam" if cov["mode"] == "complete" else f"PİLOT ({cov['attempted']}/{cov['expected']})"
        lines.append(
            f"| {stats['label']} | {mode} | {cov['ok']}/{cov['expected']} "
            f"| {_count(stats, 'asr_transcripts')} | {_count(stats, 'similarity')} | {_count(stats, 'human_naturalness')} "
            f"| {_count(stats, 'human_similarity')} | {_count(stats, 'human_pronunciation')} | {_count(stats, 'human_extra_sound')} |"
        )
    lines += [
        "",
        "## Ses kalitesi",
        "",
        "| Motor | CER | Ekleme oranı | Metin dışı şüphe (ASR) | Süre uyarısı | Benzerlik (ECAPA) "
        "| Doğallık | Benzerlik (insan) | Telaffuz | Metin dışı ses (insan) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stats in report["engines"].values():
        q, h = stats["quality"], stats["quality"]["human"]
        extra = h["extra_sound_share"]
        lines.append(
            f"| {stats['label']} | {_fmt(q['cer_mean'])} | {_fmt(q['insertion_rate_mean'])} | {q['extra_speech_suspects']} "
            f"| {q['duration_flags']} | {_fmt(q['similarity_mean'])} | {_fmt(h['naturalness'], 2)} | {_fmt(h['similarity'], 2)} "
            f"| {_fmt(h['pronunciation'], 2)} | {'—' if extra is None else f'{extra:.0%}'} |"
        )
    lines += [
        "",
        "## Üretim performansı (kaliteden ayrı değerlendirilir)",
        "",
        "RTF bu çalışmada: madde başına üretim süresi (model yükleme hariç, CUDA senkronlu) / üretilen ses süresi. "
        "Küçük değer hızlıdır; 1'in altı gerçek zamandan hızlıdır. Model kartlarındaki RTF/hız rakamları başka donanım ve "
        "tanımlarla verildiği için bu tabloyla karşılaştırılmaz.",
        "",
        "| Motor | RTF ort. | RTF medyan | Model yükleme (sn) | VRAM tepe (MB) | Cihaz |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for stats in report["engines"].values():
        p = stats["performance"]
        lines.append(
            f"| {stats['label']} | {_fmt(p['rtf_mean'], 2)} | {_fmt(p['rtf_median'], 2)} | {_fmt(p['load_seconds'])} "
            f"| {_fmt(p['vram_peak_mb'])} | {p['device'] or '—'} |"
        )
    lines += ["", "## Karar (önceden sabitlenen kalite ölçütleri; performans ayrıca belirtilir)", ""]
    lines.append(
        f"Kalite adayı için: 12 maddelik tam koşu (aday ve XTTS), eksiksiz kanıt, başarısız çıktı yok, kör dinlemede doğallık ve "
        f"benzerlikte XTTS'e göre en az +{MIN_HUMAN_GAIN}, CER'de en fazla +{MAX_CER_LOSS:.2f}, "
        f"metin dışı ses ≤ %{int(MAX_EXTRA_SOUND_SHARE * 100)}. RTF kaliteyi elemez."
    )
    if report.get("pilot_listening_ignored"):
        lines.append("")
        lines.append("_Pilot dinleme formu (`listening_pilot/`) bulundu; karara katılmadı._")
    lines.append("")
    for name, verdict in report["verdicts"].items():
        quality, perf = verdict["quality"], verdict["performance"]
        text = f"- **{report['engines'][name]['label']}** — kalite: **{quality['decision']}**"
        if quality.get("reasons"):
            text += " (" + "; ".join(quality["reasons"]) + ")"
        text += f" · performans: {perf['label']}"
        if perf["rtf_mean"] is not None:
            text += f" (RTF {perf['rtf_mean']:.2f})"
        lines.append(text)
        if quality.get("missing"):
            lines.append("  - eksik: " + "; ".join(quality["missing"]))
    return "\n".join(lines) + "\n"
