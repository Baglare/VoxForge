"""Voice Takes (VF-018): alternatives of one utterance, user selection and undo.

A *take group* belongs to one unit of text (a Playground generation today, a Studio
segment later) and lives in ``<root>/<group_id>.json``:

- ``recipe``: everything needed to synthesise the unit again except the seed (engine
  variant, profile/reference, controls, preset, normalisation, device ...);
- ``takes``: every take ever made (original + alternatives) with its seed, settings,
  audio path, quality report and speaker similarity. Takes are never removed and their
  WAVs are never deleted, so no take is lost;
- ``selected`` + ``selection_history``: the user's choice; ``revert`` goes back to the
  previous choice (history is append-only, the revert itself is recorded too).

``TakeDirector`` makes 2-3 alternatives with *different seeds* and the same recipe through
a ``Synthesizer`` callable, checks each one with the ``QualityDirector`` and ranks them.
It recommends but never selects on its own: only the user changes ``selected``.
"""

from __future__ import annotations

import json
import random
import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from voxforge.playground.rhythm import split_sentences
from voxforge.quality.checks import CONFIDENCES, VERDICT_LABELS, Finding, _verdict
from voxforge.quality.director import QualityDirector
from voxforge.runs import utc_now_iso, write_json_atomic

SCHEMA_VERSION = 1
GROUP_ID_PATTERN = r"[A-Za-z0-9_.-]{1,160}"
MIN_ALTERNATIVES, MAX_ALTERNATIVES = 1, 3
DEFAULT_ALTERNATIVES = 2
IDENTITY_DROP = 0.05  # orijinal take'e göre ECAPA düşüşü: kimlik kayması şüphesi
IDENTITY_DROP_STRONG = 0.10
VERDICT_ORDER = {"ok": 0, "suspect": 1, "problem": 2, "unchecked": 1}

# recipe + seed -> {"id", "wav", "seed", ...} (Playground: PlaygroundService.generate kaydı)
Synthesizer = Callable[[dict[str, Any], int], dict[str, Any]]


class TakeError(ValueError):
    """User-facing take problem."""


def derive_seeds(base: int, existing: set[int], count: int) -> list[int]:
    """Distinct new seeds, reproducible from the base seed and how many takes exist."""
    rng = random.Random(int(base) * 7919 + len(existing))
    seeds: list[int] = []
    while len(seeds) < count:
        seed = rng.randint(0, 2**31 - 1)
        if seed not in existing and seed not in seeds:
            seeds.append(seed)
    return seeds


def rank_key(take: dict[str, Any]) -> tuple[Any, ...]:
    """Lower is better: verdict, confident major findings, CER, then voice similarity."""
    report = take.get("quality") or {}
    findings = report.get("findings") or []
    metrics = report.get("metrics") or {}
    major = sum(1 for f in findings if f["severity"] == "major" and f["confidence"] != "low")
    minor = sum(1 for f in findings if f["severity"] == "minor" and f["confidence"] != "low")
    cer = metrics.get("cer")
    similarity = take.get("similarity")
    return (
        VERDICT_ORDER.get(report.get("verdict", "unchecked"), 1),
        major,
        minor,
        cer if isinstance(cer, (int, float)) else 1.0,
        -(similarity if isinstance(similarity, (int, float)) else 0.0),
    )


class TakeStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()

    def _path(self, group_id: str) -> Path:
        if not re.fullmatch(GROUP_ID_PATTERN, group_id or ""):
            raise TakeError(f"Geçersiz take grubu: {group_id!r}")
        return self.root / f"{group_id}.json"

    def exists(self, group_id: str) -> bool:
        return self._path(group_id).is_file()

    def load(self, group_id: str) -> dict[str, Any]:
        path = self._path(group_id)
        if not path.is_file():
            raise TakeError(f"Take grubu bulunamadı: {group_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, group: dict[str, Any]) -> dict[str, Any]:
        group["updated_at"] = utc_now_iso()
        write_json_atomic(self._path(group["group_id"]), group)
        return group

    def list(self) -> list[dict[str, Any]]:
        groups = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                groups.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return groups

    def create(
        self,
        group_id: str,
        text: str,
        spoken_text: str,
        recipe: dict[str, Any],
        source: dict[str, Any],
        original: dict[str, Any] | None = None,
        parent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.exists(group_id):
                return self.load(group_id)
            now = utc_now_iso()
            group: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "group_id": group_id,
                "created_at": now,
                "text": text,
                "spoken_text": spoken_text,
                "recipe": recipe,
                "source": source,
                "parent": parent,
                "takes": [],
                "selected": None,
                "selection_history": [],
                "recommended": None,
            }
            if original is not None:
                take = {**original, "take_id": "t1", "origin": "original", "added_at": now}
                group["takes"].append(take)
                group["selected"] = "t1"
                group["selection_history"].append({"take_id": "t1", "at": now, "reason": "original"})
            return self.save(group)

    def add_take(self, group_id: str, take: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            group = self.load(group_id)
            take = {**take, "take_id": f"t{len(group['takes']) + 1}", "added_at": utc_now_iso()}
            group["takes"].append(take)
            group["recommended"] = recommend(group)
            self.save(group)
            return take

    def update_take(self, group_id: str, take_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            group = self.load(group_id)
            take = _find(group, take_id)
            take.update(fields)
            group["recommended"] = recommend(group)
            return self.save(group)

    def select(self, group_id: str, take_id: str, reason: str = "user") -> dict[str, Any]:
        with self._lock:
            group = self.load(group_id)
            _find(group, take_id)
            if group.get("selected") == take_id:
                return group
            group["selected"] = take_id
            group["selection_history"].append({"take_id": take_id, "at": utc_now_iso(), "reason": reason})
            return self.save(group)

    def revert(self, group_id: str) -> dict[str, Any]:
        """Back to the selection before the current one (recorded as a new history entry)."""
        with self._lock:
            group = self.load(group_id)
            current = group.get("selected")
            previous = _previous_selection(group["selection_history"], current)
            if previous is None:
                raise TakeError("Geri dönülecek önceki seçim yok.")
            group["selected"] = previous
            group["selection_history"].append({"take_id": previous, "at": utc_now_iso(), "reason": f"revert from {current}"})
            return self.save(group)


def selection_stack(history: list[dict[str, Any]]) -> list[str]:
    """Replay the history: a selection pushes, a revert pops back to the previous choice."""
    stack: list[str] = []
    for entry in history:
        if str(entry.get("reason", "")).startswith("revert"):
            if stack:
                stack.pop()
            if not stack or stack[-1] != entry["take_id"]:
                stack.append(entry["take_id"])
        else:
            stack.append(entry["take_id"])
    return stack


def _previous_selection(history: list[dict[str, Any]], current: str | None) -> str | None:
    stack = selection_stack(history)
    if stack and stack[-1] == current:
        stack.pop()
    return stack[-1] if stack else None


def _find(group: dict[str, Any], take_id: str) -> dict[str, Any]:
    for take in group["takes"]:
        if take["take_id"] == take_id:
            return take
    raise TakeError(f"Take bulunamadı: {take_id}")


def recommend(group: dict[str, Any]) -> str | None:
    """Best checked take, or None when nothing is checked."""
    checked = [t for t in group.get("takes", []) if t.get("quality")]
    if not checked:
        return None
    return min(checked, key=rank_key)["take_id"]


def apply_identity_check(report: dict[str, Any], similarity: float | None, original_similarity: float | None) -> dict[str, Any]:
    """Add a speaker-similarity finding when a take sounds less like the voice than the original."""
    if not isinstance(similarity, (int, float)) or not isinstance(original_similarity, (int, float)):
        return report
    drop = round(original_similarity - similarity, 4)
    if drop <= IDENTITY_DROP:
        return report
    finding = Finding(
        "identity_drift",
        "minor",
        "medium" if drop > IDENTITY_DROP_STRONG else "low",
        "speaker",
        f"Referans sese benzerlik orijinal take'ten {drop:.3f} düşük ({similarity:.3f} / {original_similarity:.3f}).",
        evidence={"similarity": similarity, "original_similarity": original_similarity},
    )
    findings = [*report.get("findings", []), {**asdict(finding), "label": finding.label}]
    rebuilt = [Finding(**{k: v for k, v in item.items() if k != "label"}) for item in findings]
    verdict = _verdict(rebuilt)
    return {**report, "findings": findings, "verdict": verdict, "verdict_label": VERDICT_LABELS[verdict], "retake_recommended": verdict == "problem"}


class TakeDirector:
    """Generate, check and rank alternatives; selection stays with the user."""

    def __init__(self, store: TakeStore, quality: QualityDirector, synthesize: Synthesizer) -> None:
        self.store = store
        self.quality = quality
        self.synthesize = synthesize

    def check_take(self, group: dict[str, Any], take: dict[str, Any]) -> dict[str, Any]:
        recipe = group["recipe"]
        reference = recipe.get("reference")
        report = self.quality.check(
            Path(take["wav"]), take.get("spoken_text") or group["spoken_text"], take.get("intended_pauses"),
            Path(reference) if reference else None,
        )
        similarity = report["metrics"].get("similarity")
        original = next((t for t in group["takes"] if t.get("origin") == "original"), None)
        if original is not None and original["take_id"] != take.get("take_id"):
            report = apply_identity_check(report, similarity, original.get("similarity"))
        return {"quality": report, "similarity": similarity}

    def alternatives(self, group_id: str, count: int = DEFAULT_ALTERNATIVES, check: bool = True) -> tuple[dict[str, Any], list[str]]:
        """Make ``count`` (1-3) new takes with fresh seeds; one failing take does not stop the rest."""
        count = int(count)
        if not MIN_ALTERNATIVES <= count <= MAX_ALTERNATIVES:
            raise TakeError(f"Alternatif sayısı {MIN_ALTERNATIVES}-{MAX_ALTERNATIVES} olmalı.")
        group = self.store.load(group_id)
        existing = {int(t["seed"]) for t in group["takes"] if t.get("seed") is not None}
        base = int(group["recipe"].get("seed") or 0)
        errors: list[str] = []
        for seed in derive_seeds(base, existing, count):
            try:
                record = self.synthesize(group["recipe"], seed)
                take = take_from_record(record, origin="alternative")
                if check:
                    group = self.store.load(group_id)
                    take.update(self.check_take(group, take))
                self.store.add_take(group_id, take)
            except Exception as exc:  # noqa: BLE001 - bir alternatif diğerlerini durdurmaz
                errors.append(f"tohum {seed}: {exc}")
        return self.store.load(group_id), errors


def take_from_record(record: dict[str, Any], origin: str) -> dict[str, Any]:
    """Generation record (Playground/Studio) -> take entry with the settings that made it."""
    settings = record.get("settings") or {}
    return {
        "origin": origin,
        "record_id": record.get("id"),
        "wav": record.get("wav"),
        "seed": settings.get("seed", record.get("seed")),
        "engine": record.get("engine"),
        "model": record.get("model"),
        "variant": record.get("variant"),
        "settings": {
            "label": settings.get("label"),
            "applied": settings.get("applied"),
            "controls": settings.get("controls"),
            "preset": settings.get("preset"),
            "normalize": settings.get("normalize"),
        },
        "spoken_text": record.get("spoken_text"),
        "intended_pauses": (record.get("join") or {}).get("pauses"),
        "timing": {k: (record.get("timing") or {}).get(k) for k in ("generation_seconds", "audio_seconds", "rtf")},
        "vram_peak_mb": (record.get("vram") or {}).get("generation_peak_mb"),
        "created_at": record.get("created_at"),
    }


def sentence_texts(text: str, spoken_text: str) -> tuple[list[str], bool]:
    """Sentences of the original text when they line up with the spoken ones (else spoken, already normalised)."""
    original, spoken = split_sentences(text), split_sentences(spoken_text)
    if len(original) == len(spoken):
        return original, True
    return spoken, False


__all__ = [
    "CONFIDENCES",
    "DEFAULT_ALTERNATIVES",
    "MAX_ALTERNATIVES",
    "TakeDirector",
    "TakeError",
    "TakeStore",
    "derive_seeds",
    "rank_key",
    "recommend",
    "sentence_texts",
    "take_from_record",
]
