"""Playground adapter for the Quality Director and Voice Takes (VF-018).

Connects ``PlaygroundService`` records to the engine/UI independent services in
``voxforge.quality``. Reports go to ``outputs/playground/quality/<record id>.json`` and
take groups to ``outputs/playground/takes/<group id>.json``; generation records and WAVs
stay where the Playground keeps them and are never deleted.

Automation rule (Issue #18): alternatives start on their own only for a ``problem``
verdict (a major finding with high confidence); a ``suspect`` result is shown and the
user decides. The best take is *recommended*, never selected automatically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from voxforge.playground.prosody import VoxControls
from voxforge.playground.service import PlaygroundError, PlaygroundService
from voxforge.quality.asr import AsrClient
from voxforge.quality.director import QualityDirector
from voxforge.quality.takes import (
    DEFAULT_ALTERNATIVES,
    TakeDirector,
    TakeError,
    TakeStore,
    sentence_texts,
    take_from_record,
)
from voxforge.runs import write_json_atomic


def recipe_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Everything needed to regenerate a Playground record except the seed."""
    settings = record.get("settings") or {}
    return {
        "kind": "playground",
        "text": record.get("text"),
        "variant": record.get("variant"),
        "engine": record.get("engine"),
        "profile": record.get("profile"),
        "reference": record.get("reference"),
        "reference_text": record.get("reference_text"),
        "normalize": settings.get("normalize", True),
        "device": record.get("device") if record.get("engine") != "mock" else "cuda",
        "controls": settings.get("controls"),
        "preset": settings.get("preset"),
        "seed": settings.get("seed"),
    }


class PlaygroundQuality:
    def __init__(self, service: PlaygroundService, asr: AsrClient | None = None, director: QualityDirector | None = None) -> None:
        self.service = service
        self.asr = asr if asr is not None else AsrClient(service.root / "logs")
        self.director = director or QualityDirector(self.asr)
        self.report_dir = service.root / "quality"
        self.store = TakeStore(service.root / "takes")
        self.takes = TakeDirector(self.store, self.director, self.synthesize)

    # -- generation through the Playground service --------------------------------------
    def synthesize(self, recipe: dict[str, Any], seed: int) -> dict[str, Any]:
        controls = VoxControls(**recipe["controls"]) if recipe.get("controls") else None
        reference = Path(recipe["reference"]) if recipe.get("reference") and not recipe.get("profile") else None
        return self.service.generate(
            recipe["text"],
            recipe["variant"],
            recipe.get("profile"),
            seed=seed,
            normalize=bool(recipe.get("normalize", True)),
            reference=reference,
            device=recipe.get("device") or "cuda",
            reference_text=recipe.get("reference_text"),
            controls=controls if controls is not None and recipe["variant"] in ("voxcpm2", "voxcpm2-ultimate") else None,
            preset_key=recipe.get("preset"),
        )

    # -- reports --------------------------------------------------------------------------
    def _record(self, record_id: str) -> dict[str, Any]:
        record = self.service.get(record_id)
        if record is None:
            raise PlaygroundError(f"Üretim bulunamadı: {record_id}")
        return record

    def report_path(self, record_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", record_id or ""):
            raise PlaygroundError(f"Geçersiz kayıt: {record_id!r}")
        return self.report_dir / f"{record_id}.json"

    def report(self, record_id: str) -> dict[str, Any] | None:
        path = self.report_path(record_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def check(self, record_id: str, force: bool = False) -> dict[str, Any]:
        if not force:
            cached = self.report(record_id)
            if cached is not None:
                return cached
        record = self._record(record_id)
        reference = record.get("reference")
        report = self.director.check(
            Path(record["wav"]), record.get("spoken_text") or record.get("text") or "",
            (record.get("join") or {}).get("pauses"), Path(reference) if reference else None,
        )
        report["record_id"] = record_id
        write_json_atomic(self.report_path(record_id), report)
        return report

    # -- take groups ---------------------------------------------------------------------
    @staticmethod
    def group_id(record_id: str, sentence: int | None = None) -> str:
        return f"pg-{record_id}" + (f"-s{int(sentence) + 1}" if sentence is not None else "")

    def group_for(self, record_id: str) -> dict[str, Any]:
        """Take group of a Playground generation; created with the original as take t1."""
        group_id = self.group_id(record_id)
        if self.store.exists(group_id):
            return self.store.load(group_id)
        record = self._record(record_id)
        original = take_from_record(record, origin="original")
        report = self.check(record_id)
        original.update(quality=report, similarity=report["metrics"].get("similarity"))
        return self.store.create(
            group_id, record["text"], record.get("spoken_text") or record["text"], recipe_from_record(record),
            {"kind": "playground", "record_id": record_id}, original=original,
        )

    def sentence_group(self, record_id: str, sentence: int) -> dict[str, Any]:
        """Take group for ONE sentence of a generation (only the problematic sentence is regenerated)."""
        parent = self.group_for(record_id)
        sentences, original_text = sentence_texts(parent["text"], parent["spoken_text"])
        if not 0 <= int(sentence) < len(sentences):
            raise TakeError(f"Cümle {int(sentence) + 1} yok (toplam {len(sentences)}).")
        group_id = self.group_id(record_id, sentence)
        if self.store.exists(group_id):
            return self.store.load(group_id)
        recipe = {**parent["recipe"], "text": sentences[int(sentence)]}
        if not original_text:
            recipe["normalize"] = False  # okunuş zaten normalize edilmiş cümle
        return self.store.create(
            group_id, sentences[int(sentence)], sentences[int(sentence)], recipe,
            {"kind": "playground-sentence", "record_id": record_id, "sentence": int(sentence)},
            parent={"group_id": parent["group_id"], "sentence": int(sentence)},
        )

    def alternatives(self, record_id: str, count: int = DEFAULT_ALTERNATIVES, sentence: int | None = None) -> tuple[dict[str, Any], list[str]]:
        group = self.sentence_group(record_id, sentence) if sentence is not None else self.group_for(record_id)
        return self.takes.alternatives(group["group_id"], count)

    def after_generation(self, record: dict[str, Any], auto_check: bool, auto_retake: bool, count: int) -> dict[str, Any]:
        """Hook after a Playground generation: check, and retake only on a high-confidence problem."""
        result: dict[str, Any] = {"report": None, "group": None, "errors": [], "auto_retake": False}
        if not auto_check or record.get("engine") == "mock":
            return result
        report = self.check(record["id"])
        result["report"] = report
        if auto_retake and report.get("retake_recommended"):
            group, errors = self.alternatives(record["id"], count)
            result.update(group=group, errors=errors, auto_retake=True)
        return result

    def select(self, group_id: str, take_id: str) -> dict[str, Any]:
        return self.store.select(group_id, take_id)

    def revert(self, group_id: str) -> dict[str, Any]:
        return self.store.revert(group_id)

    def close(self) -> str | None:
        return self.asr.close()
