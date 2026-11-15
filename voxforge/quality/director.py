"""Quality Director (VF-018): check one generated utterance, independent of who generated it.

``QualityDirector.check`` takes a WAV, the text the engine was asked to say (the
*spoken* form after normalisation), the pauses the engine inserted on purpose and,
optionally, the reference voice. It returns the explainable report of
``voxforge.quality.checks`` plus speaker similarity. Nothing here generates audio or
knows about Gradio, the Playground or Studio; both use the same contract.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from voxforge.quality.asr import Recogniser
from voxforge.quality.checks import evaluate
from voxforge.runs import utc_now_iso


class QualityDirector:
    def __init__(self, recogniser: Recogniser | None) -> None:
        self.recogniser = recogniser

    def check(
        self,
        wav: Path,
        spoken_text: str,
        intended_pauses: list[float] | None = None,
        reference: Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        words: list[dict[str, Any]] | None = None
        text: str | None = None
        info: dict[str, Any] = {}
        similarity = None
        if self.recogniser is not None:
            try:
                reply = self.recogniser.transcribe(Path(wav), Path(reference) if reference else None)
                words, text = list(reply.get("words") or []), reply.get("text") or ""
                info = {**(reply.get("asr") or {}), "seconds": reply.get("seconds")}
                similarity = reply.get("similarity")
                if reply.get("similarity_error"):
                    info["similarity_error"] = reply["similarity_error"]
            except Exception as exc:  # noqa: BLE001 - ASR yoksa ses ölçümleriyle devam
                info = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            info = {"error": "ASR yapılandırılmadı"}
        report = evaluate(spoken_text, Path(wav), words, text, intended_pauses, info)
        report["metrics"]["similarity"] = similarity
        report["checked_at"] = utc_now_iso()
        report["wav"] = str(wav)
        report["check_seconds"] = round(time.perf_counter() - started, 2)
        return report
