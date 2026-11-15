"""ASR client for the Quality Director (VF-018).

One long-lived faster-whisper worker (``scripts/quality/asr_worker.py``) inside the
existing ``VoxForge-bench-eval`` environment, managed with the Playground's
``EngineManager`` protocol. It runs on the **CPU by default** and the manager hides the
GPU from it (``CUDA_VISIBLE_DEVICES=-1``): the 8 GB card stays reserved for the single
TTS engine (VoxCPM2 hot ~5.8 GB + desktop apps leave no room for large-v3).
Measured on this machine (large-v3 int8, 16 threads): ~8.5 s load, ~2x real time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from voxforge.playground.workers import REPO_ROOT, EngineManager, WorkerError, bench_python

ASR_WORKER = REPO_ROOT / "scripts" / "quality" / "asr_worker.py"
ASR_ENGINE = "asr"
ASR_LOAD_TIMEOUT = 600.0
ASR_TIMEOUT = 900.0


class Recogniser(Protocol):
    """Anything that turns a WAV into words; ``reference`` asks for speaker similarity too."""

    def transcribe(self, wav: Path, reference: Path | None = None) -> dict[str, Any]: ...


def eval_python() -> Path:
    return bench_python("eval")


class AsrClient:
    def __init__(self, log_dir: Path, manager: EngineManager | None = None, device: str = "cpu") -> None:
        self.device = device
        self.manager = manager or EngineManager(
            Path(log_dir),
            python_for=lambda _engine: eval_python(),
            worker_script=ASR_WORKER,
            load_timeout=ASR_LOAD_TIMEOUT,
            synth_timeout=ASR_TIMEOUT,
        )

    def problem(self) -> str | None:
        """Why the ASR cannot run (missing environment), or None."""
        python = self.manager.python_for(ASR_ENGINE)
        if not Path(python).is_file():
            return (
                f"Değerlendirme ortamı yok: {python}. Kurulum: scripts/benchmark/setup_env.ps1 -Engine eval "
                "(docs/VOICE_QUALITY_BENCHMARK.md). Yalnız ses ölçümleri yapıldı."
            )
        return None

    def transcribe(self, wav: Path, reference: Path | None = None) -> dict[str, Any]:
        problem = self.problem()
        if problem:
            raise WorkerError(problem)
        payload = {"wav": str(wav), "reference": str(reference) if reference else None}
        reply = self.manager.call(ASR_ENGINE, "transcribe", payload, device=self.device)
        worker = reply.get("worker") or {}
        reply["asr"] = {
            **(reply.get("asr") or {}),
            "model": worker.get("model") or (reply.get("asr") or {}).get("model"),
            "device": worker.get("device", self.device),
            "load_seconds": worker.get("load_seconds") if reply.get("loaded_now") else None,
        }
        return reply

    @property
    def loaded(self) -> bool:
        return self.manager.current_engine == ASR_ENGINE

    def close(self) -> str | None:
        return self.manager.unload()
