"""Expose the Playground worker as a Studio TTS engine."""

from __future__ import annotations

import wave
from dataclasses import asdict
from typing import Any

from scripts.benchmark.sources import load_pins
from voxforge.engines import EngineCapabilities, EngineError, SynthesisRequest, SynthesisResult, engine_spec
from voxforge.playground.prosody import VoxControls, plan_chunks
from voxforge.playground.service import DEFAULT_SEED, VARIANTS
from voxforge.playground.workers import EngineManager, worker_env


def worker_engine_spec(engine_id: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    if engine_id != "voxcpm2":
        return engine_spec(engine_id)
    pin = load_pins()["voxcpm2"]
    revision = worker_env(env).get("VOXFORGE_VOXCPM_REVISION", "").strip() or pin["revision"]
    return {
        "engine_id": engine_id,
        "model_id": f"{pin['model']}@{revision}",
        "needs_reference": True,
        "supports_presets": True,
    }


class WorkerEngineAdapter:
    def __init__(self, engine_id: str, manager: EngineManager) -> None:
        self.manager = manager
        self.capabilities = EngineCapabilities(
            **worker_engine_spec(engine_id, manager.env),
            device="cpu" if engine_id == "mock" else "cuda",
        )

    def synthesize(
        self, request: SynthesisRequest, *, seed: int = DEFAULT_SEED, max_chars: int | None = None,
    ) -> SynthesisResult:
        caps = self.capabilities
        if request.language not in caps.languages:
            raise EngineError(f"{caps.engine_id} dili desteklemiyor: {request.language}")
        reference = request.reference_path
        if caps.needs_reference and (reference is None or not reference.is_file()):
            raise EngineError(f"{caps.engine_id} referans ses dosyası gerektirir.")
        is_vox = caps.engine_id == "voxcpm2"
        controls = VoxControls.from_dict(request.preset) if is_vox else VoxControls()
        plan = plan_chunks(request.text, controls, max_chars or VARIANTS[caps.engine_id].max_chars, normalize=False)
        settings = controls.engine_settings() if is_vox else request.preset if caps.supports_presets else {}
        # Studio previously joined long XTTS/Mock segments without inserted silence.
        pauses = plan.pauses if is_vox else [0.0] * (len(plan.chunks) - 1)
        reply = self.manager.synthesize(
            caps.engine_id,
            {
                "chunks": plan.chunks,
                "reference": str(reference) if reference else None,
                "settings": settings,
                "seed": seed,
                "out": str(request.out_path.resolve()),
                "style": controls.style or None,
                "pauses": pauses,
                "trim": plan.trim,
            },
            device=caps.device,
        )
        worker = reply["worker"]
        source = worker.get("source") or {}
        actual_model = f"{source.get('model')}@{source.get('revision')}" if is_vox else worker.get("model")
        if actual_model != caps.model_id:
            raise EngineError(f"Model değişti: {actual_model} / {caps.model_id}")
        with wave.open(str(request.out_path), "rb") as handle:
            if not handle.getnframes():
                raise EngineError("Üretilen WAV boş.")
            sample_rate = handle.getframerate()
        return SynthesisResult(
            request.out_path, caps.engine_id, caps.model_id,
            {
                "engine": caps.engine_id, "model": actual_model, "source": source, "device": worker.get("device"),
                "seed": reply["seed"], "applied": reply.get("applied_settings"), "preset": request.preset,
                "controls": asdict(controls) if is_vox else None, "sample_rate": sample_rate,
                "audio_seconds": reply.get("audio_seconds"), "rtf": reply.get("rtf"),
                "join": reply.get("join"), "vram_peak_mb": reply.get("vram_peak_mb"), "vram": reply.get("vram"),
                "model_loaded_now": reply.get("loaded_now"), "runtime": worker.get("runtime"),
            },
            reply.get("warnings", []),
        )
