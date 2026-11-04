"""TTS engine contract and adapters.

``create_engine`` keeps the existing selection rule: ``VOXFORGE_TTS_ENGINE``
(``xtts`` default, ``mock`` for model-free runs). Heavy imports stay lazy.
"""

from __future__ import annotations

from typing import Any

from voxforge.engines.base import (
    EngineCapabilities,
    EngineError,
    SynthesisRequest,
    SynthesisResult,
    ToFileEngineAdapter,
    TTSEngine,
)

XTTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
MOCK_MODEL_ID = "mock-sine-v1"
ENGINE_SPECS: dict[str, dict[str, Any]] = {
    "mock": {"model_id": MOCK_MODEL_ID, "needs_reference": False, "supports_presets": False},
    "xtts": {"model_id": XTTS_MODEL_ID, "needs_reference": True, "supports_presets": True},
}

__all__ = [
    "EngineCapabilities",
    "EngineError",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSEngine",
    "ToFileEngineAdapter",
    "create_engine",
    "engine_spec",
    "wrap_backend",
]


def wrap_backend(engine_name: str, backend: Any, device: str) -> ToFileEngineAdapter:
    """Attach the capability description for a raw backend object."""
    if engine_name not in ENGINE_SPECS:
        raise EngineError(f"Bilinmeyen motor: {engine_name}")
    caps = EngineCapabilities(engine_id=engine_name, device=device, **ENGINE_SPECS[engine_name])
    return ToFileEngineAdapter(backend, caps)


def engine_spec(engine_name: str | None = None) -> dict[str, Any]:
    """Identity of the selected engine *without* loading a model (for project settings/UI)."""
    from scripts.tts_engine_factory import selected_tts_engine_name

    name = selected_tts_engine_name(engine_name)
    return {"engine_id": name, **ENGINE_SPECS[name]}


def create_engine(engine_name: str | None = None) -> ToFileEngineAdapter:
    from scripts.tts_engine_factory import create_tts_engine, selected_tts_engine_name

    name = selected_tts_engine_name(engine_name)
    backend, device = create_tts_engine(name)
    return wrap_backend(name, backend, device)
