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

__all__ = [
    "EngineCapabilities",
    "EngineError",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSEngine",
    "ToFileEngineAdapter",
    "create_engine",
    "wrap_backend",
]


def wrap_backend(engine_name: str, backend: Any, device: str) -> ToFileEngineAdapter:
    """Attach the capability description for a raw backend object."""
    if engine_name == "mock":
        caps = EngineCapabilities(
            engine_id="mock",
            model_id=MOCK_MODEL_ID,
            device=device,
            needs_reference=False,
            supports_presets=False,
        )
    elif engine_name == "xtts":
        caps = EngineCapabilities(
            engine_id="xtts",
            model_id=XTTS_MODEL_ID,
            device=device,
            needs_reference=True,
            supports_presets=True,
        )
    else:
        raise EngineError(f"Bilinmeyen motor: {engine_name}")
    return ToFileEngineAdapter(backend, caps)


def create_engine(engine_name: str | None = None) -> ToFileEngineAdapter:
    from scripts.tts_engine_factory import create_tts_engine, selected_tts_engine_name

    name = selected_tts_engine_name(engine_name)
    backend, device = create_tts_engine(name)
    return wrap_backend(name, backend, device)
