# -*- coding: utf-8 -*-
"""Lazy TTS engine selection, without importing torch or Coqui for Mock mode."""

from __future__ import annotations

import os
from typing import Any


ENGINE_ENV_VAR = "VOXFORGE_TTS_ENGINE"
SUPPORTED_ENGINES = frozenset({"xtts", "mock"})


def selected_tts_engine_name(engine_name: str | None = None) -> str:
    raw = engine_name if engine_name is not None else os.environ.get(ENGINE_ENV_VAR, "xtts")
    name = raw.strip().lower()
    if name not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unsupported VoxForge TTS engine: {name!r}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_ENGINES))}."
        )
    return name


def create_tts_engine(engine_name: str | None = None) -> tuple[Any, str]:
    """Return ``(engine, device)``; XTTS loads only when explicitly selected."""
    name = selected_tts_engine_name(engine_name)
    if name == "mock":
        from scripts.mock_tts_engine import MockTTSEngine

        return MockTTSEngine(), "mock"

    # Delay heavyweight imports so unit tests and synthetic runs require only
    # the Python standard library.
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    engine.to(device)
    return engine, device
