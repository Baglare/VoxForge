"""Engine contract shared by every TTS backend (VF-003).

The application only talks to ``TTSEngine.synthesize``. Backends that expose a
Coqui-style ``tts_to_file`` call are wrapped by ``ToFileEngineAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable


class EngineError(RuntimeError):
    """Raised when a request does not fit the engine's capabilities."""


@dataclass(frozen=True)
class EngineCapabilities:
    engine_id: str
    model_id: str
    device: str
    languages: tuple[str, ...] = ("tr",)
    needs_reference: bool = False
    supports_presets: bool = False


@dataclass
class SynthesisRequest:
    text: str
    out_path: Path
    language: str = "tr"
    reference_path: Path | None = None
    preset: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    path: Path
    engine_id: str
    model_id: str
    params: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class TTSEngine(Protocol):
    capabilities: EngineCapabilities

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...


class ToFileEngineAdapter:
    """Wraps an object with ``tts_to_file(text=, speaker_wav=, language=, file_path=, **kw)``.

    One lock per adapter serialises inference; Coqui models are not safe to
    call concurrently.
    """

    def __init__(self, backend: Any, capabilities: EngineCapabilities) -> None:
        self.backend = backend
        self.capabilities = capabilities
        self._lock = Lock()

    def _validate(self, request: SynthesisRequest) -> None:
        caps = self.capabilities
        if not request.text.strip():
            raise EngineError("Bos metin sentezlenemez.")
        if request.language not in caps.languages:
            raise EngineError(f"{caps.engine_id} dili desteklemiyor: {request.language}")
        if caps.needs_reference:
            if request.reference_path is None or not Path(request.reference_path).is_file():
                raise EngineError(f"{caps.engine_id} referans ses dosyasi gerektirir.")

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self._validate(request)
        caps = self.capabilities
        call: dict[str, Any] = {
            "text": request.text,
            "language": request.language,
            "file_path": str(request.out_path),
        }
        if request.reference_path is not None:
            call["speaker_wav"] = str(request.reference_path)
        params = dict(request.preset) if caps.supports_presets else {}
        warnings: list[str] = []

        request.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                self.backend.tts_to_file(**call, **params)
            except TypeError as exc:
                if not params:
                    raise
                print(f"Preset parametreleri desteklenmedi, default uretime dusuluyor: {exc}")
                self.backend.tts_to_file(**call)
                params = {}
                warnings.append(
                    "Seçilen üretim modu bu XTTS API sürümünde tamamen desteklenmedi; "
                    "default üretime geri düşüldü."
                )

        return SynthesisResult(
            path=request.out_path,
            engine_id=caps.engine_id,
            model_id=caps.model_id,
            params=params,
            warnings=warnings,
        )
