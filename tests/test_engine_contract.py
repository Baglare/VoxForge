"""VF-003: engine contract, adapters and engine-agnostic generation core."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from scripts.mock_tts_engine import MockTTSEngine
from voxforge.engines import EngineError, SynthesisRequest, TTSEngine, create_engine, wrap_backend
from voxforge.generation import synthesize_text


class FakeCoquiBackend:
    """Mimics XTTS ``tts_to_file`` without torch; rejects preset kwargs like an old API."""

    def __init__(self, accept_presets: bool) -> None:
        self.accept_presets = accept_presets
        self.calls: list[dict] = []
        self.inner = MockTTSEngine()

    def tts_to_file(self, *, text, language, file_path, speaker_wav=None, **kwargs):
        self.calls.append({"speaker_wav": speaker_wav, **kwargs})
        if kwargs and not self.accept_presets:
            raise TypeError(f"unexpected keyword {next(iter(kwargs))}")
        return self.inner.tts_to_file(text=text, language=language, file_path=file_path)


class EngineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mock_engine_follows_contract(self) -> None:
        engine = create_engine("mock")
        self.assertIsInstance(engine, TTSEngine)
        self.assertEqual(engine.capabilities.engine_id, "mock")
        self.assertFalse(engine.capabilities.needs_reference)
        result = engine.synthesize(
            SynthesisRequest(text="Merhaba.", out_path=self.tmp / "a.wav", preset={"temperature": 0.7})
        )
        self.assertTrue(result.path.is_file())
        self.assertEqual(result.params, {})  # mock ignores presets explicitly

    def test_xtts_adapter_requires_reference(self) -> None:
        engine = wrap_backend("xtts", FakeCoquiBackend(accept_presets=True), "cpu")
        with self.assertRaises(EngineError):
            engine.synthesize(SynthesisRequest(text="Merhaba.", out_path=self.tmp / "a.wav"))

    def test_xtts_adapter_falls_back_when_presets_unsupported(self) -> None:
        reference = self.tmp / "ref.wav"
        MockTTSEngine().tts_to_file(text="referans", file_path=str(reference))
        backend = FakeCoquiBackend(accept_presets=False)
        engine = wrap_backend("xtts", backend, "cpu")
        result = engine.synthesize(
            SynthesisRequest(
                text="Merhaba.",
                out_path=self.tmp / "b.wav",
                reference_path=reference,
                preset={"temperature": 0.7},
            )
        )
        self.assertEqual(result.params, {})
        self.assertTrue(result.warnings)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0]["speaker_wav"], str(reference))

    def test_generation_core_chunks_and_validates_with_mock(self) -> None:
        engine = create_engine("mock")
        text = "Bu bir deneme cümlesidir. " * 20
        with mock.patch("voxforge.generation.shutil.which", return_value=None):
            result = synthesize_text(
                engine,
                text,
                output_dir=self.tmp,
                output_stem="gen",
                reference_path=None,
                post_processing_enabled=True,
                max_chars=120,
            )
        self.assertTrue(result["chunking_used"])
        self.assertEqual(len(result["parts"]), result["chunk_count"])
        self.assertTrue(all(part["ok"] for part in result["parts"]))
        self.assertFalse(result["post_processing_applied"])  # no ffmpeg -> raw kept, warned
        self.assertTrue(result["warnings"])
        with wave.open(str(result["final_output_path"]), "rb") as handle:
            self.assertGreater(handle.getnframes(), 0)

    def test_generation_core_stops_on_engine_failure(self) -> None:
        engine = wrap_backend("mock", MockTTSEngine(fail_on_substring="HATA"), "mock")
        with self.assertRaises(Exception):
            synthesize_text(
                engine,
                "İlk cümle. İkinci cümle HATA içeriyor. Üçüncü cümle.",
                output_dir=self.tmp,
                output_stem="fail",
                reference_path=None,
                max_chars=20,
            )
        self.assertFalse((self.tmp / "fail_raw.wav").exists())


if __name__ == "__main__":
    unittest.main()
