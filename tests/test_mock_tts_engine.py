from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from scripts.mock_tts_engine import MockTTSEngine
from scripts.tts_engine_factory import create_tts_engine, selected_tts_engine_name


class MockEngineTests(unittest.TestCase):
    def test_model_free_factory(self):
        with patch.dict(os.environ, {"VOXFORGE_TTS_ENGINE": "mock"}):
            engine, device = create_tts_engine()
        self.assertIsInstance(engine, MockTTSEngine)
        self.assertEqual(device, "mock")

    def test_default_engine_is_xtts(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(selected_tts_engine_name(), "xtts")

    def test_unknown_engine_rejected(self):
        with self.assertRaises(ValueError):
            selected_tts_engine_name("xxtts")

    def test_wav_format_and_reference_is_not_required(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nested" / "example.wav"
            engine = MockTTSEngine()
            engine.tts_to_file(text="Merhaba dünya", speaker_wav="does-not-exist.wav", file_path=str(output))
            self.assertEqual(len(engine.calls), 1)
            with wave.open(str(output), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 24000)
                self.assertGreater(wav.getnframes(), 0)

    def test_same_text_has_deterministic_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path_a = Path(temp) / "a.wav"
            path_b = Path(temp) / "b.wav"
            engine = MockTTSEngine()
            engine.tts_to_file(text="Aynı Türkçe metin", file_path=str(path_a))
            engine.tts_to_file(text="Aynı Türkçe metin", file_path=str(path_b))
            self.assertEqual(path_a.read_bytes(), path_b.read_bytes())

    def test_injected_failure_does_not_leave_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "failed.wav"
            engine = MockTTSEngine(fail_on_substring="HATA")
            with self.assertRaises(RuntimeError):
                engine.tts_to_file(text="HATA ile deneme", file_path=str(path))
            self.assertFalse(path.exists())
            self.assertEqual(len(engine.calls), 0)

    def test_empty_text_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                MockTTSEngine().tts_to_file(text=" \t", file_path=str(Path(temp) / "empty.wav"))

    def test_non_turkish_language_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                MockTTSEngine().tts_to_file(text="hello", language="en", file_path=str(Path(temp) / "en.wav"))


if __name__ == "__main__":
    unittest.main()
