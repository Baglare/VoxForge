from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from scripts.audio_concat_utils import concatenate_wavs
from scripts.mock_tts_engine import MockTTSEngine
from scripts.mock_tts_smoke import run_mock_smoke


class MockPipelineTests(unittest.TestCase):
    def test_end_to_end_uses_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "test.wav"
            chunks, frames = run_mock_smoke("Birinci cümle. İkinci cümle. " * 17, output)
            self.assertGreater(chunks, 1)
            self.assertGreater(frames, 0)
            self.assertTrue(output.is_file())

    def test_missing_part_refuses_to_omit_content(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = MockTTSEngine()
            valid = Path(temp) / "valid.wav"
            engine.tts_to_file(text="Bir", file_path=str(valid))
            output = Path(temp) / "joined.wav"
            ok, message = concatenate_wavs([valid, Path(temp) / "missing.wav"], output)
            self.assertFalse(ok)
            self.assertIn("eksik", message.lower())
            self.assertFalse(output.exists())

    def test_one_part_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = MockTTSEngine()
            original = Path(temp) / "original.wav"
            target = Path(temp) / "target.wav"
            engine.tts_to_file(text="Tek WAV", file_path=str(original))
            with patch("scripts.audio_concat_utils.shutil.which", return_value=None):
                ok, _ = concatenate_wavs([original], target)
            self.assertTrue(ok)
            self.assertEqual(original.read_bytes(), target.read_bytes())

    def test_multi_part_without_ffmpeg_uses_pcm_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = MockTTSEngine()
            parts = [Path(temp) / f"part{i}.wav" for i in range(3)]
            for part in parts:
                engine.tts_to_file(text="Merhaba", file_path=str(part))
            output = Path(temp) / "joined.wav"
            with patch("scripts.audio_concat_utils.shutil.which", return_value=None):
                ok, _ = concatenate_wavs(parts, output)
            self.assertTrue(ok)
            with wave.open(str(output), "rb") as wav:
                total = wav.getnframes()
            expected = 0
            for part in parts:
                with wave.open(str(part), "rb") as wav:
                    expected += wav.getnframes()
            self.assertEqual(total, expected)

    def test_mismatched_pcm_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = MockTTSEngine()
            a = Path(temp) / "a.wav"
            b = Path(temp) / "b.wav"
            engine.tts_to_file(text="Bir", file_path=str(a))
            with wave.open(str(b), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 4000)
            output = Path(temp) / "joined.wav"
            with patch("scripts.audio_concat_utils.shutil.which", return_value=None):
                ok, msg = concatenate_wavs([a, b], output)
            self.assertFalse(ok)
            self.assertIn("uyumsuz", msg.lower())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
