"""VF-Q1: engine outputs must be 16-bit PCM WAV (Chatterbox float WAV broke validation)."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.benchmark import run_engine
from scripts.mock_tts_engine import MockTTSEngine
from voxforge.benchmark.experiment import prepare_run
from voxforge.studio.project import file_sha256


def write_float_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """IEEE float WAV (format tag 3), as torchaudio.save writes float tensors."""
    data = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


class FakeTensor:
    """Minimal torch-like tensor: detach().cpu().float().numpy()."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.array


class PcmWavTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_float_wav_is_what_broke_validation(self) -> None:
        path = self.tmp / "float.wav"
        write_float_wav(path, np.zeros(100, dtype=np.float32), 24000)
        with self.assertRaisesRegex(wave.Error, "unknown format: 3"):
            run_engine.wav_seconds(path)

    def test_save_pcm16_writes_readable_mono_pcm(self) -> None:
        path = self.tmp / "out.wav"
        audio = np.array([[0.0, 0.5, -0.5, 1.7, -2.0, 0.25]], dtype=np.float32)  # (1, N), kırpılacak uçlar
        run_engine.save_pcm16(path, FakeTensor(audio), 24000)
        self.assertEqual(run_engine.wav_format(path), "pcm16/24000Hz/1ch")
        with wave.open(str(path), "rb") as handle:
            frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        self.assertEqual(frames.tolist(), [0, 16384, -16384, 32767, -32767, 8192])
        self.assertFalse(path.with_name("out.wav.partial").exists())
        with self.assertRaises(ValueError):
            run_engine.save_pcm16(self.tmp / "bad.wav", np.array([np.nan], dtype=np.float32), 24000)

    def test_chatterbox_loader_saves_pcm16(self) -> None:
        class FakeChatterbox:
            sr = 24000

            @classmethod
            def from_pretrained(cls, device, t3_model=None):
                return cls()

            def generate(self, text, language_id=None, audio_prompt_path=None):
                return FakeTensor(np.full((1, 2400), 0.1, dtype=np.float32))  # float, kanal önde

        fake_module = types.ModuleType("chatterbox.mtl_tts")
        fake_module.ChatterboxMultilingualTTS = FakeChatterbox
        pinned = {"source": "git", "url": "https://github.com/resemble-ai/chatterbox",
                  "commit": run_engine.load_pins()["chatterbox"]["commit"], "version": "0.1.7"}
        with mock.patch.dict(sys.modules, {"chatterbox": types.ModuleType("chatterbox"), "chatterbox.mtl_tts": fake_module}), \
                mock.patch.object(run_engine, "source_revision", lambda name: pinned):
            synth, settings = run_engine.load_chatterbox({"device": "cpu", "reference": self.tmp / "ref.wav"})
            out = self.tmp / "t01.wav"
            synth("Merhaba.", out, 1234)
        self.assertEqual(run_engine.wav_format(out), "pcm16/24000Hz/1ch")
        self.assertAlmostEqual(run_engine.wav_seconds(out), 0.1, places=3)
        # Ayarlar değişmedi: önceki (başarısız) koşuya devam edilebilir.
        self.assertEqual(set(settings), {"model", "source", "language_id", "sample_rate", "note"})

    def test_failed_items_are_retried_in_the_same_run(self) -> None:
        reference = self.tmp / "ref.wav"
        MockTTSEngine().tts_to_file(text="referans", file_path=str(reference))
        run_dir = prepare_run(self.tmp / "bench", reference)
        self.assertEqual(run_engine.run(run_dir, "mock", False, None), 0)
        results_path = run_dir / "mock" / "results.json"
        results = json.loads(results_path.read_text("utf-8"))
        for entry in results["items"]:
            if entry["item_id"] == "t02":
                entry.update(ok=False, error="Error: unknown format: 3")
        results_path.write_text(json.dumps(results), "utf-8")
        t01 = run_dir / "mock" / "t01.wav"
        before = (file_sha256(t01), t01.stat().st_mtime_ns)

        self.assertEqual(run_engine.run(run_dir, "mock", False, None), 0)
        after = json.loads(results_path.read_text("utf-8"))
        self.assertEqual(after["sessions"][-1]["items"], ["t02"])  # yalnız başarısız madde
        self.assertTrue(all(entry["ok"] for entry in after["items"]))
        self.assertEqual(next(e for e in after["items"] if e["item_id"] == "t02")["format"], "pcm16/24000Hz/1ch")
        self.assertEqual((file_sha256(t01), t01.stat().st_mtime_ns), before)  # başarılı çıktıya dokunulmadı


if __name__ == "__main__":
    unittest.main()
