# -*- coding: utf-8 -*-
"""Voice Playground: real worker processes with model-free engines (no GPU, no downloads)."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.playground.service import VARIANTS, PlaygroundError, PlaygroundService  # noqa: E402
from voxforge.playground.workers import EngineManager, WorkerError  # noqa: E402

# Every engine name is served by the Mock synthesizer; settings are echoed back so the
# test can see exactly what reached the "library". The protocol/concat code is the real one.
FAKE_WORKER = textwrap.dedent(
    f"""
    import os, sys
    sys.path.insert(0, {str(REPO_ROOT / "scripts" / "playground")!r})
    import engine_worker as w
    from scripts.mock_tts_engine import MockTTSEngine

    def fake(device):
        engine = MockTTSEngine()
        def synth(text, out_path, reference, settings, seed):
            if settings.get("explode"):
                raise RuntimeError("bilerek hata")
            engine.tts_to_file(text=text, file_path=str(out_path))
            return {{"applied": dict(settings)}}
        return synth, {{"model": "fake-" + str(os.getpid()), "defaults": {{"min_p": 0.05}}}}

    for name in w.ENGINES:
        w.LOADERS[name] = fake
    raise SystemExit(w.main(sys.argv[1:]))
    """
)


def wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / handle.getframerate()


class VoicePlaygroundTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vf_playground_"))
        self.reference = self.tmp / "ref.wav"
        with wave.open(str(self.reference), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\x00\x00" * 2400)

    def manager(self, fake: bool = True) -> EngineManager:
        kwargs = {"python_for": lambda engine: Path(sys.executable), "load_timeout": 60, "synth_timeout": 60}
        if fake:
            script = self.tmp / "fake_worker.py"
            script.write_text(FAKE_WORKER, encoding="utf-8")
            kwargs["worker_script"] = script
        manager = EngineManager(self.tmp / "logs", **kwargs)
        self.addCleanup(manager.close)
        return manager

    def test_mock_generation_record_and_wav(self) -> None:
        service = PlaygroundService(self.tmp / "pg", self.manager(fake=False))
        text = "Saat 14:30'da 5 kişi geldi. " + "Uzun bir cümle daha geliyor ve devam ediyor. " * 8
        record = service.generate(text, "mock", None, seed=7)
        self.assertGreater(record["chunks"], 1)  # uzun metin parçalandı, tek WAV'da birleşti
        self.assertGreater(wav_seconds(record["wav"]), 0)
        self.assertAlmostEqual(record["timing"]["audio_seconds"], wav_seconds(record["wav"]), places=2)
        self.assertIn("beş", record["spoken_text"])  # VF-004 okunuşu
        self.assertEqual(record["model"], "mock-sine-v1")
        self.assertEqual(record["settings"]["seed"], 7)
        self.assertIsNotNone(record["timing"]["model_load_seconds"])
        self.assertIn("vram", record)
        saved = json.loads(Path(record["wav"]).with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(saved["id"], record["id"])
        self.assertEqual(service.history()[0]["id"], record["id"])
        again = service.generate("Evet.", "mock", None)
        self.assertIsNone(again["timing"]["model_load_seconds"])  # aynı motor yeniden yüklenmedi

    def test_only_one_engine_process_at_a_time(self) -> None:
        manager = self.manager()
        service = PlaygroundService(self.tmp / "pg", manager)
        first = service.generate("Merhaba.", "xtts", None, reference=self.reference)
        first_proc = manager._current.process
        minp = service.generate("Evet.", "chatterbox-minp", None, reference=self.reference)
        self.assertIsNotNone(first_proc.poll(), "önceki motor süreci yeni motor başlamadan kapanmalı")
        self.assertEqual(manager.current_engine, "chatterbox-v3")
        self.assertEqual(minp["settings"]["applied"], {"min_p": 0.02})
        default = service.generate("Evet.", "chatterbox-default", None, reference=self.reference)
        self.assertEqual(default["settings"]["applied"], {})  # varsayılan: ayar gönderilmez
        self.assertEqual(default["model"], minp["model"])  # aynı süreç, yeniden yükleme yok
        self.assertNotEqual(first["model"], minp["model"])
        self.assertEqual(manager.started_count, 2)
        self.assertEqual(manager.unload(), "chatterbox-v3")
        self.assertIsNone(manager.current_engine)

    def test_compare_orders_by_engine_and_keeps_going_after_errors(self) -> None:
        manager = self.manager()
        service = PlaygroundService(self.tmp / "pg", manager)
        records, errors = service.generate_many(
            "Tamam.", ["chatterbox-minp", "mock", "chatterbox-default"], None, seed=3, reference=self.reference
        )
        self.assertEqual(errors, [])
        self.assertEqual([r["engine"] for r in records], ["chatterbox-v3", "chatterbox-v3", "mock"])
        self.assertEqual({r["settings"]["seed"] for r in records}, {3})
        self.assertEqual(manager.started_count, 2)
        _, errors = service.generate_many("Tamam.", ["xtts"], None)
        self.assertEqual(errors, ["XTTS-v2: XTTS-v2 için bir ses profili seçin."])

    def test_worker_errors(self) -> None:
        manager = self.manager()
        with self.assertRaises(WorkerError):
            manager.synthesize("mock", {"chunks": ["a"], "settings": {"explode": 1}, "out": str(self.tmp / "x.wav")})
        self.assertEqual(manager.current_engine, "mock")  # sentez hatası süreci öldürmez
        missing = EngineManager(self.tmp / "logs", python_for=lambda e: self.tmp / "yok" / "python.exe")
        with self.assertRaisesRegex(WorkerError, "ortamı bulunamadı"):
            missing.synthesize("voxcpm2", {"chunks": ["a"], "out": str(self.tmp / "y.wav")})
        service = PlaygroundService(self.tmp / "pg", manager)
        with self.assertRaises(PlaygroundError):
            service.generate("   ", "mock", None)

    def test_variants_keep_engine_defaults(self) -> None:
        self.assertEqual(VARIANTS["chatterbox-default"].settings, {})
        self.assertEqual(VARIANTS["chatterbox-minp"].settings, {"min_p": 0.02})
        self.assertEqual(VARIANTS["xtts"].settings, {})


if __name__ == "__main__":
    unittest.main()
