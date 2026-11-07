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
from voxforge.playground.workers import EngineManager, WorkerError, prepend_path  # noqa: E402

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
        def synth(text, out_path, reference, settings, seed, **extra):
            if settings.get("explode"):
                raise RuntimeError("bilerek hata")
            engine.tts_to_file(text=text, file_path=str(out_path))
            return {{"applied": {{**settings, **extra}}, "timings": {{"prompt_seconds": 0.0}}}}
        return synth, {{"model": "fake-" + str(os.getpid()), "defaults": {{"min_p": 0.05}},
                       "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}}

    w.check_device = lambda engine, device, helper: None  # testler GPU gerektirmez
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

    def test_sentences_share_a_chunk_up_to_limit(self) -> None:
        # Benchmark her maddeyi tek parça üretir; kısa çok cümleli metin cümle cümle bölünmemeli.
        from voxforge.playground.service import split_for_engine

        spoken, chunks = split_for_engine("Merhaba! Saat 14:30'da 5 kişi geldi. Evet.", 250, normalize=True)
        self.assertEqual(chunks, [spoken])
        self.assertIn("beş", spoken)
        long_text = "Bu cümle parçalama sınırını aşmak için yazıldı. " * 12
        _, long_chunks = split_for_engine(long_text, 250, normalize=True)
        self.assertGreater(len(long_chunks), 1)
        self.assertTrue(all(len(c) <= 250 for c in long_chunks))

    def test_ffmpeg_dir_moves_to_front_of_path(self) -> None:
        # Klasör PATH'te zaten olsa bile başa alınmalı (TorchCodec system32'den önce ister).
        shared = Path(r"C:\ff\bin")
        path = r"C:\Windows\system32;C:\x;c:\FF\bin;C:\y"
        self.assertEqual(prepend_path(path, shared), r"C:\ff\bin;C:\Windows\system32;C:\x;C:\y")
        self.assertEqual(prepend_path("", shared), r"C:\ff\bin")

    def test_variants_keep_engine_defaults(self) -> None:
        self.assertEqual(VARIANTS["chatterbox-default"].settings, {})
        self.assertEqual(VARIANTS["chatterbox-minp"].settings, {"min_p": 0.02})
        self.assertEqual(VARIANTS["xtts"].settings, {})
        self.assertEqual(VARIANTS["voxcpm2"].settings, {})  # VF-Q1 cfg 2.0 / 10 adım worker'da sabit
        self.assertFalse(VARIANTS["voxcpm2"].ultimate)
        self.assertTrue(VARIANTS["voxcpm2-ultimate"].ultimate)

    # ---- VF-015 ---------------------------------------------------------------------------
    def test_device_choice_restarts_worker_and_hides_gpu_for_cpu(self) -> None:
        manager = self.manager()
        service = PlaygroundService(self.tmp / "pg", manager)
        gpu = service.generate("Evet.", "voxcpm2", None, reference=self.reference)
        self.assertEqual(gpu["device"], "cuda")
        cpu = service.generate("Evet.", "voxcpm2", None, reference=self.reference, device="cpu")
        self.assertEqual(cpu["device"], "cpu")
        self.assertTrue(cpu["id"].endswith("_cpu"))
        self.assertEqual(manager.started_count, 2)  # cihaz değişince süreç yeniden başlar
        self.assertEqual(manager._current.ready.get("visible_gpu"), "-1")  # CPU süreci GPU'yu görmez
        self.assertTrue(any("CPU'da üretildi" in note for note in cpu["diagnosis"]))
        with self.assertRaises(PlaygroundError):
            service.generate("Evet.", "voxcpm2", None, reference=self.reference, device="tpu")

    def test_ultimate_requires_and_remembers_transcript(self) -> None:
        manager = self.manager()
        service = PlaygroundService(self.tmp / "pg", manager)
        with self.assertRaisesRegex(PlaygroundError, "transkript"):
            service.generate("Evet.", "voxcpm2-ultimate", None, reference=self.reference)
        record = service.generate(
            "Evet.", "voxcpm2-ultimate", None, reference=self.reference, reference_text="  Merhaba ben.  "
        )
        self.assertEqual(record["settings"]["applied"].get("reference_text"), "Merhaba ben.")
        self.assertTrue(record["settings"]["ultimate"])
        plain = service.generate("Evet.", "voxcpm2", None, reference=self.reference, reference_text="yok sayılır")
        self.assertNotIn("reference_text", plain["settings"]["applied"])  # yalnız Ultimate'e gider
        service.save_transcript("mimo", "Kayıttaki metin.")
        self.assertEqual(service.get_transcript("mimo"), "Kayıttaki metin.")
        self.assertEqual(service.get_transcript("baska"), "")
        with self.assertRaises(PlaygroundError):
            service.save_transcript("../profiles/x", "kaçış")

    def test_worker_refuses_cuda_without_gpu(self) -> None:
        engine_worker = _import_script("playground", "engine_worker")

        class NoCuda:
            class torch:
                __version__ = "2.15.0+cpu"

                class cuda:
                    @staticmethod
                    def is_available() -> bool:
                        return False

        with self.assertRaisesRegex(RuntimeError, "CPU'ya geçirilmedi"):
            engine_worker.check_device("voxcpm2", "cuda", NoCuda())
        engine_worker.check_device("voxcpm2", "cpu", NoCuda())  # açık CPU seçimi serbest

    def test_prompt_cache_reuses_same_reference_only(self) -> None:
        engine_worker = _import_script("playground", "engine_worker")
        calls: list[tuple] = []

        def build(prompt_text=None, prompt_wav_path=None, reference_wav_path=None):
            calls.append((prompt_text, prompt_wav_path, reference_wav_path))
            return {"n": len(calls)}

        stats = {"prompt_seconds": 0.0, "prompt_cache_hits": 0.0}
        cached = engine_worker.memoized_prompt_builder(build, stats, True, None, "cpu")
        ref = str(self.reference)
        first = cached(reference_wav_path=ref)
        self.assertIs(cached(reference_wav_path=ref), first)
        self.assertEqual((len(calls), stats["prompt_cache_hits"]), (1, 1))
        cached(prompt_text="metin", prompt_wav_path=ref, reference_wav_path=ref)  # Ultimate ayrı anahtar
        self.assertEqual(len(calls), 2)
        with wave.open(str(self.reference), "wb") as handle:  # referans değişti: yeniden kodlanır
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(bytes([1, 0]) * 4800)
        cached(reference_wav_path=ref)
        self.assertEqual(len(calls), 3)
        off = engine_worker.memoized_prompt_builder(build, stats, False, None, "cpu")
        off(reference_wav_path=ref)
        off(reference_wav_path=ref)
        self.assertEqual(len(calls), 5)

    def test_diagnosis_flags_full_vram(self) -> None:
        from voxforge.playground.service import diagnose

        record = {
            "engine": "voxcpm2",
            "device": "cuda",
            "timing": {"rtf": 9.6, "generation_seconds": 41.0, "stage_seconds": {"prompt_seconds": 34.0}},
            "vram": {"device_total_mb": 8188.0, "device_used_mb": 8187.5, "reserved_mb": 7138.0},
        }
        notes = " | ".join(diagnose(record))
        self.assertIn("VRAM tamamen dolu", notes)
        self.assertIn("RTF 9.6 > 1", notes)
        self.assertIn("kodlaması sürenin %83", notes)


def _import_script(folder: str, name: str):
    path = str(REPO_ROOT / "scripts" / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
    return __import__(name)


class TorchCudaIndexTest(unittest.TestCase):
    def test_index_follows_published_wheels_and_driver(self) -> None:
        torch_cuda = _import_script("benchmark", "torch_cuda")
        self.assertEqual(
            torch_cuda.parse_driver_cuda("| NVIDIA-SMI 617.42   KMD Version: 617.42   CUDA UMD Version: 13.4 |"), (13, 4)
        )
        self.assertEqual(torch_cuda.parse_driver_cuda("Driver Version: 560.94   CUDA Version: 12.6"), (12, 6))
        self.assertIsNone(torch_cuda.parse_driver_cuda("no gpu"))

        def wheel(index: str) -> str:
            return torch_cuda.wheel_name("torch", "2.15.0", index, "cp312", "win_amd64")

        listings = {"cu134": wheel("cu134"), "cu132": wheel("cu132"), "cu126": "torch-2.14.1+cu126-cp312-cp312-win_amd64.whl"}

        def choose(driver):
            return torch_cuda.choose_index("2.15.0", driver, listings, "cp312", "win_amd64")

        self.assertEqual(choose((13, 4)), "cu134")
        self.assertEqual(choose((13, 2)), "cu132")  # sürücü cu134'ü çalıştıramaz
        self.assertIsNone(choose((12, 6)))  # eski sabit cu126'da 2.15.0 yok: sessiz CPU yerine hata


if __name__ == "__main__":
    unittest.main()
