# -*- coding: utf-8 -*-
"""VF-018 Quality Director + Voice Takes: rules, confidence, takes, undo, Playground bridge.

No model and no GPU: ASR words are given directly (or by a fake recogniser / fake ASR
worker speaking the real protocol); audio is synthetic 16-bit PCM.
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import textwrap
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.playground.service import PlaygroundService  # noqa: E402
from voxforge.playground.workers import EngineManager  # noqa: E402
from voxforge.quality.asr import AsrClient  # noqa: E402
from voxforge.quality.checks import align, describe_report, evaluate, summarize  # noqa: E402
from voxforge.quality.director import QualityDirector  # noqa: E402
from voxforge.quality.takes import TakeDirector, TakeError, TakeStore, derive_seeds, take_from_record  # noqa: E402

RATE = 16000


def write_wav(path: Path, pattern: list[tuple[float, bool]], amplitude: float = 0.3) -> Path:
    """(seconds, voiced) pieces -> 16-bit mono WAV (voiced = 220 Hz tone)."""
    frames = bytearray()
    t = 0
    for seconds, voiced in pattern:
        for _ in range(int(seconds * RATE)):
            value = amplitude * math.sin(2 * math.pi * 220 * t / RATE) if voiced else 0.0
            frames += struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
            t += 1
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes(frames))
    return path


def words(text: str, prob: float = 0.95, start: float | None = None, step: float = 0.35) -> list[dict]:
    """ASR words; without ``start`` no timestamps (the unheard-speech check is then skipped)."""
    if start is None:
        return [{"word": w, "start": None, "end": None, "prob": prob} for w in text.split()]
    return [{"word": w, "start": start + i * step, "end": start + (i + 1) * step - 0.05, "prob": prob} for i, w in enumerate(text.split())]


TEXT = "Yaşlı muhafız oğluna son kez sarıldı. Kayıp ve özlem dolu bu vedada ikisi de konuşamadı."


def codes(report: dict) -> set[str]:
    return {f["code"] for f in report["findings"]}


class ChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vf_quality_"))
        # ~0,085 sn/karakter: normal okuma süresi, sonda doğal sessizlik
        self.normal = write_wav(self.tmp / "normal.wav", [(0.1, False), (3.2, True), (0.3, False), (3.6, True), (0.25, False)])

    def test_clean_take_is_ok(self) -> None:
        report = evaluate(TEXT, self.normal, words(TEXT), TEXT)
        self.assertEqual(report["verdict"], "ok", describe_report(report))
        self.assertEqual(report["metrics"]["cer"], 0.0)
        self.assertTrue(report["asr_checked"])

    def test_unheard_speech_catches_repeat_whisper_dropped(self) -> None:
        # 15 kelime 0,1-6,8 sn'ye yayılır: ses tamamen kapsanır
        clean = evaluate(TEXT, self.normal, words(TEXT, start=0.1, step=6.8 / 15), TEXT)
        self.assertNotIn("unheard_speech", codes(clean))
        self.assertLess(clean["metrics"]["unheard_speech_seconds"], 0.3)
        # Aynı transkript, ama sonda 1,5 sn fazladan konuşma (Whisper tekrarı yazmadı)
        extra = write_wav(self.tmp / "extra.wav", [(0.1, False), (3.2, True), (0.3, False), (3.6, True), (1.5, True), (0.25, False)])
        report = evaluate(TEXT, extra, words(TEXT, start=0.1, step=6.8 / 15), TEXT)
        finding = next(f for f in report["findings"] if f["code"] == "unheard_speech")
        self.assertEqual((finding["severity"], finding["confidence"]), ("major", "high"))
        self.assertEqual(report["verdict"], "problem")

    def test_minor_asr_difference_is_only_a_note(self) -> None:
        heard = TEXT.replace("muhafız", "muhafiz")
        report = evaluate(TEXT, self.normal, words(heard), heard)
        self.assertEqual(report["verdict"], "ok")
        self.assertIn("pronunciation", codes(report))

    def test_multi_word_name_change_is_not_a_near_word(self) -> None:
        # GPU doğrulamasında görülen durum: özel isim başka iki kelime olarak okundu
        heard = TEXT.replace("Yaşlı muhafız", "Burcu yanar")
        report = evaluate(TEXT, self.normal, words(heard), heard)
        finding = next(f for f in report["findings"] if f["code"] == "wrong_word")
        self.assertEqual(finding["severity"], "major")
        self.assertEqual(report["verdict"], "suspect")

    def test_word_alignment_operations(self) -> None:
        ops = align(["a", "b", "c", "d"], ["a", "c", "c", "x"])
        kinds = [op[0] for op in ops]
        self.assertEqual(kinds.count("equal"), 2)
        self.assertEqual(len(ops), 4)

    def test_missing_word_is_suspect_not_problem(self) -> None:
        heard = TEXT.replace("özlem ", "")
        report = evaluate(TEXT, self.normal, words(heard), heard)
        finding = next(f for f in report["findings"] if f["code"] == "missing_words")
        self.assertEqual(finding["confidence"], "medium")
        self.assertEqual(finding["sentence"], 1)
        self.assertEqual(report["verdict"], "suspect")
        self.assertFalse(report["retake_recommended"])  # ASR tek başına otomatik işlem başlatmaz
        self.assertEqual(report["problem_sentences"], [1])

    def test_short_function_word_drop_is_low_confidence(self) -> None:
        heard = TEXT.replace(" ve ", " ")
        report = evaluate(TEXT, self.normal, words(heard), heard)
        finding = next(f for f in report["findings"] if f["code"] == "missing_words")
        self.assertEqual((finding["severity"], finding["confidence"]), ("minor", "low"))
        self.assertEqual(report["verdict"], "ok")

    def test_truncation_with_audio_evidence_is_problem(self) -> None:
        cut = write_wav(self.tmp / "cut.wav", [(0.1, False), (3.2, True), (0.3, False), (0.6, True)])  # sessizliksiz bitiş
        heard = "Yaşlı muhafız oğluna son kez sarıldı. Kayıp ve"
        report = evaluate(TEXT, cut, words(heard), heard)
        finding = next(f for f in report["findings"] if f["code"] == "truncated")
        self.assertEqual(finding["confidence"], "high")
        self.assertEqual(finding["source"], "asr+audio")
        self.assertTrue(report["metrics"]["abrupt_end"])
        self.assertEqual(report["verdict"], "problem")
        self.assertTrue(report["retake_recommended"])

    def test_repetition_detected(self) -> None:
        long_audio = write_wav(self.tmp / "long.wav", [(0.1, False), (3.2, True), (0.3, False), (12.0, True), (0.3, False)])
        heard = TEXT.replace("dolu bu vedada", "dolu bu vedada bu vedada bu vedada")
        report = evaluate(TEXT, long_audio, words(heard), heard)
        finding = next(f for f in report["findings"] if f["code"] == "repeated_words")
        self.assertEqual(finding["confidence"], "high")
        self.assertEqual(report["verdict"], "problem")

    def test_unsure_asr_never_makes_a_problem(self) -> None:
        cut = write_wav(self.tmp / "cut2.wav", [(0.1, False), (3.2, True), (0.3, False), (0.6, True)])
        heard = "Yaşlı muhafız oğluna son kez sarıldı. Kayıp ve"
        report = evaluate(TEXT, cut, words(heard, prob=0.3), heard)
        asr = [f for f in report["findings"] if f["source"].startswith("asr")]
        self.assertTrue(asr)
        self.assertTrue(all(f["confidence"] == "low" for f in asr))
        self.assertIn("asr_unsure", codes(report))
        self.assertNotEqual(report["verdict"], "problem")

    def test_near_substitution_is_pronunciation(self) -> None:
        heard = TEXT.replace("muhafız", "muhafiz")
        report = evaluate(TEXT, self.normal, words(heard), heard)
        finding = next(f for f in report["findings"] if f["code"] == "pronunciation")
        self.assertEqual(finding["severity"], "minor")

    def test_silent_output_is_problem_without_asr(self) -> None:
        silent = write_wav(self.tmp / "silent.wav", [(2.0, False)])
        report = evaluate(TEXT, silent, None)
        self.assertEqual(report["verdict"], "problem")
        self.assertFalse(report["asr_checked"])
        self.assertIn("ASR yok", summarize(report))

    def test_intended_pauses_are_not_long_silences(self) -> None:
        paused = write_wav(self.tmp / "paused.wav", [(0.05, False), (3.2, True), (1.3, False), (3.6, True), (0.1, False)])
        self.assertIn("long_silence", codes(evaluate(TEXT, paused, words(TEXT), TEXT)))
        report = evaluate(TEXT, paused, words(TEXT), TEXT, intended_pauses=[0.95])
        self.assertNotIn("long_silence", codes(report))
        gap = write_wav(self.tmp / "gap.wav", [(0.05, False), (3.2, True), (3.5, False), (3.6, True), (0.1, False)])
        report = evaluate(TEXT, gap, words(TEXT), TEXT, intended_pauses=[0.55])
        finding = next(f for f in report["findings"] if f["code"] == "long_silence")
        self.assertEqual(finding["severity"], "major")

    def test_level_and_clipping(self) -> None:
        quiet = write_wav(self.tmp / "quiet.wav", [(0.1, False), (6.8, True), (0.25, False)], amplitude=0.01)
        self.assertIn("low_level", codes(evaluate(TEXT, quiet, words(TEXT), TEXT)))
        loud = write_wav(self.tmp / "loud.wav", [(0.1, False), (6.8, True), (0.25, False)], amplitude=1.5)
        self.assertIn("clipping", codes(evaluate(TEXT, loud, words(TEXT), TEXT)))


class FakeRecogniser:
    """Hears exactly ``heard[wav name]`` (default: the expected text); fixed similarity."""

    def __init__(self, text: str, heard: dict[str, str] | None = None, similarity: dict[str, float] | None = None) -> None:
        self.text, self.heard, self.sim = text, heard or {}, similarity or {}
        self.calls = 0

    def transcribe(self, wav: Path, reference: Path | None = None) -> dict:
        self.calls += 1
        text = self.heard.get(Path(wav).name, self.text)
        return {"text": text, "words": words(text), "similarity": self.sim.get(Path(wav).name, 0.8) if reference else None,
                "asr": {"model": "fake"}, "seconds": 0.0}


class TakesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vf_takes_"))
        self.store = TakeStore(self.tmp / "takes")
        self.made: list[int] = []

    def synth(self, recipe: dict, seed: int) -> dict:
        if seed in self.fail_seeds:
            raise RuntimeError("bilerek hata")
        self.made.append(seed)
        wav = write_wav(self.tmp / f"take_{seed}.wav", [(0.1, False), (3.2, True), (0.3, False), (3.6, True), (0.25, False)])
        return {"id": f"rec{seed}", "wav": str(wav), "engine": "voxcpm2", "variant": "voxcpm2", "spoken_text": recipe["text"],
                "settings": {"seed": seed, "label": "cfg 2.0 · 10 adım", "controls": {"cfg": 2.0}}}

    fail_seeds: set[int] = set()

    def group(self, director: TakeDirector) -> dict:
        original = self.synth({"text": TEXT}, 1234)
        take = take_from_record(original, "original")
        take["take_id"] = "t1"
        group = self.store.create("pg-x", TEXT, TEXT, {"text": TEXT, "seed": 1234, "reference": str(self.tmp / "ref.wav")},
                                  {"kind": "test"}, original=take)
        take.update(director.check_take(group, take))
        return self.store.update_take("pg-x", "t1", quality=take["quality"], similarity=take["similarity"])

    def test_alternatives_selection_and_revert(self) -> None:
        recogniser = FakeRecogniser(TEXT, heard={"take_1234.wav": TEXT.replace("özlem ", "")})
        director = TakeDirector(self.store, QualityDirector(recogniser), self.synth)
        group = self.group(director)
        self.assertEqual(group["selected"], "t1")
        self.assertEqual(group["takes"][0]["quality"]["verdict"], "suspect")
        group, errors = director.alternatives("pg-x", 3)
        self.assertEqual(errors, [])
        self.assertEqual(len(group["takes"]), 4)
        seeds = [t["seed"] for t in group["takes"]]
        self.assertEqual(len(set(seeds)), 4)  # her alternatif farklı tohum
        self.assertEqual(group["selected"], "t1")  # öneri var ama seçim kullanıcıda
        self.assertNotEqual(group["recommended"], "t1")
        self.assertEqual(group["takes"][1]["settings"]["controls"], {"cfg": 2.0})  # ayarlar take ile saklanır
        self.store.select("pg-x", "t3")
        self.store.select("pg-x", "t4")
        self.assertEqual(self.store.revert("pg-x")["selected"], "t3")
        self.assertEqual(self.store.revert("pg-x")["selected"], "t1")
        with self.assertRaises(TakeError):
            self.store.revert("pg-x")
        for take in self.store.load("pg-x")["takes"]:
            self.assertTrue(Path(take["wav"]).is_file())  # hiçbir take kaybolmaz

    def test_failure_does_not_stop_other_alternatives(self) -> None:
        director = TakeDirector(self.store, QualityDirector(FakeRecogniser(TEXT)), self.synth)
        self.group(director)
        seeds = derive_seeds(1234, {1234}, 2)
        self.fail_seeds = {seeds[0]}
        group, errors = director.alternatives("pg-x", 2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(group["takes"]), 2)
        with self.assertRaises(TakeError):
            director.alternatives("pg-x", 4)

    def test_identity_drift_is_flagged_and_ranked_down(self) -> None:
        seeds = derive_seeds(1234, {1234}, 2)
        recogniser = FakeRecogniser(TEXT, similarity={"take_1234.wav": 0.85, f"take_{seeds[0]}.wav": 0.70, f"take_{seeds[1]}.wav": 0.84})
        director = TakeDirector(self.store, QualityDirector(recogniser), self.synth)
        self.group(director)
        group, _ = director.alternatives("pg-x", 2)
        drifted = next(t for t in group["takes"] if t["seed"] == seeds[0])
        finding = next(f for f in drifted["quality"]["findings"] if f["code"] == "identity_drift")
        self.assertEqual(finding["confidence"], "medium")
        self.assertNotEqual(group["recommended"], drifted["take_id"])

    def test_group_id_is_validated(self) -> None:
        with self.assertRaises(TakeError):
            self.store.load("../x")


FAKE_ASR_WORKER = textwrap.dedent(
    f"""
    import sys
    sys.path.insert(0, {str(REPO_ROOT / "scripts" / "quality")!r})
    import os
    import asr_worker as w

    class Fake:
        def __init__(self, device, beam_size):
            self.info = {{"model": "fake-whisper", "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}}
        def transcribe(self, wav):
            return {{"text": "merhaba dünya", "words": [{{"word": "merhaba", "start": 0.0, "end": 0.4, "prob": 0.9}}], "segments": [], "duration": 1.0}}
        def similarity(self, reference, wav):
            return 0.77

    w.Recogniser = Fake
    raise SystemExit(w.main(sys.argv[1:]))
    """
)


class PlaygroundBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vf_pgq_"))

    def manager(self, script: Path | None = None) -> EngineManager:
        kwargs = {"python_for": lambda engine: Path(sys.executable), "load_timeout": 60, "synth_timeout": 60}
        if script is not None:
            kwargs["worker_script"] = script
        manager = EngineManager(self.tmp / "logs", **kwargs)
        self.addCleanup(manager.close)
        return manager

    def test_asr_client_speaks_worker_protocol_on_cpu(self) -> None:
        script = self.tmp / "fake_asr.py"
        script.write_text(FAKE_ASR_WORKER, encoding="utf-8")
        client = AsrClient(self.tmp / "logs", manager=self.manager(script))
        wav = write_wav(self.tmp / "a.wav", [(1.0, True)])
        reply = client.transcribe(wav, reference=wav)
        self.assertEqual(reply["text"], "merhaba dünya")
        self.assertEqual(reply["similarity"], 0.77)
        self.assertEqual(reply["asr"]["device"], "cpu")
        self.assertEqual(reply["worker"]["visible_gpu"], "-1")  # ASR GPU'yu göremez: VRAM TTS'e kalır
        self.assertTrue(client.loaded)
        client.close()
        self.assertFalse(client.loaded)

    def test_after_generation_retakes_only_on_problem(self) -> None:
        from voxforge.playground.quality import PlaygroundQuality

        service = PlaygroundService(self.tmp / "pg", self.manager())
        text = "Merhaba dünya. Bugün hava çok güzel."
        silent_heard = FakeRecogniser("")  # Mock sesi konuşma değil: ASR boş duyar
        quality = PlaygroundQuality(service, director=QualityDirector(silent_heard))
        record = service.generate(text, "mock", None, seed=5)
        self.assertEqual(quality.after_generation(record, True, True, 2)["report"], None)  # Mock denetlenmez

        quality.synthesize({"text": text, "variant": "mock", "seed": 5}, 9)  # tarif -> üretim
        report = quality.check(record["id"])
        self.assertIn(report["verdict"], ("suspect", "problem"))
        self.assertEqual(quality.check(record["id"]), report)  # önbellekten
        group, errors = quality.alternatives(record["id"], 2)
        self.assertEqual(errors, [])
        self.assertEqual([t["origin"] for t in group["takes"]], ["original", "alternative", "alternative"])
        self.assertEqual(group["recipe"]["variant"], "mock")
        sentence_group, _ = quality.alternatives(record["id"], 1, sentence=1)
        self.assertEqual(sentence_group["text"], "Bugün hava çok güzel.")
        self.assertEqual(sentence_group["parent"]["group_id"], group["group_id"])
        self.assertEqual(len(sentence_group["takes"]), 1)
        with self.assertRaises(TakeError):
            quality.sentence_group(record["id"], 5)


if __name__ == "__main__":
    unittest.main()
