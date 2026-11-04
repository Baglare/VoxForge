"""VF-Q1: benchmark pipeline without GPU (Mock engine, fake ASR/similarity)."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark import run_engine
from scripts.mock_tts_engine import MockTTSEngine
from voxforge.benchmark.experiment import (
    BenchmarkError,
    build_report,
    create_blind_set,
    prepare_run,
    read_human_scores,
)
from voxforge.benchmark.metrics import edit_counts, normalize_for_cer, score_transcript
from voxforge.benchmark.testset import TEST_ITEMS
from voxforge.studio.project import file_sha256

ALL_IDS = [item.item_id for item in TEST_ITEMS]


class MetricTests(unittest.TestCase):
    def test_turkish_normalisation_and_edit_counts(self) -> None:
        self.assertEqual(normalize_for_cer("İstanbul'da IŞIK 3 kez!"), "istanbul da ışık üç kez")
        counts = edit_counts("abc", "abxcdd")
        self.assertEqual((counts.substitutions, counts.deletions, counts.insertions), (0, 0, 3))
        self.assertEqual(edit_counts("abcd", "abd").deletions, 1)

    def test_extra_speech_is_flagged_by_insertions(self) -> None:
        clean = score_transcript("Evet.", "evet")
        babble = score_transcript("Evet.", "evet evet tabii ki öyle")
        self.assertEqual(clean["cer"], 0.0)
        self.assertTrue(babble["extra_speech_suspect"])
        self.assertGreater(babble["insertion_rate"], 1.0)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.reference = tmp / "profil_referansi.wav"
        MockTTSEngine().tts_to_file(text="referans ses", file_path=str(self.reference))
        self.reference_sha = file_sha256(self.reference)
        self.run_dir = prepare_run(tmp / "voice_benchmark", self.reference, "referans ses")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ---- yardımcılar: GPU/model olmadan başka motorların sonuçlarını taklit eder ----
    def fake_engine(self, name: str, ids=ALL_IDS, fail: str | None = None, rtf: float = 0.4) -> None:
        out = self.run_dir / name
        out.mkdir(exist_ok=True)
        items = []
        for item in TEST_ITEMS:
            if item.item_id not in ids:
                continue
            path = out / f"{item.item_id}.wav"
            ok = item.item_id != fail
            if ok:
                MockTTSEngine().tts_to_file(text=item.spoken + name, file_path=str(path))
            items.append({"item_id": item.item_id, "path": str(path), "ok": ok, "rtf": rtf if ok else None,
                          "audio_seconds": run_engine.wav_seconds(path) if ok else None,
                          "error": None if ok else "CUDA out of memory"})
        (out / "results.json").write_text(json.dumps({"engine": name, "status": "completed", "items": items}), "utf-8")

    def fake_evaluation(self, name: str, transcript_suffix: str, similarity: float) -> None:
        results = json.loads((self.run_dir / name / "results.json").read_text("utf-8"))
        spoken = {item.item_id: item.spoken for item in TEST_ITEMS}
        items = [{"item_id": e["item_id"], "transcript": spoken[e["item_id"]] + transcript_suffix, "similarity": similarity}
                 for e in results["items"] if e["ok"]]
        (self.run_dir / "evaluation").mkdir(exist_ok=True)
        (self.run_dir / "evaluation" / f"{name}.json").write_text(json.dumps({"items": items}), "utf-8")

    def fill_sheet(self, sheet: Path, candidate: str, extra_sound: str = "0") -> None:
        key = json.loads((sheet.parent / "blind_key.json").read_text("utf-8"))["items"]
        with sheet.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))
        for row in rows:
            better = key[row["item_id"]][row["label"]] == candidate
            row.update(naturalness="4,5" if better else "3", similarity="4" if better else "3", pronunciation="4",
                       extra_sound=extra_sound)
        with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

    # ---- testler ----
    def test_prepare_mock_generation_blind_and_report(self) -> None:
        manifest = json.loads((self.run_dir / "manifest.json").read_text("utf-8"))
        self.assertEqual(manifest["reference"]["sha256"], self.reference_sha)
        self.assertEqual(file_sha256(self.reference), self.reference_sha)  # kaynak profil sesi değişmedi
        self.assertIn("on dört otuz", next(i["spoken"] for i in manifest["items"] if i["item_id"] == "t07"))

        self.assertEqual(run_engine.run(self.run_dir, "mock", False, None), 0)
        mock_results = json.loads((self.run_dir / "mock" / "results.json").read_text("utf-8"))
        self.assertEqual(mock_results["coverage"]["mode"], "complete")

        self.fake_engine("xtts")
        self.fake_engine("chatterbox-v3", fail="t11")
        self.fake_evaluation("xtts", "", 0.70)
        self.fake_evaluation("chatterbox-v3", " ve sonra", 0.82)
        sheet = create_blind_set(self.run_dir)
        with self.assertRaises(BenchmarkError):
            create_blind_set(self.run_dir)  # dolu form korunur
        key = json.loads((self.run_dir / "listening" / "blind_key.json").read_text("utf-8"))["items"]
        self.assertEqual(sorted(key["t01"].values()), ["chatterbox-v3", "xtts"])  # mock dinlemeye girmez
        self.assertNotIn("chatterbox-v3", key["t11"].values())  # başarısız çıktı dinlemeye girmez
        self.fill_sheet(sheet, "chatterbox-v3")
        self.assertEqual(read_human_scores(self.run_dir)["chatterbox-v3"]["naturalness"]["t01"], 4.5)

        report = build_report(self.run_dir)
        self.assertEqual(report["engines"]["xtts"]["quality"]["cer_mean"], 0.0)
        quality = report["verdicts"]["chatterbox-v3"]["quality"]
        self.assertEqual(quality["decision"], "uygun değil (kalite)")  # 1 başarısız çıktı + CER kaybı
        self.assertTrue(any("başarısız" in reason for reason in quality["reasons"]))

    def test_pilot_grows_into_full_run_without_touching_pilot_outputs(self) -> None:
        self.assertEqual(run_engine.run(self.run_dir, "mock", False, {"t01", "t03"}), 0)
        results_path = self.run_dir / "mock" / "results.json"
        pilot = json.loads(results_path.read_text("utf-8"))
        self.assertEqual(pilot["coverage"]["mode"], "pilot")
        self.assertEqual(pilot["coverage"]["attempted"], 2)
        t01 = self.run_dir / "mock" / "t01.wav"
        before = (file_sha256(t01), t01.stat().st_mtime_ns)

        self.assertEqual(run_engine.run(self.run_dir, "mock", False, None), 0)  # aynı komut, --only yok
        full = json.loads(results_path.read_text("utf-8"))
        self.assertEqual(full["coverage"]["mode"], "complete")
        self.assertEqual([entry["item_id"] for entry in full["items"]], ALL_IDS)  # manifest sırası
        self.assertEqual((file_sha256(t01), t01.stat().st_mtime_ns), before)  # pilot çıktısı yeniden üretilmedi
        self.assertEqual([len(s["items"]) for s in full["sessions"]], [2, 10])
        self.assertEqual(run_engine.run(self.run_dir, "mock", False, None), 0)  # yapacak iş yok

        # Farklı paket sürümüyle devam etmek karışık sonuç üretmez.
        full["versions"] = {"python": "0.0"}
        full["items"] = full["items"][:-1]
        results_path.write_text(json.dumps(full), "utf-8")
        snapshot = results_path.read_text("utf-8")
        self.assertEqual(run_engine.run(self.run_dir, "mock", False, None), 3)
        self.assertEqual(results_path.read_text("utf-8"), snapshot)

    def test_partial_runs_never_get_a_final_verdict(self) -> None:
        self.fake_engine("xtts")
        self.fake_engine("chatterbox-v3", ids=["t01", "t02"])
        self.fake_evaluation("xtts", "", 0.7)
        self.fake_evaluation("chatterbox-v3", "", 0.9)
        with self.assertRaises(BenchmarkError):
            create_blind_set(self.run_dir)  # karar seti tam koşu ister
        pilot_sheet = create_blind_set(self.run_dir, pilot=True)
        self.assertEqual(pilot_sheet.parent.name, "listening_pilot")
        self.fill_sheet(pilot_sheet, "chatterbox-v3")

        report = build_report(self.run_dir)
        quality = report["verdicts"]["chatterbox-v3"]["quality"]
        self.assertEqual(quality["decision"], "pilot (nihai karar yok)")
        self.assertIn("aday 2/12", quality["missing"][0])
        self.assertTrue(report["pilot_listening_ignored"])
        self.assertIsNone(report["engines"]["chatterbox-v3"]["quality"]["human"]["naturalness"])  # pilot puanı karara girmez

    def test_missing_evidence_and_speed_do_not_decide_quality(self) -> None:
        self.fake_engine("xtts", rtf=0.5)
        self.fake_engine("chatterbox-v3", rtf=1.8)  # gerçek zamandan yavaş ama kaliteli
        self.fake_evaluation("xtts", "", 0.70)
        self.fake_evaluation("chatterbox-v3", "", 0.85)
        sheet = create_blind_set(self.run_dir)
        self.fill_sheet(sheet, "chatterbox-v3", extra_sound="")  # metin dışı ses işaretlenmedi

        verdict = build_report(self.run_dir)["verdicts"]["chatterbox-v3"]
        self.assertEqual(verdict["quality"]["decision"], "eksik kanıt")
        self.assertTrue(any("metin dışı ses işareti 0/12" in m for m in verdict["quality"]["missing"]))

        self.fill_sheet(sheet, "chatterbox-v3", extra_sound="0")
        report = build_report(self.run_dir)
        verdict = report["verdicts"]["chatterbox-v3"]
        self.assertEqual(verdict["quality"]["decision"], "kalite adayı (VF-009 için)")
        self.assertTrue(verdict["performance"]["label"].startswith("gerçek zamandan yavaş"))
        self.assertIn("RTF 1.80", (self.run_dir / "report.md").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
