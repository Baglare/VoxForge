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

    def fake_engine(self, name: str, fail: str | None = None) -> None:
        """Simulate another engine's results.json from Mock audio (no GPU, no model)."""
        out = self.run_dir / name
        out.mkdir()
        items = []
        for item in TEST_ITEMS:
            path = out / f"{item.item_id}.wav"
            ok = item.item_id != fail
            if ok:
                MockTTSEngine().tts_to_file(text=item.spoken + name, file_path=str(path))
            items.append({"item_id": item.item_id, "path": str(path), "ok": ok, "rtf": 0.4 if ok else None,
                          "audio_seconds": run_engine.wav_seconds(path) if ok else None,
                          "error": None if ok else "CUDA out of memory"})
        (out / "results.json").write_text(json.dumps({"engine": name, "status": "completed", "items": items}), "utf-8")

    def fake_evaluation(self, name: str, transcript_suffix: str, similarity: float) -> None:
        results = json.loads((self.run_dir / name / "results.json").read_text("utf-8"))
        items = [
            {"item_id": e["item_id"], "transcript": next(i.spoken for i in TEST_ITEMS if i.item_id == e["item_id"]) + transcript_suffix,
             "similarity": similarity}
            for e in results["items"] if e["ok"]
        ]
        (self.run_dir / "evaluation").mkdir(exist_ok=True)
        (self.run_dir / "evaluation" / f"{name}.json").write_text(json.dumps({"items": items}), "utf-8")

    def test_prepare_mock_generation_blind_and_report(self) -> None:
        manifest = json.loads((self.run_dir / "manifest.json").read_text("utf-8"))
        self.assertEqual(manifest["reference"]["sha256"], self.reference_sha)
        self.assertEqual(file_sha256(self.reference), self.reference_sha)  # kaynak profil sesi değişmedi
        self.assertEqual(len(manifest["items"]), 12)
        self.assertIn("on dört otuz", next(i["spoken"] for i in manifest["items"] if i["item_id"] == "t07"))

        # Gerçek çalıştırıcı, Mock motoruyla uçtan uca (GPU yok).
        self.assertEqual(run_engine.run(self.run_dir, "mock", False, None), 0)
        mock_results = json.loads((self.run_dir / "mock" / "results.json").read_text("utf-8"))
        self.assertEqual(mock_results["status"], "completed")
        self.assertTrue(all(item["ok"] and item["rtf"] is not None for item in mock_results["items"]))
        self.assertEqual(run_engine.run(self.run_dir, "mock", False, None), 2)  # sonuç üzerine yazılmaz

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

        rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig"), delimiter=";"))
        for row in rows:
            better = key[row["item_id"]][row["label"]] == "chatterbox-v3"
            row.update(naturalness="4" if better else "3", similarity="4,5" if better else "3", pronunciation="4",
                       extra_sound="0")
        with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(read_human_scores(self.run_dir)["chatterbox-v3"]["similarity"][0], 4.5)

        report = build_report(self.run_dir)
        xtts, chatterbox = report["engines"]["xtts"], report["engines"]["chatterbox-v3"]
        self.assertEqual(xtts["cer_mean"], 0.0)
        self.assertGreater(chatterbox["insertion_rate_mean"], 0.0)
        self.assertEqual(chatterbox["failures"], 1)
        self.assertEqual(chatterbox["human"]["naturalness"], 4.0)
        verdict = report["verdicts"]["chatterbox-v3"]
        self.assertEqual(verdict["decision"], "uygun değil")  # 1 başarısız çıktı + CER kaybı
        self.assertTrue((self.run_dir / "report.md").read_text("utf-8").startswith("# Ses kalitesi"))


if __name__ == "__main__":
    unittest.main()
