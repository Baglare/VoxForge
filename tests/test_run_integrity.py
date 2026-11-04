"""VF-002: run records must not mix runs or report false success."""

from __future__ import annotations

import csv
import os
import tempfile
import time
import unittest
import wave
from pathlib import Path

from scripts import create_human_eval_report as human_eval
from voxforge.runs import (
    finish_training_run,
    new_run_id,
    start_training_run,
    training_checkpoint_candidates,
    wav_part_status,
)


def touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TrainingRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.experiment = Path(self._tmp.name) / "exp"
        self.output = self.experiment / "training_output"
        touch(self.output / "old_run" / "best_model.pth", b"old")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_stale_checkpoint_is_not_success(self) -> None:
        manifest = start_training_run(self.experiment, self.output, settings={})
        result = finish_training_run(self.experiment, manifest)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["checkpoints"], [])
        self.assertTrue(result["warnings"])

    def test_only_new_checkpoints_are_recorded_and_selected(self) -> None:
        manifest = start_training_run(self.experiment, self.output, settings={})
        new_file = touch(self.output / "new_run" / "checkpoint_3.pth", b"new")
        result = finish_training_run(self.experiment, manifest)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual([item["path"] for item in result["checkpoints"]], ["new_run/checkpoint_3.pth"])

        candidates, provenance = training_checkpoint_candidates(self.experiment, self.output)
        self.assertEqual(provenance["source"], "run_manifest")
        self.assertEqual(provenance["run_id"], result["run_id"])
        self.assertEqual(candidates, [new_file])

    def test_modified_checkpoint_counts_as_produced(self) -> None:
        old = self.output / "old_run" / "best_model.pth"
        manifest = start_training_run(self.experiment, self.output, settings={})
        old.write_bytes(b"retrained-and-longer")
        stamp = time.time() + 5
        os.utime(old, (stamp, stamp))
        result = finish_training_run(self.experiment, manifest)
        self.assertEqual(result["status"], "succeeded")

    def test_without_manifest_legacy_discovery_is_flagged(self) -> None:
        candidates, provenance = training_checkpoint_candidates(self.experiment, self.output)
        self.assertEqual(provenance["source"], "legacy_discovery")
        self.assertIsNotNone(provenance["warning"])
        self.assertEqual(len(candidates), 1)

    def test_run_ids_are_unique(self) -> None:
        self.assertNotEqual(new_run_id(), new_run_id())


class HumanEvalIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def matrix(self, name: str) -> Path:
        root = self.root / name
        for variant in ("base", "best_model"):
            (root / variant).mkdir(parents=True)
        return root

    def write_scores(self, matrix_root: Path, base: int, best: int) -> None:
        path = human_eval.human_eval_paths(matrix_root)["scorecard"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=human_eval.CSV_COLUMNS, delimiter=";")
            writer.writeheader()
            for variant, score in (("base", base), ("best_model", best)):
                row = {column: "" for column in human_eval.CSV_COLUMNS}
                row.update({"variant": variant, **{m: score for m in human_eval.METRIC_COLUMNS}})
                writer.writerow(row)

    def summary_for(self, matrix_root: Path, use_default_scores: bool = False) -> dict:
        rows = human_eval.load_rows(matrix_root, use_default_scores)
        source = human_eval.detect_score_source(rows, use_default_scores)
        return human_eval.build_summary(
            matrix_root, rows, source, human_eval.human_eval_paths(matrix_root)["scorecard"]
        )

    def test_two_runs_do_not_mix_scores(self) -> None:
        run_a, run_b = self.matrix("run_a"), self.matrix("run_b")
        self.write_scores(run_a, base=2, best=4)
        self.write_scores(run_b, base=5, best=1)
        summary_a, summary_b = self.summary_for(run_a), self.summary_for(run_b)
        self.assertEqual(summary_a["highest_total_score"]["variant"], "best_model")
        self.assertEqual(summary_b["highest_total_score"]["variant"], "base")
        self.assertEqual(summary_a["score_source"], "manual")

    def test_unscored_run_is_not_given_a_verdict(self) -> None:
        summary = self.summary_for(self.matrix("empty"))
        self.assertEqual(summary["score_source"], "empty")
        self.assertTrue(summary["overall_result"].startswith("Değerlendirilmedi"))
        self.assertNotIn(human_eval.EXAMPLE_CONCLUSION, summary["overall_result"])

    def test_example_scores_are_labelled(self) -> None:
        summary = self.summary_for(self.matrix("example"), use_default_scores=True)
        self.assertEqual(summary["score_source"], "example")
        self.assertTrue(summary["overall_result"].startswith("ÖRNEK PUAN"))


class WavPartStatusTests(unittest.TestCase):
    def test_missing_empty_and_valid_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertFalse(wav_part_status(folder / "missing.wav")["ok"])
            self.assertFalse(wav_part_status(touch(folder / "broken.wav", b"not a wav"))["ok"])

            empty = folder / "empty.wav"
            valid = folder / "valid.wav"
            for path, frames in ((empty, b""), (valid, b"\x00\x00" * 240)):
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(24000)
                    handle.writeframes(frames)
            self.assertEqual(wav_part_status(empty)["error"], "bos ses")
            self.assertTrue(wav_part_status(valid)["ok"])


if __name__ == "__main__":
    unittest.main()
