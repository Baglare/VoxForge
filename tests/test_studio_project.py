"""VF-005: persistent project, queue recovery and segment-level cache with Mock TTS."""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path

from scripts.mock_tts_engine import MockTTSEngine
from voxforge.engines import MOCK_MODEL_ID, wrap_backend
from voxforge.studio import ProjectSettings, StudioProject, run_pending

FAULT = "ARIZA"


def mock_engine(fail_on: str | None = None):
    backend = MockTTSEngine(fail_on_substring=fail_on)
    return wrap_backend("mock", backend, "mock"), backend


class StudioProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name) / "kitap"
        sentences = [f"Cümle {i} burada." for i in range(100)]
        sentences[50] = f"Cümle {FAULT} burada."
        StudioProject.create(
            self.folder,
            "Kitap",
            " ".join(sentences),
            ProjectSettings(engine_id="mock", model_id=MOCK_MODEL_ID),
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def states(self, project: StudioProject) -> dict[str, int]:
        counts: dict[str, int] = {}
        for segment in project.segments():
            counts[segment["state"]] = counts.get(segment["state"], 0) + 1
        return counts

    def test_failure_mid_run_recovery_and_single_segment_edit(self) -> None:
        project = StudioProject.open(self.folder)
        self.assertEqual(len(project.segments()), 100)
        self.assertEqual(project.segments()[3]["spoken_text"], "Cümle üç burada.")

        # 1) Ortadaki segment düşer; önceki ve sonraki başarılar korunur.
        self.assertEqual(project.enqueue_missing(), 100)
        engine, _ = mock_engine(fail_on=FAULT)
        self.assertEqual(run_pending(project, engine), {"done": 99, "failed": 1, "cancelled": 0})
        self.assertEqual(self.states(project), {"done": 99, "failed": 1})
        done_paths = {s["id"]: s["audio_path"] for s in project.segments() if s["state"] == "done"}

        # 2) Çökme benzetimi: running kalmış iş yeniden açılışta kuyruğa döner.
        failed_id = next(s["id"] for s in project.segments() if s["state"] == "failed")
        self.assertEqual(project.enqueue_missing(), 1)
        project.claim_next_job()
        with closing(sqlite3.connect(project.store.db_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM job WHERE status='running'").fetchone()[0], 1)

        reopened = StudioProject.open(self.folder)
        self.assertEqual(self.states(reopened), {"done": 99, "queued": 1})
        engine, backend = mock_engine()
        self.assertEqual(run_pending(reopened, engine)["done"], 1)
        self.assertEqual(len(backend.calls), 1)  # yalnız başarısız segment üretildi
        self.assertEqual(self.states(reopened), {"done": 100})

        # 3) Tek cümle düzenlenince yalnız o segment geçersizleşir.
        edited_id = reopened.segments()[10]["id"]
        reopened.update_segment_text(edited_id, "Cümle on yeniden yazıldı.")
        self.assertEqual(self.states(reopened), {"done": 99, "pending": 1})
        self.assertEqual(reopened.enqueue_missing(), 1)
        engine, backend = mock_engine()
        run_pending(reopened, engine)
        self.assertEqual([call["text"] for call in backend.calls], ["Cümle on yeniden yazıldı."])

        after = {s["id"]: s["audio_path"] for s in reopened.segments()}
        for segment_id, path in done_paths.items():
            if segment_id != edited_id:
                self.assertEqual(after[segment_id], path)
        self.assertNotEqual(after[edited_id], done_paths[edited_id])
        self.assertIn(failed_id, after)
        self.assertEqual(reopened.orphan_audio_files(), [])  # eski dosya korunur ama artefakt kaydı da durur

    def test_cancel_and_engine_mismatch(self) -> None:
        project = StudioProject.open(self.folder)
        project.enqueue_missing()
        self.assertEqual(project.cancel_queued(), 100)
        self.assertEqual(self.states(project), {"cancelled": 100})

        xtts_like = wrap_backend("xtts", MockTTSEngine(), "cpu")
        with self.assertRaises(ValueError):
            run_pending(project, xtts_like)


if __name__ == "__main__":
    unittest.main()
