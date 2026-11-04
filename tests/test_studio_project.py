"""VF-005: persistent project, queue recovery and segment-level cache with Mock TTS."""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path

from scripts.mock_tts_engine import MockTTSEngine
from voxforge.engines import MOCK_MODEL_ID, wrap_backend
from voxforge.studio import ProjectSettings, StudioProject, process_job, run_pending
from voxforge.studio.project import LEASE_SECONDS, file_sha256
from voxforge.studio.store import DB_FILENAME, MIGRATIONS, SCHEMA_VERSION, ProjectStore

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
        self.assertEqual(run_pending(project, engine), {"done": 99, "failed": 1, "cancelled": 0, "lost": 0})
        self.assertEqual(self.states(project), {"done": 99, "failed": 1})
        done_paths = {s["id"]: s["audio_path"] for s in project.segments() if s["state"] == "done"}

        # 2) Aktif worker'ın işi yeniden açılışta alınmaz; yalnız kalp atışı bayatlamış iş kurtarılır.
        failed_id = next(s["id"] for s in project.segments() if s["state"] == "failed")
        self.assertEqual(project.enqueue_missing(), 1)
        held = project.claim_next_job()
        while_active = StudioProject.open(self.folder)
        self.assertEqual(self.states(while_active), {"done": 99, "running": 1})
        self.assertIsNone(while_active.claim_next_job())  # ikinci kez başlatılmaz

        with closing(sqlite3.connect(project.store.db_path)) as db:
            db.execute(
                "UPDATE job SET heartbeat_at = heartbeat_at - ? WHERE id = ?",
                (LEASE_SECONDS + 1, held["id"]),
            )
            db.commit()

        reopened = StudioProject.open(self.folder)
        self.assertEqual(self.states(reopened), {"done": 99, "queued": 1})
        engine, backend = mock_engine()
        self.assertEqual(run_pending(reopened, engine)["done"], 1)
        self.assertEqual(len(backend.calls), 1)  # yalnız başarısız segment üretildi
        self.assertEqual(self.states(reopened), {"done": 100})
        self.assertFalse(project.complete_job(held, Path("x.wav"), "0", 0.0))  # kiralaması düşen eski worker yazamaz

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

    def test_reference_content_change_invalidates_cache(self) -> None:
        reference = Path(self._tmp.name) / "ref.wav"
        MockTTSEngine().tts_to_file(text="birinci referans", file_path=str(reference))
        folder = Path(self._tmp.name) / "ref_project"
        project = StudioProject.create(
            folder,
            "Ref",
            "Bir. İki. Üç.",
            ProjectSettings(engine_id="mock", model_id=MOCK_MODEL_ID, reference_path=str(reference)),
        )
        project.enqueue_missing()
        run_pending(project, mock_engine()[0])
        self.assertEqual(self.states(project), {"done": 3})
        old_paths = {s["audio_path"] for s in project.segments()}

        # Aynı yol, farklı içerik: aynı proje nesnesi bile eski önbelleği kullanmamalı.
        MockTTSEngine().tts_to_file(text="ikinci farkli referans sesi", file_path=str(reference))
        self.assertEqual(self.states(project), {"pending": 3})
        project.enqueue_missing()
        engine, backend = mock_engine()
        run_pending(project, engine)
        self.assertEqual(len(backend.calls), 3)
        self.assertTrue(old_paths.isdisjoint({s["audio_path"] for s in project.segments()}))

    def test_expired_worker_cannot_overwrite_new_workers_wav(self) -> None:
        folder = Path(self._tmp.name) / "race"
        project_a = StudioProject.create(
            folder, "Yaris", "Tek cümle.", ProjectSettings(engine_id="mock", model_id=MOCK_MODEL_ID)
        )
        project_a.enqueue_missing()
        job_a = project_a.claim_next_job()
        winner: dict = {}

        class SlowOldWorker:
            """A'nın sentezi sürerken kiralaması düşer ve B aynı işi tamamlar."""

            def tts_to_file(self, *, text, file_path, **kwargs):
                with closing(sqlite3.connect(folder / DB_FILENAME)) as db:
                    db.execute("UPDATE job SET heartbeat_at = heartbeat_at - ?", (LEASE_SECONDS + 1,))
                    db.commit()
                project_b = StudioProject.open(folder)
                job_b = project_b.claim_next_job()
                self_status = process_job(project_b, mock_engine()[0], job_b)
                winner.update(status=self_status, segment=project_b.segments()[0])
                return MockTTSEngine().tts_to_file(text=text + " eski worker", file_path=file_path)

        status_a = process_job(project_a, wrap_backend("mock", SlowOldWorker(), "mock"), job_a)

        self.assertEqual(winner["status"], "done")
        self.assertEqual(status_a, "lost")
        segment = project_a.segments()[0]
        self.assertEqual(segment["state"], "done")
        self.assertEqual(segment["audio_path"], winner["segment"]["audio_path"])
        with closing(sqlite3.connect(folder / DB_FILENAME)) as db:
            recorded_sha = db.execute("SELECT sha256 FROM artifact").fetchone()[0]
        self.assertEqual(file_sha256(Path(segment["audio_path"])), recorded_sha)
        self.assertEqual(list(project_a.store.audio_dir.glob("*.wav")), [Path(segment["audio_path"])])

    def test_cancel_and_engine_mismatch(self) -> None:
        project = StudioProject.open(self.folder)
        project.enqueue_missing()
        self.assertEqual(project.cancel_queued(), 100)
        self.assertEqual(self.states(project), {"cancelled": 100})

        xtts_like = wrap_backend("xtts", MockTTSEngine(), "cpu")
        with self.assertRaises(ValueError):
            run_pending(project, xtts_like)


class MigrationBackupTests(unittest.TestCase):
    def test_backup_includes_commits_still_in_wal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            db_path = folder / DB_FILENAME
            holder = sqlite3.connect(db_path, isolation_level=None)  # açık bağlantı WAL checkpoint'ini engeller
            try:
                holder.execute("PRAGMA journal_mode = WAL")
                holder.executescript("BEGIN;" + MIGRATIONS[1] + "PRAGMA user_version = 1; COMMIT;")
                holder.execute(
                    "INSERT INTO project (id, name, created_at, settings_json) VALUES (1, 'WAL', 'now', '{}')"
                )
                self.assertGreater((folder / f"{DB_FILENAME}-wal").stat().st_size, 0)

                store = ProjectStore(folder)  # v1 -> güncel sürüm, önce yedek
                self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            finally:
                holder.close()

            backup = folder / f"{DB_FILENAME}.bak-v1"
            with closing(sqlite3.connect(backup)) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT name FROM project").fetchone()[0], "WAL")


if __name__ == "__main__":
    unittest.main()
