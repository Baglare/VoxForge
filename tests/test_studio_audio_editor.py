"""VF-023 timeline, edited exports, isolated regeneration and portable project backups."""

from __future__ import annotations

import shutil
import stat
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from test_studio_takes import FakeSynthesisManager, pcm, write_wav

from voxforge.quality.director import QualityDirector
from voxforge.studio.project import file_sha256
from voxforge.studio.service import StudioError, StudioService

RATE = 24000


def silence(seconds: float) -> bytes:
    return bytes(round(seconds * RATE) * 2)


class StudioAudioEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        reference = self.root / "reference.wav"
        write_wav(reference, 1000)
        self.book = self.root / "book.txt"
        self.book.write_text(
            "# Gece\n\nİlk kapıyı açtım. Işıkları söndürdüm.\n\n# Sabah\n\nGün doğdu. Dışarı çıktım.",
            encoding="utf-8",
        )
        self.manager = FakeSynthesisManager()
        self.manager.resume.set()
        self.service = StudioService(self.root / "projects", manager=self.manager, quality=QualityDirector(None))
        self.slug = self.service.create(
            "Kitap", "", reference, {"pacing": "narration", "sentence_pause": 0.35, "paragraph_pause": 0.85},
            document=self.book,
        )
        character = self.service.add_voice(self.slug, "Ada", reference)
        last = self.service.segments(self.slug)[-1]
        self.service.assign_voice(self.slug, [last["id"]], character)
        self.service.start(self.slug)
        self.service.wait(self.slug, 5)

    def test_two_chapter_pcm_mix_undo_and_restore_preserve_active_takes(self) -> None:
        service, slug = self.service, self.slug
        project = service.open(slug)
        chapter_one, chapter_two = [chapter_id for chapter_id, _ in project.chapters()]
        first_chapter = project.segments(chapter_id=chapter_one)
        second_chapter = project.segments(chapter_id=chapter_two)
        first, item, third = first_chapter
        self.assertEqual(len(second_chapter), 3)
        service.start_take(slug, item["id"], item["cache_key"])
        service.wait(slug, 5)
        _, candidate = service.take_view(slug, item["id"])["takes"]
        service.change_take(slug, item["id"], item["cache_key"], candidate["id"])

        original_chapter = pcm(service.export_audio(slug, chapter_id=chapter_one))
        original_book = pcm(service.export_audio(slug))
        original_chapter_two = pcm(service.export_audio(slug, chapter_id=chapter_two))
        active_ids = [row["selected_take"] for row in service.segments(slug)]
        source_files = {path: file_sha256(path) for path in project.store.audio_dir.glob("*.wav")}
        self.assertEqual(
            original_chapter,
            pcm(Path(first["audio_path"])) + silence(0.85) + pcm(Path(candidate["path"]))
            + silence(0.35) + pcm(Path(third["audio_path"])),
        )

        service.change_mix(slug, item["id"], (-6.0, 0.5))
        reopened = StudioService(service.root, manager=self.manager)
        mixed = reopened.snapshot(slug)["segments"]
        self.assertEqual((mixed[1]["gain_db"], mixed[1]["pause_s"], mixed[1]["cache_key"]), (-6, 0.5, item["cache_key"]))
        self.assertAlmostEqual(mixed[2]["timeline_start_s"] - mixed[1]["timeline_start_s"], 1.0)
        self.assertEqual(mixed[1]["selected_take"], candidate["id"])
        take = pcm(Path(candidate["path"]))
        reduced = b"".join(struct.pack("<h", round(sample * 10 ** (-6 / 20))) for sample, in struct.iter_unpack("<h", take))
        prefix = pcm(Path(first["audio_path"])) + silence(0.85)
        expected_chapter = prefix + reduced + silence(0.5) + pcm(Path(third["audio_path"]))
        expected_book = expected_chapter + silence(0.85) + original_chapter_two
        self.assertEqual(pcm(reopened.export_audio(slug, chapter_id=chapter_one)), expected_chapter)
        self.assertEqual(pcm(reopened.export_audio(slug, chapter_id=chapter_two)), original_chapter_two)
        self.assertEqual(pcm(reopened.export_audio(slug)), expected_book)
        self.assertEqual({path: file_sha256(path) for path in source_files}, source_files)

        bundle = reopened.backup(slug)
        with ZipFile(bundle) as archive:
            self.assertIn("source.txt", archive.namelist())
            self.assertIn("reference.wav", archive.namelist())
            self.assertIn("voices/ada.wav", archive.namelist())
            self.assertNotIn("project.sqlite-wal", archive.namelist())
        other = StudioService(self.root / "other-projects", manager=self.manager)
        restored = other.restore(bundle, "Taşınan kitap")
        self.assertEqual(other.open(restored).name, "Taşınan kitap")
        self.assertEqual((other.project_dir(restored) / "source.txt").read_bytes(), self.book.read_bytes())
        restored_items = other.segments(restored)
        self.assertEqual(restored_items[-1]["voice_name"], "Ada")
        restored_voice = Path(other.open(restored).voices()["ada"]["reference_path"])
        self.assertEqual(restored_voice, other.project_dir(restored) / "voices" / "ada.wav")
        self.assertEqual([row["selected_take"] for row in restored_items], active_ids)
        self.assertTrue(all(Path(row["audio_path"]).is_relative_to(other.project_dir(restored)) for row in restored_items))
        self.assertEqual(pcm(other.export_audio(restored)), expected_book)
        other.change_mix(restored, item["id"], None)
        self.assertEqual(pcm(other.export_audio(restored)), original_book)
        other.change_take(restored, item["id"], item["cache_key"], None)
        self.assertEqual(
            pcm(other.export_audio(restored, chapter_id=chapter_one)),
            prefix + pcm(Path(item["audio_path"])) + silence(0.35) + pcm(Path(third["audio_path"])),
        )
        reopened.change_mix(slug, item["id"], None)
        self.assertEqual(pcm(reopened.export_audio(slug, chapter_id=chapter_one)), original_chapter)
        self.assertEqual(pcm(reopened.export_audio(slug)), original_book)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg gerekli")
    def test_chapter_and_book_mp3_use_edited_pcm(self) -> None:
        project = self.service.open(self.slug)
        chapter_id = project.chapters()[0][0]
        item = project.segments(chapter_id=chapter_id)[1]
        self.service.change_mix(self.slug, item["id"], (-12, 0.7))
        ffmpeg = shutil.which("ffmpeg")
        for scope in (chapter_id, None):
            with self.subTest(scope=scope):
                wav = self.service.export_audio(self.slug, chapter_id=scope)
                mp3 = self.service.export_audio(self.slug, chapter_id=scope, audio_format="mp3")
                expected = self.root / f"expected-{scope}.mp3"
                subprocess.run(
                    [ffmpeg, "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "libmp3lame", "-q:a", "2", str(expected)],
                    check=True, capture_output=True,
                )
                self.assertEqual(mp3.read_bytes(), expected.read_bytes())
                with wave.open(str(wav), "rb") as combined:
                    self.assertEqual(combined.getframerate(), RATE)

    def test_single_take_does_not_touch_another_queued_segment_in_same_chapter(self) -> None:
        service, slug = self.service, self.slug
        project = service.open(slug)
        first, second, *_ = project.segments()
        self.assertEqual(first["chapter_id"], second["chapter_id"])
        project.enqueue_take(first["id"], first["cache_key"])
        original = [row["audio_path"] for row in project.segments()]
        calls = len(self.manager.calls)
        service.start_take(slug, second["id"], second["cache_key"])
        service.wait(slug, 5)
        self.assertEqual(len(self.manager.calls), calls + 1)
        self.assertEqual(self.manager.calls[-1][1]["chunks"], [second["spoken_text"]])
        self.assertEqual(service.take_view(slug, first["id"])["job"]["status"], "queued")
        self.assertEqual(len(service.take_view(slug, first["id"])["takes"]), 1)
        self.assertEqual(len(service.take_view(slug, second["id"])["takes"]), 2)
        self.assertEqual([row["audio_path"] for row in project.segments()], original)

    def test_single_edited_segment_does_not_drain_other_missing_jobs(self) -> None:
        service, slug = self.service, self.slug
        project = service.open(slug)
        first, second, *unchanged = service.segments(slug)
        service.set_spoken(slug, first["id"], "İlk okunuş değişti.")
        service.set_spoken(slug, second["id"], "Yalnız bu replik üretilecek.")
        self.assertEqual(project.enqueue_missing([first["id"]]), 1)
        calls = len(self.manager.calls)
        self.assertEqual(service.start_segment(slug, second["id"]), 1)
        service.wait(slug, 5)
        updated = service.segments(slug)
        self.assertEqual(len(self.manager.calls), calls + 1)
        self.assertEqual(updated[0]["state"], "queued")
        self.assertEqual(updated[1]["state"], "done")
        self.assertEqual([r["audio_path"] for r in updated[2:]], [r["audio_path"] for r in unchanged])

    def test_restore_rejects_invalid_members_corrupt_audio_and_overwrite(self) -> None:
        service, slug = self.service, self.slug
        bundle = service.backup(slug)
        original = pcm(service.export_audio(slug))
        with self.assertRaisesRegex(StudioError, "zaten"):
            service.restore(bundle, "Kitap")
        for filename in ("../escape.wav", "audio/linked.wav"):
            with self.subTest(filename=filename):
                bad = self.root / "bad.zip"
                shutil.copyfile(bundle, bad)
                entry = ZipInfo(filename)
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                with ZipFile(bad, "a") as archive:
                    archive.writestr(entry, "../escape.wav")
                with self.assertRaisesRegex(StudioError, "geçersiz"):
                    service.restore(bad, "Hatalı")
        corrupt = self.root / "corrupt.zip"
        with ZipFile(bundle) as original_archive, ZipFile(corrupt, "w") as changed:
            for name in original_archive.namelist():
                changed.writestr(name, b"invalid" if name.startswith("audio/") else original_archive.read(name))
        with self.assertRaisesRegex(StudioError, "WAV özeti"):
            service.restore(corrupt, "Hatalı")
        self.assertEqual([key for _, key in service.list_projects()], [slug])
        self.assertFalse((service.root / "escape.wav").exists())
        self.assertFalse(any(service.root.glob(".restore-*")))
        self.assertEqual(pcm(service.export_audio(slug)), original)

    def test_backup_does_not_carry_a_live_worker_lease_into_restored_project(self) -> None:
        service, slug = self.service, self.slug
        project = service.open(slug)
        item = service.segments(slug)[0]
        job_id = project.enqueue_take(item["id"], item["cache_key"])
        self.assertEqual(project.claim_next_job(job_id=job_id)["id"], job_id)
        bundle = service.backup(slug)
        restored = service.restore(bundle, "Kurtarılan")
        self.assertEqual(service.take_view(slug, item["id"])["job"]["status"], "running")
        self.assertEqual(service.take_view(restored, item["id"])["job"]["status"], "queued")
        self.assertEqual(pcm(service.export_audio(restored)), pcm(service.export_audio(slug)))

    def test_mix_values_validate_and_multiple_edits_undo_in_order(self) -> None:
        service, slug = self.service, self.slug
        item = service.segments(slug)[0]
        for invalid in ((float("nan"), 1), (13, None), (0, -1), (0, float("inf"))):
            with self.assertRaises(StudioError):
                service.change_mix(slug, item["id"], invalid)
        service.change_mix(slug, item["id"], (0, 0))
        service.change_mix(slug, item["id"], (3, 0.5))
        service.change_mix(slug, item["id"], (3, 0.5))
        service.change_mix(slug, item["id"], None)
        self.assertEqual(service.segment(slug, item["id"])["pause_s"], 0)
        service.change_mix(slug, item["id"], None)
        self.assertIsNone(service.segment(slug, item["id"])["pause_s"])
        self.assertFalse(service.segment(slug, item["id"])["can_undo_mix"])
