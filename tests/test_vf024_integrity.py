"""VF-024: ses silme/değiştirme sonrası bağımlılıklar, geçici referanslar, rapor koruma (model yok)."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import finetune_readiness_report as readiness
from tests.test_voice_library import LibraryFixture, write_wav
from voxforge.playground.prosody import ControlError
from voxforge.playground.scenes import Scene, SceneNarrator, SceneProject
from voxforge.playground.service import PlaygroundService
from voxforge.voices import VoiceLibrary, VoiceLibraryError

DESIGN_KEY = f"design:{'a' * 32}"


class DependencyFixture(LibraryFixture):
    def setUp(self) -> None:
        super().setUp()
        self.playground = self.tmp / "playground"
        self.service = PlaygroundService(root=self.playground)
        self.library = VoiceLibrary(designs=self.designs, trash_root=self.tmp / "trash", playground_root=self.playground)
        self.slug = self.library.create_profile("Anlatıcı", self.source, permission=True)
        self.narrator = SceneNarrator(self.service)

    def scene_project(self, profile: str | None = None) -> SceneProject:
        project = SceneProject(name="Kitap", slug="kitap", profile=profile or self.slug,
                               scenes=[Scene(id="s01", title="Bir", text="Merhaba dünya.")])
        scene = project.scenes[0]
        scene.wav = str(write_wav(self.playground / "scenes" / "kitap" / "s01.wav", value=2))
        scene.fingerprint = self.narrator.fingerprint(project, scene)
        self.narrator.store.save(project)
        return project


class SceneDependencyTest(DependencyFixture):
    def test_delete_is_blocked_while_scenes_use_the_voice(self) -> None:
        self.scene_project()
        self.assertEqual(self.library.usages(self.slug), {"scenes": ["Kitap"]})
        with self.assertRaises(VoiceLibraryError) as raised:
            self.library.delete(self.slug, self.slug)
        self.assertIn("Kitap", str(raised.exception))
        self.assertTrue((self.profiles / self.slug).is_dir())  # hiçbir şey taşınmadı

    def test_forced_delete_marks_scenes_orphan_and_refuses_merge(self) -> None:
        project = self.scene_project()
        self.assertEqual(self.narrator.scene_state(project, project.scenes[0]), "ready")
        self.library.delete(self.slug, self.slug, force=True)
        self.assertEqual(self.narrator.scene_state(project, project.scenes[0]), "orphan")
        with self.assertRaises(ControlError) as merge:
            self.narrator.merge(project)
        self.assertIn("kütüphanede yok", str(merge.exception))
        with self.assertRaises(ControlError):
            self.narrator.generate_scene(project, "s01")
        self.assertEqual(len(self.narrator.generate_all(project)), 1)  # sahne başına tekrar eden hata yok
        self.assertTrue(Path(project.scenes[0].wav).is_file())  # mevcut sahne sesi korunur

    def test_rename_keeps_scenes_ready(self) -> None:
        project = self.scene_project()
        self.library.rename(self.slug, "Başka ad")
        self.assertEqual(self.narrator.scene_state(project, project.scenes[0]), "ready")

    def test_design_voice_scene_usage_blocks_archive(self) -> None:
        self.scene_project(profile=DESIGN_KEY)
        self.assertEqual(self.library.usages(DESIGN_KEY), {"scenes": ["Kitap"]})
        with self.assertRaises(VoiceLibraryError):
            self.library.delete(DESIGN_KEY, "a" * 8)


class TranscriptLifecycleTest(DependencyFixture):
    def write_transcript(self, key: str) -> Path:
        path = self.playground / "transcripts" / f"{key.replace(':', '-')}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Eski kaydın transkripti", encoding="utf-8")
        return path

    def test_deleted_profile_does_not_leak_transcript_to_a_reused_slug(self) -> None:
        transcript = self.write_transcript(self.slug)
        self.library.delete(self.slug, self.slug)
        self.assertFalse(transcript.exists())
        parked = list((self.tmp / "trash" / "transcripts").glob(f"{self.slug}-*.txt"))
        self.assertEqual([p.read_text(encoding="utf-8") for p in parked], ["Eski kaydın transkripti"])
        self.assertEqual(self.service.get_transcript(self.slug), "")

    def test_replaced_reference_parks_old_transcript_but_reprocess_keeps_it(self) -> None:
        transcript = self.write_transcript(self.slug)
        self.library.reprocess(self.slug)
        self.assertTrue(transcript.exists())
        self.library.replace_reference(self.slug, write_wav(self.tmp / "yeni.wav", value=9), permission=True)
        self.assertFalse(transcript.exists())
        self.assertEqual(len(list((self.tmp / "trash" / "transcripts").glob("*.txt"))), 1)

    def test_design_archive_parks_its_transcript_with_a_valid_windows_name(self) -> None:
        self.write_transcript(DESIGN_KEY)
        self.library.delete(DESIGN_KEY, "a" * 8)
        parked = list((self.tmp / "trash" / "transcripts").glob("*.txt"))
        self.assertEqual(len(parked), 1)
        self.assertNotIn(":", parked[0].name)


class OneOffReferenceTest(DependencyFixture):
    def make_reference(self, name: str, age_days: float) -> Path:
        path = write_wav(self.playground / "references" / name)
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))
        return path

    def test_prune_removes_only_old_unreferenced_files(self) -> None:
        used = self.make_reference("reference_20260101-000000_000001.wav", 30)
        unused = self.make_reference("reference_20260101-000000_000002.wav", 30)
        young = self.make_reference("reference_20261120-000000_000003.wav", 1)
        (self.playground / "audio").mkdir(parents=True)
        (self.playground / "audio" / "kayit.json").write_text(json.dumps({"reference": str(used)}), encoding="utf-8")
        removed = self.service.prune_references(max_age_days=14)
        self.assertEqual(removed, [unused])
        self.assertTrue(used.is_file() and young.is_file())
        self.assertTrue((self.playground / "audio" / "kayit.json").is_file())

    def test_prune_keeps_references_used_by_take_recipes(self) -> None:
        used = self.make_reference("reference_20260101-000000_000009.wav", 40)
        (self.playground / "takes").mkdir(parents=True)
        (self.playground / "takes" / "grup.json").write_text(json.dumps({"recipe": {"reference": str(used)}}), encoding="utf-8")
        self.assertEqual(self.service.prune_references(max_age_days=14), [])

    def test_failed_generation_discards_its_fresh_reference(self) -> None:
        fresh = self.make_reference("reference_20261121-000000_000004.wav", 0)
        kept = self.make_reference("reference_20261121-000000_000005.wav", 0)
        self.assertTrue(self.service.discard_reference_if_unused(fresh, []))
        self.assertFalse(fresh.exists())
        self.assertFalse(self.service.discard_reference_if_unused(kept, [{"reference": str(kept)}]))
        self.assertTrue(kept.exists())
        outside = write_wav(self.tmp / "baska" / "reference_x.wav")
        self.assertFalse(self.service.discard_reference_if_unused(outside, []))  # kendi klasörü dışına dokunmaz
        self.assertTrue(outside.exists())


class ReadinessReportTest(DependencyFixture):
    def test_previous_reports_are_preserved_before_overwrite(self) -> None:
        reports = self.tmp / "reports"
        reports.mkdir()
        json_path, md_path = reports / "finetune_readiness_report.json", reports / "finetune_readiness_report.md"
        with patch.object(readiness, "REPORTS_DIR", reports), patch.object(readiness, "JSON_REPORT_PATH", json_path), \
                patch.object(readiness, "MARKDOWN_REPORT_PATH", md_path):
            self.assertEqual(readiness.preserve_previous_reports(), [])  # ilk çalıştırma: yedek yok
            json_path.write_text('{"dataset_path": "eski"}', encoding="utf-8")
            md_path.write_text("eski", encoding="utf-8")
            first = readiness.preserve_previous_reports()
            second = readiness.preserve_previous_reports()  # aynı saniyede ikinci kez: üzerine yazmaz
        self.assertEqual(len(first), 2)
        self.assertEqual(len({*first, *second}), 4)
        self.assertEqual(json.loads(first[0].read_text(encoding="utf-8"))["dataset_path"], "eski")

    def test_panel_rejects_a_report_from_another_dataset_or_run(self) -> None:
        from app.gradio_xtts_demo import readiness_report_is_current

        report_path = self.tmp / "r.json"
        report_path.write_text("{}", encoding="utf-8")
        dataset = self.tmp / "ds"
        dataset.mkdir()
        report = {"dataset_path": str(dataset)}
        self.assertTrue(readiness_report_is_current(report, report_path, dataset, time.time() - 5))
        self.assertFalse(readiness_report_is_current({"dataset_path": str(self.tmp / "baska")}, report_path, dataset, time.time() - 5))
        self.assertFalse(readiness_report_is_current(report, report_path, dataset, time.time() + 60))


if __name__ == "__main__":
    unittest.main()
