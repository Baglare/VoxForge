"""VF-024: ses silme/değiştirme sonrası bağımlılıklar, geçici referanslar, rapor koruma (model yok)."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import finetune_readiness_report as readiness
from tests.test_voice_library import LibraryFixture, StubManager, fake_preprocess, write_wav
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


class UiFixture(DependencyFixture):
    """Scene and quick-voice handlers wired to a temporary library (no model)."""

    def setUp(self) -> None:
        super().setUp()
        from app.voice_playground import build_app
        from tests.test_quick_voice_ui import handlers

        self.service = PlaygroundService(root=self.playground, manager=StubManager())
        for target in (patch("voxforge.voices.VoiceLibrary", lambda *a, **k: self.library),
                       patch("scripts.audio_preprocessing_utils.preprocess_reference_audio", fake_preprocess)):
            target.start()
            self.addCleanup(target.stop)
        self.events = handlers(build_app(self.service, library=self.library))

    def references(self) -> list[Path]:
        return sorted((self.playground / "references").glob("*.wav"))


class OrphanSceneUiTest(UiFixture):
    def stored_profile(self) -> str | None:
        return json.loads((self.playground / "scenes" / "kitap.json").read_text(encoding="utf-8"))["profile"]

    def load_orphan(self) -> list:
        self.scene_project()
        self.library.delete(self.slug, self.slug, force=True)
        loaded = self.events["on_scene_load"].fn("kitap")
        self.assertIsNone(loaded[1])  # menü boş gelir (Gradio bilinmeyen değeri reddeder)
        self.assertIn("kütüphanede yok", loaded[-2])
        return loaded

    def test_failed_scene_generation_keeps_the_original_voice_id(self) -> None:
        loaded = self.load_orphan()
        name, profile, variant, seed, gap, _text, project, rows = loaded[:8]
        out = self.events["on_scene_generate_one"].fn(project, name, profile, variant, seed, gap, rows, 1, "cpu")
        self.assertIn("Hata", out[-2])
        self.assertEqual(out[0].profile, self.slug)
        self.assertEqual(self.stored_profile(), self.slug)  # başarısız üretim sonrası kaydedilen dosya
        self.assertTrue(Path(out[0].scenes[0].wav).is_file())

    def test_generate_all_and_merge_keep_the_original_voice_id(self) -> None:
        loaded = self.load_orphan()
        name, profile, variant, seed, gap, _text, project, rows = loaded[:8]
        self.events["on_scene_generate_all"].fn(project, name, profile, variant, seed, gap, rows, "cpu")
        self.assertEqual(self.stored_profile(), self.slug)
        merged = self.events["on_scene_merge"].fn(project, name, profile, variant, seed, gap, rows, "cpu")
        self.assertIn("kütüphanede yok", merged[-2])
        self.assertEqual(self.stored_profile(), self.slug)

    def test_explicit_new_voice_replaces_the_orphan_id(self) -> None:
        loaded = self.load_orphan()
        name, _profile, variant, seed, gap, _text, project, rows = loaded[:8]
        new_slug = self.library.create_profile("Yeni Ses", self.source, permission=True)
        self.events["on_scene_generate_one"].fn(project, name, new_slug, variant, seed, gap, rows, 1, "cpu")
        self.assertEqual(self.stored_profile(), new_slug)


class OneOffReferenceUiTest(UiFixture):
    def generate_args(self, **values) -> list:
        names = {"engine": 1, "cfg": 10, "source": 17, "upload": 18, "permission": 19}
        event = self.events["on_generate"]
        args = [component.value for component in event.inputs]
        args[0] = "Merhaba dünya."
        args[17], args[18], args[19] = "upload", str(write_wav(self.tmp / "tek.wav", value=3)), True
        for name, value in values.items():
            args[names[name]] = value
        return args

    def test_invalid_controls_leave_no_reference_behind(self) -> None:
        result = self.events["on_generate"].fn(*self.generate_args(cfg=999))
        self.assertIn("CFG", result[5])
        self.assertEqual(self.references(), [])

    def test_unexpected_failure_before_generation_discards_the_reference(self) -> None:
        with patch.object(self.service, "generate_many", side_effect=RuntimeError("motor yok")):
            with self.assertRaises(RuntimeError):
                self.events["on_generate"].fn(*self.generate_args())
        self.assertEqual(self.references(), [])

    def test_failed_generation_discards_and_success_keeps_the_reference(self) -> None:
        with patch.object(self.service, "generate_many", return_value=([], ["hata"])):
            self.events["on_generate"].fn(*self.generate_args())
        self.assertEqual(self.references(), [])
        self.events["on_generate"].fn(*self.generate_args())
        kept = self.references()
        self.assertEqual(len(kept), 1)
        self.assertEqual(self.service.history()[0]["reference"], str(kept[0]))

    def test_preprocessing_crash_removes_the_partial_file(self) -> None:
        from voxforge.playground.service import prepare_uploaded_reference

        def crash(source, target, mode="safe"):
            Path(target).write_bytes(b"yarim")
            raise RuntimeError("ffprobe çöktü")

        upload = write_wav(self.tmp / "kaynak.wav", value=4)
        with patch("scripts.audio_preprocessing_utils.preprocess_reference_audio", crash):
            with self.assertRaises(RuntimeError):
                prepare_uploaded_reference(str(upload), self.playground / "references")
        self.assertEqual(self.references(), [])
        self.assertTrue(upload.is_file())  # kullanıcının kaynak dosyasına dokunulmaz


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

        dataset = self.tmp / "ds"
        dataset.mkdir()
        report = {"dataset_path": str(dataset), "run_id": "bu-calistirma"}
        self.assertTrue(readiness_report_is_current(report, dataset, "bu-calistirma"))
        self.assertFalse(readiness_report_is_current({**report, "dataset_path": str(self.tmp / "baska")}, dataset, "bu-calistirma"))
        self.assertFalse(readiness_report_is_current(report, dataset, "sonraki-calistirma"))
        self.assertFalse(readiness_report_is_current({"dataset_path": str(dataset)}, dataset, "bu-calistirma"))  # eski biçim
        self.assertFalse(readiness_report_is_current(report, dataset, ""))

    def test_quick_rerun_with_failing_script_never_shows_the_previous_report(self) -> None:
        """Same dataset, back-to-back runs: the second script writes nothing, the old file is fresh."""
        import subprocess

        from app import gradio_xtts_demo as demo

        reports = self.tmp / "reports"
        dataset = self.tmp / "ds"
        dataset.mkdir()
        json_path = reports / "finetune_readiness_report.json"
        calls: list[list[str]] = []

        def first_run(script, args, timeout_seconds):
            calls.append(args)
            reports.mkdir(exist_ok=True)
            run_id = args[args.index("--run-id") + 1]
            json_path.write_text(json.dumps({"dataset_path": str(dataset), "run_id": run_id,
                                             "readiness_level": "READY"}), encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")

        def failing_run(script, args, timeout_seconds):
            calls.append(args)
            return subprocess.CompletedProcess(args, 2, "", "betik çöktü")

        with patch.object(demo, "REPORTS_DIR", reports), patch.object(demo, "dataset_slug_to_dir", lambda _s: dataset):
            with patch.object(demo, "run_finetune_subprocess", first_run):
                self.assertIn("READY", demo.run_readiness_report_from_gradio("ds"))
            with patch.object(demo, "run_finetune_subprocess", failing_run):
                second = demo.run_readiness_report_from_gradio("ds")
        self.assertTrue(second.startswith("HATA"))
        self.assertNotEqual(calls[0][-1], calls[1][-1])  # her çalıştırma yeni kimlik

    def test_script_echoes_the_run_id(self) -> None:
        reports = self.tmp / "reports"
        dataset = self.tmp / "ds"
        dataset.mkdir()
        json_path, md_path = reports / "finetune_readiness_report.json", reports / "finetune_readiness_report.md"
        with patch.object(readiness, "REPORTS_DIR", reports), patch.object(readiness, "JSON_REPORT_PATH", json_path), \
                patch.object(readiness, "MARKDOWN_REPORT_PATH", md_path), patch.object(readiness, "print_report", lambda *a: None):
            readiness.main(["--dataset", str(dataset), "--run-id", "abc123"])
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["run_id"], "abc123")


if __name__ == "__main__":
    unittest.main()
