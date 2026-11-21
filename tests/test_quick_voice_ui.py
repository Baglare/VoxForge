"""Regression checks for quick generation, the voice library tab and developer tools wiring."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.voice_playground import build_app
from tests.test_voice_library import DESIGN_ID, LibraryFixture, StubManager, fake_preprocess, write_wav
from voxforge.playground.service import PlaygroundService


def handlers(app):
    return {event.fn.__name__: event for event in app.fns.values() if event.fn}


class QuickVoiceTest(unittest.TestCase):
    def test_reference_gate_and_generation_handler(self):
        with tempfile.TemporaryDirectory() as temp:
            service = PlaygroundService(Path(temp))
            app = build_app(service)
            events = handlers(app)
            gate = events["generation_button"]
            engine = gate.inputs[0]
            self.assertEqual(engine.value, "voxcpm2")
            self.assertFalse(gate.outputs[0].interactive)
            for name in ("xtts", "voxcpm2", "chatterbox-v3"):
                self.assertFalse(gate.fn(name, None)["interactive"])
                self.assertTrue(gate.fn(name, "voice")["interactive"])
                self.assertFalse(gate.fn(name, "voice", "upload", None)["interactive"])
                self.assertTrue(gate.fn(name, None, "upload", "kayit.wav")["interactive"])
            self.assertTrue(gate.fn("mock", None)["interactive"])
            event = events["on_generate"]
            args = [component.value for component in event.inputs]
            with patch.object(service, "generate_many", return_value=([], [])) as generate:
                self.assertIsNone(event.fn(*args)[-1])
                generate.assert_not_called()
                args[6] = "voice"
                event.fn(*args)
                self.assertEqual(generate.call_args.args[1:3], (["voxcpm2"], "voice"))
                self.assertIsNone(generate.call_args.kwargs["engine_settings"])
                self.assertFalse(generate.call_args.kwargs["post_process"])


class QuickVoiceLibraryTest(LibraryFixture):
    def setUp(self):
        super().setUp()
        self.manager = StubManager()
        self.service = PlaygroundService(self.tmp / "playground", manager=self.manager)
        for target in (patch("voxforge.voices.VoiceLibrary", lambda *a, **k: self.library),
                       patch("scripts.audio_preprocessing_utils.preprocess_reference_audio", fake_preprocess)):
            target.start()
            self.addCleanup(target.stop)
        self.app = build_app(self.service, library=self.library)
        self.events = handlers(self.app)
        self.slug = self.library.create_profile("Anlatıcı", self.source, permission=True)

    def generate_args(self, **values):
        event = self.events["on_generate"]
        names = {
            "text": 0, "engine": 1, "profile": 6, "seed": 7, "source": 17, "upload": 18, "permission": 19,
            "xtts_mode": 20, "post": 21, "ab": 22, "ab_choice": 23,
        }
        args = [component.value for component in event.inputs]
        for name, value in values.items():
            args[names[name]] = value
        return event, args

    def test_xtts_mode_and_ab_share_voice_and_seed(self):
        event, args = self.generate_args(
            engine="xtts", profile=self.slug, seed=77, xtts_mode="Daha doğal deneme", ab=True, ab_choice="xtts:Daha stabil",
        )
        result = event.fn(*args)
        (_e1, a), (_e2, b) = self.manager.requests
        self.assertEqual(a["settings"], {"temperature": 0.75, "top_p": 0.9, "top_k": 80})
        self.assertEqual(b["settings"]["repetition_penalty"], 5.0)
        self.assertEqual((a["seed"], a["reference"]), (b["seed"], b["reference"]))
        self.assertTrue(result[-3]["visible"])  # B audio
        self.assertTrue(result[-1])  # A record id feeds the quality chain

    def test_design_voice_in_quick_voice_and_menus(self):
        key = f"design:{DESIGN_ID}"
        event, args = self.generate_args(profile=key, ab=True, ab_choice="preset:yavas-anlatici")
        event.fn(*args)
        self.assertTrue(all(Path(r["reference"]).name == "voice.wav" for _e, r in self.manager.requests))
        self.assertEqual(len(self.manager.requests), 2)

    def test_one_off_upload_needs_permission_and_is_preprocessed(self):
        upload = write_wav(self.tmp / "tek.wav", value=3)
        event, args = self.generate_args(source="upload", upload=str(upload), permission=False)
        result = event.fn(*args)
        self.assertEqual(self.manager.requests, [])
        self.assertIn("onaylayın", result[5])
        args[19] = True
        result = event.fn(*args)
        reference = Path(self.manager.requests[-1][1]["reference"])
        self.assertEqual(reference.parent, self.service.root / "references")
        self.assertIn("Tek seferlik referans", result[5])
        self.assertIsNone(self.service.history()[0]["profile"])

    def test_library_changes_refresh_every_voice_menu(self):
        create = self.events["on_create"]
        upload = write_wav(self.tmp / "yeni.wav", value=4)
        targets = create.inputs[4:]
        self.assertGreaterEqual(len(targets), 4)  # quick, scenes, studio narrator + character (+ design menu)
        result = create.fn("İkinci Ses", str(upload), True, None, *[t.value for t in targets])
        self.assertIn("Profil oluşturuldu", result[0])
        new_keys = {value for _label, value in result[-len(targets)]["choices"]}
        self.assertIn("ikinci-ses", new_keys)
        self.assertIn(f"design:{DESIGN_ID}", new_keys)
        studio_choices = {value for _label, value in result[-len(targets) + 2]["choices"]}
        self.assertIn("", studio_choices)
        delete = self.events["on_delete"]
        result = delete.fn("ikinci-ses", "ikinci-ses", False, "ikinci-ses", None, "", "", None)
        self.assertIsNone(result[-len(targets)]["value"])  # deleted voice is no longer selected
        self.assertNotIn("ikinci-ses", {value for _label, value in result[-len(targets)]["choices"]})

    def test_developer_tools_handlers_are_wired(self):
        names = set(self.events)
        for name in ("run_readiness_report_from_gradio", "export_experiment_from_gradio",
                     "run_training_dry_run_from_gradio", "refresh_datasets"):
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()
