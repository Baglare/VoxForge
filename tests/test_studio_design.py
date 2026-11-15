"""Voice design uses a native preview and immutable project-local character audio."""

import asyncio
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

from test_studio_takes import FakeSynthesisManager, pcm

from voxforge.studio.service import StudioError, StudioService


class DesignManager(FakeSynthesisManager):
    def __init__(self):
        super().__init__()
        self.resume.set()
        self.accept_native_design = True

    def synthesize(self, engine, request, device):
        reply = super().synthesize(engine, {"settings": {}, **request}, device)
        if self.accept_native_design and request.get("voice_design"):
            reply["applied_settings"] = {"mode": "voice_design"}
        return reply


class StudioDesignTests(TestCase):
    def test_rejects_preview_without_native_mode(self):
        with TemporaryDirectory() as directory:
            manager = DesignManager()
            manager.accept_native_design = False
            service = StudioService(Path(directory), manager=manager)
            with self.assertRaisesRegex(StudioError, "referanssız ses tasarımı doğrulanamadı"):
                service.preview_design("Sakin bir anlatıcı", "Günaydın.")
            self.assertEqual(list((service.designs.root / "previews").iterdir()), [])
            self.assertEqual(service.designs.versions(), [])

    def test_saves_distinct_versions_and_reopens_character_with_same_reference(self):
        with TemporaryDirectory() as directory:
            manager = DesignManager()
            service = StudioService(Path(directory), manager=manager)
            sample = service.preview_design("Sakin bir anlatıcı", "Selam, ben Duru.", 42)
            engine, request, device = manager.calls[0]
            self.assertEqual((engine, device, request["seed"]), ("voxcpm2", "cuda", 42))
            self.assertTrue(request["voice_design"])
            self.assertNotIn("reference", request)
            self.assertNotIn("reference_text", request)
            first = service.save_design(sample["id"], "Duru")
            sample = service.preview_design("Sakin ve tok bir anlatıcı", "Selam, ben Duru.", 44)
            second = service.save_design(sample["id"], "Duru")
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(len(service.designs.versions()), 2)
            self.assertNotEqual(pcm(first["wav"]), pcm(second["wav"]))

            slug = service.create("Kitap", "Gün doğdu. Rüzgâr dindi.", first["wav"])
            key = service.add_designed_voice(slug, "", first["id"])
            studio = StudioService(service.root, manager=manager)
            character, = studio.voices(slug)
            self.assertEqual((character["name"], character["design_version"]), ("Duru", first["id"]))
            copied = Path(character["reference_path"])
            self.assertNotEqual(copied, first["wav"])
            self.assertEqual(pcm(copied), pcm(first["wav"]))
            ids = [item["id"] for item in studio.segments(slug)]
            studio.assign_voice(slug, ids, key)
            self.assertEqual(studio.start(slug), 2)
            studio.wait(slug, 5)
            self.assertEqual([item["state"] for item in studio.segments(slug)], ["done", "done"])
            self.assertEqual([request["reference"] for _, request, _ in manager.calls[2:]], [str(copied)] * 2)
            self.assertTrue(all(not request.get("voice_design") for _, request, _ in manager.calls[2:]))

            mock = service.create("Mock", "Bir cümle.", None, engine_id="mock")
            with self.assertRaisesRegex(StudioError, "aynı VoxCPM2 model revizyonu"):
                service.add_designed_voice(mock, "Duru", first["id"])
            with self.assertRaisesRegex(StudioError, "Geçersiz ses tasarımı kimliği"):
                service.add_designed_voice(slug, "Duru", "../voice")

    @skipUnless(find_spec("gradio"), "Studio UI test requires Gradio")
    def test_studio_preview_save_and_attach_controls(self):
        from gradio.state_holder import SessionState

        from app.studio_ui import build_studio

        with TemporaryDirectory() as directory:
            service = StudioService(Path(directory), manager=DesignManager())
            app = build_studio(
                service, engine_label="voxcpm2", profile_choices=lambda: [("Varsayılan", "")],
                resolve_reference=lambda *_: None, preset_choices=["normal"], preset_kwargs={},
                default_preset="normal", vox_presets={"normal": ("Normal", {})},
            )
            state = SessionState(app)

            def event(name, values):
                fn = next(fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == name)
                return asyncio.run(app.process_api(fn, values, state=state, explicit_call=True))["data"]

            status, audio, _, save_button = event("on_design_preview", ["Sakin bir ses", "Merhaba.", 42])
            self.assertIn("Referanssız örnek hazır", status)
            self.assertTrue(Path(audio["path"]).is_file())
            self.assertTrue(save_button["interactive"])
            _, update = event("on_save_design", [None, "Duru"])
            version = service.designs.version(update["value"])
            audio, = event("on_design_version", [version["id"]])
            self.assertEqual(pcm(Path(audio["path"])), pcm(version["wav"]))

            slug = service.create("Kitap", "Güneş doğdu.", version["wav"])
            fn = next(fn for fn in app.fns.values() if getattr(fn.fn, "__name__", None) == "on_add_design")
            state[fn.inputs[0]._id] = slug
            _, panel, _ = event("on_add_design", [None, version["id"], "Ada", "normal"])
            self.assertIn("Tasarlanmış ses", panel)
