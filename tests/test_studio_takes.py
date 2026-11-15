"""VF-019 regressions for legacy projects and the selected take's PCM in chapter exports."""

from __future__ import annotations

import json
import sqlite3
import struct
import tempfile
import threading
import unittest
import wave
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from voxforge.engines import MOCK_MODEL_ID
from voxforge.playground.adapter import worker_engine_spec
from voxforge.quality.director import QualityDirector
from voxforge.studio import ProjectSettings
from voxforge.studio.project import file_sha256
from voxforge.studio.service import StudioError, StudioService
from voxforge.studio.store import DB_FILENAME, MIGRATIONS


def write_wav(path: Path, amplitude: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", ""))
        wav.writeframes(struct.pack("<hh", amplitude, -amplitude) * 6000)


def pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


class FakeSynthesisManager:
    env = {}

    def __init__(self) -> None:
        self.calls = []
        self.started = threading.Event()
        self.resume = threading.Event()

    def synthesize(self, engine, request, device):
        self.calls.append((engine, request, device))
        if len(self.calls) == 1:
            self.started.set()
            if not self.resume.wait(timeout=5):
                raise TimeoutError("Test did not release the first job.")
        write_wav(Path(request["out"]), 1500 + len(self.calls) * 100)
        model_id = worker_engine_spec(engine, self.env)["model_id"]
        model, _, revision = model_id.partition("@")
        return {
            "worker": {"model": model_id, "source": {"model": model, "revision": revision}, "device": device},
            "seed": request["seed"], "applied_settings": request["settings"],
        }


class StudioTakeTests(unittest.TestCase):
    def test_narration_chapter_inserts_paragraph_and_sentence_pauses_around_selected_takes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.wav"
            write_wav(reference, 1000)
            manager = FakeSynthesisManager()
            manager.resume.set()
            service = StudioService(root / "projects", manager=manager)
            slug = service.create(
                "Anlatım", "Sabah oldu. Kapıyı açtım.\n\nDışarı çıktım.", reference,
                {"pacing": "narration", "sentence_pause": 0.35, "paragraph_pause": 0.85},
            )
            self.assertEqual(service.start(slug), 3)
            service.wait(slug, 5)
            segments = service.segments(slug)
            self.assertEqual([s["paragraph"] for s in segments], [0, 0, 1])
            paths = [Path(s["audio_path"]) for s in segments]
            original_parts = [pcm(path) for path in paths]
            first, second, third = original_parts
            sentence_pause = bytes(8400 * 2)  # 0.35 s at 24 kHz, mono int16
            paragraph_pause = bytes(20400 * 2)  # 0.85 s at 24 kHz, mono int16
            original = first + sentence_pause + second + paragraph_pause + third
            self.assertEqual(pcm(service.build_preview(slug)), original)

            segment_id, key = segments[1]["id"], segments[1]["cache_key"]
            service.start_take(slug, segment_id, key)
            service.wait(slug, 5)
            _, alternative = service.take_view(slug, segment_id)["takes"]
            alternative_pcm = pcm(Path(alternative["path"]))
            self.assertNotEqual(alternative_pcm, second)
            service.change_take(slug, segment_id, key, alternative["id"])
            self.assertEqual(
                pcm(service.build_preview(slug)),
                first + sentence_pause + alternative_pcm + paragraph_pause + third,
            )
            service.change_take(slug, segment_id, key, None)
            self.assertEqual(pcm(service.build_preview(slug)), original)
            self.assertEqual([pcm(path) for path in paths], original_parts)
            self.assertEqual(len(manager.calls), 4)
            self.assertTrue(all(request["pauses"] == [] for _, request, _ in manager.calls))

    def test_xtts_chapter_keeps_gapless_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.wav"
            write_wav(reference, 1000)
            manager = FakeSynthesisManager()
            manager.resume.set()
            service = StudioService(root / "projects", manager=manager)
            slug = service.create(
                "XTTS", "Sabah oldu. Kapıyı açtım.\n\nDışarı çıktım.", reference,
                {"temperature": 0.7}, engine_id="xtts",
            )
            self.assertEqual(service.start(slug), 3)
            service.wait(slug, 5)
            expected = b"".join(pcm(Path(s["audio_path"])) for s in service.segments(slug))
            self.assertEqual(pcm(service.build_preview(slug)), expected)

    def test_worker_stop_resume_character_take_and_chapter_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            narrator_ref, character_ref = root / "narrator.wav", root / "character.wav"
            write_wav(narrator_ref, 1000)
            write_wav(character_ref, 2000)
            manager = FakeSynthesisManager()
            service = StudioService(root / "projects", manager=manager, quality=QualityDirector(None))
            slug = service.create("Kitap", "Sabah oldu.\n\nKapıyı açtım.", narrator_ref)
            project = service.open(slug)
            first, second = service.segments(slug)
            voice = service.add_voice(slug, "Ada", character_ref, {"cfg": 2.5})
            service.assign_voice(slug, [second["id"]], voice)
            self.assertEqual(service.start(slug), 2)
            self.assertTrue(manager.started.wait(timeout=5))
            self.assertEqual(service.stop(slug), 1)
            manager.resume.set()
            service.wait(slug, 5)
            self.assertEqual([s["state"] for s in service.segments(slug)], ["done", "cancelled"])
            self.assertEqual(service.start(slug), 1)
            service.wait(slug, 5)
            original = service.build_preview(slug)
            first_path, second_path = (Path(s["audio_path"]) for s in service.segments(slug))
            self.assertEqual(pcm(original), pcm(first_path) + pcm(second_path))

            segment_id = second["id"]
            item = service.segment(slug, segment_id)
            project.enqueue_take(segment_id, item["cache_key"])
            self.assertEqual(service.stop(slug), 1)
            self.assertEqual(service.start(slug), 1)
            service.wait(slug, 5)
            after = service.take_view(slug, segment_id)
            base, alternative = after["takes"]
            self.assertIsNotNone(base["quality"])
            self.assertIsNotNone(alternative["quality"])
            self.assertEqual(after["selected_take"], 0)
            self.assertEqual(len(manager.calls), 3)
            self.assertTrue(all(engine == "voxcpm2" and device == "cuda" for engine, _, device in manager.calls))
            second_request, take_request = [call[1] for call in manager.calls[1:]]
            self.assertEqual(second_request["chunks"], ["Kapıyı açtım."])
            self.assertEqual(pcm(Path(second_request["reference"])), pcm(character_ref))
            self.assertEqual(take_request["reference"], second_request["reference"])
            self.assertEqual(take_request["settings"], {"cfg_value": 2.5})
            self.assertNotEqual(take_request["seed"], second_request["seed"])

            service.change_take(slug, segment_id, item["cache_key"], alternative["id"])
            reopened = StudioService(service.root, manager=manager)
            selected = reopened.build_preview(slug)
            self.assertEqual(pcm(selected), pcm(first_path) + pcm(Path(alternative["path"])))
            self.assertNotEqual(pcm(selected), pcm(original))
            self.assertEqual(reopened.open(slug).orphan_audio_files(), [])
            Path(alternative["path"]).unlink()
            with self.assertRaises(StudioError):
                reopened.build_preview(slug)
            reopened.change_take(slug, segment_id, item["cache_key"], None)
            self.assertEqual(pcm(reopened.build_preview(slug)), pcm(original))
            reopened.set_spoken(slug, segment_id, "Yeni metin.")
            with self.assertRaisesRegex(StudioError, "Segment değişti"):
                reopened.change_take(slug, segment_id, item["cache_key"], alternative["id"])
            with self.assertRaises(StudioError):
                reopened.build_preview(slug)

    def test_v3_artifact_keeps_its_key_and_voice_after_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"VOXFORGE_TTS_ENGINE": "mock"}):
            root = Path(tmp)
            folder = root / "legacy"
            wav = folder / "audio" / "original.wav"
            write_wav(wav, 1500)
            settings = ProjectSettings("mock", MOCK_MODEL_ID)
            service = StudioService(root)
            with closing(sqlite3.connect(folder / DB_FILENAME)) as db:
                db.executescript("".join(MIGRATIONS[v] for v in (1, 2, 3)) + "PRAGMA user_version = 3;")
                db.execute("INSERT INTO project VALUES (1, 'Eski', 'now', ?)", (json.dumps(settings.__dict__),))
                db.execute("INSERT INTO voice VALUES ('ada', 'Ada', NULL, '{}', 'now')")
                db.execute(
                    "INSERT INTO segment (id, ord, source_text, spoken_text, voice_id, updated_at)"
                    " VALUES (1, 0, 'Merhaba.', 'Merhaba.', 'ada', 'now')"
                )
                # Key from v3's canonical cache payload, independent of the code under test.
                key = "ff116a364e2687fb4197920d95835551b57ac115114d663a8c5d4167d54db075"
                db.execute("INSERT INTO artifact VALUES (1, 1, ?, ?, ?, 0.5, 'now')",
                           (key, str(wav), file_sha256(wav)))
                db.commit()
            self.assertEqual(service.list_projects(), [("Eski", "legacy")])
            backup = folder / f"{DB_FILENAME}.bak-v3"
            self.assertFalse(backup.exists())
            item = service.take_view("legacy", 1)
            self.assertEqual(item["state"], "done")
            self.assertEqual(item["voice_name"], "Ada")
            self.assertEqual(item["selected_take"], 0)
            service.check_take("legacy", 1, item["cache_key"], 0)
            self.assertIsNotNone(service.take_view("legacy", 1)["takes"][0]["quality"])
            self.assertEqual(pcm(service.build_preview("legacy")), pcm(wav))
            with closing(sqlite3.connect(backup)) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 3)
                self.assertEqual(db.execute("SELECT cache_key, sha256 FROM artifact").fetchone(), (key, file_sha256(wav)))
