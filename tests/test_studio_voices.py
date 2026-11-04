"""VF-007: narrator + three characters keep correct segment/voice mapping (Mock engine)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.mock_tts_engine import MockTTSEngine
from voxforge.engines import wrap_backend
from voxforge.studio.service import StudioError, StudioService

SCRIPT = (
    "Akşam oldu ve herkes masaya oturdu.\n"
    "Ayşe: Bugün 3 kitap okudum.\n"
    "Mehmet: Ben de 2 tane bitirdim.\n"
    "Zeynep: Harika!\n"
    "— Gerçekten mi? diye sordu biri.\n"
    "Ali: Ben hiç okumadım.\n"
    "Not: bu satır konuşma değil.\n"
    "Anlatıcı gülümsedi."
)


class StudioVoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.backend = MockTTSEngine()
        self.engine = wrap_backend("mock", self.backend, "mock")
        self._env = mock.patch.dict(os.environ, {"VOXFORGE_TTS_ENGINE": "mock"})
        self._env.start()
        self.service = StudioService(tmp / "studio_projects", lambda: self.engine)
        self.refs = {}
        for name in ("Ayşe", "Mehmet", "Zeynep"):
            path = tmp / f"{name}.wav"
            MockTTSEngine().tts_to_file(text=f"{name} referansı", file_path=str(path))
            self.refs[name] = path

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def run_all(self, slug: str) -> None:
        self.service.start(slug)
        self.service.wait(slug, timeout=30)

    def test_narrator_and_three_characters(self) -> None:
        slug = self.service.create("Akşam", SCRIPT, None)
        segments = self.service.segments(slug)
        self.assertEqual(len(segments), 8)  # her senaryo satırı ayrı segment

        keys = {name: self.service.add_voice(slug, name, ref) for name, ref in self.refs.items()}
        with self.assertRaises(StudioError):
            self.service.add_voice(slug, "ayşe", None)  # aynı karakter ikinci kez eklenmez

        suggestions = {item["number"]: item for item in self.service.dialogue_suggestions(slug)}
        self.assertEqual(sorted(suggestions), [2, 3, 4, 5, 6])  # "Not:" ve düz anlatım öneri değil
        self.assertEqual(suggestions[5]["kind"], "dialogue")
        self.assertIsNone(suggestions[6]["voice_key"])  # Ali için karakter yok

        result = self.service.apply_suggestions(slug)
        self.assertEqual(result, {"assigned": 3, "relabelled": 3, "unmatched": ["Ali"]})

        by_number = {item["ord"] + 1: item for item in self.service.segments(slug)}
        self.assertEqual(by_number[2]["source_text"], "Ayşe: Bugün 3 kitap okudum.")
        self.assertEqual(by_number[2]["spoken_text"], "Bugün üç kitap okudum.")
        expected_voice = {1: None, 2: keys["Ayşe"], 3: keys["Mehmet"], 4: keys["Zeynep"], 5: None, 6: None, 7: None, 8: None}
        self.assertEqual({n: item["voice_id"] for n, item in by_number.items()}, expected_voice)

        self.run_all(slug)
        reference_by_text = {call["text"]: call["speaker_wav"] for call in self.backend.calls}
        project_dir = self.service.project_dir(slug)
        self.assertIsNone(reference_by_text["Akşam oldu ve herkes masaya oturdu."])
        for number, name in ((2, "Ayşe"), (3, "Mehmet"), (4, "Zeynep")):
            used = Path(reference_by_text[by_number[number]["spoken_text"]])
            self.assertEqual(used, project_dir / "voices" / f"{keys[name]}.wav")  # projeye kopyalanan referans
        self.assertEqual(self.service.progress(slug)["counts"], {"done": 8})

        # Zeynep'in satırı Ayşe'ye atanınca yalnız o segment yeniden üretilir, Ayşe'nin referansıyla.
        self.backend.calls.clear()
        ids = self.service.segment_ids_from_numbers(slug, "4")
        self.assertEqual(self.service.assign_voice(slug, ids, keys["Ayşe"]), 1)
        self.assertEqual(self.service.progress(slug)["counts"], {"done": 7, "pending": 1})
        self.run_all(slug)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(Path(self.backend.calls[0]["speaker_wav"]), project_dir / "voices" / f"{keys['Ayşe']}.wav")

        # Uygulama yeniden açıldığında eşleşmeler korunur.
        reopened = StudioService(self.service.root, lambda: self.engine)
        names = [item["voice_name"] for item in reopened.segments(slug)]
        self.assertEqual(names, [None, "Ayşe", "Mehmet", "Ayşe", None, None, None, None])
        self.assertEqual(reopened.progress(slug)["counts"], {"done": 8})

    def test_segment_number_parsing(self) -> None:
        slug = self.service.create("Numara", "Bir. İki. Üç. Dört. Beş.", None)
        ids = [item["id"] for item in self.service.segments(slug)]
        self.assertEqual(self.service.segment_ids_from_numbers(slug, "2, 4-5"), [ids[1], ids[3], ids[4]])
        for bad in ("", "9", "a-b"):
            with self.assertRaises(StudioError):
                self.service.segment_ids_from_numbers(slug, bad)


if __name__ == "__main__":
    unittest.main()
