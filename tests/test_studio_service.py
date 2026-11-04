"""VF-006: Studio service flow behind the Gradio tab, with the Mock engine."""

from __future__ import annotations

import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from scripts.mock_tts_engine import MockTTSEngine
from voxforge.engines import wrap_backend
from voxforge.studio.service import StudioError, StudioService, parse_lexicon, slugify

TEXT = "Toplantı 14:30'da başladı. Katılım %25 arttı. VoxForge sunuldu.\n\nİkinci paragraf burada."


class StudioServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "studio_projects"
        self.backend = MockTTSEngine()
        self.engine = wrap_backend("mock", self.backend, "mock")
        self._env = mock.patch.dict(os.environ, {"VOXFORGE_TTS_ENGINE": "mock"})
        self._env.start()
        self.service = StudioService(self.root, lambda: self.engine)

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def test_full_studio_flow(self) -> None:
        slug = self.service.create("Bölüm 1: Giriş", TEXT, None, lexicon_text="VoxForge = voks forc")
        self.assertEqual(slug, "bolum-1-giris")
        self.assertEqual(self.service.list_projects(), [("Bölüm 1: Giriş", slug)])

        segments = self.service.segments(slug)
        self.assertEqual(len(segments), 4)
        self.assertEqual(segments[0]["spoken_text"], "Toplantı on dört otuz'da başladı.")
        self.assertEqual(segments[0]["transformations"][0]["rule"], "time")
        self.assertEqual(segments[2]["spoken_text"], "voks forc sunuldu.")
        self.assertEqual(segments[3]["paragraph"], 1)

        # Başlat -> hepsi üretilir, önizleme doğru sırada ve tam uzunlukta birleşir.
        with self.assertRaises(StudioError):
            self.service.build_preview(slug)  # henüz eksik segment var
        self.assertEqual(self.service.start(slug), 4)
        self.service.wait(slug, timeout=30)
        progress = self.service.progress(slug)
        self.assertEqual(progress["counts"], {"done": 4})
        self.assertFalse(progress["running"])
        preview = self.service.build_preview(slug)
        self.assertAlmostEqual(self.service.preview_seconds(preview), progress["duration_s"], places=2)

        # Tek segment düzenlenir -> yalnız o yeniden üretilir; ikinci başlatma tek iş kuyruğa alır.
        first = segments[1]["id"]
        self.service.update_source(slug, first, "Katılım %30 arttı.")
        self.assertEqual(self.service.segment(slug, first)["spoken_text"], "Katılım yüzde otuz arttı.")
        self.service.set_spoken(slug, segments[3]["id"], "İkinci paragraf, elle okunuş.")
        self.assertEqual(self.service.segment(slug, segments[3]["id"])["transformations"][0]["rule"], "manual")
        self.backend.calls.clear()
        self.assertEqual(self.service.start(slug), 2)
        self.service.wait(slug, timeout=30)
        self.assertEqual(
            sorted(call["text"] for call in self.backend.calls),
            ["Katılım yüzde otuz arttı.", "İkinci paragraf, elle okunuş."],
        )
        self.assertEqual(self.service.start(slug), 0)  # her şey güncel

        # Yeniden açılan servis (uygulama yeniden başlatıldı) aynı durumu görür.
        reopened = StudioService(self.root, lambda: self.engine)
        self.assertEqual(reopened.progress(slug)["counts"], {"done": 4})

    def test_stop_cancels_queue_and_restart_resumes(self) -> None:
        slug = self.service.create("Durdur", "Bir. İki. Üç. Dört.", None)
        project = self.service.open(slug)
        project.enqueue_missing()
        self.assertEqual(self.service.stop(slug), 4)
        self.assertEqual(self.service.progress(slug)["counts"], {"cancelled": 4})
        self.assertEqual(self.service.start(slug), 4)
        self.service.wait(slug, timeout=30)
        self.assertEqual(self.service.progress(slug)["counts"], {"done": 4})

    def test_safety_rules(self) -> None:
        with self.assertRaises(StudioError):
            self.service.project_dir("../disari")
        with self.assertRaises(StudioError):
            self.service.create("", "metin", None)
        with self.assertRaises(StudioError):
            parse_lexicon("eşittir işareti yok")
        self.assertEqual(slugify("Çığ Öykü!"), "cig-oyku")

        # XTTS projesi referans ister; referans proje klasörüne kopyalanır.
        reference = Path(self._tmp.name) / "profil.wav"
        MockTTSEngine().tts_to_file(text="referans", file_path=str(reference))
        with mock.patch.dict(os.environ, {"VOXFORGE_TTS_ENGINE": "xtts"}):
            with self.assertRaises(StudioError):
                self.service.create("Ref yok", "Metin.", None)
            slug = self.service.create("Ref var", "Metin.", reference, preset={"temperature": 0.7})
            settings = self.service.open(slug).settings
        self.assertEqual(Path(settings.reference_path).parent, self.service.project_dir(slug))
        self.assertEqual(settings.preset, {"temperature": 0.7})
        with self.assertRaises(StudioError):
            self.service.start(slug)  # uygulama mock ile çalışıyor; motor uyuşmazlığı açıkça reddedilir
        with wave.open(settings.reference_path, "rb") as handle:
            self.assertGreater(handle.getnframes(), 0)


if __name__ == "__main__":
    unittest.main()
