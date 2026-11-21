"""Gradio dosya izinleri: junction üzerinden çözülen profil/çıktı yolları (VF-024)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from voxforge.ui_paths import gradio_allowed_paths


def make_junction(link: Path, target: Path) -> bool:
    if sys.platform != "win32":
        link.symlink_to(target, target_is_directory=True)
        return True
    return subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True).returncode == 0


class UiPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.real = self.tmp / "data" / "profiles"
        (self.real / "ses").mkdir(parents=True)
        self.wav = self.real / "ses" / "preprocessed_reference.wav"
        self.wav.write_bytes(b"RIFF....WAVE")
        self.link = self.tmp / "profiles"
        if not make_junction(self.link, self.real):
            self.skipTest("junction oluşturulamadı")

    def test_allowed_paths_are_resolved_and_narrow(self) -> None:
        with patch("scripts.voice_profile_utils.PROFILES_DIR", self.link):
            paths = gradio_allowed_paths()
        self.assertIn(str(self.real.resolve()), paths)
        self.assertNotIn(str(self.link), paths)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertNotIn(str(Path(__file__).resolve().parents[1]), paths)  # datasets/experiments açılmaz

    def _serve(self, allowed: list[str]):
        served = str(self.link / "ses" / "preprocessed_reference.wav")
        with gr.Blocks() as app:
            button = gr.Button("go")
            audio = gr.Audio(type="filepath")
            button.click(lambda: served, None, audio, api_name="go")
        app.launch(prevent_thread_lock=True, allowed_paths=allowed, quiet=True)
        return app

    def test_gradio_serves_audio_behind_junction(self) -> None:
        """Geçici klasör Gradio'da zaten izinli olduğundan bu yalnız duman testidir; eski hata
        gerçek C:\AvenoxData junction'ıyla yeniden üretilip doğrulandı (rapora bakın)."""
        from gradio_client import Client

        with patch("scripts.voice_profile_utils.PROFILES_DIR", self.link):
            allowed = gradio_allowed_paths()
        app = self._serve(allowed)
        try:
            self.assertTrue(Client(app.local_url, verbose=False).predict(api_name="/go"))
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
