"""VF-Q1: Chatterbox must come from the pinned GitHub commit, and results record the source."""

from __future__ import annotations

import json
import re
import unittest
from unittest import mock

from scripts.benchmark import check_engine, run_engine, sources

PIN = sources.load_pins()["chatterbox"]
GIT_DIRECT_URL = json.dumps(
    {"url": PIN["repo"].removesuffix(".git"), "vcs_info": {"vcs": "git", "commit_id": PIN["commit"], "requested_revision": PIN["commit"]}}
)


class SourceTests(unittest.TestCase):
    def test_pin_is_a_full_commit(self) -> None:
        self.assertRegex(PIN["commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(PIN["repo"].startswith("https://github.com/resemble-ai/chatterbox"))

    def test_direct_url_parsing(self) -> None:
        git = sources.parse_direct_url(GIT_DIRECT_URL)
        self.assertEqual((git["source"], git["commit"]), ("git", PIN["commit"]))
        self.assertEqual(sources.parse_direct_url(None)["source"], "index")  # PyPI kurulumu
        local = sources.parse_direct_url(json.dumps({"url": "file:///x", "dir_info": {"editable": True}}))
        self.assertEqual(local["source"], "local")

    def test_pypi_build_with_same_version_is_rejected(self) -> None:
        pypi = {"source": "index", "url": None, "commit": None, "version": "0.1.7"}
        problem = sources.chatterbox_pin_problem(pypi, PIN)
        self.assertIn("PyPI", problem)
        pinned = {**sources.parse_direct_url(GIT_DIRECT_URL), "version": "0.1.7"}
        self.assertIsNone(sources.chatterbox_pin_problem(pinned, PIN))
        other = {**pinned, "commit": "0" * 40}
        self.assertIsNotNone(sources.chatterbox_pin_problem(other, PIN))

    def test_results_record_source_revision_not_only_version(self) -> None:
        pinned = {**sources.parse_direct_url(GIT_DIRECT_URL), "version": "0.1.7"}
        with mock.patch.object(run_engine, "source_revision", lambda name: pinned if name == "chatterbox-tts" else None):
            versions = run_engine.package_versions()
        self.assertEqual(versions["chatterbox-tts@source"], f"{pinned['url']}@{PIN['commit']}")

    def test_runner_refuses_unpinned_chatterbox_before_loading(self) -> None:
        pypi = {"source": "index", "url": None, "commit": None, "version": "0.1.7"}
        with mock.patch.object(run_engine, "source_revision", lambda name: pypi):
            with self.assertRaisesRegex(RuntimeError, "sabitlenen kaynaktan değil"):
                run_engine.load_chatterbox({"device": "cpu", "reference": "x.wav"})

    def test_check_fails_cleanly_without_chatterbox(self) -> None:
        with mock.patch.object(check_engine, "source_revision", lambda name: None):
            report = check_engine.check_chatterbox()
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["source_ok"])
        self.assertTrue(re.search("kurulu değil|içe aktarılamadı", report["error"]))


if __name__ == "__main__":
    unittest.main()
