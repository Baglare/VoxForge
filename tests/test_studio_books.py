"""VF-020 document, migration and Studio book workflow regressions."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from importlib.util import find_spec
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

from test_studio_takes import FakeSynthesisManager, pcm, write_wav

from voxforge.engines import MOCK_MODEL_ID
from voxforge.studio.book import WORD
from voxforge.studio.project import ProjectSettings, file_sha256
from voxforge.studio.service import StudioError, StudioService
from voxforge.studio.store import DB_FILENAME, MIGRATIONS, SCHEMA_VERSION
from voxforge.text import LexiconEntry, prepare_segments

TITLES = ("Bölüm 1: Gece", "Bölüm 2: Sabah")
PARAGRAPHS = ("New York’a vardım. Işıklar yandı.", "Ada: Hazır mısın?\n— Hazırım.", "Güneş doğdu.")
EXPECTED = [TITLES[0], "New York’a vardım.", "Işıklar yandı.", "Ada: Hazır mısın?", "— Hazırım.",
            TITLES[1], PARAGRAPHS[2]]


def write_book_fixtures(root: Path) -> list[Path]:
    txt, docx, epub = (root / f"book.{ext}" for ext in ("txt", "docx", "epub"))
    txt.write_bytes(("\ufeff" + "\r\n\r\n".join(
        (TITLES[0], PARAGRAPHS[0], PARAGRAPHS[1].replace("\n", "\r\n"), TITLES[1], PARAGRAPHS[2])
    )).encode("utf-8"))
    with ZipFile(docx, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="word/document.xml"/></Relationships>',
        )
        blocks = []
        for heading, text in ((True, TITLES[0]), (False, PARAGRAPHS[0]), (False, PARAGRAPHS[1]),
                              (True, TITLES[1]), (False, PARAGRAPHS[2])):
            style = '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>' if heading else ""
            contents = "<w:br/>".join(f"<w:t>{escape(line)}</w:t>" for line in text.splitlines())
            blocks.append(f"<w:p>{style}<w:r>{contents}</w:r></w:p>")
        archive.writestr(
            "word/document.xml", f'<w:document xmlns:w="{WORD[1:-1]}"><w:body>{"".join(blocks)}</w:body></w:document>',
        )
    with ZipFile(epub, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
            '<rootfiles><rootfile full-path="OPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        archive.writestr(
            "OPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Deneme</dc:title>'
            '<dc:identifier id="id">urn:example:vf020</dc:identifier><dc:language>tr</dc:language>'
            '<meta property="dcterms:modified">2026-11-15T00:00:00Z</meta></metadata><manifest>'
            '<item id="two" href="Text/chapter%202.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="one" href="Text/chapter1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
            '</manifest><spine><itemref idref="one"/><itemref idref="two"/></spine></package>',
        )
        archive.writestr("OPS/nav.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml"><body>İçindekiler</body></html>')
        archive.writestr(
            "OPS/Text/chapter1.xhtml",
            f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Gece</title></head><body>'
            f'<section><h1>{TITLES[0]}</h1><p><em>New York</em>’a vardım. Işıklar yandı.</p>'
            '<p>Ada: Hazır mısın?<br/>— Hazırım.</p></section></body></html>',
        )
        archive.writestr(
            "OPS/Text/chapter 2.xhtml",
            f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Sabah</title></head><body>'
            f'<h1>{TITLES[1]}</h1><p>{PARAGRAPHS[2]}</p></body></html>',
        )
    return [txt, docx, epub]


class FailOnceManager(FakeSynthesisManager):
    def __init__(self):
        super().__init__()
        self.failed = False

    def synthesize(self, engine, request, device):
        if request["chunks"] == ["Işıklar yandı."] and not self.failed:
            self.failed = True
            self.calls.append((engine, request, device))
            raise RuntimeError("sentetik motor hatası")
        return super().synthesize(engine, request, device)


class StudioBookTests(unittest.TestCase):
    def test_three_formats_and_immutable_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = StudioService(root / "projects", manager=FakeSynthesisManager())
            for document in write_book_fixtures(root):
                with self.subTest(document=document.suffix):
                    slug = service.create(
                        "Kitap", "", None, lexicon_text="~ New = Niyu\n! New York = Niyu York",
                        engine_id="mock", document=document,
                    )
                    self.assertEqual(service.chapters(slug), list(enumerate(TITLES, 1)))
                    segments = service.segments(slug)
                    self.assertEqual([item["source_text"] for item in segments], EXPECTED)
                    self.assertEqual([item["paragraph"] for item in segments], [0, 1, 1, 2, 2, 0, 1])
                    self.assertEqual(segments[1]["spoken_text"], "Niyu York’a vardım.")
                    suggestions = service.dialogue_suggestions(slug)
                    self.assertEqual([(s["speaker"], s["line"]) for s in suggestions],
                                     [("Ada", "Hazır mısın?"), (None, "— Hazırım.")])
                    original = service.project_dir(slug) / f"source{document.suffix}"
                    self.assertEqual(original.read_bytes(), document.read_bytes())
                    with self.assertRaisesRegex(StudioError, "kaynak metin sabittir"):
                        service.update_source(slug, segments[1]["id"], "Başka metin")
                    service.set_spoken(slug, segments[3]["id"], "Hazır mısın?")
                    self.assertEqual(service.update_lexicon(slug, "! New York = Nü York\n! Ada = Aada"), 1)
                    updated = service.segments(slug)
                    self.assertEqual(updated[1]["spoken_text"], "Nü York’a vardım.")
                    self.assertEqual(updated[3]["spoken_text"], "Hazır mısın?")
                    self.assertEqual([item["source_text"] for item in updated], EXPECTED)
                    self.assertEqual(original.read_bytes(), document.read_bytes())

    def test_known_name_stays_whole_at_length_boundary(self):
        text = "Daha sonra hemen eve New York’a vardım."
        self.assertIn("New", prepare_segments(text, max_chars=26)[0].spoken)
        items = prepare_segments(
            text, max_chars=26,
            lexicon=[LexiconEntry("New York", "Niyu York", True)], protect_names=True,
        )
        self.assertEqual([item.spoken for item in items], ["Daha sonra hemen eve", "Niyu York’a vardım."])

    def test_invalid_import_leaves_no_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = root / "broken.docx"
            document.write_text("Word olmayan dosya", encoding="utf-8")
            service = StudioService(root / "projects", manager=FakeSynthesisManager())
            with self.assertRaisesRegex(StudioError, "DOCX belgesi okunamadı"):
                service.create("Kitap", "", None, engine_id="mock", document=document)
            self.assertFalse(service.root.exists())

    def test_v4_bookkeeping_migration_preserves_selected_take_and_undo(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "legacy"
            original, alternative = folder / "audio" / "base.wav", folder / "audio" / "alt.wav"
            write_wav(original, 1500)
            write_wav(alternative, 2500)
            settings = ProjectSettings("mock", MOCK_MODEL_ID)
            key = "ff116a364e2687fb4197920d95835551b57ac115114d663a8c5d4167d54db075"
            with closing(sqlite3.connect(folder / DB_FILENAME)) as db:
                db.executescript("".join(MIGRATIONS[v] for v in (1, 2, 3, 4)) + "PRAGMA user_version = 4;")
                db.execute("INSERT INTO project VALUES (1, 'Eski', 'now', ?)", (json.dumps(settings.__dict__),))
                db.execute(
                    "INSERT INTO segment (id, ord, source_text, spoken_text, updated_at)"
                    " VALUES (1, 0, 'Merhaba.', 'Merhaba.', 'now')"
                )
                db.execute(
                    "INSERT INTO artifact (id, segment_id, cache_key, path, sha256, duration_s, created_at)"
                    " VALUES (1, 1, ?, ?, ?, 0.5, 'now')", (key, str(original), file_sha256(original)),
                )
                db.execute(
                    "INSERT INTO take VALUES (1, 1, 123, ?, ?, 0.5, NULL, '{}', 'now')",
                    (str(alternative), file_sha256(alternative)),
                )
                db.execute("INSERT INTO take_selection VALUES (1, 1, 1)")
                db.commit()
            service = StudioService(folder.parent)
            self.assertEqual(service.chapters("legacy"), [(1, "Eski")])
            self.assertEqual(service.open("legacy").store.schema_version(), SCHEMA_VERSION)
            self.assertEqual(pcm(service.export_audio("legacy", chapter_id=1)), pcm(alternative))
            service.change_take("legacy", 1, key, None)
            self.assertEqual(pcm(service.export_audio("legacy")), pcm(original))
            self.assertEqual(len(service.take_view("legacy", 1)["takes"]), 2)
            with closing(sqlite3.connect(folder / f"{DB_FILENAME}.bak-v4")) as backup:
                self.assertEqual(backup.execute("SELECT * FROM take_selection").fetchall(), [(1, 1, 1)])

    @unittest.skipUnless(find_spec("gradio"), "Studio UI test requires Gradio")
    def test_two_chapter_studio_ui_workflow(self):
        from gradio.data_classes import FileData
        from gradio.state_holder import SessionState

        from app.studio_ui import build_studio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "ref.wav"
            write_wav(reference, 1100)
            _, _, document = write_book_fixtures(root)
            manager = FailOnceManager()
            service = StudioService(root / "projects", manager=manager)
            app = build_studio(
                service, engine_label="voxcpm2", profile_choices=lambda: [("Varsayılan", "")],
                resolve_reference=lambda *_: reference,
                preset_choices=["normal"], preset_kwargs={}, default_preset="normal",
                vox_presets={"normal": ("Normal", {}), "narration": ("Anlatım", {
                    "pacing": "narration", "sentence_pause": 0.35, "paragraph_pause": 0.85,
                })},
            )
            state = SessionState(app)

            def event(name, *values):
                fn = next(fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == name)
                if name in {"on_create", "on_project_change", "on_book_controls"}:
                    return asyncio.run(app.process_api(fn, list(values), state=state, explicit_call=True))["data"]
                if name == "on_preview":
                    slug, chapter_id, audio_format = values
                    self.assertEqual(state[fn.inputs[0]._id], slug)
                    status, update = asyncio.run(
                        app.process_api(fn, [None, chapter_id, audio_format], state=state, explicit_call=True)
                    )["data"]
                    if update.get("value"):
                        update["value"] = update["value"]["path"]
                    return status, update
                return asyncio.run(app.call_function(fn, list(values)))["prediction"]

            def take_event(action, *values):
                fn = next(fn for fn in app.fns.values() if getattr(fn.fn, "__defaults__", None) == (action,))
                return asyncio.run(app.call_function(fn, list(values)))["prediction"]

            upload = FileData(path=str(document), orig_name=document.name).model_dump()
            message, choice = event("on_create", "Kitap", "", upload, None, None, "narration", "", "voxcpm2")
            self.assertIn("7 segment", message)
            slug = choice["value"]
            event("on_project_change", slug)
            controls, *_ = event("on_book_controls", slug)
            self.assertEqual([tuple(choice) for choice in controls["choices"]], [("Tüm kitap / proje", 0), *[
                (f"{i} · {title}", i) for i, title in enumerate(TITLES, 1)
            ]])
            self.assertIn("1 segment", event("on_save_lexicon", slug, "! New York = Niyu York", None)[0])
            self.assertEqual(service.segments(slug)[1]["spoken_text"], "Niyu York’a vardım.")
            project = service.open(slug)
            self.assertEqual(project.enqueue_missing(chapter_id=2), 2)  # left by a previous run
            self.assertIn("5 segment", event("on_start", slug, 1)[0])
            self.assertTrue(manager.started.wait(5))
            self.assertIn("4 segment", event("on_stop", slug)[0])
            manager.resume.set()
            service.wait(slug, 5)
            self.assertEqual([s["state"] for s in project.segments(chapter_id=2)], ["queued", "queued"])
            self.assertIn("4 segment", event("on_start", slug, 1)[0])
            service.wait(slug, 5)
            self.assertEqual([s["state"] for s in project.segments(chapter_id=1)],
                             ["done", "done", "failed", "done", "done"])
            self.assertIn("1 segment", event("on_start", slug, 1)[0])
            service.wait(slug, 5)
            self.assertIn("Eksik segment", event("on_preview", slug, 0, "WAV")[0])
            self.assertIn("WAV hazır", event("on_preview", slug, 1, "WAV")[0])
            self.assertIn("2 segment", event("on_start", slug, 2)[0])
            service.wait(slug, 5)
            self.assertIn("güncel", event("on_start", slug, 0)[0])
            self.assertEqual(len(manager.calls), 8)

            segments = service.segments(slug)
            base = [pcm(Path(item["audio_path"])) for item in segments]
            item = segments[2]
            key = item["cache_key"]
            self.assertIn("kuyruğa alındı", take_event("generate", slug, item["id"], None)[0])
            service.wait(slug, 5)
            _, alternative = service.take_view(slug, item["id"])["takes"]
            active = base.copy()
            active[2] = pcm(Path(alternative["path"]))
            token = f'{key}:{alternative["id"]}'
            self.assertIn("Aktif take", take_event("select", slug, item["id"], token)[0])

            sentence_gap, paragraph_gap = bytes(8400 * 2), bytes(20400 * 2)
            chapter_one = paragraph_gap.join((active[0], active[1] + sentence_gap + active[2],
                                              active[3] + sentence_gap + active[4]))
            chapter_two = paragraph_gap.join(active[5:])
            expected_pcm = {1: chapter_one, 2: chapter_two, 0: paragraph_gap.join((chapter_one, chapter_two))}
            for scope, expected in expected_pcm.items():
                _, file = event("on_preview", slug, scope, "WAV")
                self.assertEqual(pcm(Path(file["value"])), expected)

            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                for scope, expected in expected_pcm.items():
                    message, file = event("on_preview", slug, scope, "MP3")
                    self.assertIn("MP3 hazır", message)
                    path = Path(file["value"])
                    expected_mp3 = root / f"expected-{scope}.mp3"
                    subprocess.run(
                        [ffmpeg, "-nostdin", "-v", "error", "-f", "s16le", "-ar", "24000", "-ac", "1",
                         "-i", "pipe:0", "-c:a", "libmp3lame", "-q:a", "2", str(expected_mp3)],
                        input=expected, check=True, capture_output=True,
                    )
                    self.assertEqual(path.read_bytes(), expected_mp3.read_bytes())
                    decoded = subprocess.run(
                        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le", "pipe:1"],
                        check=True, capture_output=True,
                    ).stdout
                    self.assertEqual(len(decoded), len(expected))
            self.assertIn("Aktif take", take_event("undo", slug, item["id"], token)[0])
            _, file = event("on_preview", slug, 0, "WAV")
            self.assertEqual(pcm(Path(file["value"])), expected_pcm[0].replace(active[2], base[2], 1))
            self.assertEqual(len(manager.calls), 9)
