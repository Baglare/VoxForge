"""Voice library: profiles + VoxCPM2 designs, safe changes, Playground integration (no models)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import voice_profile_utils as vpu
from voxforge.playground.service import PlaygroundError, PlaygroundService
from voxforge.studio.design import DesignLibrary
from voxforge.voices import VoiceLibrary, VoiceLibraryError

DESIGN_ID = "a" * 32


def write_wav(path: Path, seconds: float = 0.5, value: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(value.to_bytes(2, "little", signed=True) * int(24000 * seconds))
    return path


def fake_preprocess(source, target, mode="safe"):
    """Stand-in for FFmpeg preprocessing: copy and report like the real helper."""
    shutil.copy2(source, target)
    report = {"quality": "GOOD", "duration_seconds": 0.5, "sample_rate": 24000, "channels": 1}
    return {"selected_output_path": str(target), "selected_variant": "safe",
            "candidate_reports": {"original": dict(report), "safe": dict(report)}, "preprocessing_warning": None}


class LibraryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vf_library_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.profiles = self.tmp / "profiles"
        for target in (patch.object(vpu, "PROFILES_DIR", self.profiles),
                       patch.object(vpu, "preprocess_reference_audio", fake_preprocess)):
            target.start()
            self.addCleanup(target.stop)
        self.designs = DesignLibrary(self.tmp / "studio")
        self.library = VoiceLibrary(designs=self.designs, trash_root=self.tmp / "trash")
        self.source = write_wav(self.tmp / "upload.wav", value=5)
        version = self.designs.root / "versions" / DESIGN_ID
        write_wav(version / "voice.wav", value=7)
        from voxforge.studio.project import file_sha256

        (version / "voice.json").write_text(json.dumps({
            "id": DESIGN_ID, "name": "Tok anlatıcı", "created_at": "2026-11-20T10:00:00+00:00",
            "sha256": file_sha256(version / "voice.wav"), "model_id": "openbmb/VoxCPM2@rev",
            "description": "sıcak, tok", "text": "Merhaba, ben tasarlanmış bir sesim.", "seed": 1234,
        }), encoding="utf-8")


class VoiceLibraryTest(LibraryFixture):
    def test_lists_both_kinds_and_resolves_references(self) -> None:
        slug = self.library.create_profile("Anlatıcı Ses", self.source, permission=True)
        self.assertEqual(slug, "anlatici-ses")
        kinds = {item["key"]: item["kind"] for item in self.library.entries()}
        self.assertEqual(kinds, {"anlatici-ses": "profile", f"design:{DESIGN_ID}": "design"})
        labels = dict((key, label) for label, key in self.library.choices())
        self.assertIn("tasarım", labels[f"design:{DESIGN_ID}"])
        self.assertEqual(self.library.reference(slug).name, "preprocessed_reference.wav")
        self.assertEqual(self.library.reference(f"design:{DESIGN_ID}").name, "voice.wav")
        self.assertEqual(self.library.design_transcript(f"design:{DESIGN_ID}"), "Merhaba, ben tasarlanmış bir sesim.")

    def test_create_requires_permission_and_audio(self) -> None:
        with self.assertRaises(VoiceLibraryError):
            self.library.create_profile("X", self.source, permission=False)
        with self.assertRaises(VoiceLibraryError):
            self.library.create_profile("X", None, permission=True)
        self.assertEqual(self.library.entries()[0]["kind"], "design")

    def test_rename_keeps_slug_and_design_hash(self) -> None:
        slug = self.library.create_profile("Eski", self.source, permission=True)
        self.library.rename(slug, "Yeni ad")
        self.assertEqual(self.library.get(slug)["name"], "Yeni ad")
        self.assertTrue((self.profiles / slug).is_dir())
        self.library.rename(f"design:{DESIGN_ID}", "Sakin anlatıcı")
        version = self.designs.version(DESIGN_ID)  # verifies WAV hash too
        self.assertEqual(version["name"], "Sakin anlatıcı")
        with self.assertRaises(VoiceLibraryError):
            self.library.rename(slug, "  ")

    def test_replace_reference_keeps_previous_in_trash(self) -> None:
        slug = self.library.create_profile("Ses", self.source, permission=True)
        created = json.loads((self.profiles / slug / "profile.json").read_text(encoding="utf-8"))["created_at"]
        new_source = write_wav(self.tmp / "new.wav", seconds=0.7, value=9)
        with self.assertRaises(VoiceLibraryError):
            self.library.replace_reference(slug, new_source, permission=False)
        self.library.replace_reference(slug, new_source, permission=True)
        metadata = json.loads((self.profiles / slug / "profile.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["created_at"], created)
        self.assertIn("reference_replaced_at", metadata)
        self.assertEqual((self.profiles / slug / "original_reference.wav").read_bytes(), new_source.read_bytes())
        kept = list((self.tmp / "trash" / "replaced-references").glob(f"{slug}-*"))
        self.assertEqual(len(kept), 1)
        self.assertEqual((kept[0] / "original_reference.wav").read_bytes(), self.source.read_bytes())
        self.assertEqual(sorted(p.name for p in (self.profiles / slug).iterdir()),
                         ["original_reference.wav", "preprocessed_reference.wav", "profile.json"])

    def test_failed_replacement_leaves_profile_unchanged(self) -> None:
        slug = self.library.create_profile("Ses", self.source, permission=True)
        before = {p.name: p.read_bytes() for p in (self.profiles / slug).iterdir()}

        def broken(*_args, **_kwargs):
            raise vpu.PreprocessingError("bozuk kayıt")

        with patch.object(vpu, "preprocess_reference_audio", broken), self.assertRaises(VoiceLibraryError):
            self.library.replace_reference(slug, write_wav(self.tmp / "bad.wav"), permission=True)
        self.assertEqual({p.name: p.read_bytes() for p in (self.profiles / slug).iterdir()}, before)

    def test_delete_needs_confirmation_and_moves_to_trash(self) -> None:
        slug = self.library.create_profile("Silinecek", self.source, permission=True)
        with self.assertRaises(VoiceLibraryError):
            self.library.delete(slug, "evet")
        self.assertTrue((self.profiles / slug).is_dir())
        target = self.library.delete(slug, slug)
        self.assertFalse((self.profiles / slug).exists())
        self.assertTrue((target / "profile.json").is_file())
        key = f"design:{DESIGN_ID}"
        with self.assertRaises(VoiceLibraryError):
            self.library.delete(key, DESIGN_ID)  # token is the 8-character short id
        archived = self.library.delete(key, DESIGN_ID[:8])
        self.assertTrue((archived / "voice.wav").is_file())
        self.assertEqual(self.library.entries(), [])

    def test_slug_path_containment_still_applies(self) -> None:
        for bad in ("../x", "A", ""):
            with self.assertRaises(VoiceLibraryError):
                self.library.rename(bad, "ad")


class StubManager:
    """Writes a short WAV like an engine worker; records every request."""

    current_engine = None

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    def status(self) -> dict:
        return {}

    def unload(self):
        return None

    def close(self) -> None:
        pass

    def synthesize(self, engine, request, device="cuda"):
        self.requests.append((engine, request))
        write_wav(Path(request["out"]), 0.3, 1000)
        return {"worker": {"model": f"stub-{engine}", "device": device}, "seconds": 0.1, "audio_seconds": 0.3,
                "rtf": 0.3, "applied_settings": request["settings"], "seed": request["seed"]}


class PlaygroundLibraryTest(LibraryFixture):
    def setUp(self) -> None:
        super().setUp()
        self.manager = StubManager()
        self.service = PlaygroundService(self.tmp / "playground", manager=self.manager)
        resolver = patch("voxforge.voices.VoiceLibrary", lambda *a, **k: self.library)
        resolver.start()
        self.addCleanup(resolver.stop)

    def test_design_voice_is_a_reference_and_transcript(self) -> None:
        key = f"design:{DESIGN_ID}"
        record = self.service.generate("Kısa deneme.", "voxcpm2", key)
        self.assertEqual(Path(self.manager.requests[-1][1]["reference"]).name, "voice.wav")
        self.assertEqual(record["profile"], key)
        self.assertEqual(self.service.get_transcript(key), "Merhaba, ben tasarlanmış bir sesim.")
        self.service.generate("Kısa deneme.", "voxcpm2-ultimate", key, reference_text="Elle düzeltilmiş metin.")
        self.assertEqual(self.service.get_transcript(key), "Elle düzeltilmiş metin.")
        self.assertTrue((self.service.transcript_dir / f"design-{DESIGN_ID}.txt").is_file())

    def test_xtts_mode_only_for_xtts_and_known_keys(self) -> None:
        record = self.service.generate("Deneme.", "xtts", None, reference=self.source,
                                       engine_settings={"temperature": 0.7, "top_k": 50}, preset_key="xtts:Daha stabil")
        self.assertEqual(self.manager.requests[-1][1]["settings"], {"temperature": 0.7, "top_k": 50})
        self.assertEqual(record["settings"]["preset"], "xtts:Daha stabil")
        with self.assertRaises(PlaygroundError):
            self.service.generate("Deneme.", "voxcpm2", None, reference=self.source, engine_settings={"temperature": 0.7})
        with self.assertRaises(PlaygroundError):
            self.service.generate("Deneme.", "xtts", None, reference=self.source, engine_settings={"speed": 2})

    def test_output_normalisation_keeps_raw_and_falls_back(self) -> None:
        def ok(source, target):
            shutil.copy2(source, target)
            return True, "Çıktı sesi normalize edildi."

        with patch("voxforge.generation.normalize_wav", ok):
            record = self.service.generate("Deneme.", "mock", None, post_process=True)
        self.assertTrue(record["wav"].endswith("_norm.wav"))
        self.assertTrue(Path(record["raw_wav"]).is_file())
        self.assertTrue(record["post_processing"]["applied"])
        with patch("voxforge.generation.normalize_wav", lambda s, t: (False, "FFmpeg bulunamadı; ham çıktı korundu.")):
            record = self.service.generate("Deneme.", "mock", None, post_process=True)
        self.assertEqual(record["wav"], record["raw_wav"])
        self.assertFalse(record["post_processing"]["applied"])


if __name__ == "__main__":
    unittest.main()


class ReplaceRecoveryTest(LibraryFixture):
    """Fault injection: the old reference must survive every failure and interruption."""

    def setUp(self) -> None:
        super().setUp()
        self.slug = self.library.create_profile("Ses", self.source, permission=True)
        self.folder = self.profiles / self.slug
        self.before = self.snapshot()
        self.new_source = write_wav(self.tmp / "new.wav", seconds=0.7, value=9)
        self.real_replace, self.real_copy, self.real_move = vpu._replace_file, vpu._copy_file, vpu._move_tree

    def snapshot(self) -> dict[str, bytes]:
        return {name: (self.folder / name).read_bytes() for name in vpu.PROFILE_FILE_NAMES}

    def backups(self) -> list[Path]:
        return vpu.pending_profile_replacements(self.slug)

    def fail_swap_of(self, name: str):
        def replace(src, dst):
            if Path(src).parent.name == vpu.REPLACEMENT_STAGING_NAME and Path(dst).name == name:
                raise OSError("enjekte: dosya kilitli")
            return self.real_replace(src, dst)
        return replace

    def test_failed_archive_keeps_old_files_in_profile(self) -> None:
        def move(*_args):
            raise OSError("enjekte: çöp kutusu yazılamaz")

        with patch.object(vpu, "_move_tree", move):
            warning = self.library.replace_reference(self.slug, self.new_source, permission=True)
        self.assertIn("korundu", warning)
        self.assertEqual((self.folder / "original_reference.wav").read_bytes(), self.new_source.read_bytes())
        [backup] = self.backups()
        self.assertEqual({n: (backup / n).read_bytes() for n in vpu.PROFILE_FILE_NAMES}, self.before)
        self.assertEqual(vpu._read_journal(backup)["state"], "committed")
        self.assertEqual(self.library.pending_replacements(), {self.slug: 1})
        # Sonraki işlem yedeği güvenle arşivler; yeni profil olduğu gibi kalır.
        self.library.recover_replacements(self.slug)
        self.assertEqual(self.backups(), [])
        [kept] = (self.tmp / "trash" / "replaced-references").glob(f"{self.slug}-*")
        self.assertEqual((kept / "original_reference.wav").read_bytes(), self.source.read_bytes())
        self.assertEqual((self.folder / "original_reference.wav").read_bytes(), self.new_source.read_bytes())

    def test_partial_archive_move_is_not_reported_as_safe(self) -> None:
        def move(src, dst):
            Path(dst).mkdir(parents=True)
            raise OSError("enjekte: kopya yarıda kaldı")

        with patch.object(vpu, "_move_tree", move):
            self.library.replace_reference(self.slug, self.new_source, permission=True)
        [backup] = self.backups()  # kaynak silinmedi, hedef yarım
        self.assertEqual((backup / "original_reference.wav").read_bytes(), self.source.read_bytes())

    def test_swap_failure_restores_old_files(self) -> None:
        with patch.object(vpu, "_replace_file", self.fail_swap_of("preprocessed_reference.wav")):
            with self.assertRaises(VoiceLibraryError):
                self.library.replace_reference(self.slug, self.new_source, permission=True)
        self.assertEqual(self.snapshot(), self.before)
        self.assertEqual(self.backups(), [])
        self.assertFalse((self.folder / vpu.REPLACEMENT_STAGING_NAME).exists())

    def test_interrupted_swap_with_failed_restore_is_recoverable(self) -> None:
        def broken_copy(src, dst):
            if Path(dst).name.endswith(".restore"):
                raise OSError("enjekte: geri yükleme de başarısız")
            return self.real_copy(src, dst)

        with patch.object(vpu, "_replace_file", self.fail_swap_of("profile.json")), \
                patch.object(vpu, "_copy_file", broken_copy), self.assertRaises(VoiceLibraryError) as caught:
            self.library.replace_reference(self.slug, self.new_source, permission=True)
        self.assertIn("Hicbir sey silinmedi", str(caught.exception))
        [backup] = self.backups()
        self.assertEqual(vpu._read_journal(backup)["state"], "swapping")
        self.assertNotEqual(self.snapshot(), self.before)  # profil yarım değişti
        messages = self.library.recover_replacements(self.slug)
        self.assertIn("geri alindi", messages[0])
        self.assertEqual(self.snapshot(), self.before)
        self.assertEqual(self.backups(), [])

    def test_crash_after_commit_is_archived_on_next_replace(self) -> None:
        with patch.object(vpu, "_archive_backup", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())):
            with self.assertRaises(KeyboardInterrupt):
                self.library.replace_reference(self.slug, self.new_source, permission=True)
        [backup] = self.backups()
        self.assertEqual(vpu._read_journal(backup)["state"], "committed")
        third = write_wav(self.tmp / "third.wav", seconds=0.4, value=11)
        self.library.replace_reference(self.slug, third, permission=True)
        self.assertEqual(self.backups(), [])
        kept = sorted((self.tmp / "trash" / "replaced-references").glob(f"{self.slug}-*"))
        originals = {(k / "original_reference.wav").read_bytes() for k in kept}
        self.assertEqual(originals, {self.source.read_bytes(), self.new_source.read_bytes()})

    def test_backup_copy_failure_leaves_profile_untouched(self) -> None:
        def copy(src, dst):
            if Path(dst).parent.name.startswith(vpu.REPLACEMENT_BACKUP_PREFIX) and Path(dst).name == "profile.json":
                raise OSError("enjekte: disk dolu")
            return self.real_copy(src, dst)

        with patch.object(vpu, "_copy_file", copy), self.assertRaises(OSError):
            self.library.replace_reference(self.slug, self.new_source, permission=True)
        self.assertEqual(self.snapshot(), self.before)
        self.assertEqual(self.backups(), [])

    def test_damaged_backup_is_never_used_or_deleted(self) -> None:
        with patch.object(vpu, "_replace_file", self.fail_swap_of("profile.json")), \
                patch.object(vpu, "_restore_from_backup", lambda *a: (_ for _ in ()).throw(OSError("çöktü"))):
            with self.assertRaises(VoiceLibraryError):
                self.library.replace_reference(self.slug, self.new_source, permission=True)
        [backup] = self.backups()
        (backup / "original_reference.wav").write_bytes(b"bozuk")
        with self.assertRaises(VoiceLibraryError):
            self.library.recover_replacements(self.slug)
        self.assertTrue((backup / "journal.json").is_file())
        self.assertEqual((backup / "profile.json").read_bytes(), self.before["profile.json"])
