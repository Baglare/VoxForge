# -*- coding: utf-8 -*-
"""VF-016: advanced VoxCPM2 controls, presets, narration pacing, scenes and WAV joins.

Model free: real worker processes with the fake engine of test_voice_playground (no GPU).
"""

from __future__ import annotations

import array
import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_voice_playground import FAKE_WORKER  # noqa: E402
from voxforge.playground import audio_join, rhythm  # noqa: E402
from voxforge.playground.prosody import (  # noqa: E402
    BUILTIN_PRESETS,
    ControlError,
    PresetStore,
    VoxControls,
    plan_chunks,
)
from voxforge.playground.scenes import SceneNarrator, SceneProject, slugify, split_scenes, suggest_kind  # noqa: E402
from voxforge.playground.service import PlaygroundError, PlaygroundService, split_for_engine  # noqa: E402
from voxforge.playground.workers import EngineManager  # noqa: E402


def write_wav(path: Path, segments: list[tuple[float, float]], rate: int = 16000, amplitude: float = 0.3) -> Path:
    """segments: (seconds, 0/1 voiced) -> a sine where voiced, digital silence otherwise."""
    data = array.array("h")
    t = 0
    for seconds, voiced in segments:
        for _ in range(int(seconds * rate)):
            data.append(int(amplitude * 32767 * math.sin(2 * math.pi * 220 * t / rate)) if voiced else 0)
            t += 1
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


def seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


class AudioJoinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vf016_join_"))

    def test_legacy_join_is_byte_identical(self) -> None:
        a = write_wav(self.tmp / "a.wav", [(0.3, 0), (0.5, 1), (0.2, 0)])
        b = write_wav(self.tmp / "b.wav", [(0.4, 1)])
        out = self.tmp / "out.wav"
        audio_join.join_wavs([a, b], out)
        with wave.open(str(a), "rb") as ha, wave.open(str(b), "rb") as hb:
            expected = ha.readframes(ha.getnframes()) + b"\x00" * int(16000 * 0.2) * 2 + hb.readframes(hb.getnframes())
        with wave.open(str(out), "rb") as handle:
            self.assertEqual(handle.readframes(handle.getnframes()), expected)

    def test_trim_gives_exact_pauses_and_loudness_match(self) -> None:
        a = write_wav(self.tmp / "a.wav", [(0.3, 0), (0.5, 1), (0.6, 0)])
        b = write_wav(self.tmp / "b.wav", [(0.05, 0), (0.5, 1), (0.02, 0)], amplitude=0.05)
        out = self.tmp / "out.wav"
        info = audio_join.join_wavs([a, b], out, pauses=[0.5], trim=True, match_loudness=True)
        # 0.5 + 0.5 sn ses, iki kenar payı (<=0.04) ve tam 0.5 sn duraklama; ilk/son uzun sessizlik yok
        self.assertLess(abs(seconds(out) - 1.5), 0.17)
        stats = rhythm.audio_stats(out)
        self.assertLessEqual(stats.leading_silence, 0.05)
        self.assertEqual(len(stats.pauses), 1)
        self.assertAlmostEqual(stats.pauses[0][1] - stats.pauses[0][0], 0.5, delta=0.1)
        self.assertTrue(all(abs(g) <= audio_join.MAX_GAIN_DB for g in info["gains_db"]))
        self.assertNotEqual(info["gains_db"], [0.0, 0.0])  # seviye farkı düzeltildi (±6 dB sınırlı)

    def test_loudness_match_survives_peak_normalised_parts(self) -> None:
        # VF-017: VoxCPM2 parçaları ~0 dBFS'e normalize; eskiden pozitif kazanç tepe korumasında yok oluyordu.
        quiet = write_wav(self.tmp / "q.wav", [(0.4, 1)], amplitude=0.99)
        loud = write_wav(self.tmp / "l.wav", [(0.4, 1)], amplitude=0.99)
        with wave.open(str(quiet), "rb") as handle:  # aynı tepe, 6 dB daha düşük ortalama: yarısı 0.5 genlik
            frames = array.array("h", handle.readframes(handle.getnframes()))
        half = len(frames) // 2
        frames = array.array("h", [int(v * 0.25) if i < half else v for i, v in enumerate(frames)])
        with wave.open(str(quiet), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(frames.tobytes())
        parts = [audio_join.read_part(quiet), audio_join.read_part(loud), audio_join.read_part(loud)]
        before = [audio_join.voiced_rms_db(part) for part in parts]
        gains = audio_join.loudness_gains(parts)
        after = [audio_join.voiced_rms_db(audio_join.apply_gain(part, gain)) for part, gain in zip(parts, gains)]
        self.assertGreater(max(before) - min(before), 2.0)
        self.assertLess(max(after) - min(after), 0.3)  # fark gerçekten kapandı (eskiden tepe koruması iptal ediyordu)
        info = audio_join.join_wavs([quiet, loud, loud], self.tmp / "m.wav", pauses=0.2, trim=True, match_loudness=True)
        self.assertAlmostEqual(info["gains_db"][0] - info["gains_db"][1], before[1] - before[0], delta=0.3)
        merged = audio_join.read_part(self.tmp / "m.wav")
        self.assertLessEqual(max(abs(v) for v in merged.samples), int(audio_join.PEAK_LIMIT * 32767) + 1)

    def test_rejects_mismatched_formats_and_bad_pauses(self) -> None:
        a = write_wav(self.tmp / "a.wav", [(0.2, 1)], rate=16000)
        b = write_wav(self.tmp / "b.wav", [(0.2, 1)], rate=24000)
        with self.assertRaisesRegex(ValueError, "uyuşmuyor"):
            audio_join.join_wavs([a, b], self.tmp / "x.wav")
        with self.assertRaisesRegex(ValueError, "duraklama"):
            audio_join.join_wavs([a, a], self.tmp / "x.wav", pauses=[0.1, 0.2])


class RhythmTest(unittest.TestCase):
    def test_turkish_syllables_and_sentence_rate(self) -> None:
        self.assertEqual(rhythm.syllables("Hiç kimse oraya girmemişti."), 10)
        self.assertEqual(rhythm.split_sentences("Bir. İki! Üç?"), ["Bir.", "İki!", "Üç?"])
        words = [{"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0}, {"start": 1.5, "end": 2.0}]
        rows = rhythm.sentence_rhythm(["Ali geldi.", "Gitti."], words, [True] * 100)
        self.assertEqual([r.syllables for r in rows], [4, 2])
        self.assertEqual(rows[0].rate, 4.0)
        self.assertEqual(rows[1].pause_before, 0.5)


class ControlsTest(unittest.TestCase):
    def test_defaults_send_nothing_and_ranges_are_enforced(self) -> None:
        self.assertEqual(VoxControls().engine_settings(), {})
        self.assertTrue(VoxControls().is_default())
        self.assertEqual(VoxControls(cfg=2.5, steps=12).validated().engine_settings(), {"cfg_value": 2.5, "inference_timesteps": 12})
        for bad in (VoxControls(cfg=0.5), VoxControls(steps=40), VoxControls(pacing="x"), VoxControls(sentence_pause=3), VoxControls(style="x" * 200)):
            with self.assertRaises(ControlError):
                bad.validated()
        self.assertEqual(VoxControls(style="(yavaş) sakin").validated().style, "yavaş sakin")  # parantez öneki bozmaz
        self.assertTrue(VoxControls(steps=6).validated().warnings())

    def test_presets_builtin_and_user(self) -> None:
        self.assertEqual(BUILTIN_PRESETS["normal"].controls, VoxControls())
        self.assertTrue(all(p.evidence for p in BUILTIN_PRESETS.values()))
        store = PresetStore(Path(tempfile.mkdtemp()) / "presets.json")
        saved = store.save("Benim gece sesim", VoxControls(cfg=1.8, style="fısıltılı"))
        self.assertEqual(saved.key, "user-benim-gece-sesim")
        self.assertEqual(store.get(saved.key).controls.cfg, 1.8)
        with self.assertRaises(ControlError):
            store.delete("normal")
        with self.assertRaises(ControlError):
            store.save("../kaçış", VoxControls())
        store.delete(saved.key)
        self.assertNotIn(saved.key, store.all())

    def test_plan_standard_matches_old_split_and_narration_pauses(self) -> None:
        text = "Merhaba! Saat 14:30'da 5 kişi geldi.\n\nYeni paragraf başlıyor. Son cümle."
        legacy = split_for_engine(text, 250, True)
        standard = plan_chunks(text, VoxControls(), 250, True)
        self.assertEqual((standard.spoken, standard.chunks, standard.pauses, standard.trim), (legacy[0], legacy[1], None, False))
        narration = plan_chunks(text, VoxControls(pacing="narration", sentence_pause=0.4, paragraph_pause=0.9), 250, True)
        self.assertEqual(len(narration.chunks), 4)
        self.assertEqual(narration.pauses, [0.4, 0.9, 0.4])
        self.assertTrue(narration.trim)
        self.assertIn("beş", narration.spoken)


class ServiceAndScenesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vf016_"))
        self.reference = write_wav(self.tmp / "ref.wav", [(0.2, 1)], rate=24000)
        script = self.tmp / "fake_worker.py"
        script.write_text(FAKE_WORKER, encoding="utf-8")
        self.manager = EngineManager(
            self.tmp / "logs", python_for=lambda engine: Path(sys.executable), load_timeout=60, synth_timeout=60, worker_script=script
        )
        self.addCleanup(self.manager.close)
        self.service = PlaygroundService(self.tmp / "pg", self.manager)

    def test_style_travels_apart_from_text_and_ultimate_rejects_it(self) -> None:
        controls = VoxControls(cfg=2.4, steps=12, style="yavaş, sakin bir anlatım", pacing="narration")
        record = self.service.generate("Bir. İki cümle.", "voxcpm2", None, reference=self.reference, controls=controls, preset_key="yavas-anlatici")
        applied = record["settings"]["applied"]
        self.assertEqual(applied["style"], "yavaş, sakin bir anlatım")  # ayrı alan olarak worker'a ulaştı
        self.assertEqual((applied["cfg_value"], applied["inference_timesteps"]), (2.4, 12))
        self.assertNotIn("(", record["spoken_text"])  # seslendirilen metin değişmedi
        self.assertEqual(record["chunks"], 2)
        self.assertEqual(record["join"], {"pauses": [0.45], "trim": True})
        self.assertEqual(record["settings"]["preset"], "yavas-anlatici")
        default = self.service.generate("Bir. İki cümle.", "voxcpm2", None, reference=self.reference)
        self.assertEqual(default["settings"]["applied"], {})  # varsayılan istek değişmedi
        self.assertEqual(default["join"], {"pauses": None, "trim": False})
        with self.assertRaisesRegex(PlaygroundError, "Ultimate"):
            self.service.generate("Evet.", "voxcpm2-ultimate", None, reference=self.reference, reference_text="Merhaba.", controls=controls)
        with self.assertRaisesRegex(PlaygroundError, "geçerli değil"):
            self.service.generate("Evet.", "chatterbox-default", None, reference=self.reference, controls=VoxControls(cfg=2.5))
        ultimate = self.service.generate(
            "Evet.", "voxcpm2-ultimate", None, reference=self.reference, reference_text="Merhaba.", controls=VoxControls(cfg=2.2)
        )
        self.assertEqual(ultimate["settings"]["applied"]["cfg_value"], 2.2)  # cfg Ultimate ile uyumlu

    def test_preset_comparison_same_seed(self) -> None:
        records, errors = self.service.generate_presets(
            "Bir. İki.", ["normal", "dramatik", "yok"], None, seed=5, reference=self.reference
        )
        self.assertEqual(len(errors), 1)  # bilinmeyen preset diğerlerini durdurmaz
        self.assertEqual([r["settings"]["preset"] for r in records], ["normal", "dramatik"])
        self.assertEqual({r["settings"]["seed"] for r in records}, {5})
        self.assertNotIn("style", records[0]["settings"]["applied"])
        self.assertIn("dramatik", records[1]["settings"]["applied"]["style"])

    def test_scene_split_suggest_generate_merge(self) -> None:
        text = "## Kale\nKaranlık gölgeler, fısıltılar ve ayak sesleri.\n\n## Savaş\nDüşman kılıç çekti, saldırdı!\n\n---\n— Gel buraya!\n— Hayır."
        scenes = split_scenes(text)
        self.assertEqual([s.title for s in scenes], ["Kale", "Savaş", "Sahne 3"])
        self.assertEqual([s.kind for s in scenes], ["tension", "action", "dialogue"])
        self.assertEqual(suggest_kind("Sıradan bir sabah."), "calm")
        self.assertEqual(slugify("İlk Anlatı!"), "ilk-anlati")
        narrator = self.narrator()
        project = SceneProject(name="Deneme", slug="deneme", profile=None, scenes=scenes)
        with self.assertRaisesRegex(ControlError, "profil"):
            narrator.generate_scene(project, "s01")
        with self.assertRaisesRegex(ControlError, "Önce"):
            narrator.merge(project)

        captured = []
        original = self.service.generate

        def generate(text, variant, profile, **kwargs):
            captured.append((variant, profile, kwargs["seed"], kwargs["preset_key"]))
            return original(text, variant, None, reference=self.reference, **{k: v for k, v in kwargs.items()})

        self.service.generate = generate  # type: ignore[method-assign]
        project.profile = "mimo"
        narrator.store.create(project)
        self.assertEqual(narrator.generate_all(project), [])
        self.assertEqual({c[:3] for c in captured}, {("voxcpm2", "mimo", 1234)})  # aynı ses kimliği
        self.assertEqual([c[3] for c in captured], ["gerilim-sade", "aksiyon", "diyalog"])  # VF-017: türler stilsiz preset seçer
        first_wav = Path(project.scenes[0].wav)
        self.assertEqual(first_wav.parent, narrator.store.audio_dir("deneme"))
        merged = narrator.merge(project)
        self.assertTrue(Path(merged["wav"]).is_file())
        self.assertEqual(merged["pauses"], [0.8, 0.8])
        project.scenes[0].preset = "sakin"  # preset değişti: sahne eski sayılır
        self.assertTrue(narrator.is_stale(project, project.scenes[0]))
        with self.assertRaises(ControlError):
            narrator.merge(project)
        narrator.generate_scene(project, "s01")
        self.assertFalse(first_wav.exists())  # yalnız bu sahnenin eski sesi kaldırıldı
        reloaded = narrator.store.load("deneme")
        self.assertEqual(reloaded.scenes[0].preset, "sakin")
        self.assertIsNone(reloaded.merged_wav)

    # ---- VF-016 inceleme düzeltmeleri ------------------------------------------------------
    def narrator(self) -> SceneNarrator:
        narrator = SceneNarrator(self.service)
        # Profil referansı yerine testin referans dosyası (gerçek profiller CI'da yok).
        narrator._reference_identity = lambda profile: {  # type: ignore[method-assign]
            "profile": profile, "size": self.reference.stat().st_size, "mtime_ns": self.reference.stat().st_mtime_ns,
        }
        original = self.service.generate
        self.generated: list[str] = []

        def generate(text, variant, profile, **kwargs):
            self.generated.append(text)
            kwargs["reference"] = self.reference
            return original(text, variant, None, **kwargs)

        self.service.generate = generate  # type: ignore[method-assign]
        return narrator

    def ready_project(self, narrator: SceneNarrator) -> SceneProject:
        project = SceneProject(name="Parmak", slug="parmak", profile="mimo", scenes=split_scenes("Bir sahne.\n\nİkinci sahne."))
        narrator.store.create(project)
        self.assertEqual(narrator.generate_all(project), [])
        self.assertTrue(all(narrator.scene_state(project, s) == "ready" for s in project.scenes))
        self.generated.clear()
        return project

    def test_fingerprint_invalidates_on_any_generation_setting(self) -> None:
        narrator = self.narrator()
        changes = {
            "profil": lambda p: setattr(p, "profile", "baska"),
            "varyant": lambda p: setattr(p, "variant", "voxcpm2-ultimate"),
            "tohum": lambda p: setattr(p, "seed", 99),
            "normalizasyon": lambda p: setattr(p, "normalize", False),
        }
        for name, change in changes.items():
            with self.subTest(name):
                project = self.ready_project(narrator)
                change(project)
                self.assertTrue(all(narrator.is_stale(project, s) for s in project.scenes), name)
                with self.assertRaises(ControlError):
                    narrator.merge(project)
        # Referans dosyası değişti (profil yeniden kaydedildi): aynı profil adı yetmez.
        project = self.ready_project(narrator)
        write_wav(self.reference, [(0.3, 1)], rate=24000)
        self.assertTrue(narrator.is_stale(project, project.scenes[0]))
        # Cihaz değişti.
        project = self.ready_project(narrator)
        self.assertTrue(narrator.is_stale(project, project.scenes[0], device="cpu"))

    def test_preset_value_change_regenerates_only_affected_scenes(self) -> None:
        narrator = self.narrator()
        project = self.ready_project(narrator)
        project.scenes[1].preset = "diyalog"
        narrator.generate_scene(project, "s02")
        self.generated.clear()
        # Kullanıcı preset'i ayni adla farklı değerlere güncellenir: yalnız onu kullanan sahne eskir.
        self.service.presets.save("Ozel", VoxControls(pacing="narration", sentence_pause=0.4))
        key = PresetStore.key_for("Ozel")
        project.scenes[0].preset = key
        narrator.generate_scene(project, "s01")
        self.generated.clear()
        narrator.merge(project)
        self.service.presets.save("Ozel", VoxControls(pacing="narration", sentence_pause=0.9))
        self.assertEqual([narrator.scene_state(project, s) for s in project.scenes], ["stale", "ready"])
        with self.assertRaisesRegex(ControlError, project.scenes[0].title):
            narrator.merge(project)
        self.assertEqual(narrator.generate_all(project), [])
        self.assertEqual(self.generated, [project.scenes[0].text])  # yalnız gerekli sahne yeniden üretildi
        self.assertTrue(Path(narrator.merge(project)["wav"]).is_file())

    def test_legacy_scene_without_fingerprint_loads_and_needs_regeneration(self) -> None:
        narrator = self.narrator()
        project = self.ready_project(narrator)
        data = project.to_dict()
        for scene in data["scenes"]:
            for key in ("fingerprint", "device"):
                scene.pop(key)  # VF-016 ilk sürümünün kayıt biçimi
        path = narrator.store.path(project.slug)
        path.write_text(__import__("json").dumps(data, ensure_ascii=False), encoding="utf-8")
        legacy = narrator.store.load(project.slug)
        self.assertEqual(legacy.scenes[0].text, project.scenes[0].text)
        self.assertEqual({narrator.scene_state(legacy, s) for s in legacy.scenes}, {"stale"})
        with self.assertRaises(ControlError):
            narrator.merge(legacy)
        self.assertEqual(narrator.generate_all(legacy), [])
        self.assertTrue(Path(narrator.merge(legacy)["wav"]).is_file())

    def test_new_project_never_overwrites_existing_one(self) -> None:
        narrator = self.narrator()
        first = self.ready_project(narrator)  # slug "parmak", sesleri üretilmiş
        before = narrator.store.path("parmak").read_bytes()
        second = SceneProject(name="Parmak", slug="parmak", scenes=split_scenes("Başka metin."))
        narrator.store.create(second)
        self.assertEqual((second.slug, second.name), ("parmak-2", "Parmak-2"))
        self.assertEqual(narrator.store.path("parmak").read_bytes(), before)  # eski proje aynen duruyor
        self.assertEqual(narrator.store.load("parmak").scenes[0].text, first.scenes[0].text)
        third = SceneProject(name="Parmak", slug="parmak", scenes=split_scenes("Üçüncü."))
        narrator.store.create(third)
        self.assertEqual(third.slug, "parmak-3")
        # JSON'u silinmiş ama sesleri kalmış klasör de sahiplenilmez.
        orphan = narrator.store.audio_dir("yetim")
        orphan.mkdir(parents=True)
        fourth = SceneProject(name="yetim", slug="yetim", scenes=split_scenes("Dört."))
        narrator.store.create(fourth)
        self.assertEqual(fourth.slug, "yetim-2")
        self.assertEqual(sorted(narrator.store.list()), ["parmak", "parmak-2", "parmak-3", "yetim-2"])


if __name__ == "__main__":
    unittest.main()


class VF017PresetPolicyTest(unittest.TestCase):
    def test_default_presets_are_style_free_and_styled_ones_are_experimental(self) -> None:
        from voxforge.playground.scenes import SCENE_KINDS

        for _label, key in SCENE_KINDS.values():
            self.assertEqual(BUILTIN_PRESETS[key].controls.style, "", key)  # sahne varsayılanı stil taşımaz
        self.assertEqual(BUILTIN_PRESETS["yavas-anlatici"].controls.style, "")
        for preset in BUILTIN_PRESETS.values():
            if preset.controls.style:
                self.assertIn("experimental", preset.tags, preset.key)
                self.assertIn("deneysel", preset.name.lower(), preset.key)
                self.assertIn("DENEYSEL", preset.evidence, preset.key)
        self.assertEqual(BUILTIN_PRESETS["normal"].controls, VoxControls())  # kısa metin varsayılanı değişmedi
