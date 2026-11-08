# -*- coding: utf-8 -*-
"""VF-017 controlled narration comparison for VoxCPM2 (small, fixed GPU sample).

    .venv\\Scripts\\python.exe scripts/benchmark/vf017_narration_probe.py --out outputs/vf017/probe

Same mimo reference, same text, CFG 2.0, 10 steps, seed 1234 for every variant; only one
factor changes at a time:

  A  default request (standard pacing: sentences packed to 250 chars, no style)
  B  narration pacing (one sentence per call, trimmed edges, 0.55/0.95 s pauses), no style
  C  B + the "Yavaş anlatıcı" style instruction ("yavaş, sakin bir anlatım")

For B/C (and every scene) each sentence is ALSO requested on its own and kept *raw*
(model output, nothing trimmed or faded). The raw parts are then joined offline with the
exact Playground joiner and compared byte for byte with the worker's joined WAV: if they
are equal, the raw parts are exactly what the joined output was made from, so problems
can be attributed to the model (raw) or to the join (trim / fade / pause / level).

Scenes: the user's VF-016 scene project texts (calm, tension, action, emotional), each with
its preset AND without any style instruction (same pauses), so the effect of the style
instruction on speaker identity is measured separately from pacing.

Measure afterwards in the eval venv: scripts/benchmark/vf017_analyze.py --probe <out>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.playground.audio_join import join_wavs  # noqa: E402
from voxforge.playground.prosody import BUILTIN_PRESETS, VoxControls, plan_chunks  # noqa: E402
from voxforge.playground.service import resolve_profile_reference  # noqa: E402
from voxforge.playground.workers import EngineManager  # noqa: E402

SHORT = "Merhaba. Ben Mucan Arı ve Arı-ca. Lütfen benle hiç iyi geçinmeyin, çünkü Noodle yiyemiyorum."
STORY = (
    "Krallığın kuzeyinde, yüzyıllardır kimsenin yaklaşmaya cesaret edemediği eski bir kale vardı. "
    "Rivayete göre kalenin son hükümdarı, ölümünden hemen önce bütün kapıları mühürletmişti. "
    "Fakat o gece, nöbetçiler içeriden gelen ayak seslerini duydu. Hiç kimse orada yaşamıyordu. "
    "Hiç kimse oraya girmemişti. Buna rağmen, sabah olduğunda taht salonunda tek bir mum yanıyordu."
)
SCENES = {  # outputs/playground/scenes/vf016-ilk-deneme.json (kullanıcının dinleme testi)
    "s01_sakin": ("sakin", "Şafak söküyordu. Krallığın eski bahçesinde rüzgâr, ıslak yaprakların arasından usulca geçiyordu."),
    "s02_gerilim": ("gerilim", "Fakat koridorun sonunda bir gölge belirdi. Derin sessizliğin içinde, yaklaşan ayak sesleri gittikçe belirginleşiyordu."),
    "s03_aksiyon": ("aksiyon", "Birden düşman askerleri kapıyı kırıp içeri hücum etti! Kılıçlar çarpıştı, savaş avlunun dört bir yanına yayıldı."),
    "s04_duygusal": ("duygusal", "Gözyaşlarını gizlemeyen yaşlı muhafız, oğluna son kez sarıldı. Kayıp ve özlem dolu bu vedada ikisi de bir süre konuşamadı."),
}
SLOW = BUILTIN_PRESETS["yavas-anlatici"].controls
VARIANTS = {
    "A_default": VoxControls(),
    "B_segment": replace(SLOW, style=""),
    "C_segment_style": SLOW,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class Probe:
    def __init__(self, out: Path, reference: Path, seed: int) -> None:
        self.out, self.reference, self.seed = out, reference, seed
        self.manager = EngineManager(out / "logs")
        self.rows: list[dict] = []

    def synth(self, chunks: list[str], out: Path, controls: VoxControls, pauses, trim: bool) -> dict:
        reply = self.manager.synthesize(
            "voxcpm2",
            {
                "chunks": chunks, "reference": str(self.reference), "settings": controls.engine_settings(),
                "seed": self.seed, "out": str(out), "style": controls.style or None, "pauses": pauses, "trim": trim,
            },
        )
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error"))
        return reply

    def run(self, name: str, text: str, controls: VoxControls, group: str, raw_parts: bool) -> None:
        plan = plan_chunks(text, controls, 250, normalize=True)
        out = self.out / f"{name}.wav"
        reply = self.synth(plan.chunks, out, controls, plan.pauses, plan.trim)
        row = {
            "name": name, "group": group, "wav": str(out), "text": plan.spoken, "chunks": plan.chunks,
            "pauses": plan.pauses, "trim": plan.trim, "style": controls.style, "label": controls.label(),
            "seed": self.seed, "audio_seconds": reply.get("audio_seconds"), "seconds": reply.get("seconds"),
            "vram_peak_mb": reply.get("vram_peak_mb"), "sha": sha(out),
        }
        if raw_parts and plan.pauses is not None:
            raw_dir = self.out / "raw" / name
            raw_dir.mkdir(parents=True, exist_ok=True)
            raws = []
            for index, chunk in enumerate(plan.chunks):
                part = raw_dir / f"{index:02d}.wav"
                self.synth([chunk], part, controls, None, False)  # tek parça, kırpma yok = ham model çıktısı
                raws.append(str(part))
            rejoined = raw_dir / "rejoined.wav"
            join_wavs([Path(p) for p in raws], rejoined, pauses=plan.pauses, trim=True)
            row.update(raw_parts=raws, rejoined_equal=sha(rejoined) == row["sha"])
        self.rows.append(row)
        print(json.dumps({k: row[k] for k in ("name", "audio_seconds", "vram_peak_mb", "sha")} | {"rejoined_equal": row.get("rejoined_equal")}, ensure_ascii=False), flush=True)

    def close(self) -> None:
        self.manager.close()
        (self.out / "manifest.json").write_text(json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="mimo")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "vf017" / "probe")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--parts", default="short,abc,scenes")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    probe = Probe(args.out, resolve_profile_reference(args.profile), args.seed)
    parts = set(args.parts.split(","))
    try:
        if "short" in parts:  # süreçteki ilk üretim: VF-016 kısa metin kaydıyla bayt karşılaştırması
            probe.run("short_A_default", SHORT, VoxControls(), "short", raw_parts=False)
        if "abc" in parts:
            for name, controls in VARIANTS.items():
                probe.run(f"story_{name}", STORY, controls, "abc", raw_parts=name != "A_default")
        if "scenes" in parts:
            for key, (preset, text) in SCENES.items():
                controls = BUILTIN_PRESETS[preset].controls
                probe.run(f"{key}_preset", text, controls, "scene", raw_parts=True)
                if controls.style:
                    probe.run(f"{key}_nostyle", text, replace(controls, style=""), "scene", raw_parts=True)
                probe.run(f"{key}_A_default", text, VoxControls(), "scene", raw_parts=False)
    finally:
        probe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
