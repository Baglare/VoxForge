# -*- coding: utf-8 -*-
"""VF-016 long-narration rhythm probe for VoxCPM2 (small, fixed GPU sample).

    .venv\\Scripts\\python.exe scripts/benchmark/voxcpm_rhythm_probe.py --profile mimo --out outputs/vf016/probe

Generates the same Turkish narration with a few *segmentation* and *control* variants
through the Playground worker (one VoxCPM2 process, model loaded once, seed fixed) so
that only one factor changes per variant. Afterwards measure with
``scripts/benchmark/measure_rhythm.py --wav ... --text ...`` in the eval venv.

Variants (names are stable, used in docs/VF016_PROSODY_FINDINGS.md):
  pack250        current Playground behaviour: sentences packed up to 250 chars
  sentence       one sentence per generation, joined with the scene joiner
  pack250_ctl    pack250 + VoxCPM2 control prefix "(slow, calm narration)"
  sentence_ctl   sentence + control prefix
  *_ctltr        Turkish control text "(yavaş, sakin bir anlatım)"
  *_ctlslow      stronger English control (see CONTROLS)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.playground.rhythm import split_sentences  # noqa: E402
from voxforge.playground.service import resolve_profile_reference, split_for_engine  # noqa: E402
from voxforge.playground.workers import EngineManager  # noqa: E402

TEXTS = {
    "story": (
        "Krallığın kuzeyinde, yüzyıllardır kimsenin yaklaşmaya cesaret edemediği eski bir kale vardı. "
        "Rivayete göre kalenin son hükümdarı, ölümünden hemen önce bütün kapıları mühürletmişti. "
        "Fakat o gece, nöbetçiler içeriden gelen ayak seslerini duydu. Hiç kimse orada yaşamıyordu. "
        "Hiç kimse oraya girmemişti. Buna rağmen, sabah olduğunda taht salonunda tek bir mum yanıyordu."
    ),
    "daily": (
        "Bugün sabah erkenden kalktım ve pencereden dışarı baktım. Hava serin ama güneşliydi. "
        "Kahvemi hazırladıktan sonra yapılacaklar listemi gözden geçirdim; önce raporu bitirmem, sonra da "
        "akşam yemeği için alışveriş yapmam gerekiyordu. Öğleden sonra eski bir arkadaşımla buluştuk ve uzun "
        "uzun sohbet ettik. Eve döndüğümde yorgundum ama günün güzel geçtiğini düşünüyordum."
    ),
}
CONTROLS = {
    "ctl": "slow, calm narration",
    "ctltr": "yavaş, sakin bir anlatım",
    "ctlslow": "slow and deliberate storytelling, with natural pauses between sentences",
}


def chunks_for(text: str, variant: str) -> list[str]:
    spoken, packed = split_for_engine(text, 250, normalize=True)
    chunks = split_sentences(spoken) if variant.startswith("sentence") else packed
    control = CONTROLS.get(variant.rsplit("_", 1)[-1])
    if control:
        chunks = [f"({control}){chunk}" for chunk in chunks]
    return chunks


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="mimo")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "vf016" / "probe")
    parser.add_argument("--variants", default="pack250,sentence,pack250_ctl,sentence_ctl")
    parser.add_argument("--texts", default="story,daily")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    reference = resolve_profile_reference(args.profile)
    args.out.mkdir(parents=True, exist_ok=True)
    manager = EngineManager(args.out / "logs")
    manifest = []
    try:
        for text_key in args.texts.split(","):
            for variant in args.variants.split(","):
                chunks = chunks_for(TEXTS[text_key], variant)
                out = args.out / f"{text_key}_{variant}.wav"
                reply = manager.synthesize(
                    "voxcpm2",
                    {"chunks": chunks, "reference": str(reference), "settings": {}, "seed": args.seed, "out": str(out)},
                )
                spoken = " ".join(split_sentences(split_for_engine(TEXTS[text_key], 250, True)[0]))
                row = {
                    "name": out.stem,
                    "wav": str(out),
                    "text": spoken,
                    "chunks": len(chunks),
                    "seconds": reply.get("seconds"),
                    "audio_seconds": reply.get("audio_seconds"),
                    "vram_peak_mb": reply.get("vram_peak_mb"),
                }
                manifest.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        manager.close()
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
