# -*- coding: utf-8 -*-
"""VF-017 diagnostic: are abrupt sentence starts caused by context-free per-sentence calls?

Runs INSIDE the VoxCPM2 bench venv (loads the model once, GPU, 8 GB budget):

    <voxcpm-venv>\\python.exe scripts/benchmark/vf017_context_probe.py --out outputs/vf017/context

For each text, every sentence is generated twice with the same reference and seed:
  cold  : reference only (what narration pacing does today; must equal the B raw parts)
  ctx   : reference + the previous generated sentence as continuation prompt
          (prompt_wav = previous raw part, prompt_text = previous sentence)
Both are joined with the Playground joiner (same pauses). Not wired into the product: it
only tests a hypothesis. No style instruction (the model rejects style with a prompt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "benchmark")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from run_engine import Torch, save_pcm16  # noqa: E402

from voxforge.playground.audio_join import join_wavs  # noqa: E402

TEXT_KEYS = ("story_B_segment", "s01_sakin_nostyle", "s03_aksiyon_preset", "s04_duygusal_nostyle")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path, default=REPO_ROOT / "outputs" / "vf017" / "probe")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "vf017" / "context")
    parser.add_argument("--reference", type=Path, default=REPO_ROOT / "profiles" / "mimo" / "preprocessed_reference.wav")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    rows = {row["name"]: row for row in json.loads((args.probe / "manifest.json").read_text(encoding="utf-8"))}
    from voxcpm import VoxCPM

    torch_helper = Torch()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="cuda", optimize=False)
    rate = model.tts_model.sample_rate
    manifest = []
    for key in TEXT_KEYS:
        row = rows[key]
        for mode in ("cold", "ctx"):
            folder = args.out / key / mode
            folder.mkdir(parents=True, exist_ok=True)
            parts: list[Path] = []
            torch_helper.reset_peak()
            for index, chunk in enumerate(row["chunks"]):
                kwargs = {"reference_wav_path": str(args.reference), "cfg_value": 2.0, "inference_timesteps": 10}
                if mode == "ctx" and index:
                    kwargs.update(prompt_wav_path=str(parts[-1]), prompt_text=row["chunks"][index - 1])
                torch_helper.seed(args.seed)
                part = folder / f"{index:02d}.wav"
                save_pcm16(part, model.generate(text=chunk, **kwargs), rate)
                torch_helper.torch.cuda.empty_cache()
                parts.append(part)
            out = args.out / f"{key}__{mode}.wav"
            info = join_wavs(parts, out, pauses=row["pauses"], trim=True)
            same = None
            if mode == "cold":  # aynı istek = B ham parçaları (tekrarlanabilirlik kontrolü)
                same = [p.read_bytes() == Path(raw).read_bytes() for p, raw in zip(parts, row["raw_parts"])]
            entry = {
                "name": f"{key}__{mode}", "group": "context", "wav": str(out), "text": row["text"], "chunks": row["chunks"],
                "label": "cümle cümle, bağlamsız" if mode == "cold" else "cümle cümle + önceki cümle prompt devamı",
                "style": "", "raw_parts": [str(p) for p in parts], "audio_seconds": info["duration"],
                "vram_peak_mb": torch_helper.peak_mb(), "cold_equals_probe_raw": same,
            }
            manifest.append(entry)
            print(json.dumps({k: entry[k] for k in ("name", "audio_seconds", "vram_peak_mb", "cold_equals_probe_raw")}), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
