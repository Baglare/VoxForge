# -*- coding: utf-8 -*-
"""Isolated probe: does Chatterbox V3 repeat short Turkish utterances ("Evet." -> "Evet. Evet.")?

Runs inside the Chatterbox bench venv, one process, GPU alone. Does not touch run folders
under outputs/voice_benchmark; output goes to outputs/chatterbox_repeat/<name>/ (gitignored).

    <chatterbox-venv>\\python.exe scripts/benchmark/chatterbox_repeat_probe.py \
        --reference <reference.wav> --phase baseline|sweep|trim [--seeds 8] [--name probe1]

Each sample is keyed (phase, config, phrase, seed); finished samples are skipped on re-run.
Without an ASR model the repeat is measured acoustically: voiced bursts separated by
a pause (``bursts``), and how alike the first two bursts are (``burst_sim``, MFCC cosine).
``repeat_like`` = 2+ bursts. It is a proxy; ASR transcripts (evaluate.py) are the check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_engine import save_pcm16  # noqa: E402
from voxforge.text import normalize_text  # noqa: E402

PHRASES = ["Evet.", "Hayır.", "Tamam.", "Peki.", "Merhaba.", "Teşekkürler.", "Evet", "Evet!", "Evet, tamam."]
CONTROL_LONG = "Ağaçların gölgesinde ılık bir rüzgâr esiyordu; çocuklar şarkı söylüyordu."
DEFAULTS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0}
SWEEP = {  # one parameter at a time, around the library defaults
    "cfg_weight": [0.0, 0.3, 0.7],
    "repetition_penalty": [1.0, 1.5, 2.0],
    "temperature": [0.5, 0.65, 1.0],
    "exaggeration": [0.3, 0.7],
    "min_p": [0.02, 0.1],
}
SWEEP_PHRASES = ["Evet.", "Hayır.", "Tamam."]


def bursts(audio: np.ndarray, sr: int, thr_db: float = -35.0, gap_s: float = 0.25) -> list[tuple[float, float]]:
    hop = int(sr * 0.01)
    n = len(audio) // hop
    if n == 0:
        return []
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames**2).mean(1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    out, start, last = [], None, None
    for i, loud in enumerate(db > thr_db):
        if not loud:
            continue
        if start is None:
            start = i
        elif (i - last) * 0.01 > gap_s:
            out.append((start * 0.01, (last + 1) * 0.01))
            start = i
        last = i
    if start is not None:
        out.append((start * 0.01, (last + 1) * 0.01))
    return out


def burst_similarity(audio: np.ndarray, sr: int, spans: list[tuple[float, float]]) -> float | None:
    if len(spans) < 2:
        return None
    import librosa

    def sig(span):
        seg = audio[int(span[0] * sr): int(span[1] * sr)]
        mf = librosa.feature.mfcc(y=seg.astype(np.float32), sr=sr, n_mfcc=13)
        return mf[1:].mean(1)

    a, b = sig(spans[0]), sig(spans[1])
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def analyse(wav: np.ndarray, sr: int) -> dict:
    spans = bursts(wav, sr)
    return {
        "audio_seconds": round(len(wav) / sr, 3),
        "bursts": len(spans),
        "burst_spans": [[round(a, 2), round(b, 2)] for a, b in spans],
        "burst_sim": None if (s := burst_similarity(wav, sr, spans)) is None else round(s, 3),
        "repeat_like": len(spans) >= 2,
    }


def first_burst_trim(wav: np.ndarray, sr: int, tail_s: float = 0.15) -> np.ndarray:
    spans = bursts(wav, sr)
    if len(spans) < 2:
        return wav
    end = min(len(wav), int((spans[0][1] + tail_s) * sr))
    out = wav[:end].copy()
    fade = min(len(out), int(0.05 * sr))
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def plan(phase: str, seeds: int) -> list[dict]:
    seed_list = list(range(1, seeds + 1))
    items = []
    if phase == "baseline":
        for phrase in PHRASES + [CONTROL_LONG]:
            for seed in seed_list:
                items.append({"config": "default", "params": {}, "phrase": phrase, "seed": seed})
    elif phase == "sweep":
        for name, values in SWEEP.items():
            for value in values:
                for phrase in SWEEP_PHRASES:
                    for seed in seed_list:
                        items.append({"config": f"{name}={value}", "params": {name: value}, "phrase": phrase, "seed": seed})
    elif phase == "combo":  # filled from CLI --combo name=value,name=value
        pass
    return items


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--phase", required=True, choices=["baseline", "sweep", "combo"])
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--name", default="probe1")
    p.add_argument("--combo", default="", help="name=value,name=value (phase combo)")
    p.add_argument("--phrases", default="", help="semicolon-separated override for combo phase")
    args = p.parse_args()

    import torch

    out_dir = REPO_ROOT / "outputs" / "chatterbox_repeat" / args.name
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "samples.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["phase"], r["config"], r["phrase"], r["seed"]))

    if args.phase == "combo":
        params = {k: float(v) for k, v in (kv.split("=") for kv in args.combo.split(",") if kv)}
        phrases = args.phrases.split(";") if args.phrases else SWEEP_PHRASES
        todo = [
            {"config": args.combo or "default", "params": params, "phrase": ph, "seed": s}
            for ph in phrases for s in range(1, args.seeds + 1)
        ]
    else:
        todo = plan(args.phase, args.seeds)
    todo = [t for t in todo if (args.phase, t["config"], t["phrase"], t["seed"]) not in done]
    print(f"{args.phase}: {len(todo)} örnek üretilecek (cihaz: {'cuda' if torch.cuda.is_available() else 'cpu'})")
    if not todo:
        return 0

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    t0 = time.time()
    for idx, t in enumerate(todo, 1):
        spoken = normalize_text(t["phrase"]).spoken
        torch.manual_seed(t["seed"])
        if device == "cuda":
            torch.cuda.manual_seed_all(t["seed"])
        started = time.time()
        wav = model.generate(spoken, language_id="tr", audio_prompt_path=str(args.reference), **t["params"])
        gen_s = time.time() - started
        arr = wav.squeeze().detach().cpu().float().numpy()
        slug = f"{args.phase}__{t['config'].replace('=', '')}__{hashlib.sha1(t['phrase'].encode()).hexdigest()[:6]}__s{t['seed']}"
        slug = "".join(c if c.isalnum() or c in "_.-" else "_" for c in slug)
        save_pcm16(out_dir / "wav" / f"{slug}.wav", arr, model.sr)
        row = {"phase": args.phase, "config": t["config"], "params": t["params"], "phrase": t["phrase"], "spoken": spoken,
               "seed": t["seed"], "wav": f"wav/{slug}.wav", "gen_seconds": round(gen_s, 2), **analyse(arr, model.sr)}
        trimmed = first_burst_trim(arr, model.sr)
        row["trim_audio_seconds"] = round(len(trimmed) / model.sr, 3)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{idx}/{len(todo)}] {t['config']:<22} {t['phrase']:<14} s{t['seed']}: {row['audio_seconds']}s bursts={row['bursts']}")
    print(f"Bitti: {time.time() - t0:.0f}s · VRAM tepe {torch.cuda.max_memory_allocated() / 2**20:.0f} MB" if device == "cuda" else "Bitti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
