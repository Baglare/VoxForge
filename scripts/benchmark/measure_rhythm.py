# -*- coding: utf-8 -*-
"""Per-sentence speaking-rate / pause measurement (VF-016), inside the eval venv.

    <eval-venv>\\python.exe scripts/benchmark/measure_rhythm.py --record <playground id> [...]
    <eval-venv>\\python.exe scripts/benchmark/measure_rhythm.py --wav x.wav --text "..." [...]

faster-whisper large-v3 (already cached for VF-Q1, no download) gives word timestamps;
``voxforge/playground/rhythm.py`` turns them into syllables/second per sentence, pauses,
leading/trailing silence and the ratio "first two sentences vs the rest". ``--json``
writes everything for later comparison. The GPU is used when CUDA is available; run it
when no TTS engine holds the GPU (Playground: "Motoru kapat").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxforge.benchmark.metrics import score_transcript  # noqa: E402
from voxforge.playground.rhythm import analyze, format_table  # noqa: E402

PLAYGROUND_AUDIO = REPO_ROOT / "outputs" / "playground" / "audio"


def load_asr(device: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")


def words_of(asr: Any, path: Path) -> tuple[list[dict[str, Any]], str]:
    segments, _ = asr.transcribe(str(path), language="tr", beam_size=5, vad_filter=False, word_timestamps=True)
    words, text = [], []
    for segment in segments:
        text.append(segment.text)
        for word in segment.words or []:
            words.append({"start": word.start, "end": word.end, "word": word.word})
    return words, "".join(text).strip()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="append", default=[], help="Playground kayıt kimliği (metin kayıttan alınır)")
    parser.add_argument("--wav", action="append", default=[], type=Path)
    parser.add_argument("--text", action="append", default=[], help="--wav ile aynı sırada okunan metin")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manifest", type=Path, help="voxcpm_rhythm_probe.py manifest.json")
    args = parser.parse_args(argv)

    items: list[tuple[str, Path, str]] = []
    for record_id in args.record:
        record = json.loads((PLAYGROUND_AUDIO / f"{record_id}.json").read_text(encoding="utf-8"))
        items.append((record_id, Path(record["wav"]), record.get("spoken_text") or record["text"]))
    if len(args.text) != len(args.wav):
        parser.error("--wav ve --text sayısı aynı olmalı")
    if args.manifest:
        for row in json.loads(args.manifest.read_text(encoding="utf-8")):
            items.append((row["name"], Path(row["wav"]), row["text"]))
    items += [(wav.stem, wav, text) for wav, text in zip(args.wav, args.text)]
    if not items:
        parser.error("en az bir --record veya --wav gerekli")

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    asr = load_asr(device)
    results = {}
    for name, wav, text in items:
        words, heard = words_of(asr, wav)
        summary = analyze(wav, text, words)
        summary["asr_text"] = heard
        summary["asr"] = score_transcript(text, heard)
        results[name] = summary
        print(format_table(name, summary), flush=True)
        print(f"  ASR CER {summary['asr'].get('cer')}", flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
