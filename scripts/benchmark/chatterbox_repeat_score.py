# -*- coding: utf-8 -*-
"""ASR + speaker-similarity scoring for chatterbox_repeat_probe.py output (eval venv).

    <eval-venv>\\python.exe scripts/benchmark/chatterbox_repeat_score.py \
        --probe outputs/chatterbox_repeat/<name> --reference <reference.wav>

Reads <probe>/samples.jsonl, writes <probe>/scores.jsonl (resumable). Uses the same
faster-whisper large-v3 and ECAPA models as evaluate.py, so numbers are comparable with VF-Q1.
``repeat`` = the expected phrase occurs 2+ times in the transcript (token sequence match);
``exact`` = transcript equals the expected phrase after normalisation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import Evaluator  # noqa: E402
from voxforge.benchmark.metrics import normalize_for_cer, score_transcript  # noqa: E402


def count_phrase(tokens: list[str], phrase: list[str]) -> int:
    if not phrase:
        return 0
    n, i = 0, 0
    while i <= len(tokens) - len(phrase):
        if tokens[i: i + len(phrase)] == phrase:
            n, i = n + 1, i + len(phrase)
        else:
            i += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    args = ap.parse_args()
    probe = args.probe.resolve()
    samples = [json.loads(line) for line in (probe / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    scores_path = probe / "scores.jsonl"
    done = set()
    if scores_path.exists():
        done = {json.loads(line)["wav"] for line in scores_path.read_text(encoding="utf-8").splitlines()}
    todo = [s for s in samples if s["wav"] not in done]
    print(f"{len(todo)} örnek puanlanacak")
    if not todo:
        return 0
    ev = Evaluator(probe.parent / "_eval_models", use_asr=True, use_sim=True)
    ref_emb = ev.embedding(args.reference)
    with scores_path.open("a", encoding="utf-8") as out:
        for i, s in enumerate(todo, 1):
            wav = probe / s["wav"]
            transcript = ev.transcribe(wav)
            expected = normalize_for_cer(s["spoken"]).split()
            tokens = normalize_for_cer(transcript).split()
            row = {
                "wav": s["wav"], "phase": s["phase"], "config": s["config"], "phrase": s["phrase"], "seed": s["seed"],
                "transcript": transcript,
                "occurrences": count_phrase(tokens, expected),
                "repeat": count_phrase(tokens, expected) >= 2,
                "exact": tokens == expected,
                "similarity": ev.similarity(ref_emb, wav),
                **score_transcript(s["spoken"], transcript),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {s['config']} | {s['phrase']} s{s['seed']} -> {transcript!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
