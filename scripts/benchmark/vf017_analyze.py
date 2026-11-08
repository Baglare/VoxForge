# -*- coding: utf-8 -*-
"""VF-017 measurements for vf017_narration_probe.py outputs (eval venv, local models only).

    set HF_HOME=C:\\AvenoxData\\Models\\HF & set HF_HUB_OFFLINE=1
    <eval-venv>\\python.exe scripts/benchmark/vf017_analyze.py --probe outputs/vf017/probe

Per output: faster-whisper large-v3 word timestamps -> CER and per-sentence rhythm
(voxforge/playground/rhythm.py), ECAPA similarity to the reference AND to the default
(A) output of the same text, F0 median/spread (simple autocorrelation; a proxy for
intonation range, not an emotion score) and audio bandwidth.

Per raw sentence part (model output before any join): leading/trailing silence at the
joiner's trim threshold, how much audible signal the trim would cut (onset quieter than
-45 dB but above the noise floor), attack time, onset level and onset pitch relative to the
rest of the sentence. These separate "the model starts abruptly" from "the join cuts it".

Nothing is downloaded (HF_HUB_OFFLINE); ECAPA comes from C:\\AvenoxData\\Models\\HF.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "benchmark")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from voxforge.benchmark.metrics import score_transcript  # noqa: E402
from voxforge.playground import audio_join  # noqa: E402
from voxforge.playground.rhythm import analyze  # noqa: E402

FRAME = 0.01


def load(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    return audio.mean(axis=1), rate


def frame_db(x: np.ndarray, rate: int, frame: float = FRAME) -> np.ndarray:
    size = max(1, int(rate * frame))
    count = len(x) // size
    if not count:
        return np.array([-120.0])
    blocks = x[: count * size].reshape(count, size)
    return 10 * np.log10((blocks**2).mean(axis=1) + 1e-12)


def f0_track(x: np.ndarray, rate: int, frame: float = 0.04, hop: float = 0.01) -> np.ndarray:
    """Very small autocorrelation F0 (70-400 Hz); NaN for unvoiced/quiet frames."""
    size, step = int(rate * frame), int(rate * hop)
    lo, hi = int(rate / 400), int(rate / 70)
    levels = frame_db(x, rate, hop)
    gate = levels.max() - 30
    out = []
    for index, start in enumerate(range(0, len(x) - size, step)):
        block = x[start : start + size]
        if index >= len(levels) or levels[index] < gate:
            out.append(np.nan)
            continue
        block = block - block.mean()
        corr = np.correlate(block, block, "full")[size - 1 :]
        if corr[0] <= 0:
            out.append(np.nan)
            continue
        corr = corr / corr[0]
        lag = lo + int(np.argmax(corr[lo:hi]))
        out.append(rate / lag if corr[lag] > 0.45 else np.nan)
    return np.array(out)


def semitones(values: np.ndarray) -> float | None:
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return None
    st = 12 * np.log2(values / np.median(values))
    return round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2)


def bandwidth(x: np.ndarray, rate: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    cumulative = np.cumsum(spec) / spec.sum()
    return round(float(freqs[np.searchsorted(cumulative, 0.99)]), 0)


def onset_stats(path: Path) -> dict[str, Any]:
    x, rate = load(path)
    levels = frame_db(x, rate)
    peak = levels.max()
    trim_level = peak + audio_join.TRIM_THRESHOLD_DB
    noise = float(np.percentile(levels, 5))
    audible = max(noise + 10, peak - 60)
    above_trim = np.flatnonzero(levels >= trim_level)
    above_audible = np.flatnonzero(levels >= audible)
    first, last = above_trim[0], above_trim[-1]
    margin = int(audio_join.TRIM_MARGIN_SECONDS / FRAME)
    keep_start, keep_end = max(0, first - margin), min(len(levels), last + 1 + margin)
    voiced = levels[first : last + 1]
    median = float(np.median(voiced[voiced >= peak - 30])) if len(voiced) else peak
    rise = np.flatnonzero(levels[first:] >= median - 6)
    f0 = f0_track(x, rate)
    f0_start = int(first * FRAME / 0.01)
    head_f0 = f0[f0_start : f0_start + 30]
    rest_f0 = f0[f0_start + 30 :]
    head_level = levels[first : first + 25]
    def nanmedian(values: np.ndarray) -> float | None:
        values = values[~np.isnan(values)]
        return float(np.median(values)) if len(values) >= 3 else None
    hf0, rf0 = nanmedian(head_f0), nanmedian(rest_f0)
    return {
        "duration": round(len(x) / rate, 3),
        "lead_silence_ms": int(first * FRAME * 1000),
        "trail_silence_ms": int((len(levels) - 1 - last) * FRAME * 1000),
        # sesli ama -45 dB eşiğinin altında kalan ve kırpma payının (40 ms) dışında kalan kısım
        "trim_cuts_head_ms": int(max(0, keep_start - above_audible[0]) * FRAME * 1000) if len(above_audible) else 0,
        "trim_cuts_tail_ms": int(max(0, above_audible[-1] + 1 - keep_end) * FRAME * 1000) if len(above_audible) else 0,
        "attack_ms": int(rise[0] * FRAME * 1000) if len(rise) else None,
        "onset_level_vs_median_db": round(float(head_level.max() - median), 2) if len(head_level) else None,
        "onset_f0_vs_rest_st": round(12 * float(np.log2(hf0 / rf0)), 2) if hf0 and rf0 else None,
        "noise_floor_db_rel_peak": round(noise - peak, 1),
        "peak_dbfs": round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2),
        "clipped_samples": int((np.abs(x) >= 32767 / 32768).sum()),
    }


class Evaluator:
    def __init__(self) -> None:
        import torch
        from faster_whisper import WhisperModel
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
        kwargs: dict[str, Any] = {
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "savedir": str(Path(os.environ.get("HF_HOME", "C:/AvenoxData/Models/HF")) / "spkrec-ecapa-voxceleb"),
            "run_opts": {"device": device},
        }
        try:
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass
        self.encoder = EncoderClassifier.from_hparams(**kwargs)
        self.device = device

    def words(self, path: Path) -> tuple[list[dict[str, Any]], str]:
        segments, _ = self.asr.transcribe(str(path), language="tr", beam_size=5, vad_filter=False, word_timestamps=True)
        words, text = [], []
        for segment in segments:
            text.append(segment.text)
            words += [{"start": w.start, "end": w.end, "word": w.word} for w in segment.words or []]
        return words, "".join(text).strip()

    def embedding(self, path: Path):
        import torchaudio.functional as F

        x, rate = load(path)
        tensor = self.torch.from_numpy(x.astype("float32")).unsqueeze(0)
        if rate != 16000:
            tensor = F.resample(tensor, rate, 16000)
        with self.torch.no_grad():
            return self.encoder.encode_batch(tensor.to(self.device)).squeeze()

    def cosine(self, a, b) -> float:
        return round(float(self.torch.nn.functional.cosine_similarity(a, b, dim=0)), 4)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REPO_ROOT / "profiles" / "mimo" / "preprocessed_reference.wav")
    parser.add_argument("--extra", type=Path, action="append", default=[], help="ek manifest (ör. düzeltme sonrası)")
    args = parser.parse_args(argv)
    rows = json.loads((args.probe / "manifest.json").read_text(encoding="utf-8"))
    for extra in args.extra:
        rows += json.loads(extra.read_text(encoding="utf-8"))
    ev = Evaluator()
    reference = ev.embedding(args.reference)
    embeddings = {row["name"]: ev.embedding(Path(row["wav"])) for row in rows}
    results: dict[str, Any] = {}
    for row in rows:
        path = Path(row["wav"])
        words, heard = ev.words(path)
        summary = analyze(path, row["text"], words)
        x, rate = load(path)
        base = row["name"].rsplit("_", 1)[0] if row["name"].endswith(("_preset", "_nostyle")) else None
        default_name = (
            row["name"].replace("B_segment", "A_default").replace("C_segment_style", "A_default").replace("_fix", "")
            if row["group"] in ("abc", "fix") else f"{base}_A_default" if base else None
        )
        result = {
            "group": row["group"], "label": row.get("label"), "style": row.get("style"),
            "cer": score_transcript(row["text"], heard).get("cer"), "asr_text": heard,
            "sim_reference": ev.cosine(reference, embeddings[row["name"]]),
            "sim_default": ev.cosine(embeddings[default_name], embeddings[row["name"]]) if default_name in embeddings and default_name != row["name"] else None,
            "f0_range_st": semitones(f0_track(x, rate)),
            "bandwidth_99_hz": bandwidth(x, rate),
            "peak_dbfs": round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2),
            "clipped_samples": int((np.abs(x) >= 32767 / 32768).sum()),  # save_pcm16 ±1'de keser
            "rejoined_equal": row.get("rejoined_equal"),
            "vram_peak_mb": row.get("vram_peak_mb"),
            **{k: summary[k] for k in ("duration", "overall_rate", "articulation_rate", "pause_count", "pause_seconds",
                                       "longest_pause", "first_two_rate", "rest_rate", "head_vs_rest")},
            "sentences": [{k: s[k] for k in ("index", "rate", "articulation_rate", "pause_before", "text")} for s in summary["sentences"]],
        }
        if row.get("raw_parts"):
            result["raw_parts"] = [{"part": i, **onset_stats(Path(p))} for i, p in enumerate(row["raw_parts"])]
        results[row["name"]] = result
        print(f"{row['name']:<26} CER {result['cer']}  simRef {result['sim_reference']}  simA {result['sim_default']}  "
              f"rate {result['overall_rate']}  art {result['articulation_rate']}  pauses {result['pause_seconds']}  "
              f"head/rest {result['head_vs_rest']}  f0 {result['f0_range_st']} st  bw {result['bandwidth_99_hz']}", flush=True)
        for part in result.get("raw_parts", []):
            print(f"    raw {part['part']}: lead {part['lead_silence_ms']} ms  tail {part['trail_silence_ms']} ms  "
                  f"cut head/tail {part['trim_cuts_head_ms']}/{part['trim_cuts_tail_ms']} ms  attack {part['attack_ms']} ms  "
                  f"onset level {part['onset_level_vs_median_db']} dB  onset f0 {part['onset_f0_vs_rest_st']} st", flush=True)
    out = args.probe / "analysis.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
