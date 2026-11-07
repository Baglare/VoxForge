"""Joining generated PCM WAV parts without long artificial gaps or level jumps (VF-016).

Stdlib only: the same code runs inside every engine worker environment and in the main
.venv (scene merge). Every part must share channels, sample width (16-bit) and rate.

- ``trim``: leading/trailing silence of each part is cut to a short margin, so the
  pause between two parts is exactly the requested one (VoxCPM2 parts end with 0.0-0.3 s
  of variable silence; that randomness is what made sentence joins sound rushed or uneven).
- short linear fades at every cut avoid clicks;
- ``match_loudness``: each part's *voiced* RMS is moved toward the median of all parts,
  limited to +/- ``max_gain_db``, and a peak guard prevents clipping.

Without ``trim``/``match_loudness`` and with a single pause value the output is
byte-identical to the original Playground join (0.2 s of digital silence between parts).
"""

from __future__ import annotations

import array
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TRIM_THRESHOLD_DB = -45.0  # relative to the part's own peak frame
TRIM_MARGIN_SECONDS = 0.04
FADE_SECONDS = 0.008
FRAME_SECONDS = 0.01
MAX_GAIN_DB = 6.0
PEAK_LIMIT = 0.97


@dataclass
class Part:
    samples: array.array  # int16, interleaved
    channels: int
    rate: int


def read_part(path: Path) -> Part:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: yalnız 16-bit PCM WAV birleştirilir")
        samples = array.array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
        return Part(samples, handle.getnchannels(), handle.getframerate())


def _frame_levels(part: Part) -> list[float]:
    size = max(1, int(part.rate * FRAME_SECONDS)) * part.channels
    levels = []
    data = part.samples
    for start in range(0, len(data), size):
        block = data[start : start + size]
        energy = sum(value * value for value in block) / max(1, len(block))
        levels.append(10 * math.log10(energy / 32768.0**2) if energy > 0 else -120.0)
    return levels


def trim_silence(part: Part, threshold_db: float = TRIM_THRESHOLD_DB, margin: float = TRIM_MARGIN_SECONDS) -> Part:
    levels = _frame_levels(part)
    if not levels:
        return part
    peak = max(levels)
    voiced = [index for index, level in enumerate(levels) if level >= peak + threshold_db]
    if not voiced:
        return part
    frame = max(1, int(part.rate * FRAME_SECONDS))
    margin_frames = int(margin / FRAME_SECONDS)
    first = max(0, voiced[0] - margin_frames) * frame
    last = min(len(levels), voiced[-1] + 1 + margin_frames) * frame
    return Part(part.samples[first * part.channels : last * part.channels], part.channels, part.rate)


def fade(part: Part, seconds: float = FADE_SECONDS) -> Part:
    count = min(int(part.rate * seconds), len(part.samples) // max(1, part.channels) // 2)
    if count <= 0:
        return part
    data = array.array("h", part.samples)
    frames = len(data) // part.channels
    for index in range(count):
        gain = index / count
        for channel in range(part.channels):
            head = index * part.channels + channel
            tail = (frames - 1 - index) * part.channels + channel
            data[head] = int(data[head] * gain)
            data[tail] = int(data[tail] * gain)
    return Part(data, part.channels, part.rate)


def voiced_rms_db(part: Part, threshold_db: float = TRIM_THRESHOLD_DB + 15) -> float | None:
    levels = _frame_levels(part)
    if not levels:
        return None
    peak = max(levels)
    voiced = [level for level in levels if level >= peak + threshold_db]
    if not voiced:
        return None
    mean_energy = sum(10 ** (level / 10) for level in voiced) / len(voiced)
    return 10 * math.log10(mean_energy)


def apply_gain(part: Part, gain_db: float) -> Part:
    if abs(gain_db) < 0.05:
        return part
    peak = max((abs(value) for value in part.samples), default=0)
    factor = 10 ** (gain_db / 20)
    if peak and peak * factor > PEAK_LIMIT * 32767:
        factor = PEAK_LIMIT * 32767 / peak  # tepe koruması: kırpma yerine kazancı sınırla
    data = array.array("h", (max(-32768, min(32767, int(round(value * factor)))) for value in part.samples))
    return Part(data, part.channels, part.rate)


def loudness_gains(parts: Sequence[Part], max_gain_db: float = MAX_GAIN_DB) -> list[float]:
    levels = [voiced_rms_db(part) for part in parts]
    known = sorted(level for level in levels if level is not None)
    if not known:
        return [0.0 for _ in parts]
    target = known[len(known) // 2]
    return [0.0 if level is None else max(-max_gain_db, min(max_gain_db, target - level)) for level in levels]


def join_wavs(
    paths: Sequence[Path],
    out_path: Path,
    pauses: float | Sequence[float] = 0.2,
    trim: bool = False,
    match_loudness: bool = False,
    max_gain_db: float = MAX_GAIN_DB,
) -> dict[str, float | list[float]]:
    """Write ``paths`` joined into ``out_path`` (atomic). ``pauses``: one value or one per gap."""
    if not paths:
        raise ValueError("Birleştirilecek parça yok.")
    gaps = [float(pauses)] * (len(paths) - 1) if isinstance(pauses, (int, float)) else [float(p) for p in pauses]
    if len(gaps) != len(paths) - 1:
        raise ValueError(f"{len(paths)} parça için {len(paths) - 1} duraklama gerekir, {len(gaps)} verildi.")
    if any(gap < 0 or gap > 5 for gap in gaps):
        raise ValueError("Duraklama 0-5 sn arasında olmalı.")
    parts = [read_part(Path(path)) for path in paths]
    shape = (parts[0].channels, parts[0].rate)
    for path, part in zip(paths, parts):
        if (part.channels, part.rate) != shape:
            raise ValueError(f"Parça biçimleri uyuşmuyor: {shape} != {(part.channels, part.rate)} ({path})")
    if trim:
        parts = [fade(trim_silence(part)) for part in parts]
    gains = loudness_gains(parts, max_gain_db) if match_loudness else [0.0] * len(parts)
    parts = [apply_gain(part, gain) for part, gain in zip(parts, gains)]

    channels, rate = shape
    temporary = out_path.with_name(out_path.name + ".partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for index, part in enumerate(parts):
            if index:
                handle.writeframes(b"\x00" * int(rate * gaps[index - 1]) * channels * 2)
            handle.writeframes(part.samples.tobytes())
    os.replace(temporary, out_path)
    with wave.open(str(out_path), "rb") as handle:
        duration = handle.getnframes() / float(handle.getframerate())
    return {"duration": duration, "gains_db": [round(gain, 2) for gain in gains], "pauses": gaps}
