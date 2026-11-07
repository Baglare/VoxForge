"""Speech rhythm measurement for long narration (VF-016).

Pure functions (stdlib only) so the same code runs in the main .venv, in the eval venv
and in tests. Inputs are PCM WAV files plus, optionally, ASR word timestamps
(``scripts/benchmark/measure_rhythm.py`` gets them from faster-whisper).

Speaking rate is measured in **syllables per second**: in Turkish every syllable has
exactly one vowel, so the vowel count of the spoken (normalised) text is an exact
syllable count without a dictionary. Two rates are reported per sentence:

- ``rate``: syllables / (sentence end - sentence start), pauses inside the sentence
  included (what a listener perceives as tempo);
- ``articulation_rate``: syllables / voiced time only (pauses removed).
"""

from __future__ import annotations

import array
import math
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

TR_VOWELS = set("aeıioöuüâîûAEIİOÖUÜÂÎÛ")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
FRAME_SECONDS = 0.02
SILENCE_DB = -40.0  # relative to the loudest frame
MIN_PAUSE_SECONDS = 0.12


def syllables(text: str) -> int:
    return sum(1 for char in text if char in TR_VOWELS)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_END.split(re.sub(r"\s+", " ", text or "").strip()) if part.strip()]


# ---- audio --------------------------------------------------------------------------------
def read_pcm16_mono(path: Path) -> tuple[list[float], int]:
    """Samples in [-1, 1] (first channel) and the sample rate. 16-bit PCM only."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: yalnız 16-bit PCM WAV okunur")
        channels, rate = handle.getnchannels(), handle.getframerate()
        data = array.array("h")
        data.frombytes(handle.readframes(handle.getnframes()))
    if channels > 1:
        data = data[::channels]
    return [value / 32768.0 for value in data], rate


def frame_db(samples: list[float], rate: int, frame_seconds: float = FRAME_SECONDS) -> list[float]:
    size = max(1, int(rate * frame_seconds))
    levels = []
    for start in range(0, len(samples), size):
        block = samples[start : start + size]
        energy = sum(value * value for value in block) / max(1, len(block))
        levels.append(10 * math.log10(energy) if energy > 1e-12 else -120.0)
    return levels


def voiced_mask(levels: list[float], threshold_db: float = SILENCE_DB) -> list[bool]:
    peak = max(levels) if levels else -120.0
    return [level >= peak + threshold_db for level in levels]


def pauses(mask: list[bool], frame_seconds: float = FRAME_SECONDS, min_pause: float = MIN_PAUSE_SECONDS) -> list[tuple[float, float]]:
    """Silent stretches *between* voiced frames (leading/trailing silence excluded)."""
    found: list[tuple[float, float]] = []
    try:
        first = mask.index(True)
        last = len(mask) - 1 - mask[::-1].index(True)
    except ValueError:
        return found
    run_start = None
    for index in range(first, last + 1):
        if not mask[index] and run_start is None:
            run_start = index
        elif mask[index] and run_start is not None:
            if (index - run_start) * frame_seconds >= min_pause:
                found.append((round(run_start * frame_seconds, 3), round(index * frame_seconds, 3)))
            run_start = None
    return found


@dataclass
class AudioStats:
    duration: float
    leading_silence: float
    trailing_silence: float
    voiced_seconds: float
    pauses: list[tuple[float, float]]
    loudness_db: float  # mean dB of voiced frames (relative scale, for level jumps)


def audio_stats(path: Path) -> AudioStats:
    samples, rate = read_pcm16_mono(path)
    levels = frame_db(samples, rate)
    mask = voiced_mask(levels)
    duration = len(samples) / float(rate) if rate else 0.0
    if True in mask:
        first = mask.index(True)
        last = len(mask) - 1 - mask[::-1].index(True)
    else:
        first, last = len(mask), len(mask) - 1
    voiced = [level for level, on in zip(levels, mask) if on]
    return AudioStats(
        duration=round(duration, 3),
        leading_silence=round(first * FRAME_SECONDS, 3),
        trailing_silence=round(max(0.0, duration - (last + 1) * FRAME_SECONDS), 3),
        voiced_seconds=round(sum(mask) * FRAME_SECONDS, 3),
        pauses=pauses(mask),
        loudness_db=round(sum(voiced) / len(voiced), 2) if voiced else -120.0,
    )


# ---- sentences ------------------------------------------------------------------------------
@dataclass
class SentenceRhythm:
    index: int
    text: str
    start: float
    end: float
    syllables: int
    rate: float  # syllables per second, inner pauses included
    articulation_rate: float  # syllables per voiced second
    pause_before: float


def align_words(sentences: list[str], words: list[dict[str, Any]]) -> list[tuple[float, float] | None]:
    """Sentence time spans from ASR words.

    Words are assigned in order, proportionally to each sentence's word count (ASR
    rarely splits/merges Turkish words, and a proportional map degrades gracefully when
    it does). Returns None for a sentence that received no word.
    """
    counts = [max(1, len(sentence.split())) for sentence in sentences]
    total, available = sum(counts), len(words)
    spans: list[tuple[float, float] | None] = []
    cursor = 0.0
    for count in counts:
        begin = int(round(cursor))
        cursor += count * available / total if total else 0
        stop = int(round(cursor))
        chunk = words[begin:stop]
        spans.append((float(chunk[0]["start"]), float(chunk[-1]["end"])) if chunk else None)
    return spans


def sentence_rhythm(
    sentences: list[str], words: list[dict[str, Any]], mask: list[bool], frame_seconds: float = FRAME_SECONDS
) -> list[SentenceRhythm]:
    rows: list[SentenceRhythm] = []
    previous_end = None
    for index, (sentence, span) in enumerate(zip(sentences, align_words(sentences, words))):
        if span is None:
            continue
        start, end = span
        first, last = int(start / frame_seconds), max(int(start / frame_seconds), int(math.ceil(end / frame_seconds)))
        voiced = sum(mask[first:last]) * frame_seconds or (end - start)
        count = syllables(sentence)
        length = max(end - start, 1e-3)
        rows.append(
            SentenceRhythm(
                index=index,
                text=sentence,
                start=round(start, 3),
                end=round(end, 3),
                syllables=count,
                rate=round(count / length, 2),
                articulation_rate=round(count / max(voiced, 1e-3), 2),
                pause_before=round(start - previous_end, 3) if previous_end is not None else 0.0,
            )
        )
        previous_end = end
    return rows


def summarize(rows: Iterable[SentenceRhythm], stats: AudioStats, text: str) -> dict[str, Any]:
    rows = list(rows)
    rates = [row.rate for row in rows]
    head = rates[:2]
    tail = rates[2:]
    total_syllables = syllables(text)
    speech_span = max(stats.duration - stats.leading_silence - stats.trailing_silence, 1e-3)
    return {
        "duration": stats.duration,
        "syllables": total_syllables,
        "overall_rate": round(total_syllables / speech_span, 2),
        "articulation_rate": round(total_syllables / max(stats.voiced_seconds, 1e-3), 2),
        "pause_count": len(stats.pauses),
        "pause_seconds": round(sum(end - start for start, end in stats.pauses), 3),
        "longest_pause": round(max((end - start for start, end in stats.pauses), default=0.0), 3),
        "leading_silence": stats.leading_silence,
        "trailing_silence": stats.trailing_silence,
        "first_two_rate": round(sum(head) / len(head), 2) if head else None,
        "rest_rate": round(sum(tail) / len(tail), 2) if tail else None,
        "head_vs_rest": round((sum(head) / len(head)) / (sum(tail) / len(tail)), 3) if head and tail else None,
        "sentences": [asdict(row) for row in rows],
    }


def analyze(path: Path, text: str, words: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    stats = audio_stats(path)
    sentences = split_sentences(text)
    rows: list[SentenceRhythm] = []
    if words:
        samples, rate = read_pcm16_mono(path)
        rows = sentence_rhythm(sentences, words, voiced_mask(frame_db(samples, rate)))
    return summarize(rows, stats, text)


def format_table(name: str, summary: dict[str, Any]) -> str:
    lines = [
        f"== {name}: {summary['duration']} sn · {summary['syllables']} hece · genel {summary['overall_rate']} hece/sn"
        f" · artikülasyon {summary['articulation_rate']} · duraklama {summary['pause_count']} ({summary['pause_seconds']} sn,"
        f" en uzun {summary['longest_pause']}) · ilk-2/kalan {summary['first_two_rate']}/{summary['rest_rate']}"
        f" (oran {summary['head_vs_rest']})"
    ]
    for row in summary["sentences"]:
        lines.append(
            f"  {row['index'] + 1:>2}. {row['start']:6.2f}-{row['end']:6.2f}  {row['syllables']:>3} hece  "
            f"{row['rate']:5.2f} h/sn (art. {row['articulation_rate']:5.2f})  önce {row['pause_before']:+.2f}  {row['text'][:50]}"
        )
    return "\n".join(lines)
