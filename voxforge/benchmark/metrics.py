"""Objective, model-free metrics for the voice benchmark (VF-Q1).

- CER from an ASR transcript against the spoken text the model received,
  split into substitutions / deletions / insertions. Insertions are the main
  automatic signal for "metin dışı ses" (extra words, babble, repetition).
- Duration plausibility: seconds per character far outside the normal range
  flags truncation or runaway generation.
These are proxies. Naturalness and voice similarity are judged by listening;
speaker similarity is computed separately in the evaluation environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voxforge.text import normalize_text

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_NON_TEXT = re.compile(r"[^\wçğıöşüâîû ]+")
_SPACES = re.compile(r"\s+")

# Tipik Türkçe okuma hızı ~12-16 karakter/sn. Bu aralığın çok dışı şüphelidir.
MIN_SECONDS_PER_CHAR = 0.035
MAX_SECONDS_PER_CHAR = 0.16
EXTRA_SPEECH_INSERTION_RATIO = 0.15


def normalize_for_cer(text: str) -> str:
    """Lowercase (Turkish dotted/dotless i), digits to words, punctuation removed."""
    spoken = normalize_text(text or "").spoken
    lowered = spoken.translate(_TR_LOWER).lower()
    lowered = lowered.replace("'", " ").replace("’", " ").replace("_", " ")
    return _SPACES.sub(" ", _NON_TEXT.sub(" ", lowered)).strip()


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        return self.errors / self.reference_length if self.reference_length else float(self.errors > 0)

    @property
    def insertion_rate(self) -> float:
        return self.insertions / self.reference_length if self.reference_length else float(self.insertions > 0)


def edit_counts(reference: str, hypothesis: str) -> EditCounts:
    """Levenshtein alignment with operation counts (character level)."""
    ref, hyp = list(reference), list(hypothesis)
    rows, cols = len(ref) + 1, len(hyp) + 1
    # cost, subs, dels, ins
    previous = [(j, 0, 0, j) for j in range(cols)]
    for i in range(1, rows):
        current = [(i, 0, i, 0)]
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                diagonal = previous[j - 1]
            else:
                c, s, d, n = previous[j - 1]
                diagonal = (c + 1, s + 1, d, n)
            c, s, d, n = previous[j]
            delete = (c + 1, s, d + 1, n)
            c, s, d, n = current[j - 1]
            insert = (c + 1, s, d, n + 1)
            current.append(min(diagonal, delete, insert))
        previous = current
    _, subs, dels, ins = previous[-1]
    return EditCounts(subs, dels, ins, len(ref))


def score_transcript(spoken_text: str, transcript: str) -> dict[str, float | int | bool]:
    counts = edit_counts(normalize_for_cer(spoken_text), normalize_for_cer(transcript))
    return {
        "cer": round(counts.cer, 4),
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "insertion_rate": round(counts.insertion_rate, 4),
        "extra_speech_suspect": counts.insertion_rate > EXTRA_SPEECH_INSERTION_RATIO,
    }


def duration_check(spoken_text: str, duration_s: float | None) -> dict[str, float | bool | None]:
    characters = len(normalize_for_cer(spoken_text).replace(" ", "")) or 1
    if not duration_s:
        return {"seconds_per_char": None, "too_short": True, "too_long": False}
    per_char = duration_s / characters
    return {
        "seconds_per_char": round(per_char, 4),
        "too_short": per_char < MIN_SECONDS_PER_CHAR,
        "too_long": per_char > MAX_SECONDS_PER_CHAR,
    }
