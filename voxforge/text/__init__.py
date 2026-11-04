"""Turkish text intelligence: segmentation and spoken-text normalisation (VF-004).

``prepare_segments`` is the entry point for the future Studio pipeline: it
segments the *original* text (offsets preserved), normalises each segment into
a separate spoken text and, if expansion pushed a segment over the engine
limit, splits the spoken text again for synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from scripts.text_chunking_utils import split_text_for_tts
from voxforge.text.normalize import (
    NORMALIZER_VERSION,
    LexiconEntry,
    NormalizationResult,
    Transformation,
    apply_transformations,
    load_lexicon,
    normalize_text,
    number_to_words,
)
from voxforge.text.segment import Segment, segment_text

__all__ = [
    "NORMALIZER_VERSION",
    "LexiconEntry",
    "NormalizationResult",
    "PreparedSegment",
    "Segment",
    "Transformation",
    "apply_transformations",
    "load_lexicon",
    "normalize_text",
    "number_to_words",
    "prepare_segments",
    "segment_text",
]


@dataclass(frozen=True)
class PreparedSegment:
    segment: Segment
    normalization: NormalizationResult
    spoken_chunks: tuple[str, ...]

    @property
    def spoken(self) -> str:
        return self.normalization.spoken


def prepare_segments(
    text: str,
    max_chars: int = 220,
    lexicon: Iterable[LexiconEntry] = (),
) -> list[PreparedSegment]:
    entries = list(lexicon)
    prepared: list[PreparedSegment] = []
    for segment in segment_text(text, max_chars=max_chars):
        normalization = normalize_text(segment.text, entries)
        spoken = normalization.spoken
        chunks = tuple(split_text_for_tts(spoken, max_chars=max_chars)) if len(spoken) > max_chars else (spoken,)
        prepared.append(PreparedSegment(segment, normalization, chunks))
    return prepared
