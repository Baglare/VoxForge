"""Paragraph and sentence segmentation that keeps offsets into the original text (VF-004).

Rules are deliberately conservative: when unsure, do not split. Long sentences
fall back to the existing punctuation/space splitter.

- Paragraphs are separated by blank lines. A single newline is treated as a
  space (copy-pasted, hard-wrapped prose must not be cut mid-sentence).
- A sentence ends at ``.``, ``!``, ``?`` or ``…`` (runs such as ``?!`` or
  ``...`` count once) when followed by whitespace and an uppercase letter,
  digit, quote/bracket/dash, or the end of the paragraph.
- No split after known abbreviations (``Dr.``, ``Prof.``, ``vb.`` ...), single
  uppercase initials (``A. Yılmaz``) or bare numbers (``1. Dünya Savaşı``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.text_chunking_utils import find_split_index
from voxforge.text.dialogue import starts_dialogue_line

ABBREVIATIONS = frozenset(
    {
        "dr", "prof", "doç", "doc", "av", "op", "uzm", "yrd", "öğr", "ogr", "sn", "bkz",
        "vb", "vs", "örn", "orn", "no", "st", "mah", "cad", "sok", "apt", "tel",
        "yy", "mö", "ms", "çev", "ed", "hz", "s", "c", "bl",
    }
)
TERMINALS = ".!?…"
OPENERS = "\"'“‘«([—–-"
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Segment:
    index: int
    paragraph: int
    start: int
    end: int
    text: str  # original slice with whitespace collapsed; never rewritten


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, len(text)))
    return [_strip_span(text, start, end) for start, end in spans if text[start:end].strip()]


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _word_before(text: str, index: int, floor: int) -> str:
    position = index
    while position > floor and not text[position - 1].isspace() and text[position - 1] not in OPENERS:
        position -= 1
    return text[position:index]


def _is_sentence_end(text: str, run_start: int, run_end: int, paragraph_end: int, floor: int) -> bool:
    if run_end >= paragraph_end:
        return True
    if not text[run_end].isspace():
        return False
    next_index = run_end
    while next_index < paragraph_end and text[next_index].isspace():
        next_index += 1
    if next_index >= paragraph_end:
        return True
    following = text[next_index]
    if not (following.isupper() or following.isdigit() or following in OPENERS):
        return False
    if text[run_start:run_end] == ".":
        word = _word_before(text, run_start, floor)
        if word.lower().strip("'’") in ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha() and word.isupper():
            return False
        if word.isdigit():
            return False
    return True


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sentence_start = start
    index = start
    while index < end:
        if text[index] in TERMINALS:
            run_end = index
            while run_end < end and text[run_end] in TERMINALS:
                run_end += 1
            while run_end < end and text[run_end] in "\"'”’»)]":
                run_end += 1
            if _is_sentence_end(text, index, run_end, end, sentence_start):
                spans.append(_strip_span(text, sentence_start, run_end))
                sentence_start = run_end
            index = run_end
            continue
        index += 1
    if text[sentence_start:end].strip():
        spans.append(_strip_span(text, sentence_start, end))
    return spans


def _dialogue_line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Inside a paragraph a newline stays a space, except before a dialogue turn
    (``Ayşe: ...``, ``— ...``, opening quote): script lines never merge into one segment."""
    blocks: list[tuple[int, int]] = []
    block_start = start
    position = text.find("\n", start, end)
    while position != -1:
        if starts_dialogue_line(text[position + 1:end].split("\n", 1)[0]):
            blocks.append(_strip_span(text, block_start, position))
            block_start = position + 1
        position = text.find("\n", position + 1, end)
    blocks.append(_strip_span(text, block_start, end))
    return [block for block in blocks if block[1] > block[0]]


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    while end - start > max_chars:
        split_at = start + find_split_index(text[start:end], max_chars)
        piece = _strip_span(text, start, split_at)
        if piece[1] > piece[0]:
            spans.append(piece)
        start, end = _strip_span(text, split_at, end)
    if end > start:
        spans.append((start, end))
    return spans


def segment_text(text: str, max_chars: int = 220) -> list[Segment]:
    """Split text into paragraph-aware sentence segments of at most ``max_chars`` source chars."""
    if max_chars < 20:
        raise ValueError("max_chars en az 20 olmali.")
    segments: list[Segment] = []
    for paragraph_index, (p_start, p_end) in enumerate(_paragraph_spans(text)):
        sentence_spans = [
            span
            for b_start, b_end in _dialogue_line_spans(text, p_start, p_end)
            for span in _sentence_spans(text, b_start, b_end)
        ]
        for s_start, s_end in sentence_spans:
            for start, end in _split_long(text, s_start, s_end, max_chars):
                segments.append(
                    Segment(
                        index=len(segments),
                        paragraph=paragraph_index,
                        start=start,
                        end=end,
                        text=collapse_whitespace(text[start:end]),
                    )
                )
    return segments
