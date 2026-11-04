"""Conservative Turkish spoken-text normalisation (VF-004).

The original text is never modified. ``normalize_text`` returns the spoken text
plus a list of ``Transformation`` records (span, original, spoken, rule) so a UI
can show every change and revert any of them with ``apply_transformations``.

Only unambiguous patterns are rewritten; anything doubtful is left as written:
- integers / Turkish decimals (``3,5``) / dot thousands (``1.250.000``)
- percent ``%25``, time ``14:30``, date ``12.05.2024`` / ``12/05/2024``
- money ``₺150`` / ``150 TL`` / ``150,50 TL``
- user pronunciation lexicon entries (applied first)
Left untouched: ordinal-looking ``3. madde``, numbers glued to letters
(``A4``, ``COVID-19``, ``3kg``), leading-zero / phone-like digit groups,
negative numbers, Roman numerals, money with a suffix (``50 TL'ye``).
A suffix after an apostrophe is kept as written (``2023'te`` -> ``iki bin yirmi üç'te``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NORMALIZER_VERSION = "tr-1"

ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
SCALES = [(10**12, "trilyon"), (10**9, "milyar"), (10**6, "milyon"), (10**3, "bin")]
MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]
MAX_NUMBER = 10**15 - 1

_NUM = r"\d{1,3}(?:\.\d{3})+|\d+"
_SUFFIX = re.compile(r"['’][^\W\d_]+")
_DATE = re.compile(r"(?<![\w.,/])(\d{1,2})([./])(\d{1,2})\2(\d{4})(?![\w])(?![.,/]\d)")
_TIME = re.compile(r"(?<![\w.,:])([01]?\d|2[0-3]):([0-5]\d)(?![\w:])(?![.,]\d)")
_MONEY_PREFIX = re.compile(rf"₺\s?({_NUM})(?:,(\d{{1,2}}))?(?![\w])(?![.,]\d)")
_MONEY_SUFFIX = re.compile(rf"(?<![\w.,])({_NUM})(?:,(\d{{1,2}}))?\s?(?:TL|₺)(?![\w])")
_PERCENT = re.compile(rf"%\s?({_NUM})(?:,(\d+))?(?![\w])(?![.,]\d)")
_NUMBER = re.compile(rf"(?<![\w.,:/%₺-])({_NUM})(?:,(\d+))?(?![\w])(?![.,:/]\d)")
_PHONE_NEIGHBOUR_BEFORE = re.compile(r"\d[ -]$")
_PHONE_NEIGHBOUR_AFTER = re.compile(r"^[ -]\d")


@dataclass(frozen=True)
class Transformation:
    start: int
    end: int
    original: str
    spoken: str
    rule: str


@dataclass(frozen=True)
class NormalizationResult:
    original: str
    spoken: str
    transformations: tuple[Transformation, ...]
    version: str = NORMALIZER_VERSION


@dataclass(frozen=True)
class LexiconEntry:
    word: str
    spoken: str
    case_sensitive: bool = False


def _three_digits(value: int) -> list[str]:
    hundreds, rest = divmod(value, 100)
    tens, ones = divmod(rest, 10)
    words: list[str] = []
    if hundreds:
        if hundreds > 1:
            words.append(ONES[hundreds])
        words.append("yüz")
    if tens:
        words.append(TENS[tens])
    if ones:
        words.append(ONES[ones])
    return words


def number_to_words(value: int) -> str:
    """Turkish cardinal words for 0 <= value < 10**15 (``1000`` -> ``bin``)."""
    if value < 0 or value > MAX_NUMBER:
        raise ValueError(f"Desteklenmeyen sayi: {value}")
    if value == 0:
        return "sıfır"
    words: list[str] = []
    for scale, name in SCALES:
        count, value = divmod(value, scale)
        if count:
            if not (name == "bin" and count == 1):
                words.extend(_three_digits(count))
            words.append(name)
    words.extend(_three_digits(value))
    return " ".join(words)


def _digits_after_comma(digits: str) -> str:
    stripped = digits.lstrip("0")
    words = ["sıfır"] * (len(digits) - len(stripped))
    if stripped:
        words.append(number_to_words(int(stripped)))
    return " ".join(words)


def _integer(text: str) -> int | None:
    plain = text.replace(".", "")
    if len(plain) > 1 and plain.startswith("0"):
        return None
    value = int(plain)
    return value if value <= MAX_NUMBER else None


def _number_words(integer_text: str, fraction: str | None) -> str | None:
    value = _integer(integer_text)
    if value is None:
        return None
    words = number_to_words(value)
    if fraction:
        words += " virgül " + _digits_after_comma(fraction)
    return words


def load_lexicon(path: Path) -> list[LexiconEntry]:
    """JSON: ``{"entries": [{"word": "VoxForge", "spoken": "voks forc", "case_sensitive": true}]}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        LexiconEntry(
            word=str(item["word"]),
            spoken=str(item["spoken"]),
            case_sensitive=bool(item.get("case_sensitive", False)),
        )
        for item in data.get("entries", [])
        if str(item.get("word", "")).strip()
    ]


class _Collector:
    def __init__(self, text: str) -> None:
        self.text = text
        self.items: list[Transformation] = []

    def free(self, start: int, end: int) -> bool:
        return all(end <= item.start or start >= item.end for item in self.items)

    def add(self, start: int, end: int, spoken: str, rule: str, keep_suffix: bool = True) -> None:
        if keep_suffix:
            suffix = _SUFFIX.match(self.text, end)
            if suffix:
                spoken += suffix.group(0)
                end = suffix.end()
        if self.free(start, end):
            self.items.append(Transformation(start, end, self.text[start:end], spoken, rule))


def _apply_lexicon(collector: _Collector, entries: Iterable[LexiconEntry]) -> None:
    for entry in sorted(entries, key=lambda item: len(item.word), reverse=True):
        flags = 0 if entry.case_sensitive else re.IGNORECASE
        pattern = re.compile(rf"(?<!\w){re.escape(entry.word)}(?!\w)", flags)
        for match in pattern.finditer(collector.text):
            collector.add(match.start(), match.end(), entry.spoken, "lexicon", keep_suffix=False)


def normalize_text(text: str, lexicon: Iterable[LexiconEntry] = ()) -> NormalizationResult:
    collector = _Collector(text)
    _apply_lexicon(collector, lexicon)

    for match in _DATE.finditer(text):
        day, month, year = int(match.group(1)), int(match.group(3)), int(match.group(4))
        if 1 <= day <= 31 and 1 <= month <= 12:
            spoken = f"{number_to_words(day)} {MONTHS[month - 1]} {number_to_words(year)}"
            collector.add(match.start(), match.end(), spoken, "date")

    for match in _TIME.finditer(text):
        hour, minute = int(match.group(1)), int(match.group(2))
        spoken = number_to_words(hour)
        if minute:
            spoken += " " + (number_to_words(minute) if minute >= 10 else "sıfır " + ONES[minute])
        collector.add(match.start(), match.end(), spoken, "time")

    for pattern in (_MONEY_PREFIX, _MONEY_SUFFIX):
        for match in pattern.finditer(text):
            if _SUFFIX.match(text, match.end()):
                continue  # "50 TL'ye": ek "TL" okunuşuna göre yazılmış, dokunma
            lira = _integer(match.group(1))
            if lira is None:
                continue
            spoken = f"{number_to_words(lira)} lira"
            kurus = match.group(2)
            if kurus:
                kurus_value = int(kurus.ljust(2, "0"))
                if kurus_value:
                    spoken += f" {number_to_words(kurus_value)} kuruş"
            collector.add(match.start(), match.end(), spoken, "money", keep_suffix=False)

    for match in _PERCENT.finditer(text):
        words = _number_words(match.group(1), match.group(2))
        if words:
            collector.add(match.start(), match.end(), "yüzde " + words, "percent")

    for match in _NUMBER.finditer(text):
        start, end = match.start(), match.end()
        if _PHONE_NEIGHBOUR_BEFORE.search(text[:start]) or _PHONE_NEIGHBOUR_AFTER.match(text[end:]):
            continue
        if not match.group(2) and re.match(r"\.\s+[a-zçğıöşü]", text[end:]):
            continue  # "3. madde": sıra sayısı olabilir
        if re.match(r"\s?(?:TL|₺)(?!\w)", text[end:]):
            continue  # para kuralının bilerek atladığı tutar ("50 TL'ye")
        words = _number_words(match.group(1), match.group(2))
        if words:
            collector.add(start, end, words, "number")

    transformations = tuple(sorted(collector.items, key=lambda item: item.start))
    return NormalizationResult(
        original=text,
        spoken=apply_transformations(text, transformations),
        transformations=transformations,
    )


def apply_transformations(
    text: str,
    transformations: Iterable[Transformation],
    disabled: Iterable[int] = (),
) -> str:
    """Rebuild spoken text; indices in ``disabled`` keep the original wording (undo)."""
    skip = set(disabled)
    pieces: list[str] = []
    position = 0
    for index, item in enumerate(sorted(transformations, key=lambda t: t.start)):
        pieces.append(text[position:item.start])
        pieces.append(item.original if index in skip else item.spoken)
        position = item.end
    pieces.append(text[position:])
    return "".join(pieces)
