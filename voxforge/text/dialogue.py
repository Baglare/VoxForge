"""Dialogue hints for multi-voice narration (VF-007).

Only *suggestions*: the user confirms every assignment. Two patterns:
- script style ``Ayşe: Merhaba.`` -> speaker ``Ayşe`` (label can be dropped from the spoken text);
- prose dialogue starting with a dash or quote -> dialogue, speaker unknown.
Common label words such as ``Not:`` or ``Soru:`` are not treated as speakers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UPPER = "A-ZÇĞİÖŞÜ"
_WORD = rf"[{_UPPER}][\wçğıöşüÇĞİÖŞÜ'’.-]*"
SPEAKER_LABEL = re.compile(rf"^\s*(?P<name>{_WORD}(?:\s+{_WORD}){{0,2}})\s*:\s+(?P<line>\S.*)$", re.S)
DIALOGUE_MARKS = "—–-\"“«"
NOT_SPEAKERS = frozenset(
    {
        "not", "notlar", "örnek", "uyarı", "dikkat", "soru", "cevap", "yanıt", "bölüm", "kaynak",
        "sonuç", "tarih", "saat", "konu", "özet", "başlık", "adres", "telefon", "tel", "ek", "madde",
    }
)


@dataclass(frozen=True)
class DialogueHint:
    kind: str  # "named" | "dialogue"
    speaker: str | None
    line: str  # text without the speaker label


def speaker_label(text: str) -> tuple[str, str] | None:
    match = SPEAKER_LABEL.match(text)
    if not match:
        return None
    name = match.group("name").strip()
    if name.lower() in NOT_SPEAKERS:
        return None
    return name, match.group("line").strip()


def dialogue_hint(text: str) -> DialogueHint | None:
    labelled = speaker_label(text)
    if labelled:
        return DialogueHint("named", labelled[0], labelled[1])
    stripped = text.lstrip()
    if stripped and stripped[0] in DIALOGUE_MARKS:
        return DialogueHint("dialogue", None, stripped)
    return None


def starts_dialogue_line(text: str) -> bool:
    """True when a line opens a new dialogue turn (used to split script-style lines)."""
    stripped = text.lstrip()
    return bool(stripped) and (stripped[0] in "—–\"“«" or speaker_label(stripped) is not None)
