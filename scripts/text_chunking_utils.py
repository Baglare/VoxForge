# -*- coding: utf-8 -*-
"""XTTS icin guvenli metin bolme yardimcilari."""

from __future__ import annotations

import re
from typing import Any


SENTENCE_BREAKS = ".?!"
SECONDARY_BREAKS = ";:"
COMMA_BREAKS = ","


def normalize_tts_text(text: str) -> str:
    """TTS'e vermeden once fazla bosluklari sade hale getirir."""
    return re.sub(r"\s+", " ", text).strip()


def find_split_index(text: str, max_chars: int) -> int:
    """Oncelikli ayiricilara gore max_chars siniri icinde en iyi bolme yerini bulur."""
    window = text[: max_chars + 1]

    for break_chars in (SENTENCE_BREAKS, SECONDARY_BREAKS, COMMA_BREAKS):
        split_at = max(window.rfind(char) for char in break_chars)
        if split_at > 0:
            return split_at + 1

    split_at = window.rfind(" ")
    if split_at > 0:
        return split_at

    # Uygun ayrac yoksa siniri asmayacak sert kesim yapilir.
    return max_chars


def split_text_for_tts(text: str, max_chars: int = 220) -> list[str]:
    """Uzun metni XTTS karakter sinirini asmayan parcalara boler."""
    if max_chars < 20:
        raise ValueError("max_chars en az 20 olmali.")

    remaining = normalize_tts_text(text)
    if not remaining:
        return []

    chunks: list[str] = []
    while len(remaining) > max_chars:
        split_index = find_split_index(remaining, max_chars)
        chunk = remaining[:split_index].strip()
        remaining = remaining[split_index:].strip()
        if chunk:
            chunks.append(chunk)

    if remaining:
        chunks.append(remaining)

    return chunks


def summarize_chunks(chunks: list[str]) -> dict[str, Any]:
    """Raporlarda kullanilacak kisa chunk ozeti uretir."""
    return {
        "chunking_used": len(chunks) > 1,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "index": index,
                "char_count": len(chunk),
                "text": chunk,
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
    }
