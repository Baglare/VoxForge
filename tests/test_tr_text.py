"""VF-004: table-driven Turkish segmentation and normalisation checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from voxforge.text import (
    LexiconEntry,
    apply_transformations,
    load_lexicon,
    normalize_text,
    number_to_words,
    prepare_segments,
    segment_text,
)

NUMBER_CASES = {
    0: "sıfır",
    7: "yedi",
    10: "on",
    100: "yüz",
    101: "yüz bir",
    1000: "bin",
    1001: "bin bir",
    2023: "iki bin yirmi üç",
    1_250_000: "bir milyon iki yüz elli bin",
    3_000_000_000: "üç milyar",
}

SPOKEN_CASES = [
    ("Kitap 250 sayfa.", "Kitap iki yüz elli sayfa."),
    ("Oran %25 arttı.", "Oran yüzde yirmi beş arttı."),
    ("Değer 3,05 oldu.", "Değer üç virgül sıfır beş oldu."),
    ("Nüfus 1.250.000 kişi.", "Nüfus bir milyon iki yüz elli bin kişi."),
    ("Toplantı 14:30'da.", "Toplantı on dört otuz'da."),
    ("Saat 09:05 idi.", "Saat dokuz sıfır beş idi."),
    ("Tarih 12.05.2024 oldu.", "Tarih on iki Mayıs iki bin yirmi dört oldu."),
    ("Fiyat 150,50 TL.", "Fiyat yüz elli lira elli kuruş."),
    ("Fiyat ₺75 idi.", "Fiyat yetmiş beş lira idi."),
    ("2023'te başladı.", "iki bin yirmi üç'te başladı."),
    # Belirsiz durumlar olduğu gibi kalır.
    ("Bu 3. madde.", "Bu 3. madde."),
    ("A4 kâğıt ve COVID-19 raporu.", "A4 kâğıt ve COVID-19 raporu."),
    ("Telefon 0532 123 45 67.", "Telefon 0532 123 45 67."),
    ("Ödeme 50 TL'ye çıktı.", "Ödeme 50 TL'ye çıktı."),
    ("Yaş aralığı 10-15.", "Yaş aralığı 10-15."),
    ("Değer 3.5 civarı.", "Değer 3.5 civarı."),
]

SEGMENT_CASES = [
    (
        "Dr. Ahmet geldi. Prof. Ayşe de geldi! Peki ya sen?",
        ["Dr. Ahmet geldi.", "Prof. Ayşe de geldi!", "Peki ya sen?"],
    ),
    ("A. Yılmaz, vb. konular... Sonra bitti.", ["A. Yılmaz, vb. konular...", "Sonra bitti."]),
    ("1. Dünya Savaşı uzun sürdü.", ["1. Dünya Savaşı uzun sürdü."]),
    ("Değer 3.5 idi. Sonra arttı.", ["Değer 3.5 idi.", "Sonra arttı."]),
    ("Satır sonu\nbölmez. Yeni cümle.", ["Satır sonu bölmez.", "Yeni cümle."]),
    ("— Geldin mi? dedi. “Evet.” Bitti.", ["— Geldin mi? dedi.", "“Evet.”", "Bitti."]),
]


class TurkishTextTests(unittest.TestCase):
    def test_number_words(self) -> None:
        for value, expected in NUMBER_CASES.items():
            with self.subTest(value=value):
                self.assertEqual(number_to_words(value), expected)

    def test_spoken_text_cases(self) -> None:
        for original, expected in SPOKEN_CASES:
            with self.subTest(original=original):
                result = normalize_text(original)
                self.assertEqual(result.spoken, expected)
                self.assertEqual(result.original, original)

    def test_transformations_can_be_reverted(self) -> None:
        text = "Saat 14:30'da %25 indirim."
        result = normalize_text(text)
        self.assertEqual([t.rule for t in result.transformations], ["time", "percent"])
        for item in result.transformations:
            self.assertEqual(text[item.start:item.end], item.original)
        self.assertEqual(apply_transformations(text, result.transformations, disabled=[0, 1]), text)
        self.assertEqual(
            apply_transformations(text, result.transformations, disabled=[1]),
            "Saat on dört otuz'da %25 indirim.",
        )

    def test_lexicon_from_json_wins_over_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lexicon.json"
            path.write_text(
                json.dumps({"entries": [
                    {"word": "VoxForge", "spoken": "voks forc", "case_sensitive": True},
                    {"word": "F1", "spoken": "formula bir"},
                ]}),
                encoding="utf-8",
            )
            entries = load_lexicon(path)
        result = normalize_text("VoxForge ile f1 yarışı, voxforge değil.", entries)
        self.assertEqual(result.spoken, "voks forc ile formula bir yarışı, voxforge değil.")

    def test_segmentation_cases(self) -> None:
        for text, expected in SEGMENT_CASES:
            with self.subTest(text=text):
                self.assertEqual([s.text for s in segment_text(text)], expected)

    def test_paragraphs_and_offsets_are_preserved(self) -> None:
        text = "Birinci paragraf.  İkinci cümle.\n\n  Yeni paragraf burada."
        segments = segment_text(text)
        self.assertEqual([s.paragraph for s in segments], [0, 0, 1])
        for segment in segments:
            self.assertEqual(" ".join(text[segment.start:segment.end].split()), segment.text)

    def test_long_sentence_is_split_within_limit(self) -> None:
        text = ("Bu çok uzun bir cümle, virgüllerle devam ediyor, " * 8).strip() + "."
        segments = segment_text(text, max_chars=80)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(segment.end - segment.start <= 80 for segment in segments))

    def test_prepare_segments_keeps_original_and_respects_limit(self) -> None:
        text = "Rapor 12.05.2024 tarihli. " + "Sayı 1.250.000 oldu. " * 3
        prepared = prepare_segments(text, max_chars=40, lexicon=[LexiconEntry("Rapor", "ra por")])
        self.assertEqual(prepared[0].segment.text, "Rapor 12.05.2024 tarihli.")
        self.assertTrue(prepared[0].spoken.startswith("ra por on iki Mayıs"))
        for item in prepared:
            self.assertTrue(all(len(chunk) <= 40 for chunk in item.spoken_chunks))


if __name__ == "__main__":
    unittest.main()
