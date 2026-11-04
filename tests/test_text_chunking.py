from __future__ import annotations

import unittest

from scripts.text_chunking_utils import split_text_for_tts, summarize_chunks


class TextChunkingTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(split_text_for_tts("   \n "), [])

    def test_short_text(self):
        self.assertEqual(split_text_for_tts("  Merhaba   dünya! "), ["Merhaba dünya!"])

    def test_all_chunks_respect_limit(self):
        text = ("İzmir'de yağmur yağıyor. Uzun bir cümle söylüyoruz! " * 12).strip()
        chunks = split_text_for_tts(text, max_chars=100)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(0 < len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text)

    def test_long_single_token_fallback(self):
        chunks = split_text_for_tts("ü" * 121, max_chars=40)
        self.assertEqual([len(chunk) for chunk in chunks], [40, 40, 40, 1])

    def test_invalid_max_length(self):
        with self.assertRaises(ValueError):
            split_text_for_tts("deneme", max_chars=19)

    def test_summary(self):
        result = summarize_chunks(["ilk", "ikinci"])
        self.assertTrue(result["chunking_used"])
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(result["chunks"][1]["char_count"], 6)


if __name__ == "__main__":
    unittest.main()
