"""Fixed Turkish test set for the voice quality benchmark (VF-Q1).

Small on purpose (12 items, ~2-3 minutes of audio per engine) so a full run
fits a laptop GPU session. Every engine receives the same *spoken* text
(produced by the VoxForge normaliser), so differences come from the model,
not from number/date handling. ``TESTSET_VERSION`` goes into every manifest;
change it whenever an item changes so old and new results are never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass

from voxforge.text import normalize_text

TESTSET_VERSION = "tr-v1"


@dataclass(frozen=True)
class TestItem:
    item_id: str
    category: str
    text: str
    focus: str

    @property
    def spoken(self) -> str:
        return normalize_text(self.text).spoken


TEST_ITEMS: tuple[TestItem, ...] = (
    TestItem("t01", "kısa", "Evet.", "Çok kısa girdide uydurma/uzatma (metin dışı ses) riski."),
    TestItem("t02", "kısa", "Tamam, anladım.", "Kısa ifade, doğal bitiş."),
    TestItem(
        "t03", "türkçe sesler",
        "Ağaçların gölgesinde ılık bir rüzgâr esiyordu; çocuklar şarkı söylüyordu.",
        "ğ, ı, ş, ç, ö, ü ve uzatma işareti.",
    ),
    TestItem(
        "t04", "türkçe sesler",
        "Kışın ışıklı çarşıda dolaşırken şişeyi düşürdüm.",
        "Art arda ş/ı/ç; benzer heceler.",
    ),
    TestItem(
        "t05", "soru",
        "Yarın sabah erkenden yola çıkabilecek miyiz, yoksa biraz daha bekleyelim mi?",
        "Soru tonlaması, ayrı yazılan soru eki.",
    ),
    TestItem("t06", "ünlem", "Ne kadar güzel bir haber bu, inanamıyorum!", "Duygu ve ünlem tonlaması."),
    TestItem(
        "t07", "sayı ve tarih",
        "Toplantı 14:30'da başladı ve katılım %25 arttı; rapor 12.05.2024 tarihli.",
        "VF-004 okunuşu: saat, yüzde, tarih (model okunuş metnini alır).",
    ),
    TestItem(
        "t08", "para ve sayı",
        "Fiyatı 1.250 TL'den 999,90 TL'ye düştü.",
        "Ekli TL tutarı bilerek dönüştürülmez; model ham yazımı nasıl okuyor.",
    ),
    TestItem(
        "t09", "özel isim",
        "Eskişehir'den gelen Gökçe Yıldırım, Kahramanmaraş'taki Öğretmenevi'nde kaldı.",
        "Türkçe özel isimler ve kesme işaretli ekler.",
    ),
    TestItem(
        "t10", "yabancı kelime",
        "Yeni laptopumdaki uygulama, cloud servisine otomatik backup alıyor.",
        "Türkçe cümlede İngilizce kelimeler.",
    ),
    TestItem(
        "t11", "uzun",
        "Kasabanın kenarındaki eski değirmen, yıllardır kimsenin uğramadığı bir yer olmasına rağmen, "
        "her bahar yeniden boyanır, çevresine çiçekler dikilir ve köyün çocukları akşamüstü orada toplanıp "
        "büyüklerinden duydukları hikâyeleri birbirlerine anlatırdı.",
        "Uzun cümle: nefes, tutarlılık, kırpılma ve tekrar.",
    ),
    TestItem(
        "t12", "tekerleme",
        "Şu köşe yaz köşesi, şu köşe kış köşesi, ortada su şişesi.",
        "Tekrarlı hece; atlama ve tekrar riski.",
    ),
)


def items_by_id() -> dict[str, TestItem]:
    return {item.item_id: item for item in TEST_ITEMS}
