# VF-017 — VoxCPM2 anlatım kalitesi ve ses tutarlılığı

Tarih: 2026-11-09 · RTX 4060 Laptop (8 GB) · VoxCPM2 (voxcpm 2.0.3) · profil `mimo` · CFG 2.0 · 10 adım · tohum 1234.
Model değişmedi; eğitim, ince ayar, model indirme ve geniş parametre taraması yok. Girdi: VF-016 kullanıcı
dinleme sonuçları (kısa metin iyi; yavaş anlatıcının kimliği normale tam benzemiyor; bazı cümle başları ani;
sahne anlatımında ritim/duygu/geçişler yetersiz).

Araçlar (yeni, yerel): `scripts/benchmark/vf017_narration_probe.py` (A/B/C + sahneler, ham cümle parçaları),
`vf017_context_probe.py` (bağlam hipotezi), `vf017_scene_check.py` (gerçek sahne anlatıcısıyla uçtan uca),
`vf017_analyze.py` (eval ortamı: faster-whisper large-v3 CER + kelime zamanları, ECAPA, ritim, ham parça başlangıçları),
`vf017_report.py` (yerel dinleme sayfası). Ham ölçümler ve sesler git dışı: `outputs/vf017/`.
**Dinleme sayfası: `outputs/vf017/DINLEME.html`** (34 etiketli örnek, ölçümler yanında).

ECAPA = referansa (veya aynı metnin varsayılan A çıktısına) kosinüs benzerliği. Kısa kliplerde gürültülüdür ve
metne bağlıdır; yalnız **aynı metin içinde** karşılaştırıldı. Otomatik ölçüm doğallığın kanıtı değildir.

## 1. Kontrollü karşılaştırma (hikâye, 6 cümle)

| | hece/sn | artikülasyon | duraklama | ilk-2/kalan | ECAPA→referans | ECAPA→A | CER |
|---|---|---|---|---|---|---|---|
| A varsayılan (paket 250) | 6,93 | 8,11 | 2,2 sn | 1,09 | 0,824 | — | 0 |
| B segmentasyon + duraklama, stil kapalı | 6,58 | 8,14 | 3,3 sn | 1,02 | **0,837** | **0,933** | 0,003 |
| C = B + "yavaş, sakin bir anlatım" | 6,38 | 7,90 | 3,4 sn | 0,97 | 0,832 | 0,908 | 0,003 |

- **Tempo iyileşmesinin çoğu segmentasyon + duraklamadan geliyor** (A→B: −0,35 hece/sn; artikülasyon aynı, yani
  model daha yavaş konuşmuyor, cümleler arası nefes alanı açılıyor). Stil talimatı (B→C) küçük ek yavaşlama verir
  (−0,20 hece/sn, artikülasyon 8,14 → 7,90).
- **Ses kimliğini değiştiren stil talimatıdır**: C, normal üretime B'den belirgin uzak (0,908 vs 0,933). Kullanıcının
  "yavaş anlatıcı normale benzemiyor" gözlemiyle tutarlı.
- Kısa metin ve hikâye varsayılanı VF-016 kayıtlarıyla **bayt bayt aynı** (sha256 eşit); varsayılan değişmedi.

## 2. Sahne teşhisi (kullanıcının 4 sahnesi)

Preset'li çıktılar kullanıcının VF-016'da dinlediği dosyalarla bayt bayt aynı yeniden üretildi; yani ölçülen,
kullanıcının duyduğu sestir.

| sahne | preset (stilli) ECAPA→ref / →A | aynı duraklama stilsiz | varsayılan A | not |
|---|---|---|---|---|
| sakin | 0,724 / 0,859 | **0,784 / 0,912** | 0,759 | stil tınıyı değiştirdi (enerji dağılımı 1,5 → 4,0 kHz: nefesli) |
| gerilim | 0,798 / 0,859 | 0,805 / 0,882 | 0,791 | stil tempoyu belirgin düşürdü (7,37 → 6,34) ama tınıyı kaydırdı |
| aksiyon | (stil yok) 0,753 / 0,929 | — | 0,752 | "Birden" A'da da yanlış (ASR "Böre'den"): ham model hatası |
| duygusal | 0,768 / 0,887 | **0,804 / 0,929** | 0,814 | tempo aynı; ölçülebilir tek etki kimlik kayması |

- **Ham WAV ile birleşik çıktı**: her preset'li sahne ve B/C için cümleler ayrıca ham (kırpmasız) üretildi; ham
  parçalar Playground birleştiricisiyle birleştirilince worker çıktısıyla **bayt bayt aynı** (9/9). Yani sahne
  içindeki sorunlar ya ham model çıktısında ya da birleştirmede; ikisi ayrı ölçülebilir.
- **Kırpma konuşma kesmiyor**: duyulabilir sinyal (gürültü tabanı +10 dB) hiçbir parçada −45 dB eşiği + 40 ms payın
  dışında kalmadı (baş/son kesilen 0 ms, 26 parça).
- **Ani cümle başları ham model çıktısında**: VoxCPM2 her cümleyi 0-30 ms içinde, ön nefes/geçiş olmadan konuşarak
  başlatıyor (kısa cümlelerde 0 ms). Birleştirme bunu yaratmıyor, ama düzeltemiyor da. Cümleler arası seviye farkı
  küçük (stilsiz ≤1,4 dB, stilli ≤2,5 dB).
- **Sahne geçişleri birleştirmede bozuluyordu (seviye)**: VoxCPM2 parçaları ~0 dBFS'e normalize (tepe 0,98-0,995).
  `apply_gain`'in tepe koruması her pozitif kazancı iptal ediyordu; sakin sahne diğerlerinden ~2,8 dB sessiz kaldı
  ama arayüz "+2,5 dB" gösterdi (VF-016 raporundaki seviye düzeltmeleri de uygulanmış değil, istenmiş değerlerdi).
- **Aynı tohum ses kimliğini garanti etmiyor**: aynı tohum ve referansla varsayılan çıktının ECAPA'sı sahne metnine göre
  0,75-0,81 arasında; aynı istek bile süreç durumuna bağlı: yeni süreçteki ilk sahne, önceki süreçteki aynı istekle bayt
  olarak farklı çıktı (diğer 3 sahne aynı). Kimlik tutarlılığı için kalan tek kaldıraç stil talimatını kullanmamaktır.
- **Duygu kontrolü**: stilli preset'lerin otomatik ölçülebilen etkileri yavaşlama + kimlik/tını kayması; duygu aktarımı
  kullanıcı dinlemesinde yetersiz. Bu yüzden başarılı gösterilmiyor (aşağıda "deneysel").

## 3. Uygulanan düzeltmeler (en küçük)

1. **Seviye eşleme** (`voxforge/playground/audio_join.py`): kazançlar tümüne ortak bir miktar düşülerek tepe sınırına
   sığdırılır, böylece farklar gerçekten kapanır; raporlanan kazanç uygulanan kazançtır. Yalnız sahne birleştirmeyi
   etkiler (anlatım/kısa metin birleşiminde seviye eşleme yok → bayt eşitliği korunur). Kullanıcı sahneleri:
   −17,7/−16,1/−15,2/−14,9 dB → hepsi −17,9 dB. Regresyon testi eski kodda FAIL, yenide PASS.
2. **Stilsiz varsayılanlar** (`prosody.py`, `scenes.py`): "Yavaş anlatıcı" = B (stil yok, 0,55/0,95 sn). Eski stilli
   sürüm "Yavaş anlatıcı + stil (deneysel)". Sakin/Dramatik/Gerilim/Duygusal stilli preset'ler ad ve kanıt metninde
   "deneysel". Sahne türleri stilsiz preset seçer; gerilim/dramatik ve duygusal için yalnız duraklaması uzun yeni
   preset'ler ("Gerilim (stilsiz)", "Duygusal (stilsiz)") — ton üretmez, sesi değiştirmez. Mevcut projeler kayıtlı
   preset'lerini korur; preset değerleri değişen sahneleri parmak izi "ayar değişti" olarak işaretler.
3. "Normal" preset ve varsayılan istek değişmedi.

Uçtan uca doğrulama (gerçek `SceneNarrator`, yeni proje, kullanıcının sahne metinleri, GPU):

| | sakin | gerilim | aksiyon | duygusal | birleşik ECAPA | birleşik kazançlar |
|---|---|---|---|---|---|---|
| eski (VF-016) | 0,724 | 0,798 | 0,753 | 0,768 | 0,830 | (gösterilen +2,5…; uygulanmadı) |
| yeni (VF-017) | **0,784** | **0,808** | 0,753 | **0,804** | **0,843** | +0,1/−0,7/−2,1/−1,3 dB (uygulandı) |

Bilinen ödünler: stilsiz gerilim sahnesi daha hızlı (7,19 vs 6,34 hece/sn; yavaşlatan stildi) ve bu tohumda bir
kelime bozuk ("Derin" → ASR "Derenin", CER 0,035; stilli sürümde 0). Sahne tek tohumla üretildiği için kullanıcı
dinleyip gerekirse o sahnede deneysel preset'e dönebilir.

## 4. Denenen ve reddedilen

- **Önceki cümleyi prompt devamı olarak vermek** (ani başlangıçlar bağlamsızlıktan mı?): aynı süreçte bağlamsız ile
  karşılaştırıldı. ECAPA→referans 4/4 örnekte düştü (hikâye 0,838 → 0,801, sakin 0,791 → 0,778, aksiyon 0,740 → **0,636**,
  duygusal 0,809 → 0,788), CER biraz arttı. Üretilmiş sesle koşullama kimliği kaydırıyor; ürüne eklenmedi.
  Örnekler dinleme sayfasında (bölüm 4).
- F0 tabanlı "cümle başı perde" ölçümü: basit otokorelasyon izleyicisi cümle başlarında çoğunlukla değer üretemedi;
  güvenilir olmadığı için karar için kullanılmadı.

## 5. Referans kalitesi (mimo; orijinal değiştirilmedi, ön işleme denenmedi)

| | orijinal | ön işlenmiş (modele giden) |
|---|---|---|
| biçim | **Ogg Opus, 16 kHz** (0,07 MB; sesli not sıkıştırması) | PCM 24 kHz |
| süre / baş-son sessizlik | 31,7 sn / 0,18-0,35 sn | 31,7 sn / 0,16-0,35 sn |
| tepe / kırpılmış örnek | 0,0 dBFS / 2 örnek | −2,0 dBFS / 0 |
| konuşma medyanı / gürültü tabanı | −20,7 / −48,4 dB (~28 dB SNR) | −22,3 / −50,0 dB |
| bant genişliği (%99,9 enerji) | ~6,1 kHz (7 kHz üstü %0,02) | ~6,0 kHz |
| 5 sn pencerelerde seviye | −23,6 … −17,7 dB (~6 dB oynama) | −25,8 … −19,4 dB |
| duraklama | 8 adet, toplam 2,2 sn, en uzun 0,54 sn | aynı |

Değerlendirme: uzunluk ve SNR yeterli, kırpma ihmal edilebilir. **Asıl sınır sıkıştırma ve bant genişliği**: referans
16 kHz Opus sesli not, ~6 kHz üstü yok; model 48 kHz üretse de tınıyı bu dar banttan öğreniyor. Kayıt boyunca ~6 dB
seviye oynaması (mikrofon mesafesi) ve okuma tonu (sakin, sabit, 6,3 hece/sn) da üretimin ifade aralığını sınırlar:
duygu/gerilim için referansta örnek yok. Orijinal referansa dokunulmadı. Öneri (kullanıcı kararı): sıkıştırmasız
(WAV, ≥24 kHz) 30-60 sn yeni kayıt; ön işleme (gürültü azaltma, seviye düzleme) yalnız ayrı profil kopyasıyla,
geri alınabilir deney olarak — bu turda yapılmadı.

## Doğrulama

- Testler: `tests/test_vf016_prosody.py` (17, yeni seviye ve preset politikası testleri dahil) + `tests/test_voice_playground.py` PASS; ruff PASS.
- GPU: 15 üretim + 26 ham cümle parçası (probe), 8 bağlam denemesi, 4 sahne uçtan uca. VRAM tepe 5,55-6,31 GB (worker),
  bağlam denemesi 6,67 GB (prompt önbelleksiz) — 8 GB sınırı içinde.
- Çalıştırılmayan: kullanıcı kör dinlemesi (asıl kabul), Gradio arayüzünde görsel kontrol, XTTS/Chatterbox (kod değişmedi),
  referans ön işleme deneyi, ikinci tohum taraması.
