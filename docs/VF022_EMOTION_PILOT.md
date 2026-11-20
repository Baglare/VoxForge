# VF-022 · Türkçe ve İngilizce duygu yönergeleri

[Issue #18](https://github.com/Baglare/VoxForge/issues/18) · Pilot · 20 Kasım 2026

## Deney

Üç Türkçe replik seçildi. Her replik bir kez stilsiz, bir kez Türkçe duygu yönergesiyle, bir kez de aynı yönergenin İngilizce anlam eşdeğeriyle üretildi. Hedefler sevinç, kaygı ve üzüntü. Replik ve ayarlar her üçlü içinde aynı. VoxCPM2 revizyonu `32279effe8c19989596f05d353d1447f51d9e915`, seed `1234`, CFG `2.0`, 10 difüzyon adımı, tek parça, kırpma yok. Her klip aynı kayıtlı WAV’la, yalnız referanslı *controllable cloning* modunda üretildi. Ultimate Cloning referans transkriptiyle çalışır ve stil yönergesi kabul etmez.

Kör ZIP dokuz numaralı WAV, kimlik referansı ve puanlama formu içerir. Üç koşulun her biri bir kez birinci, bir kez ikinci, bir kez üçüncü sırada dinlenir. Hedef duygu ve koşul eşlemesi ayrı anahtardadır. Formda algılanan duygu ve belirginliği, doğallık, referans kimliğine benzerlik ve “başka biri gibi” işareti ayrı alanlardır. Kimlik değişimi duyulan varyant, belirginlik artsa da iyileşme sayılmaz.

## Ölçümler

Türkçe faster-whisper large-v3 ile karakter hata oranı (CER) ve edit sayıları ölçülür. CER, ASR’nin metne uyum ölçüsüdür. Telaffuz nüansları, vurgu ve doğallık kör dinlemeyle değerlendirilir. Ritim raporu klip süresi, genel ve artikülasyon hızı, iç duraklamalar ve F0 aralığını verir. Duraklama eşiği, zirve pencerenin 40 dB altında en az 120 ms’dir. Hızlar beklenen Türkçe metnin hece sayısı üzerinden hesaplanır. F0 aralığı p10–p90 yarım ton farkıdır.

ECAPA kosinüsü her klip için iki ayrı karşılaştırma verir. İlki kayıtlı kimlik referansına, ikincisi aynı repliğin stilsiz okumasına göre hesaplanır. Stilsiz satırlardaki ikinci değer bu nedenle 1,000’dır. Kimlik için evrensel bir ECAPA kabul eşiği kullanılmaz. Duygu aktarımı ile doğallık için dinleyici puanı gerekir.

### Pilot sonucu

| Hedef | Koşul | CER | ECAPA referans | ECAPA stilsiz | Süre (sn) | Hız / artik, (hece/sn) | Duraklama (adet / sn) | F0 (yarım ton) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sevinç | Stilsiz | 0,00% | 0,6792 | 1,0000 | 3,36 | 6,97 / 8,04 | 1 / 0,28 | 5,78 |
| Sevinç | Türkçe | 0,00% | 0,6474 | 0,9071 | 3,36 | 6,97 / 7,42 | 1 / 0,18 | 5,68 |
| Sevinç | İngilizce | 0,00% | 0,6330 | 0,7664 | 3,84 | 6,05 / 7,37 | 2 / 0,56 | 7,08 |
| Kaygı | Stilsiz | 13,21% | 0,5381 | 1,0000 | 2,56 | 8,47 / 8,97 | 0 / 0,00 | 15,60 |
| Kaygı | Türkçe | 0,00% | 0,6614 | 0,6451 | 2,72 | 8,02 / 8,61 | 0 / 0,00 | 8,56 |
| Kaygı | İngilizce | 0,00% | 0,4872 | 0,6182 | 2,56 | 8,47 / 8,54 | 0 / 0,00 | 9,29 |
| Üzüntü | Stilsiz | 0,00% | 0,7131 | 1,0000 | 3,52 | 5,92 / 6,94 | 1 / 0,34 | 7,45 |
| Üzüntü | Türkçe | 0,00% | 0,6595 | 0,8066 | 3,36 | 6,10 / 7,14 | 1 / 0,32 | 3,22 |
| Üzüntü | İngilizce | 0,00% | 0,5297 | 0,6703 | 3,36 | 6,21 / 7,81 | 1 / 0,40 | 4,55 |

- **Ses kimliği.** Türkçe yönerge, üç eşleşmenin tamamında İngilizceye göre hem kayıtlı referansa hem stilsiz okumaya daha yakındı. Referansa ortalama kosinüs stilsizde 0,6435, Türkçede 0,6561, İngilizcede 0,5500. Türkçe koşul, sevinç ve üzüntüde stilsiz okumaya kıyasla referanstan uzaklaştı. İngilizce koşul üç replikte de uzaklaştı. Kaygının stilsiz okumasında ASR metin farkı bulunduğundan o okumaya karşı benzerlik söylenen içeriğin etkisini de taşıyabilir.
- **Telaffuz ve ritim.** Sekiz klipte normalize ASR metni hedefle eşleşti. Stilsiz kaygı klibinde Whisper başlangıçtaki sözcüğü yakalamadı ve sonraki sözcüğü farklı yazdı (%13,21 CER). Bunun kaynağı dinlemeyle kontrol edilmeli. İngilizce sevinç okuması stilsiz klibe göre 0,48 sn daha uzun ve toplam iç duraklaması 0,28 sn daha fazla. Diğer repliklerde hız ve F0 etkisi hedefe göre değişti.
- **Duygu ve doğallık.** Kör dinleyici puanı henüz yok. Ritim ve F0 değişimi duygu aktarımının kanıtı sayılmaz. Dinleme kimlik değişimi gösterirse varyant elenir. Bu pilot hiçbir stil koşulunu Studio varsayılanı olarak benimsemiyor.

RTX 4060 Laptop GPU üzerinde dokuz adet 48 kHz WAV üretildi. Zirve PyTorch CUDA ayırımı 6.307,7 MiB. Yerel önbellekler kullanıldı. ZIP bütünlüğü, WAV ve kimlik referansı hash'leri ve sıra dengesi doğrulandı. Hedefli Voice Playground testleri (13) ve Ruff geçti.

## Yeniden üretim

Depo kökünde mevcut ortamları ve yerel model önbelleğini kullan:

```powershell
py -3 scripts/benchmark/vf022_emotion_probe.py synth --out outputs/vf022-yeni --reference <mevcut-yerel-wav>
& 'C:/AvenoxData/Environments/VoxForge-bench-eval/Scripts/python.exe' scripts/benchmark/vf022_emotion_probe.py measure --out outputs/vf022-yeni
```

Üretim için boşta bir CUDA GPU ve yeni bir çıktı dizini kullan. ZIP dinleyiciye ayrılır. `anahtar.json` ve `olcumler.json` değerlendirmeciye ayrılır. WAV’lar ve ham sonuçlar yerel Git dışı çıktı dizininde kalır.
