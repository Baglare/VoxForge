# VF-016 — VoxCPM2 Prosody Studio, sahne anlatımı ve çalışma alanı düzeni

Tarih: 2026-11-07 · Donanım: RTX 4060 Laptop (8 GB) · VoxCPM2 (voxcpm 2.0.3, torch 2.15.0+cu134)
Profil: `mimo` (30 sn referans) · tohum 1234 (stil denemelerinde ek olarak 77).
Model eğitimi, ince ayar ve yeni model indirmesi yok. Ölçüm araçları yeni ve yerel:

- `voxforge/playground/rhythm.py`: cümle başına **hece/sn** (Türkçe'de hece = ünlü sayısı; sözlük gerekmez),
  artikülasyon hızı (yalnız sesli kareler), cümle arası duraklama, baş/son sessizlik.
- `scripts/benchmark/measure_rhythm.py` (eval ortamı): önbellekteki faster-whisper large-v3 ile kelime zaman
  damgaları + CER. `scripts/benchmark/voxcpm_rhythm_probe.py`: tek değişkenli küçük GPU örneği.
- Konuşmacı benzerliği: VF-Q1 ile aynı SpeechBrain ECAPA (referansa kosinüs). Kısa kliplerde ECAPA gürültülüdür;
  yalnız aynı metin içi karşılaştırmada kullanıldı.

Ham ölçümler (git dışı): `outputs/vf016/{probe,probe2,verify,style_alt}/rhythm.json` ve `manifest.json`.

## Aşama 1 — Uzun anlatımda acele okuma

### Ölçüm (mevcut davranış, 250 karakterlik paketleme)

| | genel hece/sn | artikülasyon | cümle arası duraklama | ilk 2 cümle / kalan |
|---|---|---|---|---|
| Referans kaydı (mimo, okuma) | 6,26 | 7,08 | 0,00-0,48 sn | 0,98 |
| Hikâye metni (6 cümle) | 6,93 | 8,11 | **0,04-0,72** sn | **1,09** (1. cümle 8,46) |
| Günlük metin (5 cümle) | 7,05 | 8,01 | 0,08-0,36 sn | 0,90 |

### Neden

1. **Modelin bu referansla doğal temposu referanstan hızlı**: artikülasyon 8,0-8,1 hece/sn, referansta 7,1 (~%13).
2. **Paket içindeki cümle sınırlarında neredeyse duraklama yok** (0,02-0,4 sn; “Fakat o gece …” öncesi 0,04 sn).
   Dinleyicinin “acele” algısının asıl kaynağı budur: cümleler birbirine yapışıyor.
3. **İlk cümle etkisi metne bağlı**, sistematik değil: hikâyede uzun ve virgüllü açılış cümlesi en hızlı (8,46),
   günlük metinde açılış cümleleri kalanından yavaş (oran 0,90). Parça başı tohum yeniden uygulaması etkisiz:
   cümle başına üretimde (her cümlede aynı tohum) oranlar aynı kaldı (1,07 / 0,89).
4. **Metin hazırlığı neden değil**: VF-004 normalizasyonu noktalamayı korur, ASR CER 0-0,003; aynı metin ve tohumla
   eski kayıt bayt bayt yeniden üretildi (19,24 sn).

### Denenen düzeltmeler (tek değişken, tohum 1234)

| Varyant | hikâye hece/sn | günlük hece/sn | benzerlik (hikâye/günlük) | CER |
|---|---|---|---|---|
| paket 250 (eski) | 6,93 | 7,05 | 0,824 / 0,852 | 0 / 0,003 |
| cümle cümle, 0,2 sn ek | 7,17 | 6,97 | 0,854 / 0,864 | 0,003 / 0 |
| paket + `(slow, calm narration)` | 6,60 | 6,65 | 0,815 / 0,828 | 0 |
| paket + `(yavaş, sakin bir anlatım)` | **6,15** | — | 0,793 / — | 0,008 |
| **Yavaş anlatıcı preset'i** (Türkçe yönerge + cümle cümle + kırpılmış kenar + 0,55/0,95 sn) | **6,38** | **6,22** | **0,832 / 0,845** | 0,003 / 0 |
| Diyalog preset'i (yönergesiz, cümle cümle, 0,35 sn) | 6,93 | 6,89 | 0,848 / 0,861 | 0,003 / 0 |

Sonuç: cümle cümle üretim tek başına tempoyu düşürmez ama konuşmacı benzerliğini artırır; kırpılmış kenar +
açık duraklama cümle sınırını doğallaştırır; Türkçe yavaşlık yönergesi artikülasyonu düşürür. Üçü birlikte
(**Yavaş anlatıcı**) hikâyede ilk iki cümlenin acelesini giderdi (1,09 → 0,97), toplam cümle arası duraklama
~2 sn'den ~3,6 sn'ye çıktı, benzerlik eski düzeyde kaldı.

**Varsayılan değişmedi**: Normal preset aynı isteği gönderir; kısa metin (92 karakter, mimo, tohum 1234) yeni kodla
eski kayıtla **bayt bayt aynı** üretildi. Uzun VoxCPM2 metni standart ritimde üretilince kayıtta “Yavaş anlatıcı /
Anlatım ritmi” önerisi görünür.

## Aşama 2 — Gelişmiş kontroller

Kurulu kodda doğrulanan gerçek kontroller (`voxcpm/core.py`, model kartı):

| Kontrol | Aralık | Not |
|---|---|---|
| CFG (`cfg_value`) | 1,0-3,0 (varsayılan 2,0) | yalnız varsayılandan farklıysa gönderilir |
| Difüzyon adımı (`inference_timesteps`) | 4-20 (varsayılan 10) | < 8 uyarı: VF-015'te 6 adımda telaffuz bozuldu |
| Tohum | ≥ 0, -1 rastgele (varsayılan 1234) | süreç başı ilk üretim tekrarlanabilir (VF-015) |
| Stil yönergesi | ≤ 160 karakter | modele `(yönerge)metin` olarak verilir; **metinden ayrı alan** olarak taşınır, parantezler temizlenir |
| Anlatım ritmi | standart / anlatım | anlatım: VF-004 cümleleri, paragraf (boş satır) ve cümle duraklaması 0-2 sn |

Model **sayısal hız veya duygu parametresi sunmaz**; model kartı stil sonuçlarının üretimden üretime değiştiğini
belirtir. **Ultimate Cloning uyumluluğu**: CFG, adım, tohum ve anlatım ritmi uyumlu; stil yönergesi **uyumsuz**
(prompt devamı; kütüphane CLI'si de `--control` ile `--prompt-text`'i reddeder) ve servis açık hata verir.
“Hızlı (6 adım)” sabit karşılaştırma varyantıdır, panel uygulanmaz (sessizce yok sayılmaz, arayüzde yazılı).

Kullanıcı preset'leri `outputs/playground/presets.json` (git dışı); hazır preset'ler silinemez. “Preset
karşılaştırması” aynı metni aynı ses ve tohumla seçilen preset'lerle üretir ve yan yana dinleme yuvalarına koyar.

### Stil yönergeleri gerçekte ne yaptı (kısa metin, 28 hece)

| Preset | hece/sn | benzerlik | yorum |
|---|---|---|---|
| Normal | 7,49 | 0,737 | referans |
| Sakin | 6,09 | 0,737 | yavaşladı, kimlik aynı |
| Dramatik | 6,09 | 0,728 | yavaşladı; “dramatik” etki otomatik ölçülemez |
| Gerilim | 6,11 | 0,728 | yavaşladı; etki dinlenmeli |
| `hızlı, enerjik ve heyecanlı` | 6,36 | 0,703 | **hızlanmadı** |
| `duygusal, yumuşak ve hüzünlü` | 6,01 | 0,712 | yavaşladı |

Aksiyon sahnesinde (2 tohum) yönergesiz 0,657/0,611; “hızlı, enerjik” 0,475/0,689; “canlı ve enerjik”
0,433/0,684; “heyecanlı, gergin” 0,399/0,629 ve hiçbiri hızlandırmadı. Bu yüzden **Aksiyon preset'i yönergesizdir**
(yalnız kısa duraklama) ve bu açıkça yazılır. Türkçe'de denenen yönergelerin ortak, ölçülebilir etkisi
“yavaşla + duraklat”tır; duygu tonu dinleyerek doğrulanmalıdır. Her preset'in kanıt metni arayüzde görünür.

## Aşama 3 — Sahne anlatımı

- `voxforge/playground/scenes.py` (arayüzden bağımsız, Studio'ya taşınabilir JSON proje):
  `## Başlık`, `---`, `[sahne: Başlık]` işaretleri; işaret yoksa paragraflar. En fazla 40 sahne × 2000 karakter.
- Tür önerisi yerel anahtar kelimelerle (LLM/servis yok), tablodan düzenlenir. Türler kısayoldur ve yalnız
  varsayılan preset'i seçer: sakin→sakin, gerilim→gerilim, aksiyon→aksiyon, duygusal→duygusal, diyalog→diyalog,
  dramatik→dramatik.
- **Ses kimliği**: tüm sahneler aynı profil referansı, aynı VoxCPM2 varyantı ve aynı tohumla üretilir; yalnız
  preset değişir. Ultimate varyantında stil kullanılamaz (preset'in stili varsa sahne hata verir).
- Sahne başına üret / yeniden üret, dinle. Her sahne üretimde bir **parmak izi** saklar: metin, profil + referans
  dosyasının kimliği (yol, boyut, değiştirilme zamanı), varyant ve sabit ayarları, tohum, normalizasyon, preset'in
  gerçek kontrol değerleri, Ultimate transkripti ve cihaz. Biri değişince sahne “ayar değişti” olur, birleştirme
  engellenir ve “Değişen/eksik sahneleri üret” yalnız o sahneleri yeniden üretir. Parmak izi olmayan eski kayıtlar
  yüklenir ama doğrulanamadığı için yeniden üretilmesi gerekir. Yeniden üretimde yalnız o projenin eski sahne sesi silinir.
- Yeni proje mevcut bir projenin üzerine yazmaz: ad dosyası özel oluşturmayla ayrılır, ad (veya sesleri kalmış klasörü)
  kullanılıyorsa otomatik `-2`, `-3` … verilir ve kullanıcıya yazılır.
- Birleştirme (`audio_join.join_wavs`): kenar sessizliği −45 dB'de kırpılır (40 ms pay, 8 ms geçiş), sahneler arası
  sabit duraklama (varsayılan 0,8 sn), sesli kısımların RMS'i medyana ±6 dB içinde eşlenir, tepe koruması 0,97.
- Gerçek GPU denemesi (4 sahne): birleşik 18,85 sn, en uzun boşluk 0,96 sn, seviye düzeltmeleri +0,4/0/−2,3/+1,0 dB,
  ASR CER 0, birleşik benzerlik 0,776. Sahne bazında benzerlik: sakin 0,811, gerilim 0,765, duygusal 0,635 ve eski
  aksiyon yönergesiyle **0,475** (bu bulgu Aksiyon preset'ini yönergesiz yaptı).

## VRAM

Tüm GPU denemelerinde üretim tepesi 5,55-6,31 GB (8 GB sınırı içinde); referans önbelleği ve `empty_cache`
(VF-015) korunuyor. Sahne/preset işleri tek VoxCPM2 sürecinde art arda çalışır, model bir kez yüklenir.

## Doğrulama

- `tests/test_vf016_prosody.py` 14 test (modelsiz, gerçek worker süreçleri) + `tests/test_voice_playground.py` 13 test PASS; ruff PASS.
- Gerçek GPU: kısa metin bayt eşitliği, 2 uzun metin × 3 preset, 6 stil, 4 sahne + birleştirme, 7 yönerge × 2 tohum.
- Tarayıcı: gelişmiş panel, preset seçimi, sahne bölme tablosu (gerçek Gradio).
- Çalıştırılmayan: XTTS/Chatterbox yeniden denenmedi (kodları değişmedi); kör dinleme kullanıcıda; Studio'ya
  taşıma yapılmadı (tasarım hazır).
