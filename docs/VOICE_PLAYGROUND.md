# Voice Playground

Serbest Türkçe metni XTTS-v2, Chatterbox V3 veya VoxCPM2 ile üretip dinlemek, WAV indirmek ve
motor/ayar çıktılarını yan yana karşılaştırmak için bağımsız Gradio arayüzü. Studio
(`run_gradio_demo.ps1`) ve VF-Q1 benchmark akışından ayrıdır; onların dosyalarına, profillere ve
motor varsayılanlarına yazmaz.

```
.\run_voice_playground.ps1          # http://127.0.0.1:7861
.\run_voice_playground.ps1 -Port 7870
```

## Ön koşul
- XTTS-v2: ana `.venv` (Studio ile aynı kurulum).
- Chatterbox V3 ve VoxCPM2: VF-Q1 izole ortamları
  (`scripts/benchmark/setup_env.ps1 -Engine chatterbox|voxcpm`, `C:\AvenoxData\Environments\VoxForge-bench-*`).
  Ortam yoksa arayüz kurulum komutunu gösteren bir hata verir; diğer motorlar çalışmaya devam eder.
- Model önbellekleri benchmark ile aynıdır: `HF_HOME=AvenoxData\Models\HF`, Coqui için `TTS_HOME`;
  XTTS'in TorchCodec'i için Gyan.FFmpeg.Shared DLL klasörü PATH'e eklenir.

## Mimari
- `app/voice_playground.py` (ana `.venv`): arayüz. Tüm istekler tek kuyruktan geçer.
- `voxforge/playground/workers.py` `EngineManager`: her motor kendi ortamında kalıcı bir alt süreç
  (`scripts/playground/engine_worker.py`). **Aynı anda en fazla bir süreç**: başka motor istenince önce
  mevcut süreç kapatılır (CUDA belleği süreçle birlikte boşalır), sonra yenisi yüklenir. Aynı motorla
  art arda üretimde model yeniden yüklenmez. "Motoru kapat" düğmesi VRAM'i elle boşaltır.
- `voxforge/playground/service.py`: varyantlar (motor + ayar), VF-004 okunuş normalizasyonu ve
  parçalama, kayıtlar. Çıktılar `outputs/playground/audio/<id>.wav` + `<id>.json` (git dışı);
  yeniden açılışta geçmiş yüklenir. Worker günlükleri `outputs/playground/logs/<motor>.log`.

## Varyantlar
| Varyant | Ayar |
|---|---|
| XTTS-v2 | ürün adaptörü, preset yok (VF-Q1 ile aynı) |
| Chatterbox V3 · varsayılan | `t3_model=v3`, `language_id=tr`, kütüphane varsayılanları |
| Chatterbox V3 · min_p=0.02 | yalnız `min_p=0.02` eklenir (docs/CHATTERBOX_REPEAT_PROBE.md) |
| VoxCPM2 (varsayılan) | controllable mod, `cfg_value=2.0`, `inference_timesteps=10` (VF-Q1) |
| VoxCPM2 · hızlı (6 adım) | yalnız `inference_timesteps=6` (hız denemesi; VF-015 pilotunda kalite düştü) |
| VoxCPM2 · Ultimate Cloning | aynı ayarlar + `prompt_wav_path`=referans, `prompt_text`=birebir transkript (model kartının önerisi) |
| Mock | modelsiz test sesi |

Kurulu kütüphane bir ayarı kabul etmezse üretim hata verir; sessizce düşürülmez. Chatterbox kaynağı
`engine_pins.json` commit'ine bağlıdır (benchmark ile aynı kontrol).

## Cihaz (VF-015)
Varsayılan cihaz **GPU (CUDA)**. Worker CUDA'lı torch yoksa başlamaz ve onarım komutunu gösterir;
model yüklendikten sonra da gerçekten CUDA'da olduğunu doğrular. CPU'ya **sessiz geçiş yoktur**.
"CPU (yalnız karşılaştırma)" bilerek seçilir; o süreç `CUDA_VISIBLE_DEVICES=-1` ile GPU'yu görmez
(Windows'ta boş değer GPU'yu gizlemiyor). Cihaz değişince süreç yeniden başlar.

CPU torch'lu bir ortamı onarmak (yalnız torch değişir, önceki paket listesi kaydedilir):
```
powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine voxcpm -TorchOnly
```
`scripts/benchmark/torch_cuda.py`, kurulu torch sürümünü gerçekten yayımlayan ve sürücünün çalıştırabildiği
en yeni CUDA indeksini seçer (ör. torch 2.15.0 → cu134/cu132; eski sabit cu126'da yok).

## VoxCPM2 ve 8 GB VRAM (VF-015)
- Ağırlıklar ~5,1 GB (bf16 LM ~4,6 GB, fp32 AudioVAE 0,36 GB, sabit KV önbelleği 0,29 GB).
- 30 sn'lik referansın AudioVAE ile kodlanması geçici olarak +1,2 GB ister. Worker kodlanmış referansı
  önbellekte tutar (yol + boyut + değiştirilme zamanı + transkript anahtarı; kodlayıcı deterministik,
  sonuç CPU'da ve salt okunur) ve kodlamadan/üretimden sonra `torch.cuda.empty_cache()` çağırır.
  Sıcak üretimde tepe ~5,6–5,8 GB, ayrılmış ~5,8 GB.
- Laptop GPU'su masaüstüyle paylaşılır (canlı duvar kağıdı, tarayıcı, oyun istemcisi boşta ~0,6–1,3 GB
  tutabiliyor). Toplam VRAM dolunca Windows belleği sistem RAM'ine taşır ve üretim 10 kattan fazla
  yavaşlar (ölçüldü: aynı cümle 3,5 sn yerine 40 sn ve üstü). Arayüz bunu "Darboğaz" notunda gösterir.
- `torch.compile` varsayılan kapalı (Windows'ta triton yok); `VOXFORGE_VOXCPM_COMPILE=1` + triton ile
  denenebilir, ölçülmedi. Önbelleği kapatmak için `VOXFORGE_VOXCPM_PROMPT_CACHE=0`.
- Aynı süreçte aynı tohumla art arda üretimler bit düzeyinde aynı değildir (GPU çekirdek seçimi); her
  sürecin ilk üretimi tekrarlanabilir. CPU üretimi önceki CPU örneğiyle birebir aynı çıktı.

## Ultimate Cloning
VoxCPM2 ayarı "Ultimate Cloning" seçilince referansın birebir transkripti istenir; profil başına
`outputs/playground/transcripts/<profil>.txt` dosyasına kaydedilir (profiller salt okunur kalır).
Yanlış ya da eksik transkript benzerliği düşürür. Mevcut "yalnız referans" modu varsayılandır.

## VF-016: gelişmiş kontroller, preset'ler ve sahne anlatımı
Ayrıntı ve ölçümler: [VF016_PROSODY_FINDINGS.md](VF016_PROSODY_FINDINGS.md).
- **VoxCPM2 gelişmiş kontroller** (motor VoxCPM2 iken açılır): CFG 1,0-3,0, difüzyon adımı 4-20, tohum,
  stil yönergesi (metinden ayrı alan; modele `(yönerge)metin` olarak verilir), anlatım ritmi
  (standart = eski paketleme; anlatım = cümle cümle üretim, kenar kırpma, cümle/paragraf duraklaması).
  Varsayılan değerler eski isteği birebir gönderir. Ultimate Cloning ile stil yönergesi kullanılamaz.
- **Preset'ler**: Normal, Sakin, Yavaş anlatıcı (uzun anlatım için önerilen), Dramatik, Gerilim, Aksiyon
  (yönergesiz), Duygusal, Diyalog; her birinin kanıt notu arayüzde görünür. Kullanıcı preset'leri
  `outputs/playground/presets.json`. "Preset karşılaştırması" aynı metni aynı ses/tohumla üretir.
- **Sahne anlatımı** sekmesi: metni sahnelere ayır, tür/preset ata (yerel öneri, düzenlenebilir), sahne
  başına üret/yeniden üret/dinle, tek WAV'da birleştir (kırpılmış kenar, sabit sahne arası duraklama,
  ±6 dB seviye eşleme). Projeler `outputs/playground/scenes/<ad>.json` + `<ad>/` altında.

## VF-017: stilsiz varsayılanlar ve seviye eşleme düzeltmesi
Ayrıntı ve ölçümler: [VF017_NARRATION_FINDINGS.md](VF017_NARRATION_FINDINGS.md).
- **Yavaş anlatıcı** artık stil yönergesi göndermez (cümle cümle + 0,55/0,95 sn duraklama). Eski stilli sürüm
  "Yavaş anlatıcı + stil (deneysel)". Sakin/Dramatik/Gerilim/Duygusal stilli preset'ler "deneysel" etiketlidir.
- Sahne türlerinin varsayılan preset'leri stilsizdir: sakin → Yavaş anlatıcı, gerilim/dramatik → Gerilim (stilsiz,
  uzun duraklama), duygusal → Duygusal (stilsiz, uzun duraklama), aksiyon/diyalog aynı. Mevcut projeler kayıtlı
  preset'lerini korur.
- Sahne birleştirmede seviye eşleme gerçekten uygulanır (önceden ~0 dBFS'e normalize parçalarda tepe koruması
  pozitif kazancı iptal ediyordu); gösterilen kazanç uygulanan kazançtır.

## Her üretimde gösterilen
Model, kaynak commit (Chatterbox), uygulanan ayarlar ve kütüphane varsayılanları (JSON'da), tohum,
profil, parça sayısı, üretim süresi, ses süresi, RTF (üretim / ses süresi), bu istekte model yüklendiyse
yükleme süresi; VRAM için üretim tepe (`torch.cuda.max_memory_allocated`), ayrılmış bellek ve GPU'nun
toplam kullanımı; cihaz, torch/GPU, dtype, parça süreleri, referans kodlama süresi ve kısa
darboğaz notları (CPU, RTF > 1, VRAM dolu, başka süreçlerin GPU belleği, model yükleme). Karşılaştırmada aynı tohum kullanılır; `-1` rastgele tohum seçer ve kaydeder.

## Sınırlar
- Playground kendi GPU kullanımını sınırlar; aynı anda açık Studio veya benchmark süreçlerini göremez.
  8 GB kartta VoxCPM2 için GPU kullanan diğer uygulamaları kapatmak hızı belirgin artırır (yukarıya bakın).
- Normalizasyon cümle bazında yapılır ama cümleler motor sınırına (~220-250 karakter) kadar aynı parçada
  üretilir; yalnız daha uzun metinler bölünüp 0,2 sn sessizlikle birleştirilir (standart ritim). Chatterbox'ta
  her cümleyi ayrı üretmek ~0,9 sn boşluk ve tonlama kopukluğu ekliyordu; VoxCPM2'de ise cümle cümle üretim +
  kenar kırpma ölçümle daha iyi çıktı (VF-016 "Anlatım" ritmi, isteğe bağlı). loudnorm uygulanmaz; yalnız
  sahne birleştirmede ±6 dB RMS eşlemesi var.
- XTTS TorchCodec için FFmpeg shared bin PATH'te zaten olsa bile en başa alınır (system32'den sonra
  kalınca `Could not load libtorchcodec` hatası veriyordu).
- Çıktılar Chatterbox'ta PerTh filigranı içerir.

## Test
`tests/test_voice_playground.py`: gerçek worker süreçleri ve modelsiz motorla tek-motor kuralı, min_p
aktarımı, kayıt/WAV, hata yolları; VF-015: cihaz seçimi ve CPU sürecinde GPU gizleme, CUDA yokken
reddetme, Ultimate transkript zorunluluğu/saklama, prompt önbelleği anahtarı, darboğaz notları, CUDA
indeks seçimi. `tests/test_vf016_prosody.py`: WAV birleştirme (eski birleştirme bayt eşitliği, kırpma,
seviye), ritim ölçümü, kontrol aralıkları, preset deposu, parça planı, stilin metinden ayrı taşınması,
Ultimate uyumsuzluğu, preset karşılaştırması ve sahne üret/birleştir akışı. Gerçek motorlarla GPU üretimi
bu testlerde yoktur (ölçümler PR açıklamasında ve VF016_PROSODY_FINDINGS.md'de).
