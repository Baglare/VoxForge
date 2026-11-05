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
| VoxCPM2 | controllable mod, `cfg_value=2.0`, `inference_timesteps=10` (VF-Q1) |
| Mock | modelsiz test sesi |

Kurulu kütüphane bir ayarı kabul etmezse üretim hata verir; sessizce düşürülmez. Chatterbox kaynağı
`engine_pins.json` commit'ine bağlıdır (benchmark ile aynı kontrol).

## Her üretimde gösterilen
Model, kaynak commit (Chatterbox), uygulanan ayarlar ve kütüphane varsayılanları (JSON'da), tohum,
profil, parça sayısı, üretim süresi, ses süresi, RTF (üretim / ses süresi), bu istekte model yüklendiyse
yükleme süresi; VRAM için üretim tepe (`torch.cuda.max_memory_allocated`), ayrılmış bellek ve GPU'nun
toplam kullanımı. Karşılaştırmada aynı tohum kullanılır; `-1` rastgele tohum seçer ve kaydeder.

## Sınırlar
- Playground kendi GPU kullanımını sınırlar; aynı anda açık Studio veya benchmark süreçlerini göremez.
  8 GB kartta VoxCPM2 yüklenirken başka bir uygulamanın GPU'yu kullanmaması gerekir.
- Normalizasyon cümle bazında yapılır ama cümleler motor sınırına (~220-250 karakter) kadar aynı parçada
  üretilir; yalnız daha uzun metinler bölünüp 0,2 sn sessizlikle birleştirilir. Her cümleyi ayrı üretmek
  cümle aralarına ~0,9 sn boşluk ve tonlama kopukluğu ekliyordu. loudnorm uygulanmaz.
- XTTS TorchCodec için FFmpeg shared bin PATH'te zaten olsa bile en başa alınır (system32'den sonra
  kalınca `Could not load libtorchcodec` hatası veriyordu).
- Çıktılar Chatterbox'ta PerTh filigranı içerir.

## Test
`tests/test_voice_playground.py`: gerçek worker süreçleri ve modelsiz motorla tek-motor kuralı, min_p
aktarımı, kayıt/WAV, hata yolları. Gerçek motorlarla GPU üretimi bu testlerde yoktur.
