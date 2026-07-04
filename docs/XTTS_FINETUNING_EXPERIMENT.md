# VoxForge Deneysel XTTS Fine-Tuning

Bu doküman, VoxForge içindeki mevcut yerel fine-tuning datasetini kullanarak deneysel XTTS-v2 GPT encoder fine-tuning altyapısını açıklar. Bu aşama kalite garantisi vermez; amaç önce training pipeline'ın yerel makinede başlayıp başlamadığını görmektir.

## Bu deney ne yapar?

Bu deney, `datasets/baglare-finetune-v1` içindeki doğrulanmış WAV ve transkriptleri Coqui/LJSpeech benzeri formata export eder. Ardından kullanıcı isterse ayrı bir PowerShell komutuyla XTTS-v2 GPT encoder fine-tuning denemesini başlatabilir.

Bu akış Gradio arayüzüne bağlı değildir. Yeni kullanıcı özelliği, veritabanı veya public servis eklemez.

## Dataset neden önce export ediliyor?

VoxForge dataset formatı proje içi hazırlık için sade tutulur:

```text
audio_path|text
```

Coqui training tarafında ise LJSpeech formatter ile çalışmak daha pratiktir. Bu yüzden export adımı kaynak dataset dosyalarını değiştirmez; bunun yerine deney klasörü altında ayrı bir kopya üretir.

Export edilen yapı:

```text
experiments/<run_slug>/
|-- dataset/
|   |-- wavs/
|   |-- metadata_train.csv
|   `-- metadata_eval.csv
|-- checkpoints/
|-- training_output/
`-- experiment_manifest.json
```

## LJSpeech metadata formatı

`metadata_train.csv` ve varsa `metadata_eval.csv` başlıksız yazılır:

```text
VF_TR_001|metin|metin
```

İlk sütunda `.wav` uzantısı bulunmaz. `VF_TR_001`, `dataset/wavs/VF_TR_001.wav` dosyasını temsil eden audio id değeridir.

## Neden 7.45 dakika veri deneysel sayılır?

Mevcut dataset teknik olarak geçerlidir:

- 80 geçerli örnek
- 0 uyarı
- 0 hata
- Yaklaşık 446.88 saniye / 7.448 dakika
- Ortalama örnek süresi yaklaşık 5.586 saniye

Bu, pipeline denemesi için anlamlıdır; ancak gerçek fine-tuning kalitesi için küçük bir veri miktarıdır. Bu yüzden readiness seviyesi `DATASET_VALID_BUT_SMALL` olarak değerlendirilir.

## Training komutları

Önce dataset export edilir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_xtts_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1 -RunName baglare_xtts_exp01
```

Training başlatmadan önce dry-run ile manifest, dataset klasörü, train/eval metadata dosyaları, checkpoint dosyaları, CUDA bilgisi, GPT trainer importları ve config oluşturma adımı kontrol edilebilir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -BatchSize 2 -GradAccum 8 -DryRun
```

Dry-run modeli eğitmez, `load_tts_samples` çalıştırmaz ve checkpoint indirme başlatmaz. Başarılı olduğunda terminalin sonunda şu satır görünmelidir:

```text
XTTS fine-tuning dry-run completed successfully
```

Training başlatmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -BatchSize 2 -GradAccum 8
```

Bu komut gerçekten training sürecini başlatır. Küçük dataset ve düşük adım sayısı nedeniyle çıkan model deneysel kabul edilmelidir.

## Beklenen çıktı klasörleri

```text
experiments/<run_slug>/dataset/
experiments/<run_slug>/checkpoints/
experiments/<run_slug>/training_output/
fine_tuned_models/
```

`experiments/` ve `fine_tuned_models/` altındaki gerçek dosyalar GitHub'a eklenmemelidir. Bu klasörler büyük checkpoint, model, log ve trainer çıktıları içerebilir.

## Olası hatalar

### CUDA OOM

GPU belleği yetmezse batch size değerini düşürün:

```powershell
-BatchSize 1
```

Gerekirse `-GradAccum` değerini artırarak efektif batch davranışı korunabilir.

### Coqui import hataları

Kurulu `coqui-tts`, `TTS` veya `trainer` API'si beklenen recipe ile aynı değilse script hangi importun eksik olduğunu terminalde açık yazar. Dry-run şu importları özellikle ayrı ayrı kontrol eder:

- `GPTArgs`
- `GPTTrainer`
- `GPTTrainerConfig`
- `XttsAudioConfig` ve bulunduğu import kaynağı
- `Trainer`
- `TrainerArgs`
- `BaseDatasetConfig`
- `load_tts_samples`

Bu durumda kurulu paket sürümü ve resmi XTTS GPT training recipe kontrol edilmelidir.

### XttsAudioConfig uyumluluğu

Bazı `coqui-tts` sürümlerinde `XttsAudioConfig`, `TTS.tts.layers.xtts.trainer.gpt_trainer` içinde bulunmaz. Script bu sınıfı sırayla şu modüllerde arar:

- `TTS.tts.layers.xtts.trainer.gpt_trainer`
- `TTS.tts.models.xtts`
- `TTS.tts.configs.xtts_config`

Bulunan kaynak terminalde `XttsAudioConfig import source: ...` satırıyla yazılır. Fallback ile bulunması hata değildir; yalnızca kurulu Coqui sürümünün sınıfı farklı modülde tuttuğunu gösterir.

`XttsAudioConfig` oluşturulurken `sample_rate`, `dvae_sample_rate` ve `output_sample_rate` alanları constructor imzasına göre filtrelenir. Kurulu sürüm bu alanlardan birini desteklemiyorsa script o alanı vermeden devam eder ve desteklenen alanları `DEBUG: XttsAudioConfig desteklenen alanlar: ...` satırıyla gösterir.

### XttsArgs unexpected keyword hatası

`TypeError: XttsArgs.__init__() got an unexpected keyword argument 'max_conditioning_length'` hatası, XTTS GPT training için genel `XttsArgs` sınıfının yanlış yerde kullanılmasından kaynaklanır. Training scripti bu aşamada `TTS.tts.layers.xtts.trainer.gpt_trainer` içindeki `GPTArgs`, `GPTTrainer` ve `GPTTrainerConfig` yolunu kullanır. `XttsAudioConfig` ise kurulu `coqui-tts` sürümüne göre fallback import ile bulunur.

Script, kurulu Coqui sürümünün desteklemediği config argümanlarını terminalde uyarı olarak gösterip atlamaya çalışır. Zorunlu bir argüman eksikse training başlamadan anlaşılır hata verir.

### Max steps sınırı

`-MaxSteps` küçük deneylerin uzun çalışmasını önlemek için kullanılır. Kurulu `trainer` veya `GPTTrainerConfig` sürümü `max_steps` alanını destekliyorsa script bunu ayarlar. Desteklemiyorsa terminalde açık uyarı verir; bu durumda eğitim süresi ilgili Coqui trainer sürümünün epoch/step davranışına bağlı olabilir.

### Checkpoint indirme hataları

Training scripti eksik XTTS dosyalarını `experiments/<run_slug>/checkpoints/` altına indirmeyi dener. Ağ bağlantısı, Hugging Face erişimi veya sertifika sorunları indirmeyi engelleyebilir.

### FFmpeg / torchcodec sorunları

Training runner, Gyan.FFmpeg.Shared yolunu PATH başına eklemeye çalışır. Ses okuma tarafında FFmpeg, TorchCodec veya PyTorch/CUDA uyumsuzluğu çıkarsa önce FFmpeg yolu ve PyTorch CUDA kurulumu kontrol edilmelidir.

## Training sonrası model nasıl değerlendirilecek?

Bu aşamada otomatik model değerlendirme eklenmemiştir. Training sonrası ayrı bir kapsamda şu kontroller yapılabilir:

- Aynı test metinlerini base XTTS ve fine-tuned checkpoint ile karşılaştırmak
- Ses benzerliğini kulakla değerlendirmek
- Telaffuz, gürültü, hız ve stabilite notları tutmak
- Kısa datasetin overfit davranışı üretip üretmediğini kontrol etmek

## Bu aşamanın portfolyo değeri

Bu çalışma, yalnızca demo üretiminden ileri gidip veri hazırlama, dataset doğrulama, export, deney manifesti, checkpoint hijyeni ve kontrollü training başlatma altyapısını gösterir. Model kalitesini abartmadan, yerel ve etik bir fine-tuning deneyi için gereken mühendislik iskeletini ortaya koyar.
