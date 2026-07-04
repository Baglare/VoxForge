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
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1 -DryRun
```

Dry-run modeli eğitmez, `load_tts_samples` çalıştırmaz ve checkpoint indirme başlatmaz. Başarılı olduğunda terminalin sonunda şu satır görünmelidir:

```text
XTTS fine-tuning dry-run completed successfully
```

Dry-run çıktısında varsayılan akış için `Start with eval: False` görünmelidir. `TrainerArgs ozeti` altında da şu değerler kontrol edilmelidir:

- `start_with_eval: False`
- `skip_train_epoch: False`
- `grad_accum_steps: <GradAccum degeri>`

Training başlatmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Bu komut gerçekten training sürecini başlatır. Runner varsayılan olarak `-StartWithEval` geçmez; bu yüzden Python tarafında `start_with_eval=False` kullanılır. `-SaveStep 1`, kısa denemede checkpoint yazımını erken tetiklemek için kullanılır. Küçük dataset ve düşük adım sayısı nedeniyle çıkan model deneysel kabul edilmelidir.

Sadece bilinçli bir karşılaştırma yapmak istenirse runner'a `-StartWithEval` eklenebilir. Küçük dataset denemelerinde varsayılan önerilen akış `start_with_eval=False` olarak kalmalıdır.

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

`-MaxSteps` küçük deneylerin uzun çalışmasını önlemek için kullanılır. Script, `GPTTrainerConfig` ve `TrainerArgs` constructor imzalarını `inspect.signature` ile kontrol eder ve mümkünse `max_steps` sınırını uygular.

Dry-run çıktısında hangi limit mekanizmasının kullanılacağı açıkça yazılır:

- `limit_mode: max_steps`
- `limit_mode: epochs_fallback`
- `limit_mode: unsupported`

Kurulu `coqui-tts/trainer` sürümü `max_steps` desteklemiyorsa script `epochs` veya `num_epochs` alanı arar. Bunlardan biri destekleniyorsa güvenli fallback olarak `epochs=1` kullanılır ve terminalde şu mesaj görünür:

```text
max_steps desteklenmiyor; güvenli fallback olarak epochs=1 kullanılacak.
```

`max_steps` ve epoch tabanlı sınırların hiçbiri desteklenmiyorsa dry-run yine tamamlanabilir; ama gerçek training başlatılmaz. Bu durumda script exit code `1` ile çıkar ve şu hatayı verir:

```text
Bu coqui-tts/trainer sürümünde max_steps güvenli şekilde uygulanamıyor. Eğitim başlatılmadı.
```

Bu durumda daha güvenli kısa deneme için epoch sınırı desteği eklenmeli veya kurulu trainer API'sine göre adım/epoch sınırı yeniden uyarlanmalıdır.

Dry-run ve gerçek training öncesinde script şu config bilgilerini yazar:

- requested max_steps
- requested epochs
- resolved limit_mode
- config.epochs
- config.num_epochs
- save_step
- save_checkpoints
- save_n_checkpoints

`TrainerArgs ozeti` bölümünde ayrıca `start_with_eval`, `skip_train_epoch` ve `grad_accum_steps` değerleri yazılır. Küçük deneyde beklenen güvenli varsayılan `start_with_eval: False` ve `skip_train_epoch: False` değerleridir.

`limit_mode: epochs_fallback` ise script config üzerinde `epochs` veya `num_epochs` alanını gerçekten `1` veya daha büyük görmeyi bekler. Bu doğrulanamazsa gerçek training başlamaz ve şu hata verilir:

```text
Epoch fallback config üzerinde doğrulanamadı. Eğitim başlatılmadı.
```

### EPOCH: 0/0 ve checkpoint oluşmaması

Bazı trainer sürümlerinde config limitleri doğru uygulanmadığında veya training eval ile başladığında trainer açılıp yalnızca eval çalıştırabilir. Bu durumda `EPOCH: 0/0` görünebilir ve `training_output/` altında checkpoint üretilmeyebilir. Bu durum gerçek training pass olarak kabul edilmez.

Bu senaryoda önce dry-run çıktısını kontrol edin:

- `Start with eval: False`
- `TrainerArgs ozeti` altında `start_with_eval: False`
- `TrainerArgs ozeti` altında `skip_train_epoch: False`
- `save_step: 1`
- `save_checkpoints: True`

Gerçek training başlatırken normal komutta `-StartWithEval` kullanmayın:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Checkpoint yine oluşmazsa şu üç noktayı özellikle kontrol edin:

- `start_with_eval=False` kullanılıyor mu?
- `limit_mode: epochs_fallback` görünüyorsa `epochs` veya `num_epochs` gerçekte `1` veya daha büyük mü?
- Kısa deneyde `save_step` değeri `1` olarak tutuluyor mu?

Training bittikten sonra script `training_output/` klasörünü recursive tarar. `.pth`, `.pt`, `.ckpt`, `.safetensors` veya checkpoint benzeri artifact bulunursa yolları terminale yazar ve şu mesajı verir:

```text
Training completed and checkpoint artifacts were found.
```

Checkpoint bulunamazsa script şu mesajı verir ve exit code `1` ile çıkar:

```text
Training finished but no checkpoint artifact was found.
```

### Checkpoint indirme hataları

Training scripti eksik XTTS dosyalarını `experiments/<run_slug>/checkpoints/` altına indirmeyi dener. Ağ bağlantısı, Hugging Face erişimi veya sertifika sorunları indirmeyi engelleyebilir.

### FFmpeg / torchcodec sorunları

Training runner, Gyan.FFmpeg.Shared yolunu PATH başına eklemeye çalışır. Ses okuma tarafında FFmpeg, TorchCodec veya PyTorch/CUDA uyumsuzluğu çıkarsa önce FFmpeg yolu ve PyTorch CUDA kurulumu kontrol edilmelidir.

## Training sonrası inference testi

Training başarıyla checkpoint ürettiyse ilk amaç kalite iddiası değildir; fine-tuned checkpoint ile inference pipeline çalışıyor mu bunu görmektir. Bu adım yeni training başlatmaz, Gradio UI değiştirmez ve yeni paket eklemez.

İlk fine-tuned checkpoint inference denemesi için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_finetuned.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Text "Merhaba, bu ilk fine-tuned testidir."
```

Script `training_output/` altında checkpointi şu öncelikle seçer:

1. `best_model.pth`
2. En yeni `best_model_*.pth`
3. En yüksek numaralı `checkpoint_*.pth`

Base XTTS dosyaları `experiments/<run_slug>/checkpoints/` içinden okunur. Fine-tuned GPT checkpoint olarak seçilen training output checkpointi kullanılır. Speaker wav verilmezse öncelik sırası şudur:

1. `profiles/baglare/preprocessed_reference.wav`
2. `samples/my_voice.wav`
3. Experiment dataset içinden kısa bir WAV

Başarılı denemede şu dosyalar oluşur:

```text
outputs/finetuned_eval/base_test.wav
outputs/finetuned_eval/finetuned_test.wav
outputs/reports/finetuned_eval_report.json
```

Base output üretimi başarısız olursa fine-tuned deneme yine devam eder; hata JSON raporuna yazılır. Fine-tuned output başarısız olursa script exit code `1` ile biter ve hangi import, checkpoint yükleme veya synthesize çağrısında patladığını sade şekilde yazar.

`best_model.pth` kalite garantisi değildir. Base output ve fine-tuned output mutlaka dinlenerek karşılaştırılmalıdır. Mevcut 72 train / 8 eval örnekli küçük dataset nedeniyle ses benzerliği, telaffuz ve stabilite sınırlı olabilir.

## Checkpoint matrix evaluation

Tek cümlelik testte fine-tuned ses robotik gelirse hemen yeni training başlatmadan önce birden fazla checkpoint ve birden fazla Türkçe test cümlesiyle karşılaştırma yapılmalıdır. Matrix evaluation akışı kaliteyi otomatik ölçmez; insan kulağıyla base ve fine-tuned çıktılar karşılaştırılır.

Matrix testi için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_matrix.ps1 -Experiment .\experiments\baglare-xtts-exp01
```

Opsiyonel özel referans ses vermek için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_matrix.ps1 -Experiment .\experiments\baglare-xtts-exp01 -SpeakerWav .\profiles\baglare\preprocessed_reference.wav
```

Script şu varyantları dener:

- `base`
- `best_model.pth`
- `best_model_72.pth`, varsa
- En yüksek numaralı `checkpoint_*.pth`, varsa

Aynı checkpoint iki kez seçilmez. Eksik checkpoint varyantı atlanır ve rapora yazılır.

Çıktılar zaman damgalı klasöre yazılır:

```text
outputs/finetuned_eval/matrix/<timestamp>/
|-- base/
|   |-- test_01.wav
|   `-- test_02.wav
|-- best_model/
|-- best_model_72/
`-- checkpoint_71/
```

Rapor dosyaları:

```text
outputs/reports/finetuned_matrix_report.json
outputs/reports/finetuned_matrix_report.md
```

Dinleme sırası:

1. Önce `base` klasörünü dinleyin.
2. Sonra fine-tuned varyantları sırayla dinleyin.
3. Benzerlik, doğallık, telaffuz ve robotiklik açısından 1-5 puan verin.
4. Fine-tuned çıktı base'den kötüyse daha fazla training basmadan önce veri, referans ve checkpoint seçimi değerlendirilmelidir.

Robotiklik duyuluyorsa bunun nedeni yalnızca checkpoint olmayabilir. Kayıt temizliği, referans ses seçimi, dataset çeşitliliği, transkript uyumu ve kısa dataset sınırı birlikte incelenmelidir.

## Human evaluation scorecard

Matrix evaluation çıktıları dinlendikten sonra manuel puanları tek bir scorecard dosyasına kaydetmek için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_human_eval_report.ps1 -MatrixRoot .\outputs\finetuned_eval\matrix\<timestamp> -UseDefaultScores
```

Bu komut training başlatmaz, yeni model eğitmez ve Gradio UI'a dokunmaz. `-UseDefaultScores`, ilk manuel dinleme puanlarını kullanarak şu dosyaları üretir:

```text
outputs/reports/human_eval_scorecard.csv
outputs/reports/human_eval_summary.json
outputs/reports/human_eval_summary.md
```

CSV dosyası noktalı virgül (`;`) ile ayrılır ve sonradan elle düzenlenebilir:

```text
variant;naturalness;similarity;pronunciation;human_likeness;text_accuracy;total_score;notes
```

Puanlama 1 kötü, 5 iyi olacak şekilde yorumlanır. `human_likeness` alanında 1 daha robotik, 5 daha insan gibi kabul edilir.

Bu otomatik kalite ölçümü değildir. Ses benzerliği insan kulağıyla değerlendirilir ve sonuçlar küçük dataset ile deneysel fine-tuning bağlamında yorumlanmalıdır. İlk puanlara göre fine-tuning pipeline teknik olarak başarılıdır; kalite artışı sınırlıdır. `best_model` base'e göre küçük iyileşme gösterir. `checkpoint_71` daha yüksek puan alsa da bazı kayıtlarda cümle erken kesiliyor gibi olduğu için güvenilir kabul edilmemelidir. Daha fazla veri, daha iyi referans kayıt ve inference ayarı değerlendirilmelidir.

## Training sonrası model nasıl değerlendirilecek?

İlk inference scripti yalnızca teknik pipeline kontrolüdür. Kalite değerlendirmesi için şu kontroller ayrıca yapılmalıdır:

- Aynı test metinlerini base XTTS ve fine-tuned checkpoint ile karşılaştırmak
- Ses benzerliğini kulakla değerlendirmek
- Telaffuz, gürültü, hız ve stabilite notları tutmak
- Kısa datasetin overfit davranışı üretip üretmediğini kontrol etmek

## Bu aşamanın portfolyo değeri

Bu çalışma, yalnızca demo üretiminden ileri gidip veri hazırlama, dataset doğrulama, export, deney manifesti, checkpoint hijyeni ve kontrollü training başlatma altyapısını gösterir. Model kalitesini abartmadan, yerel ve etik bir fine-tuning deneyi için gereken mühendislik iskeletini ortaya koyar.
