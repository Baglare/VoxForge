# VoxForge Deneysel XTTS Fine-Tuning

Bu doküman, VoxForge içindeki yerel dataset ile yapılan deneysel XTTS-v2 GPT encoder fine-tuning akışını açıklar. Akış local çalışır, Gradio arayüzüne bağlı değildir ve kalite garantisi vermez.

## 1. Amaç

Bu deneyin amacı, mevcut kayıt datasetinin XTTS fine-tuning formatına export edilmesini, training pipeline'ın yerel ortamda kontrollü şekilde çalışmasını, checkpoint üretilmesini ve üretilen checkpoint ile inference yapılmasını test etmektir.

Bu akış şunları hedeflemez:

- Public servis veya hesap sistemi eklemek
- Gradio UI davranışını değiştirmek
- Ses kalitesini garanti etmek
- Küçük dataset ile production seviyesinde model üretmek

## 2. Dataset durumu

Mevcut deney datasetinin özeti:

- 80 geçerli örnek
- Yaklaşık 446.88 saniye / 7.45 dakika toplam kayıt
- Ortalama örnek süresi yaklaşık 5.59 saniye
- Readiness yorumu: `DATASET_VALID_BUT_SMALL`

Bu veri miktarı pipeline testi için yeterlidir; ancak güçlü ve kararlı fine-tuning sonucu beklemek için küçüktür. Kalite değerlendirmesi bu sınır dikkate alınarak yapılmalıdır.

## 3. Export akışı

VoxForge dataset formatı proje içi hazırlık için şu yapıyı kullanır:

```text
audio_path|text
```

Coqui training tarafında LJSpeech formatter ile çalışmak daha pratik olduğu için dataset önce deney klasörüne export edilir. Export adımı kaynak dataset dosyalarını değiştirmez.

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_xtts_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1 -RunName baglare_xtts_exp01
```

Beklenen yapı:

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

`metadata_train.csv` ve `metadata_eval.csv` başlıksız yazılır:

```text
VF_TR_001|metin|metin
```

İlk sütunda `.wav` uzantısı bulunmaz. `VF_TR_001`, `dataset/wavs/VF_TR_001.wav` dosyasını temsil eder.

## 4. Training dry-run

Dry-run, training başlatmadan önce ortamı ve config üretimini kontrol eder.

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1 -DryRun
```

Dry-run şu kontrolleri yapar:

- Experiment ve dataset yolları
- Train/eval metadata dosyaları
- Checkpoint dosyalarının varlığı
- CUDA bilgisi
- GPT trainer importları
- `XttsAudioConfig` import kaynağı
- Training config oluşturma
- `limit_mode`, epoch fallback ve checkpoint ayarları
- `TrainerArgs` içinde `start_with_eval`, `skip_train_epoch`, `grad_accum_steps`

Başarılı dry-run sonunda şu satır görünür:

```text
XTTS fine-tuning dry-run completed successfully
```

Dry-run modeli eğitmez, `load_tts_samples` çalıştırmaz ve training checkpoint üretmez. Bu nedenle dry-run başarısı training başarısı değildir.

## 5. Gerçek training

Training yalnızca kullanıcı aşağıdaki komutu çalıştırırsa başlar:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Varsayılan akışta `-StartWithEval` kullanılmaz. Beklenen değerler:

- `start_with_eval: False`
- `skip_train_epoch: False`
- `save_step: 1`
- `save_checkpoints: True`

`-SaveStep 1`, kısa deneylerde checkpoint yazımını erken tetiklemek için kullanılır. Küçük dataset ve düşük adım sayısı nedeniyle çıkan model deneysel kabul edilmelidir.

## 6. Training başarı kriterleri

Training başarısı artifact temellidir.

Başarılı sayılmak için:

1. Training komutu hata ile durmamalıdır.
2. `training_output/` altında checkpoint artifact oluşmalıdır.
3. Script checkpoint artifact yollarını terminale yazmalıdır.

Checkpoint artifact örnekleri:

```text
*.pth
*.pt
*.ckpt
*.safetensors
checkpoint_*
best_model*
```

Checkpoint bulunursa beklenen mesaj:

```text
Training completed and checkpoint artifacts were found.
```

Checkpoint bulunamazsa beklenen hata:

```text
Training finished but no checkpoint artifact was found.
```

`EPOCH: 0/0`, yalnızca evaluation çalışması veya trainer logunun tamamlanması başarı sayılmaz. Checkpoint yoksa deney başarısız kabul edilir.

## 7. Training limit davranışı

`-MaxSteps`, kısa deneylerin kontrolsüz uzamasını önlemek için kullanılır. Script, kurulu `coqui-tts` ve `trainer` API'sine göre limit mekanizmasını açıkça raporlar:

- `limit_mode: max_steps`
- `limit_mode: epochs_fallback`
- `limit_mode: unsupported`

`max_steps` desteklenmiyorsa ve config üzerinde güvenli `epochs` veya `num_epochs` alanı doğrulanabiliyorsa fallback olarak `epochs=1` kullanılır.

Beklenen fallback mesajı:

```text
max_steps desteklenmiyor; güvenli fallback olarak epochs=1 kullanılacak.
```

Güvenli limit uygulanamıyorsa gerçek training başlatılmaz:

```text
Bu coqui-tts/trainer sürümünde max_steps güvenli şekilde uygulanamıyor. Eğitim başlatılmadı.
```

Epoch fallback config üzerinde doğrulanamazsa gerçek training yine durdurulur:

```text
Epoch fallback config üzerinde doğrulanamadı. Eğitim başlatılmadı.
```

## 8. Sık görülebilecek hatalar

### CUDA OOM

GPU belleği yetmezse `-BatchSize 1` korunmalı, gerekirse `-GradAccum` artırılmalıdır.

### Coqui import uyumsuzluğu

Kurulu `coqui-tts`, `TTS` veya `trainer` sürümü beklenen XTTS GPT recipe ile aynı olmayabilir. Dry-run şu importları ayrı ayrı kontrol eder:

- `GPTArgs`
- `GPTTrainer`
- `GPTTrainerConfig`
- `XttsAudioConfig`
- `Trainer`
- `TrainerArgs`
- `BaseDatasetConfig`
- `load_tts_samples`

`XttsAudioConfig` farklı modülden import edilebilir. Script bulunan kaynağı terminalde yazar; fallback kaynak kullanılması tek başına hata değildir.

### EPOCH: 0/0

`EPOCH: 0/0` görünürse ve checkpoint oluşmazsa bu gerçek training pass değildir. Önce şu değerler kontrol edilmelidir:

- `start_with_eval: False`
- `skip_train_epoch: False`
- `save_step: 1`
- `save_checkpoints: True`
- `limit_mode` güvenli mi?
- `epochs` veya `num_epochs` gerçekten `1` veya daha büyük mü?

### Checkpoint indirme hataları

Eksik XTTS dosyaları `experiments/<run_slug>/checkpoints/` altına indirilmeye çalışılır. Ağ, sertifika veya Hugging Face erişim sorunları indirmeyi engelleyebilir.

### FFmpeg / TorchCodec sorunları

Ses okuma tarafında FFmpeg, TorchCodec veya PyTorch/CUDA uyumsuzluğu görülebilir. Bu durumda önce FFmpeg yolu ve PyTorch CUDA kurulumu kontrol edilmelidir.

## 9. Training sonrası inference

Training checkpoint ürettiyse ilk hedef kalite iddiası değildir; fine-tuned checkpoint ile inference pipeline'ın çalıştığını doğrulamaktır.

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_finetuned.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Text "Merhaba, bu ilk fine-tuned testidir."
```

Checkpoint seçme önceliği:

1. `best_model.pth`
2. En yeni `best_model_*.pth`
3. En yüksek numaralı `checkpoint_*.pth`

Speaker wav önceliği:

1. `profiles/baglare/preprocessed_reference.wav`
2. `samples/my_voice.wav`
3. Experiment dataset içinden kısa bir WAV

Beklenen çıktılar:

```text
outputs/finetuned_eval/base_test.wav
outputs/finetuned_eval/finetuned_test.wav
outputs/reports/finetuned_eval_report.json
```

`best_model.pth` kalite garantisi değildir. Base ve fine-tuned dosyalar dinlenerek karşılaştırılmalıdır.

## 10. Checkpoint matrix evaluation

Matrix evaluation, birden fazla checkpoint varyantını ve Türkçe test cümlesini karşılaştırmak için kullanılır. Bu akış otomatik kalite ölçümü değildir.

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_matrix.ps1 -Experiment .\experiments\baglare-xtts-exp01
```

Opsiyonel referans ses:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_matrix.ps1 -Experiment .\experiments\baglare-xtts-exp01 -SpeakerWav .\profiles\baglare\preprocessed_reference.wav
```

Denenen varyantlar:

- `base`
- `best_model.pth`
- `best_model_72.pth`, varsa
- En yüksek numaralı `checkpoint_*.pth`, varsa

Çıktılar:

```text
outputs/finetuned_eval/matrix/<timestamp>/
outputs/reports/finetuned_matrix_report.json
outputs/reports/finetuned_matrix_report.md
```

Önerilen dinleme sırası:

1. Önce `base` çıktılarını dinleyin.
2. Sonra fine-tuned varyantları dinleyin.
3. Benzerlik, doğallık, telaffuz, insanilik ve metin doğruluğu için not alın.
4. Fine-tuned çıktı base'den zayıfsa yeni training yerine önce veri, referans ve checkpoint seçimini değerlendirin.

## 11. Human evaluation scorecard

Matrix çıktıları dinlendikten sonra manuel puanları raporlamak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_human_eval_report.ps1 -MatrixRoot .\outputs\finetuned_eval\matrix\<timestamp> -UseDefaultScores
```

Oluşan dosyalar:

```text
outputs/reports/human_eval_scorecard.csv
outputs/reports/human_eval_summary.json
outputs/reports/human_eval_summary.md
```

CSV başlığı:

```text
variant;naturalness;similarity;pronunciation;human_likeness;text_accuracy;total_score;notes
```

Puanlama 1-5 aralığındadır. Bu otomatik kalite ölçümü değildir; sonuçlar insan dinlemesine dayanır ve küçük dataset sınırıyla birlikte yorumlanmalıdır.

Mevcut değerlendirme yorumu: fine-tuning pipeline teknik olarak çalışmıştır, ancak kalite artışı sınırlıdır. Bazı checkpointlerde erken kesilme veya robotiklik duyulabilir.

## 12. Inference parameter sweep

Bazı checkpointler daha iyi puan alıp uzun cümlelerde erken kesilme gösterebilir. Bu durumda yeni training başlatmadan önce inference parametreleri karşılaştırılabilir.

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_inference_params.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Variant checkpoint_71
```

Opsiyonel referans ses:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_inference_params.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Variant checkpoint_71 -SpeakerWav .\profiles\baglare\preprocessed_reference.wav
```

Parametre setleri:

- `default`
- `conservative`
- `stable`
- `longer_attempt`

Çıktılar:

```text
outputs/finetuned_eval/param_sweep/<timestamp>/<variant>/<param_set>/test_01.wav
outputs/reports/inference_param_sweep_report.json
outputs/reports/inference_param_sweep_report.md
```

Raporlarda `likely_cutoff` ve `possibly_cutoff` gibi işaretler görülebilir. Bu işaretler kalite garantisi değildir; erken kesilme karşılaştırmalı dinleme ile doğrulanmalıdır.

## 13. Uzun metinlerde chunking

XTTS Türkçe inference tarafında uzun metinler için karakter sınırı uyarısı verebilir:

```text
The text length exceeds the character limit of 226 for language 'tr', this might cause truncated audio
```

VoxForge eval scriptleri bu riski azaltmak için 220 karakter üstündeki metinleri parçalara böler. Her parça ayrı üretilir ve FFmpeg ile tek final WAV dosyasına birleştirilir.

Chunking kullanılan scriptler:

- `scripts/evaluate_xtts_finetuned_checkpoint.py`
- `scripts/evaluate_xtts_checkpoint_matrix.py`
- `scripts/evaluate_xtts_inference_params.py`

Raporlarda `chunking_used`, `chunk_count` ve `chunks` alanları yer alır.

Chunking ses kalitesini garanti etmez. Uzun metnin sessizce kırpılmasını azaltır; ancak parça geçişlerinde küçük ton veya ritim farkları duyulabilir.

## 14. Çıktı klasörleri ve GitHub sınırı

Deney sırasında oluşan gerçek dosyalar local kalmalıdır:

```text
experiments/<run_slug>/dataset/
experiments/<run_slug>/checkpoints/
experiments/<run_slug>/training_output/
fine_tuned_models/
outputs/finetuned_eval/
outputs/reports/
```

Bu klasörlerde WAV dosyaları, checkpointler, model çıktıları, trainer logları ve değerlendirme raporları bulunabilir. GitHub'a eklenmemelidir.

## 15. Sonuç nasıl yorumlanmalı?

Mevcut deney şu teknik sonucu gösterir:

- Dataset export çalışır.
- Training dry-run ortam ve config kontrolü yapar.
- Gerçek training checkpoint üretebilir.
- Fine-tuned checkpoint ile inference çalışır.
- Base ve fine-tuned çıktılar karşılaştırmalı üretilebilir.
- Matrix ve human evaluation ile manuel değerlendirme yapılabilir.

Kalite garantisi yoktur. Mevcut yaklaşık 7.45 dakikalık dataset ile kalite artışı sınırlıdır. Daha iyi sonuç için daha fazla veri, daha tutarlı kayıt, checkpoint seçimi, training ayarı ve inference parametresi denemeleri gerekir.
