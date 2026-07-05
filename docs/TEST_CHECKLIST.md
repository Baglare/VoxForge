# VoxForge Manuel Test Checklist

Bu checklist, VoxForge'un local çalışma akışlarını hızlı ve kontrollü şekilde doğrulamak için kullanılır. Testler tamamlandıktan sonra GitHub Desktop değişiklik listesinde yalnızca kaynak kod, runner, dokümantasyon ve `.gitkeep` dosyaları bulunmalıdır.

## 1. Smoke test

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_check.ps1
```

Kontrol:

- Terminalde `SMOKE CHECK PASSED` veya yalnızca uyarılar varsa `SMOKE CHECK PASSED WITH WARNINGS` görünür.
- Kritik eksik varsa `SMOKE CHECK FAILED` görünür.
- JSON rapor `outputs/reports/smoke_check_report.json` altına yazılır ve GitHub'a eklenmez.

## 2. Gradio testleri

### 2.1 Gradio açılış testi

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Kontrol:

- Local Gradio arayüzü açılır.
- Public paylaşım veya uzak servis kullanılmaz.

### 2.2 Varsayılan referans ses testi

Adım:

1. Profil seçmeden kısa bir Türkçe metin gir.
2. Harici referans ses yükleme.
3. İzin checkbox'ını işaretle ve üretimi başlat.

Kontrol:

- Sistem `samples/my_voice.wav` dosyasını kullanmaya çalışır.
- Dosya yoksa anlaşılır hata gösterilir.

### 2.3 Harici referans ses testi

Adım:

1. Profil seçme.
2. Açık izinli bir referans ses yükle.
3. Kısa bir metinle üretim başlat.

Kontrol:

- Yüklenen referans ses ön işlenir.
- Üretim bu referansla yapılır.
- Çıktı `outputs/gradio_outputs/` altında oluşur.

### 2.4 İzin ve boş metin kontrolleri

Kontrol:

- İzin checkbox'ı işaretli değilse üretim başlamaz.
- Metin alanı boşsa üretim başlamaz.
- Kullanıcıya anlaşılır uyarı verilir.

### 2.5 Kalite raporu testi

Kontrol:

- Ham ve/veya ön işlenmiş referans için `GOOD`, `WARNING` veya `BAD` sonucu görünür.
- Rapor teknik sinyal olarak yorumlanır; nihai kalite dinlenerek kontrol edilir.
- Raporlar `outputs/reports/gradio_quality_reports/` altında local kalır.

### 2.6 Inference preset testi

Adım:

1. Gradio arayüzünde `Üretim modu` alanından `Daha stabil` seç.
2. Kısa bir Türkçe metinle üretim başlat.

Kontrol:

- Durum mesajında kullanılan preset görünür.
- Ses üretimi tamamlanır veya desteklenmeyen XTTS parametresi varsa default üretime geri düşüldüğü sade şekilde yazılır.
- Kalite raporu JSON içinde `inference_preset` alanı bulunur.

### 2.7 Gradio uzun metin chunking testi

Adım:

1. Gradio metin alanına 220 karakterden uzun Türkçe metin gir.
2. İzin checkbox'ını işaretle ve üretimi başlat.

Kontrol:

- Durum mesajında chunking kullanıldığı ve parça sayısı görünür.
- Final WAV tek çıktı olarak dinlenebilir.
- Chunk parçaları `outputs/gradio_outputs/chunks/` altında local kalır.
- Kalite raporu JSON içinde `chunking_used`, `chunk_count` ve `chunks` alanları bulunur.
- Chunking kalite garantisi değildir; kırpılma riskini azaltmak için kullanılır.

### 2.8 Çıktı normalize testi

Adım:

1. `Çıktı sesini normalize et` checkbox'ını açık bırak.
2. Kısa bir metinle üretim başlat.

Kontrol:

- Durum mesajında post-processing sonucu görünür.
- Normalize başarılıysa final çıktı normalize edilmiş WAV olur.
- Normalize başarısızsa ham çıktı korunur ve kullanıcıya uyarı gösterilir.
- Kalite raporu JSON içinde `post_processing_enabled` alanı bulunur.

### 2.9 A/B karşılaştırma testi

Adım:

1. `A/B karşılaştırma üret` checkbox'ını işaretle.
2. `Üretim modu` için `Daha doğal deneme` veya `Daha uzun çıktı denemesi` seç.
3. Aynı metinle üretim başlat.

Kontrol:

- Ana çıktı seçili preset ile üretilir.
- Karşılaştırma çıktısı `Daha stabil` preset ile üretilir.
- İki WAV `outputs/gradio_outputs/` altında ayrı dosyalar olarak oluşur.
- Arayüzde ana çıktı ve karşılaştırma çıktısı ayrı audio alanlarında görünür.
- Kalite raporu JSON içinde `ab_test_enabled`, `primary_output_path` ve `comparison_output_path` alanları bulunur.

### 2.10 Experimental fine-tuning dataset dropdown testi

Adım:

1. Gradio arayüzünde `Experimental Fine-tuning Hazırlığı` bölümüne git.
2. Dataset dropdown listesini aç.

Kontrol:

- `datasets/` altında `metadata.csv` ve `wavs/` klasörü bulunan datasetler listelenir.
- Dataset yoksa `Yerel fine-tuning dataset bulunamadı.` mesajı görünür.
- Bozuk dataset varsa arayüz düşmez; eksik dosya/klasör bilgisi sade mesajla gösterilir.

### 2.11 Readiness butonu testi

Adım:

1. Geçerli bir dataset seç.
2. `Readiness raporu çalıştır` butonuna bas.

Kontrol:

- Hazırlık seviyesi, toplam örnek, geçerli örnek, toplam dakika ve öneriler görünür.
- JSON/Markdown rapor yolları `outputs/reports/` altında gösterilir.
- Bu işlem training başlatmaz.

### 2.12 Experiment export butonu testi

Adım:

1. Geçerli bir dataset seç.
2. Daha önce kullanılmamış bir run name gir.
3. `Experiment export et` butonuna bas.

Kontrol:

- Experiment path, train sample count, eval sample count ve manifest path görünür.
- Aynı run name daha önce kullanıldıysa üzerine yazılmaz ve farklı run name önerilir.
- Oluşan experiment dosyaları `experiments/` altında local kalır ve GitHub'a eklenmez.

### 2.13 Training dry-run butonu testi

Adım:

1. Export edilmiş veya mevcut bir experiment run name gir.
2. Max steps, epochs, batch size, grad accumulation ve save step değerlerini kontrol et.
3. `Training dry-run çalıştır` butonuna bas.

Kontrol:

- Sonuçta CUDA available, checkpoint OK, dataset OK, import OK, `limit_mode` ve config oluşturma sonucu görünür.
- Durum mesajında `--dry-run` kullanıldığı belirtilir.
- Bu işlem eğitim başlatmaz ve checkpoint üretmez.

### 2.14 Gerçek training butonu bulunmadığı kontrolü

Kontrol:

- Gradio arayüzünde `Training başlat`, `Gerçek training başlat` veya benzeri bir buton bulunmaz.
- Fine-tuning panelinde yalnızca readiness, export ve dry-run butonları vardır.

## 3. Profile testleri

### 3.1 Web üzerinden profil oluşturma

Adım:

1. Gradio arayüzünde profil adı gir.
2. Referans ses yükle.
3. İzin checkbox'ını işaretle.
4. `Profil oluştur` butonuna bas.

Kontrol:

- `profiles/<profile_slug>/` klasörü oluşur.
- Klasörde `original_reference.wav`, `preprocessed_reference.wav` ve `profile.json` bulunur.
- Aynı profil adı tekrar kullanılırsa mevcut profilin üzerine yazılmaz.

### 3.2 Profil seçme

Adım:

1. Gradio profil dropdown'ından oluşturulan profili seç.
2. İstersen ayrıca harici referans ses yükle.
3. Metin üretimi başlat.

Kontrol:

- Seçili profil üretimde öncelikli referans olur.
- Profil seçiliyken ayrıca yüklenen ses dosyası kullanılmaz.

### 3.3 Profil dropdown yenileme

Kontrol:

- Yeni profil oluşturulduktan sonra dropdown güncellenir.
- Mümkünse yeni profil seçili hale gelir.

### 3.4 Seçili profili yenileme

Adım:

1. Gradio'da bir profil seç.
2. `Seçili profili yenile` butonuna bas.

Kontrol:

- `original_reference.wav` korunur.
- `preprocessed_reference.wav` yeniden üretilir.
- `profile.json` kalite bilgisi güncellenir.
- XTTS modeli yüklenmez ve ses üretimi yapılmaz.

### 3.5 Seçili profili silme

Adım:

1. Gradio'da bir profil seç.
2. Silme onay checkbox'ını işaretle.
3. `Seçili profili sil` butonuna bas.

Kontrol:

- `profiles/<profile_slug>/` klasörü silinir.
- Dropdown güncellenir ve seçim temizlenir.
- `profiles/.gitkeep` korunur.
- Onay checkbox'ı işaretli değilse silme yapılmaz.

### 3.6 Terminalden profil yönetimi

Komutlar:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_voice_profile.ps1 -Name baglare -InputPath .\samples\my_voice.wav
powershell -ExecutionPolicy Bypass -File .\run_list_voice_profiles.ps1
powershell -ExecutionPolicy Bypass -File .\run_recreate_voice_profile.ps1 -Slug baglare
powershell -ExecutionPolicy Bypass -File .\run_delete_voice_profile.ps1 -Slug baglare -Yes
```

Kontrol:

- Profil komutları yalnızca `profiles/` altında local dosya üretir veya siler.
- Gerçek profil dosyaları GitHub'a eklenmez.

## 4. Fine-tuning hazırlık testleri

### 4.1 Dataset başlatma

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_init_finetune_dataset.ps1 -Name baglare_finetune_v1
```

Kontrol:

- `datasets/baglare-finetune-v1/` oluşur.
- `wavs/` alt klasörü ve `metadata.csv` oluşur.
- `metadata.csv` başlığı `audio_path|text` olur.
- Aynı isimle tekrar çalıştırılırsa üzerine yazılmaz.

### 4.2 Kayıt planı üretme

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80
```

Kontrol:

- `recording_plan.csv` oluşur.
- Başlık `clip_id;target_audio_path;text;status;notes` olur.
- Satırlar `TODO` durumuyla başlar.
- Dosya `utf-8-sig` encoding ve noktalı virgül delimiter ile yazılır.
- Dosya zaten varsa üzerine yazılmaz.

Gerekirse kontrollü yeniden üretim:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80 -Overwrite
```

### 4.3 Metadata oluşturma

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_build_metadata.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Kontrol:

- Yalnızca `recording_plan.csv` içinde `DONE` olan ve WAV dosyası bulunan satırlar `metadata.csv` içine yazılır.
- Eksik WAV dosyaları terminalde uyarı olarak görünür.

### 4.4 Dataset validation

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_validate_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Kontrol:

- Metadata satırları ve WAV dosyaları kontrol edilir.
- Süre, sample rate, mono kanal, ses seviyesi ve clipping riski raporlanır.
- Toplam süre ve ortalama örnek süresi terminalde görünür.
- Rapor `outputs/reports/finetune_dataset_report.json` altına yazılır ve GitHub'a eklenmez.

### 4.5 Readiness report

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune_readiness_report.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Kontrol:

- Toplam satır, geçerli/uyarılı/hatalı örnek sayısı ve toplam süre görünür.
- Yaklaşık 7-8 dakikalık hatasız dataset için `DATASET_VALID_BUT_SMALL` beklenir.
- JSON ve Markdown raporları `outputs/reports/` altında local kalır.

## 5. Deneysel fine-tuning testleri

### 5.1 Dataset export

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_xtts_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1 -RunName baglare_xtts_exp01
```

Kontrol:

- `experiments/baglare-xtts-exp01/` oluşur.
- `dataset/wavs/`, `metadata_train.csv`, `metadata_eval.csv` ve `experiment_manifest.json` bulunur.
- Export edilen dataset GitHub'a eklenmez.

### 5.2 Training dry-run

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1 -DryRun
```

Kontrol:

- Experiment path, dataset path, train/eval sayıları ve CUDA bilgisi görünür.
- `GPTArgs`, `GPTTrainer`, `GPTTrainerConfig` importları `Import OK` olur.
- `XttsAudioConfig import source` satırı görünür.
- `Start with eval: False` görünür.
- `TrainerArgs ozeti` altında `start_with_eval: False`, `skip_train_epoch: False`, `grad_accum_steps: 16` görünür.
- `Training config ozeti` altında `requested max_steps`, `requested epochs`, `resolved limit_mode`, `save_step`, `save_checkpoints`, `save_n_checkpoints` görünür.
- Son satırlarda `Dry-run config ve TrainerArgs olusturma OK.` ve `XTTS fine-tuning dry-run completed successfully` görünür.
- Dry-run training başlatmaz ve checkpoint üretmez.

### 5.3 Limit config kontrolü

Kontrol:

- `limit_mode: max_steps` görünüyorsa adım sınırı doğrudan uygulanır.
- `limit_mode: epochs_fallback` görünüyorsa config üzerinde `epochs` veya `num_epochs` değeri `1` veya daha büyük olmalıdır.
- `limit_mode: unsupported` görünüyorsa gerçek training başlatılmamalıdır.

### 5.4 Gerçek training

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Kontrol:

- `start_with_eval: False` ve `skip_train_epoch: False` görünür.
- `save_step: 1` ve `save_checkpoints: True` görünür.
- Güvenli limit uygulanamıyorsa script training başlatmadan durur.
- CUDA OOM durumunda `-BatchSize 1` korunur ve gerekirse `-GradAccum` artırılır.

### 5.5 Checkpoint artifact kontrolü

Kontrol:

- Training sonunda script `training_output/` altında checkpoint artifact arar.
- `.pth`, `.pt`, `.ckpt`, `.safetensors` veya checkpoint benzeri dosya bulunursa yollar terminale yazılır.
- Başarı mesajı: `Training completed and checkpoint artifacts were found.`
- Checkpoint yoksa hata mesajı: `Training finished but no checkpoint artifact was found.`
- `EPOCH: 0/0` ve checkpoint üretmeyen eval akışı başarı sayılmaz.

## 6. Inference ve değerlendirme testleri

### 6.1 Fine-tuned checkpoint inference

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_finetuned.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Text "Merhaba, bu ilk fine-tuned testidir."
```

Kontrol:

- Script seçilen checkpoint yolunu terminale yazar.
- Speaker wav önceliği doğru uygulanır.
- Fine-tuned inference başarılıysa `outputs/finetuned_eval/finetuned_test.wav` oluşur.
- Base output başarılıysa `outputs/finetuned_eval/base_test.wav` oluşur.
- Rapor `outputs/reports/finetuned_eval_report.json` altına yazılır.
- Bu test kalite garantisi değildir; çıktı dinlenerek değerlendirilir.

### 6.2 Checkpoint matrix evaluation

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_matrix.ps1 -Experiment .\experiments\baglare-xtts-exp01
```

Kontrol:

- `base`, `best_model.pth`, varsa `best_model_72.pth` ve en yüksek numaralı `checkpoint_*.pth` varyantları denenir.
- Aynı checkpoint iki kez seçilmez.
- Çıktılar `outputs/finetuned_eval/matrix/<timestamp>/` altına yazılır.
- Raporlar `outputs/reports/finetuned_matrix_report.json` ve `.md` olarak oluşur.
- Önce base, sonra fine-tuned varyantlar dinlenir.

### 6.3 Human evaluation scorecard

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_human_eval_report.ps1 -MatrixRoot .\outputs\finetuned_eval\matrix\<timestamp> -UseDefaultScores
```

Kontrol:

- `outputs/reports/human_eval_scorecard.csv` oluşur.
- `outputs/reports/human_eval_summary.json` oluşur.
- `outputs/reports/human_eval_summary.md` oluşur.
- CSV başlığı `variant;naturalness;similarity;pronunciation;human_likeness;text_accuracy;total_score;notes` olur.
- Sonuçlar insan dinlemesine dayanır; otomatik kalite ölçümü değildir.

### 6.4 Inference parameter sweep

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_inference_params.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Variant checkpoint_71
```

Kontrol:

- `default`, `conservative`, `stable` ve `longer_attempt` parametre setleri denenir.
- Çıktılar `outputs/finetuned_eval/param_sweep/<timestamp>/<variant>/<param_set>/` altına yazılır.
- Raporlar `outputs/reports/inference_param_sweep_report.json` ve `.md` olarak oluşur.
- `likely_cutoff` veya `possibly_cutoff` işaretleri dinleme ile doğrulanır.

### 6.5 Uzun metin chunking

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_finetuned.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Text "Bu uzun test metni, Türkçe XTTS karakter sınırı aşıldığında sistemin metni güvenli parçalara bölüp bölmediğini kontrol etmek için hazırlanmıştır. Cümleler doğal noktalama işaretleriyle ayrılır ve final ses dosyasında son cümlenin sessizce kırpılmadan duyulması beklenir."
```

Kontrol:

- Uzun metin 220 karakteri aşmayan parçalara bölünür.
- Final WAV `outputs/finetuned_eval/finetuned_test.wav` olarak oluşur.
- JSON raporda `chunking_used: true`, `chunk_count` ve `chunks` alanları görünür.
- Chunking kırpılma riskini azaltır; ses kalitesini garanti etmez.

### 6.6 Blind A/B experiment comparison testi

Komut:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_compare_finetune_experiments.ps1 -ExpA .\experiments\baglare-xtts-exp01 -ExpB .\experiments\baglare-xtts-exp02
```

Opsiyonel referans ses:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_compare_finetune_experiments.ps1 -ExpA .\experiments\baglare-xtts-exp01 -ExpB .\experiments\baglare-xtts-exp02 -SpeakerWav .\profiles\baglare\preprocessed_reference.wav
```

Kontrol:

- Terminalde base, exp_a ve exp_b için kullanılan checkpoint yolları görünür.
- Çıktılar `outputs/finetuned_eval/experiment_compare/<timestamp>/` altında `base/`, `exp_a/`, `exp_b/` ve `blind/` klasörlerine yazılır.
- `outputs/reports/experiment_compare_report.json` oluşur.
- `outputs/reports/experiment_compare_report.md` oluşur.
- `outputs/reports/experiment_blind_key.json` oluşur.
- `outputs/reports/experiment_blind_scorecard.csv` oluşur.
- Blind klasördeki A/B dosyaları önce experiment bilgisi bilinmeden dinlenir.
- Scorecard CSV doldurulduktan sonra blind key açılır.
- Bu test otomatik kalite ölçümü değildir; beklenti yanlılığını azaltır.
- Objektif metrikler yalnızca yardımcı teknik sinyaldir.

## 7. Yerel dosya ve GitHub kontrolü

GitHub Desktop'ta commit öncesi kontrol:

- `samples/` içindeki gerçek ses dosyaları commit listesinde olmamalıdır.
- `profiles/` içindeki gerçek profil dosyaları commit listesinde olmamalıdır.
- `datasets/` içindeki dataset dosyaları commit listesinde olmamalıdır.
- `experiments/` içindeki trainer çıktıları ve checkpointler commit listesinde olmamalıdır.
- `fine_tuned_models/` içindeki model dosyaları commit listesinde olmamalıdır.
- `outputs/` içindeki WAV, JSON, Markdown ve CSV rapor çıktıları commit listesinde olmamalıdır.
- `.venv/`, model cache dosyaları ve checkpoint uzantılı dosyalar commit listesinde olmamalıdır.

Commit listesinde beklenen dosya türleri:

- Kaynak kod değişiklikleri
- PowerShell runner değişiklikleri
- Dokümantasyon değişiklikleri
- Gerekli `.gitkeep` dosyaları
