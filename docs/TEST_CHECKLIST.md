# VoxForge Manuel Test Checklist

Bu doküman, VoxForge MVP için hızlı manuel kontrol listesidir. Amaç, demo öncesinde ana akışların çalıştığını ve kişisel ses/veri dosyalarının GitHub'a girmediğini kontrol etmektir.

## 1. Smoke test

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_check.ps1
```

Beklenen sonuç: Terminalde `SMOKE CHECK PASSED` veya yalnızca uyarılar varsa `SMOKE CHECK PASSED WITH WARNINGS` görünür. Kritik eksik varsa `SMOKE CHECK FAILED` görünür ve düzeltilmeden demo yapılmaz.

## 2. Fine-tuning dataset başlatma testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_init_finetune_dataset.ps1 -Name baglare_finetune_v1
```

Beklenen sonuç: `datasets/baglare-finetune-v1/`, `datasets/baglare-finetune-v1/wavs/` ve başlığı `audio_path|text` olan boş `metadata.csv` oluşur. Aynı isimle tekrar çalıştırılırsa üzerine yazılmaz.

## 3. Fine-tuning kayıt planı üretme testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80
```

Beklenen sonuç: `recording_plan.csv` oluşur. Dosyada `clip_id;target_audio_path;text;status;notes` başlığı bulunur, ilk 80 kayıt metni listelenir ve tüm satırların `status` alanı `TODO` olur. Dosya `utf-8-sig` encoding ve noktalı virgül (`;`) delimiter ile yazılır. Dosya zaten varsa üzerine yazılmaz.

Excel kontrolü: `recording_plan.csv` dosyasını Excel'de çift tıklayarak aç. `Bugün` gibi Türkçe karakterler bozulmadan görünmeli ve alanlar ayrı sütunlara bölünmelidir. Eski dosya bozuk görünüyorsa şu komutla kontrollü yeniden üret:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80 -Overwrite
```

## 4. Fine-tuning metadata oluşturma testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_build_metadata.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Beklenen sonuç: Yalnızca `recording_plan.csv` içinde `DONE` olan ve ses dosyası gerçekten bulunan satırlar `metadata.csv` içine yazılır. Eksik WAV dosyaları için terminalde uyarı görünür.

## 5. Fine-tuning dataset doğrulama testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_validate_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Beklenen sonuç: Metadata satırları, WAV dosyaları, fine-tuning klip süresi, sample rate, mono kanal, ses seviyesi ve clipping riski kontrol edilir. Ortalama 5-6 saniye civarındaki temiz klipler normal kabul edilir; 80 kayıt için toplu şekilde `30 saniyeden kısa` uyarısı beklenmez. Terminal özetinde toplam süre, ortalama örnek süresi ve tahmini dakika bilgisi görünür. Rapor `outputs/reports/finetune_dataset_report.json` dosyasına yazılır ve GitHub'a eklenmez.

## 6. Fine-tuning readiness report testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune_readiness_report.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Beklenen sonuç: Terminalde toplam satır, geçerli/uyarılı/hatalı örnek sayısı, toplam süre, ortalama süre ve hazırlık seviyesi görünür. Yaklaşık 7-8 dakikalık hatasız dataset için `DATASET_VALID_BUT_SMALL` beklenir. JSON ve Markdown raporları `outputs/reports/` altına yazılır ve GitHub'a eklenmez.

## 7. Deneysel XTTS dataset export testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_xtts_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1 -RunName baglare_xtts_exp01
```

Beklenen sonuç: `experiments/baglare-xtts-exp01/` altında `dataset/wavs/`, `metadata_train.csv`, `metadata_eval.csv` ve `experiment_manifest.json` oluşur. Train/eval sayıları manifest içinde görünür. Export edilen dataset GitHub'a eklenmez.

## 8. Deneysel XTTS training dry-run testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1 -DryRun
```

Beklenen sonuç: Experiment path, dataset path, train/eval sayıları, language, max steps, epoch fallback, batch size, grad accumulation, save step, `Start with eval: False` ve CUDA bilgisi görünür. Dataset klasörü, `metadata_train.csv`, varsa `metadata_eval.csv`, checkpoint dosyaları ve GPT trainer importları kontrol edilir. `GPTArgs`, `GPTTrainer`, `GPTTrainerConfig` ayrı ayrı `Import OK` olarak görünür. `XttsAudioConfig` için `Import OK: XttsAudioConfig` ve `XttsAudioConfig import source: ...` satırları görünür; fallback kaynak kullanılması hata değildir. `TrainerArgs ozeti` altında `start_with_eval: False`, `skip_train_epoch: False` ve `grad_accum_steps: 16` görünür. Config oluşturma başarılı olursa terminal sonunda `Dry-run config ve TrainerArgs olusturma OK.` ve `XTTS fine-tuning dry-run completed successfully` görünür. `-DryRun` kullanıldığı için training başlamaz, `load_tts_samples` çalıştırılmaz, checkpoint aranmaz ve checkpoint indirme yapılmaz.

## 9. Dry-run limit config kontrolü

Adım: Dry-run çıktısında `Training config ozeti` bölümünü kontrol et.

Beklenen sonuç: `requested max_steps`, `requested epochs`, `resolved limit_mode`, `config.epochs`, `config.num_epochs`, `save_step`, `save_checkpoints` ve `save_n_checkpoints` satırları görünür. `save_step` değeri `1`, `save_checkpoints` değeri `True`, `save_n_checkpoints` değeri `1` olmalıdır. `TrainerArgs ozeti` altında `start_with_eval: False`, `skip_train_epoch: False` ve `grad_accum_steps` görünmelidir. `limit_mode: epochs_fallback` görünüyorsa config üzerinde `epochs` veya `num_epochs` alanlarından en az biri `1` veya daha büyük görünmelidir.

## 10. MaxSteps / epoch limit kontrolü

Adım: Dry-run çıktısında `limit_mode` satırını kontrol et.

Beklenen sonuç: `limit_mode: max_steps` görünüyorsa adım sınırı doğrudan uygulanacaktır. `limit_mode: epochs_fallback` görünüyorsa `max_steps` desteklenmediği için güvenli fallback olarak `epochs=1` kullanılacaktır. `limit_mode: unsupported` görünüyorsa dry-run uyarıyla tamamlanabilir, fakat gerçek training başlatılmamalı ve script gerçek training modunda exit code `1` ile durmalıdır.

## 11. Deneysel XTTS training başlatma kontrolü

Adım: Dry-run başarılıysa ve kullanıcı bilinçli olarak training denemesi yapmak istiyorsa şu komut çalıştırılır:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Beklenen sonuç: Script training öncesi aynı özet bilgileri basar ve güvenli limit mekanizmasını kontrol eder. `Training baslangic akisi` altında gerçek `train sample count`, `eval sample count`, `start_with_eval: False`, `skip_train_epoch: False`, `save_step: 1` ve `save_checkpoints: True` görünür. `max_steps` veya doğrulanmış `epochs/num_epochs` sınırı uygulanabiliyorsa Coqui trainer süreci başlar. Güvenli limit uygulanamıyorsa `Bu coqui-tts/trainer sürümünde max_steps güvenli şekilde uygulanamıyor. Eğitim başlatılmadı.` veya `Epoch fallback config üzerinde doğrulanamadı. Eğitim başlatılmadı.` hatasıyla durur. CUDA OOM olursa `-BatchSize 1` korunmalı ve gerekirse `-GradAccum` artırılmalıdır. Training çıktıları ve checkpointler GitHub'a eklenmez.

## 12. Gerçek training sonrası checkpoint kontrolü

Adım: Gerçek training komutu bittikten sonra terminal çıktısını ve `experiments/baglare-xtts-exp01/training_output/` klasörünü kontrol et.

Beklenen sonuç: Script `training_output/` altında `.pth`, `.pt`, `.ckpt`, `.safetensors` veya checkpoint benzeri artifact arar. Artifact bulunursa yolları terminale yazar ve `Training completed and checkpoint artifacts were found.` mesajı görünür.

## 13. Checkpoint yoksa başarısız sayılması

Adım: Training sonunda checkpoint artifact oluşmazsa terminal çıktısını kontrol et.

Beklenen sonuç: `Training finished but no checkpoint artifact was found.` mesajı görünür ve script exit code `1` ile biter. `EPOCH: 0/0`, yalnızca eval çalışması veya checkpoint üretmeyen akış başarı sayılmaz. Hata çıktısında `start_with_eval=False` kullanılması, epoch fallback değerinin kontrol edilmesi ve `save_step` değerinin `1` tutulması önerilir.

## 14. Fine-tuned checkpoint inference testi

Adım: Training başarıyla checkpoint ürettiyse şu komut çalıştırılır:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_finetuned.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Text "Merhaba, bu ilk fine-tuned testidir."
```

Beklenen sonuç: Script `training_output/` altında `best_model.pth`, en yeni `best_model_*.pth` veya en yüksek numaralı `checkpoint_*.pth` adaylarından birini seçer ve seçilen checkpoint yolunu terminale yazar. Speaker wav olarak varsa `profiles/baglare/preprocessed_reference.wav`, yoksa `samples/my_voice.wav`, o da yoksa dataset içinden kısa bir WAV kullanılır. Fine-tuned inference başarılıysa `outputs/finetuned_eval/finetuned_test.wav` oluşur. Base output başarılıysa `outputs/finetuned_eval/base_test.wav` oluşur; base output başarısızlığı fine-tuned denemeyi engellemez. Rapor `outputs/reports/finetuned_eval_report.json` dosyasına yazılır. Bu test deneysel pipeline kontrolüdür; `best_model.pth` kalite garantisi değildir. Base ve fine-tuned çıktı dinlenerek karşılaştırılmalı, küçük dataset nedeniyle ses benzerliğinin sınırlı olabileceği unutulmamalıdır.

## 15. Gradio demo açılış testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Beklenen sonuç: Local Gradio arayüzü açılır. Demo public paylaşım açmadan local makinede çalışır.

## 16. Web üzerinden profil oluşturma testi

Adım: Gradio arayüzünde profil adı gir, referans ses yükle, izin checkbox'ını işaretle ve `Profil oluştur` butonuna bas.

Beklenen sonuç: Yeni profil `profiles/<profile_slug>/` altında oluşur. Profil klasöründe `original_reference.wav`, `preprocessed_reference.wav` ve `profile.json` bulunur.

## 17. Profil seçme testi

Adım: Gradio profil dropdown'ından oluşturulan profili seç.

Beklenen sonuç: Seçili profil üretimde öncelikli referans olur. Profil seçiliyken ayrıca yüklenen ses dosyası kullanılmaz.

## 18. Profil dropdown yenileme testi

Adım: Yeni profil oluşturduktan sonra profil dropdown'ının güncellenip güncellenmediğini kontrol et.

Beklenen sonuç: Dropdown yeni profili gösterir ve mümkünse yeni profil seçili hale gelir.

## 19. Seçili profil yenileme testi

Adım: Gradio'da bir profil seç ve `Seçili profili yenile` butonuna bas.

Beklenen sonuç: `original_reference.wav` korunur, `preprocessed_reference.wav` yeniden üretilir ve `profile.json` içindeki kalite bilgisi güncellenir. XTTS modeli yüklenmez ve ses üretimi yapılmaz.

## 20. Seçili profil silme testi

Adım: Gradio'da bir profil seç, silme onay checkbox'ını işaretle ve `Seçili profili sil` butonuna bas.

Beklenen sonuç: `profiles/<profile_slug>/` klasörü silinir, dropdown güncellenir, seçim temizlenir ve `profiles/.gitkeep` dosyası korunur. Onay checkbox'ı işaretli değilse silme yapılmaz.

## 21. Varsayılan referans ses testi

Adım: Profil seçmeden ve harici ses yüklemeden kısa bir metinle üretim denemesi yap.

Beklenen sonuç: Sistem varsayılan `samples/my_voice.wav` referansını kullanmaya çalışır. Dosya yoksa kullanıcıya anlaşılır hata gösterilir.

## 22. Harici referans ses yükleme testi

Adım: Profil seçmeden Gradio üzerinden açık izinli bir referans ses yükle ve metin seslendir.

Beklenen sonuç: Yüklenen referans ses ön işlenir ve üretimde kullanılır.

## 23. İzin checkbox testi

Adım: İzin checkbox'ını işaretlemeden üretim veya profil oluşturma denemesi yap.

Beklenen sonuç: İşlem başlamaz ve kullanıcıdan ses üzerinde hakkı veya açık izni olduğunu onaylaması istenir.

## 24. Boş metin testi

Adım: Metin alanını boş bırakıp ses üretmeyi dene.

Beklenen sonuç: Ses üretimi başlamaz ve boş metin için anlaşılır uyarı verilir.

## 25. Kalite raporu testi

Adım: Profil oluşturma veya ses üretimi sonrasında kalite raporu alanını kontrol et.

Beklenen sonuç: Ham ve/veya ön işlenmiş referans için `GOOD`, `WARNING` veya `BAD` sonucu görünür. Rapor teknik sinyal verir; nihai kalite dinlenerek kontrol edilir.

## 26. Çıktı dosyaları testi

Adım: Üretimden sonra local çıktı klasörlerini kontrol et.

Beklenen sonuç: Gradio çıktıları `outputs/gradio_outputs/`, kalite raporları `outputs/reports/`, ön işlenmiş referanslar `outputs/preprocessed_references/` altında tutulur.

## 27. GitHub'a gitmemesi gereken dosyalar kontrolü

Adım: GitHub Desktop değişiklik listesini kontrol et.

Beklenen sonuç: `samples/`, `profiles/`, `datasets/`, `experiments/`, `fine_tuned_models/`, `outputs/`, `.venv/`, gerçek ses dosyaları, checkpointler ve model dosyaları commit listesine girmez. Commit listesinde yalnızca kod, PowerShell runner, `.gitkeep` ve dokümantasyon değişiklikleri bulunur.
