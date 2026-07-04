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
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -BatchSize 2 -GradAccum 8 -DryRun
```

Beklenen sonuç: Experiment path, dataset path, train/eval sayıları, language, max steps, batch size, grad accumulation ve CUDA bilgisi görünür. Dataset klasörü, `metadata_train.csv`, varsa `metadata_eval.csv`, checkpoint dosyaları ve GPT trainer importları kontrol edilir. Config oluşturma başarılı olursa terminal sonunda `XTTS fine-tuning dry-run completed successfully` görünür. `-DryRun` kullanıldığı için training başlamaz, `load_tts_samples` çalıştırılmaz ve checkpoint indirme yapılmaz.

## 9. Deneysel XTTS training başlatma kontrolü

Adım: Dry-run başarılıysa ve kullanıcı bilinçli olarak training denemesi yapmak istiyorsa şu komut çalıştırılır:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -BatchSize 2 -GradAccum 8
```

Beklenen sonuç: Script training öncesi aynı özet bilgileri basar, eksik XTTS dosyalarını `experiments/baglare-xtts-exp01/checkpoints/` altına indirmeyi dener ve Coqui trainer sürecini başlatır. CUDA OOM olursa `-BatchSize 1` denenmelidir. Training çıktıları ve checkpointler GitHub'a eklenmez.

## 10. Gradio demo açılış testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Beklenen sonuç: Local Gradio arayüzü açılır. Demo public paylaşım açmadan local makinede çalışır.

## 11. Web üzerinden profil oluşturma testi

Adım: Gradio arayüzünde profil adı gir, referans ses yükle, izin checkbox'ını işaretle ve `Profil oluştur` butonuna bas.

Beklenen sonuç: Yeni profil `profiles/<profile_slug>/` altında oluşur. Profil klasöründe `original_reference.wav`, `preprocessed_reference.wav` ve `profile.json` bulunur.

## 12. Profil seçme testi

Adım: Gradio profil dropdown'ından oluşturulan profili seç.

Beklenen sonuç: Seçili profil üretimde öncelikli referans olur. Profil seçiliyken ayrıca yüklenen ses dosyası kullanılmaz.

## 13. Profil dropdown yenileme testi

Adım: Yeni profil oluşturduktan sonra profil dropdown'ının güncellenip güncellenmediğini kontrol et.

Beklenen sonuç: Dropdown yeni profili gösterir ve mümkünse yeni profil seçili hale gelir.

## 14. Seçili profil yenileme testi

Adım: Gradio'da bir profil seç ve `Seçili profili yenile` butonuna bas.

Beklenen sonuç: `original_reference.wav` korunur, `preprocessed_reference.wav` yeniden üretilir ve `profile.json` içindeki kalite bilgisi güncellenir. XTTS modeli yüklenmez ve ses üretimi yapılmaz.

## 15. Seçili profil silme testi

Adım: Gradio'da bir profil seç, silme onay checkbox'ını işaretle ve `Seçili profili sil` butonuna bas.

Beklenen sonuç: `profiles/<profile_slug>/` klasörü silinir, dropdown güncellenir, seçim temizlenir ve `profiles/.gitkeep` dosyası korunur. Onay checkbox'ı işaretli değilse silme yapılmaz.

## 16. Varsayılan referans ses testi

Adım: Profil seçmeden ve harici ses yüklemeden kısa bir metinle üretim denemesi yap.

Beklenen sonuç: Sistem varsayılan `samples/my_voice.wav` referansını kullanmaya çalışır. Dosya yoksa kullanıcıya anlaşılır hata gösterilir.

## 17. Harici referans ses yükleme testi

Adım: Profil seçmeden Gradio üzerinden açık izinli bir referans ses yükle ve metin seslendir.

Beklenen sonuç: Yüklenen referans ses ön işlenir ve üretimde kullanılır.

## 18. İzin checkbox testi

Adım: İzin checkbox'ını işaretlemeden üretim veya profil oluşturma denemesi yap.

Beklenen sonuç: İşlem başlamaz ve kullanıcıdan ses üzerinde hakkı veya açık izni olduğunu onaylaması istenir.

## 19. Boş metin testi

Adım: Metin alanını boş bırakıp ses üretmeyi dene.

Beklenen sonuç: Ses üretimi başlamaz ve boş metin için anlaşılır uyarı verilir.

## 20. Kalite raporu testi

Adım: Profil oluşturma veya ses üretimi sonrasında kalite raporu alanını kontrol et.

Beklenen sonuç: Ham ve/veya ön işlenmiş referans için `GOOD`, `WARNING` veya `BAD` sonucu görünür. Rapor teknik sinyal verir; nihai kalite dinlenerek kontrol edilir.

## 21. Çıktı dosyaları testi

Adım: Üretimden sonra local çıktı klasörlerini kontrol et.

Beklenen sonuç: Gradio çıktıları `outputs/gradio_outputs/`, kalite raporları `outputs/reports/`, ön işlenmiş referanslar `outputs/preprocessed_references/` altında tutulur.

## 22. GitHub'a gitmemesi gereken dosyalar kontrolü

Adım: GitHub Desktop değişiklik listesini kontrol et.

Beklenen sonuç: `samples/`, `profiles/`, `datasets/`, `experiments/`, `fine_tuned_models/`, `outputs/`, `.venv/`, gerçek ses dosyaları, checkpointler ve model dosyaları commit listesine girmez. Commit listesinde yalnızca kod, PowerShell runner, `.gitkeep` ve dokümantasyon değişiklikleri bulunur.
