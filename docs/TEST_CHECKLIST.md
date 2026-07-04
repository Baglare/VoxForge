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

Beklenen sonuç: `recording_plan.csv` oluşur. Dosyada `clip_id|target_audio_path|text|status|notes` başlığı bulunur, ilk 80 kayıt metni listelenir ve tüm satırların `status` alanı `TODO` olur. Dosya zaten varsa üzerine yazılmaz.

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

Beklenen sonuç: Metadata satırları, WAV dosyaları, süre, sample rate, mono kanal ve kalite analizi kontrol edilir. Rapor `outputs/reports/finetune_dataset_report.json` dosyasına yazılır ve GitHub'a eklenmez.

## 6. Gradio demo açılış testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Beklenen sonuç: Local Gradio arayüzü açılır. Demo public paylaşım açmadan local makinede çalışır.

## 7. Web üzerinden profil oluşturma testi

Adım: Gradio arayüzünde profil adı gir, referans ses yükle, izin checkbox'ını işaretle ve `Profil oluştur` butonuna bas.

Beklenen sonuç: Yeni profil `profiles/<profile_slug>/` altında oluşur. Profil klasöründe `original_reference.wav`, `preprocessed_reference.wav` ve `profile.json` bulunur.

## 8. Profil seçme testi

Adım: Gradio profil dropdown'ından oluşturulan profili seç.

Beklenen sonuç: Seçili profil üretimde öncelikli referans olur. Profil seçiliyken ayrıca yüklenen ses dosyası kullanılmaz.

## 9. Profil dropdown yenileme testi

Adım: Yeni profil oluşturduktan sonra profil dropdown'ının güncellenip güncellenmediğini kontrol et.

Beklenen sonuç: Dropdown yeni profili gösterir ve mümkünse yeni profil seçili hale gelir.

## 10. Seçili profil yenileme testi

Adım: Gradio'da bir profil seç ve `Seçili profili yenile` butonuna bas.

Beklenen sonuç: `original_reference.wav` korunur, `preprocessed_reference.wav` yeniden üretilir ve `profile.json` içindeki kalite bilgisi güncellenir. XTTS modeli yüklenmez ve ses üretimi yapılmaz.

## 11. Seçili profil silme testi

Adım: Gradio'da bir profil seç, silme onay checkbox'ını işaretle ve `Seçili profili sil` butonuna bas.

Beklenen sonuç: `profiles/<profile_slug>/` klasörü silinir, dropdown güncellenir, seçim temizlenir ve `profiles/.gitkeep` dosyası korunur. Onay checkbox'ı işaretli değilse silme yapılmaz.

## 12. Varsayılan referans ses testi

Adım: Profil seçmeden ve harici ses yüklemeden kısa bir metinle üretim denemesi yap.

Beklenen sonuç: Sistem varsayılan `samples/my_voice.wav` referansını kullanmaya çalışır. Dosya yoksa kullanıcıya anlaşılır hata gösterilir.

## 13. Harici referans ses yükleme testi

Adım: Profil seçmeden Gradio üzerinden açık izinli bir referans ses yükle ve metin seslendir.

Beklenen sonuç: Yüklenen referans ses ön işlenir ve üretimde kullanılır.

## 14. İzin checkbox testi

Adım: İzin checkbox'ını işaretlemeden üretim veya profil oluşturma denemesi yap.

Beklenen sonuç: İşlem başlamaz ve kullanıcıdan ses üzerinde hakkı veya açık izni olduğunu onaylaması istenir.

## 15. Boş metin testi

Adım: Metin alanını boş bırakıp ses üretmeyi dene.

Beklenen sonuç: Ses üretimi başlamaz ve boş metin için anlaşılır uyarı verilir.

## 16. Kalite raporu testi

Adım: Profil oluşturma veya ses üretimi sonrasında kalite raporu alanını kontrol et.

Beklenen sonuç: Ham ve/veya ön işlenmiş referans için `GOOD`, `WARNING` veya `BAD` sonucu görünür. Rapor teknik sinyal verir; nihai kalite dinlenerek kontrol edilir.

## 17. Çıktı dosyaları testi

Adım: Üretimden sonra local çıktı klasörlerini kontrol et.

Beklenen sonuç: Gradio çıktıları `outputs/gradio_outputs/`, kalite raporları `outputs/reports/`, ön işlenmiş referanslar `outputs/preprocessed_references/` altında tutulur.

## 18. GitHub'a gitmemesi gereken dosyalar kontrolü

Adım: GitHub Desktop değişiklik listesini kontrol et.

Beklenen sonuç: `samples/`, `profiles/`, `datasets/`, `outputs/`, `.venv/` ve gerçek ses dosyaları commit listesine girmez. Commit listesinde yalnızca kod, PowerShell runner, `.gitkeep` ve dokümantasyon değişiklikleri bulunur.
