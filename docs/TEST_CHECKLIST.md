# VoxForge Manuel Test Checklist

Bu doküman, VoxForge MVP için hızlı manuel kontrol listesidir. Amaç, demo öncesinde ana akışların çalıştığını ve kişisel ses/veri dosyalarının GitHub'a girmediğini kontrol etmektir.

## 1. Smoke test

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_check.ps1
```

Beklenen sonuç: Terminalde `SMOKE CHECK PASSED` veya yalnızca uyarılar varsa `SMOKE CHECK PASSED WITH WARNINGS` görünür. Kritik eksik varsa `SMOKE CHECK FAILED` görünür ve düzeltilmeden demo yapılmaz.

## 2. Gradio demo açılış testi

Adım:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Beklenen sonuç: Local Gradio arayüzü açılır. Demo public paylaşım açmadan local makinede çalışır.

## 3. Web üzerinden profil oluşturma testi

Adım: Gradio arayüzünde profil adı gir, referans ses yükle, izin checkbox'ını işaretle ve `Profil oluştur` butonuna bas.

Beklenen sonuç: Yeni profil `profiles/<profile_slug>/` altında oluşur. Profil klasöründe `original_reference.wav`, `preprocessed_reference.wav` ve `profile.json` bulunur.

## 4. Profil seçme testi

Adım: Gradio profil dropdown'ından oluşturulan profili seç.

Beklenen sonuç: Seçili profil üretimde öncelikli referans olur. Profil seçiliyken ayrıca yüklenen ses dosyası kullanılmaz.

## 5. Profil yenileme testi

Adım: Yeni profil oluşturduktan sonra profil dropdown'ının güncellenip güncellenmediğini kontrol et.

Beklenen sonuç: Dropdown yeni profili gösterir ve mümkünse yeni profil seçili hale gelir.

## 6. Varsayılan referans ses testi

Adım: Profil seçmeden ve harici ses yüklemeden kısa bir metinle üretim denemesi yap.

Beklenen sonuç: Sistem varsayılan `samples/my_voice.wav` referansını kullanmaya çalışır. Dosya yoksa kullanıcıya anlaşılır hata gösterilir.

## 7. Harici referans ses yükleme testi

Adım: Profil seçmeden Gradio üzerinden açık izinli bir referans ses yükle ve metin seslendir.

Beklenen sonuç: Yüklenen referans ses ön işlenir ve üretimde kullanılır.

## 8. İzin checkbox testi

Adım: İzin checkbox'ını işaretlemeden üretim veya profil oluşturma denemesi yap.

Beklenen sonuç: İşlem başlamaz ve kullanıcıdan ses üzerinde hakkı veya açık izni olduğunu onaylaması istenir.

## 9. Boş metin testi

Adım: Metin alanını boş bırakıp ses üretmeyi dene.

Beklenen sonuç: Ses üretimi başlamaz ve boş metin için anlaşılır uyarı verilir.

## 10. Kalite raporu testi

Adım: Profil oluşturma veya ses üretimi sonrasında kalite raporu alanını kontrol et.

Beklenen sonuç: Ham ve/veya ön işlenmiş referans için `GOOD`, `WARNING` veya `BAD` sonucu görünür. Rapor teknik sinyal verir; nihai kalite dinlenerek kontrol edilir.

## 11. Çıktı dosyaları testi

Adım: Üretimden sonra local çıktı klasörlerini kontrol et.

Beklenen sonuç: Gradio çıktıları `outputs/gradio_outputs/`, kalite raporları `outputs/reports/`, ön işlenmiş referanslar `outputs/preprocessed_references/` altında tutulur.

## 12. GitHub'a gitmemesi gereken dosyalar kontrolü

Adım: GitHub Desktop değişiklik listesini kontrol et.

Beklenen sonuç: `samples/`, `profiles/`, `outputs/`, `.venv/` ve gerçek ses dosyaları commit listesine girmez. Commit listesinde yalnızca kod, PowerShell runner ve dokümantasyon değişiklikleri bulunur.
