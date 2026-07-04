# VoxForge

## 1. VoxForge nedir?

VoxForge, Windows üzerinde local çalışacak şekilde tasarlanmış bir Python ses üretim MVP'sidir. Proje, Coqui XTTS-v2 modeliyle referans ses kullanarak Türkçe metin seslendirme denemesi yapar.

Bu ilk aşama, zero-shot/reference-based voice cloning yaklaşımına odaklanır. Yani model, ayrıca eğitilmiş özel bir ses modeli olmadan, verilen referans ses dosyasından konuşma karakteristiği almaya çalışır. Fine-tuning daha sonraki gelişmiş aşama olarak planlanmıştır.

## 2. Mevcut MVP durumu

Mevcut durumda proje local ortamda çalışan bir MVP seviyesindedir:

- XTTS-v2 ile referans ses kullanarak Türkçe metin seslendirme yapılır.
- Terminalden çalışan ilk minimum XTTS deneme akışı vardır.
- Gradio tabanlı local web demo vardır.
- Varsayılan referans ses yolu `samples/my_voice.wav` olarak kullanılır.
- Kullanıcı Gradio arayüzünden harici referans ses yükleyebilir.
- Kullanıcı Gradio arayüzünden profil adı, referans ses ve izin checkbox'ı ile yeni yerel voice profile oluşturabilir.
- Yerel voice profile sistemi ile sık kullanılan referans sesler `profiles/` altında saklanabilir.
- Gradio arayüzünde yerel ses profili dropdown'ı vardır.
- Profil oluşturulduktan sonra dropdown güncellenir ve mümkünse yeni profil seçili hale gelir.
- Oluşturulan profil `profiles/<profile_slug>/` altında kalıcı kalır; Gradio kapanınca silinmez.
- Aynı profil adı tekrar kullanılırsa mevcut profilin üzerine yazılmaz.
- Gradio üretim önceliği açıkça şudur: seçili profil > yüklenen referans ses > varsayılan `samples/my_voice.wav`.
- Ses üretimi için izin checkbox kontrolü vardır.
- Boş metin girişi kontrol edilir.
- FFmpeg ile referans ses ön işleme yapılır.
- Ön işlenmiş referanslar `outputs/preprocessed_references/` altına kaydedilir.
- Gradio ile üretilen sesler `outputs/gradio_outputs/` altına kaydedilir.
- Referans ses kalite analizi yapılır.
- Ham ve ön işlenmiş referans kalite raporları Gradio'da gösterilir.
- Kalite raporları `outputs/reports/gradio_quality_reports/` altına JSON olarak kaydedilir.
- Fine-tuning'e geçmeden önce local dataset iskeleti oluşturma ve doğrulama desteği vardır.
- Deneysel XTTS GPT fine-tuning için dataset export ve kontrollü training başlatma altyapısı vardır; bu akış kalite garantisi vermez ve Gradio UI'a bağlı değildir.

Proje sadece local çalışacak şekilde tasarlanmıştır. Public hosting, hesap sistemi, uzak API servisi veya bulut tabanlı ses depolama bu MVP kapsamında yoktur.

## 3. Temel özellikler

- Windows odaklı PowerShell çalıştırma dosyaları
- Coqui XTTS-v2 modeli ile Türkçe ses üretimi
- Referans ses ile zero-shot ses klonlama denemesi
- Local Gradio demo arayüzü
- Varsayılan referans ses kullanımı: `samples/my_voice.wav`
- Harici referans ses yükleme desteği
- Web arayüzünden yerel voice profile oluşturma desteği
- Terminalden yerel voice profile oluşturma alternatifi
- Gradio'da profil seçme ve profil dropdown'ını güncelleme desteği
- İzin checkbox kontrolü
- Boş metin kontrolü
- FFmpeg ile mono, 24000 Hz WAV referans hazırlama
- Ham ve ön işlenmiş referans kalite raporu
- Yerel fine-tuning dataset hazırlığı ve doğrulama scriptleri
- Deneysel XTTS fine-tuning dataset export ve training runner scriptleri
- Üretilen sesleri ve raporları local `outputs/` klasöründe tutma
- Hassas ses dosyalarını, voice profile dosyalarını ve çıktıları GitHub dışında bırakmaya uygun `.gitignore` yapısı

## 4. Kullanıcı akışı

1. Projeyi Windows makinede local olarak açın.
2. Python sanal ortamını oluşturun ve bağımlılıkları kurun.
3. Kendi sesinizi veya açık izinli bir referans sesi `samples/my_voice.wav` olarak ekleyin.
4. İlk terminal denemesini çalıştırarak XTTS akışını doğrulayın.
5. Gradio demosunu başlatın.
6. İsterseniz web arayüzünden yeni bir yerel voice profile oluşturun.
7. Profil oluşturmak için profil adını girin, referans ses yükleyin, izin checkbox'ını işaretleyin ve `Profil oluştur` butonuna basın.
8. Oluşturulan profil `profiles/<profile_slug>/` altında kalıcı olarak saklanır; Gradio kapansa bile silinmez.
9. Türkçe metni girin.
10. Gradio'da bir yerel profil seçin veya profil seçmeden harici referans ses yükleyin.
11. Sesin size ait olduğunu veya kullanma izniniz olduğunu checkbox ile onaylayın.
12. Ses üretimini başlatın.
13. Üretilen sesi Gradio arayüzünde dinleyin.
14. Ham ve ön işlenmiş referans kalite raporlarını kontrol edin.
15. Local çıktı dosyalarını `outputs/` altında inceleyin.

## 5. Proje yapısı

```text
VoxForge/
|-- app/
|   `-- gradio_xtts_demo.py
|-- scripts/
|   |-- first_xtts_test.py
|   |-- create_voice_profile.py
|   |-- list_voice_profiles.py
|   |-- voice_profile_utils.py
|   |-- delete_voice_profile.py
|   |-- recreate_voice_profile.py
|   |-- init_finetune_dataset.py
|   |-- validate_finetune_dataset.py
|   |-- finetune_readiness_report.py
|   |-- export_xtts_finetune_dataset.py
|   |-- train_xtts_gpt_experiment.py
|   |-- generate_recording_plan.py
|   |-- build_metadata_from_recording_plan.py
|   |-- smoke_check.py
|   |-- prepare_reference_audio.py
|   |-- compare_reference_quality.py
|   |-- analyze_reference_audio.py
|   |-- audio_preprocessing_utils.py
|   `-- audio_quality_utils.py
|-- samples/
|   `-- .gitkeep
|-- profiles/
|   |-- .gitkeep
|   `-- <profile-slug>/
|       |-- original_reference.wav
|       |-- preprocessed_reference.wav
|       `-- profile.json
|-- datasets/
|   |-- .gitkeep
|   `-- <dataset-slug>/
|       |-- wavs/
|       |-- metadata.csv
|       `-- dataset_report.json
|-- experiments/
|   `-- .gitkeep
|-- fine_tuned_models/
|   `-- .gitkeep
|-- outputs/
|   |-- .gitkeep
|   |-- gradio_outputs/
|   |   `-- .gitkeep
|   |-- preprocessed_references/
|   |-- reports/
|   |   `-- gradio_quality_reports/
|   `-- ab_tests/
|-- requirements.txt
|-- run_first_xtts_test.ps1
|-- run_gradio_demo.ps1
|-- run_create_voice_profile.ps1
|-- run_list_voice_profiles.ps1
|-- run_delete_voice_profile.ps1
|-- run_recreate_voice_profile.ps1
|-- run_smoke_check.ps1
|-- run_audio_quality_report.ps1
|-- run_compare_reference_quality.ps1
|-- run_init_finetune_dataset.ps1
|-- run_validate_finetune_dataset.ps1
|-- run_finetune_readiness_report.ps1
|-- run_export_xtts_finetune_dataset.ps1
|-- run_train_xtts_experiment.ps1
|-- run_generate_recording_plan.ps1
|-- run_build_metadata.ps1
|-- docs/
|   |-- DEMO_SCRIPT.md
|   |-- FINE_TUNING_PREP.md
|   |-- RECORDING_TEXT_SET_TR.md
|   |-- SETUP_WINDOWS.md
|   |-- TEST_CHECKLIST.md
|   |-- VOICE_REFERENCE_GUIDE.md
|   |-- XTTS_FINETUNING_EXPERIMENT.md
|   `-- VOICE_PROFILES.md
|-- .gitignore
`-- README.md
```

Not: `samples/`, `profiles/`, `datasets/`, `experiments/`, `fine_tuned_models/` ve `outputs/` altındaki gerçek ses, dataset, checkpoint, model ve rapor dosyaları GitHub'a yüklenmemelidir. Bu klasörler local çalışma verileri içindir. `.gitkeep` dosyaları, boş klasör niyetinin GitHub'da görünür kalması için tutulur.

## 6. Kurulum notları

Proje Windows odaklıdır ve local Python sanal ortamı ile çalıştırılır. `.venv/` klasörü GitHub'a yüklenmez.

Genel kurulum akışı:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

XTTS modeli ilk çalıştırmada model dosyalarını indirebilir. Bu dosyalar büyük olabilir ve repoya dahil edilmez.

Referans ses için:

- Kişisel ses dosyaları repoya dahil edilmez.
- Varsayılan local referans yolu `samples/my_voice.wav` olarak kullanılır.
- Bu dosya kullanıcının kendi sesi veya açık izinli bir ses olmalıdır.

## 7. Windows / FFmpeg Shared notu

PowerShell çalıştırma dosyaları Windows için hazırlanmıştır. Scriptler, `Gyan.FFmpeg.Shared` paketiyle gelen `ffmpeg.exe` dosyasını yaygın WinGet kurulum dizinlerinde otomatik bulmaya çalışır.

Bulunursa FFmpeg klasörü geçici olarak `PATH` başına eklenir ve ilgili Python scripti bu ortamla çalıştırılır. Bulunamazsa mevcut `PATH` ile devam edilir.

Kalite analizi ve referans ses ön işleme için FFmpeg/FFprobe gereklidir. FFmpeg bulunamazsa kalite analizi eksik kalabilir veya ön işleme adımı başarısız olabilir.

## 8. Çalıştırma komutları

İlk minimum XTTS terminal denemesi:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_first_xtts_test.ps1
```

Local Gradio web demosu:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Hızlı smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_check.ps1
```

Terminalden yerel voice profile oluşturma (Gradio içindeki web akışına alternatif yöntem):

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_voice_profile.ps1 -Name baglare -InputPath .\samples\my_voice.wav
```

Yerel voice profile silme:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_delete_voice_profile.ps1 -Slug baglare -Yes
```

Yerel voice profile yenileme:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_recreate_voice_profile.ps1 -Slug baglare
```

Referans ses kalite raporu:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_audio_quality_report.ps1
```

Ham ve ön işlenmiş referans ses A/B karşılaştırması:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_compare_reference_quality.ps1
```

Fine-tuning dataset iskeleti oluşturma:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_init_finetune_dataset.ps1 -Name baglare_finetune_v1
```

Fine-tuning dataset doğrulama:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_validate_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Fine-tuning readiness report:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune_readiness_report.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Deneysel XTTS fine-tuning dataset export:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_xtts_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1 -RunName baglare_xtts_exp01
```

Deneysel XTTS training dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1 -DryRun
```

Deneysel XTTS training başlatma:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Fine-tuning kayıt planı üretme:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80
```

Kayıt planından metadata oluşturma:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_build_metadata.ps1 -Dataset .\datasets\baglare-finetune-v1
```

## 9. Test / kontrol

Hızlı smoke test, model ağırlıklarını yüklemeden ve ses üretmeden temel proje sağlığını kontrol eder. Dosya yapısını, `.gitignore` kurallarını, gerekli Python importlarını, FFmpeg/FFprobe erişimini ve yerel profil klasörlerini raporlar.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_check.ps1
```

Smoke test sonucu terminalde `SMOKE CHECK PASSED`, `SMOKE CHECK PASSED WITH WARNINGS` veya `SMOKE CHECK FAILED` olarak görünür. JSON raporu local olarak `outputs/reports/smoke_check_report.json` dosyasına yazılır; bu rapor GitHub'a yüklenmemelidir.

Manuel demo kontrol listesi için `docs/TEST_CHECKLIST.md` dosyasına bakın.

## 10. Referans ses hazırlama

Varsayılan referans ses yolu:

```text
samples/my_voice.wav
```

Gradio akışı, referans sesi XTTS için daha tutarlı hale getirmek amacıyla FFmpeg ile ön işler. Bu adımda ses mono WAV formatına ve 24000 Hz sample rate değerine dönüştürülür.

Ön işlenmiş referanslar şu klasöre kaydedilir:

```text
outputs/preprocessed_references/
```

Referans ses için pratik notlar:

- 30-90 saniye arası temiz ve doğal konuşma tercih edin.
- Arka plan müziği, klavye sesi ve ortam gürültüsünden kaçının.
- Tek kişinin konuştuğu kayıt kullanın.
- Mikrofona çok yakın konuşup clipping oluşturmayın.
- Ses benzerliği model davranışına, kayıt kalitesine ve referans süresine bağlıdır.

## 11. Voice profile sistemi

Voice profile sistemi, sık kullanılan bir referans sesi yerel bir profil klasörü olarak saklama akışıdır. Bu sistem yeni bir model eğitmez ve fine-tuning yapmaz; mevcut aşama hâlâ reference-based / zero-shot voice cloning MVP'sidir.

Gradio arayüzünden profil oluşturmak için kullanıcı şu alanları doldurur:

1. Profil adı
2. Referans ses dosyası
3. Ses üzerinde hakkı veya açık izni olduğunu onaylayan izin checkbox'ı

Bu alanlar tamamlandıktan sonra `Profil oluştur` butonuna basılır. Başarılı oluşturulan profil `profiles/<profile_slug>/` altında kalıcı olarak tutulur. Gradio kapatıldığında profil silinmez; sonraki çalıştırmada aynı klasörden tekrar listelenir.

Profil klasöründe şu dosyalar bulunur:

```text
profiles/<profile_slug>/
|-- original_reference.wav
|-- preprocessed_reference.wav
`-- profile.json
```

Profil oluşturulduktan sonra Gradio profil dropdown'ı güncellenir ve mümkünse yeni profil seçili hale gelir. Aynı profil adı tekrar oluşturulursa mevcut profilin üzerine yazılmaz; bu davranış kişisel referans seslerin yanlışlıkla değiştirilmesini engeller.

Terminalden profil oluşturma hâlâ alternatif yöntem olarak kullanılabilir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_voice_profile.ps1 -Name baglare -InputPath .\samples\my_voice.wav
```

Örnek profil yapısı:

```text
profiles/
|-- .gitkeep
`-- baglare/
    |-- original_reference.wav
    |-- preprocessed_reference.wav
    `-- profile.json
```

Bu yapıda `original_reference.wav`, giriş sesinin yerel kopyasıdır. `preprocessed_reference.wav`, XTTS'e verilecek güvenli ön işlenmiş referanstır. `profile.json` içinde profil adı, slug, dosya yolları, kalite raporları, seçilen ön işleme varyantı ve ön işleme uyarısı tutulur.

Gradio arayüzünde yerel ses profili dropdown'ı vardır. Bir profil seçilirse `profiles/<slug>/preprocessed_reference.wav` doğrudan `speaker_wav` olarak kullanılır. Bu durumda yüklenen ses dosyası bilinçli olarak yok sayılır; çünkü profil seçimi kullanıcının daha önce hazırlanmış, kalite raporu alınmış ve güvenli ön işlenmiş referansı kullanmak istediği anlamına gelir.

Profil seçilmezse mevcut davranış korunur. Referans öncelik sırası şudur:

1. Seçili profil
2. Yüklenen referans ses
3. Varsayılan ses: `samples/my_voice.wav`

`profiles/` altındaki gerçek dosyalar GitHub'a yüklenmez. Bu klasörde kişisel ses kayıtları, ön işlenmiş sesler ve kalite metadata'sı bulunabilir; bunlar hem mahremiyet hem de repo boyutu nedeniyle local kalmalıdır. `.gitignore` içinde `profiles/*` kuralı gerçek profil içeriklerini dışarıda bırakır, `!profiles/.gitkeep` kuralı ise boş klasör niyetinin repoda görünmesini sağlar.

Seçili profil Gradio arayüzünden veya terminal komutlarıyla yönetilebilir. Profil yenileme işlemi `original_reference.wav` dosyasını korur, `preprocessed_reference.wav` dosyasını safe preprocessing ile yeniden üretir ve `profile.json` içindeki kalite bilgisini günceller. Profil silme işlemi yalnızca ilgili `profiles/<slug>/` klasörünü kaldırır; `profiles/.gitkeep` dosyasına dokunmaz.

Güvenli ön işleme varsayılan olarak `safe_normalized` yaklaşımını kullanır. Bu yaklaşım sesi mono, 24000 Hz, `pcm_s16le` WAV formatına getirir ve ses seviyesini dengeler. Agresif sessizlik kırpma varsayılan akışta kullanılmaz; çünkü bazı referans sesleri fazla kısaltıp konuşmacı karakterini zayıflatabilir.

Daha ayrıntılı profil dokümanı için `docs/VOICE_PROFILES.md` dosyasına bakın.

## 12. Fine-tuning hazırlığı

Bu projede fine-tuning ürünleşmiş bir kullanıcı özelliği değildir. Mevcut ana çalışma hâlâ reference-based / zero-shot voice cloning MVP'sidir. Bu bölüm local dataset hazırlığını anlatır; dataset export ve deneysel training başlatma akışı bir sonraki bölümde ayrı olarak açıklanır.

Boş dataset iskeleti oluşturmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_init_finetune_dataset.ps1 -Name baglare_finetune_v1
```

Bu komut `datasets/<dataset_slug>/`, `datasets/<dataset_slug>/wavs/` ve başlığı `audio_path|text` olan boş `metadata.csv` dosyasını oluşturur. Aynı dataset zaten varsa üzerine yazmaz.

Dataset doğrulamak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_validate_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Doğrulama scripti `metadata.csv` satırlarını, WAV dosyalarını, süreyi, sample rate değerini, mono kanal durumunu ve kalite analizini kontrol eder. Fine-tuning datasetindeki tekil klipler referans ses profili gibi 30-90 saniye beklenmez; çoğu klip 2.5-15 saniye aralığında olabilir ve datasetin toplam süresi ayrıca raporlanır. Rapor local olarak `outputs/reports/finetune_dataset_report.json` dosyasına yazılır; bu rapor ve gerçek dataset dosyaları GitHub'a yüklenmez.

Readiness report, doğrulama özetinden datasetin teknik hazırlık seviyesini çıkarır. 10 dakikadan kısa ama hatasız datasetler `DATASET_VALID_BUT_SMALL` olarak raporlanır; bu, verinin teknik olarak geçerli olduğunu fakat gerçek fine-tuning kalitesi için daha fazla veri önerildiğini anlatır.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune_readiness_report.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Kayıt toplama için `docs/RECORDING_TEXT_SET_TR.md` içinde özgün Türkçe metinler bulunur. Bu metinlerden kayıt planı üretmek için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80
```

`recording_plan.csv`, Excel'de Türkçe karakterlerin bozulmadan ve sütunların düzgün ayrılarak açılabilmesi için `utf-8-sig` encoding ve noktalı virgül (`;`) delimiter ile yazılır. Dosya zaten varsa varsayılan olarak üzerine yazılmaz; bilinçli yeniden üretim için `-Overwrite` eklenebilir.

Kayıtlar tamamlandığında `recording_plan.csv` içindeki ilgili satırların `status` alanı `DONE` yapılır. DONE satırlardan `metadata.csv` oluşturmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_build_metadata.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Ayrıntılı hazırlık rehberi için `docs/FINE_TUNING_PREP.md` dosyasına bakın.

## 13. Deneysel XTTS fine-tuning

VoxForge artık mevcut geçerli datasetten deneysel XTTS-v2 GPT encoder fine-tuning denemesi için export ve training başlatma altyapısı içerir. Bu, ürünleşmiş veya kalite garantili bir fine-tuned model akışı değildir; amaç önce local training pipeline'ın başlayıp başlamadığını görmektir.

Export adımı kaynak dataset dosyalarını değiştirmez. `experiments/<run_slug>/dataset/` altında LJSpeech uyumlu `metadata_train.csv`, varsa `metadata_eval.csv`, WAV kopyaları ve `experiment_manifest.json` üretir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_xtts_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1 -RunName baglare_xtts_exp01
```

Training komutu kullanıcı çalıştırmadıkça eğitim başlamaz. Önce dry-run ile manifest, dataset dosyaları, checkpoint varlığı, CUDA, GPT trainer importları ve config oluşturma kontrolü yapılabilir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1 -DryRun
```

Dry-run başarılıysa terminalde `XTTS fine-tuning dry-run completed successfully` satırı görünür. Bu adım modeli eğitmez ve checkpoint indirme başlatmaz. Script, XTTS GPT recipe tarafında `GPTArgs`, `GPTTrainer` ve `GPTTrainerConfig` yolunu kullanır. Bazı `coqui-tts` sürümlerinde `XttsAudioConfig` farklı modülde bulunduğu için script fallback import kullanır ve import kaynağını terminalde yazar; desteklenmeyen config argümanlarını da uyarı olarak gösterir. Dry-run ayrıca `limit_mode`, `config.epochs`, `config.num_epochs`, `save_step`, `save_checkpoints` ve `save_n_checkpoints` bilgisini yazar. Gerçek training, güvenli bir `max_steps` veya doğrulanmış `epochs/num_epochs` sınırı bulunamazsa başlatılmaz.

Training başlatmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Training bittikten sonra script `training_output/` klasörünü recursive tarar. `.pth`, `.pt`, `.ckpt`, `.safetensors` veya checkpoint benzeri artifact bulunmazsa işlem başarısız sayılır ve exit code `1` ile biter. Mevcut yaklaşık 7.45 dakikalık dataset teknik olarak geçerlidir, ancak gerçek fine-tuning kalitesi için küçük kabul edilir. Deney çıktıları, checkpointler, trainer logları ve model dosyaları `experiments/` veya `fine_tuned_models/` altında local kalmalı ve GitHub'a yüklenmemelidir.

Ayrıntılı deney rehberi için `docs/XTTS_FINETUNING_EXPERIMENT.md` dosyasına bakın.

## 14. Kalite raporu sistemi

VoxForge, referans ses dosyası için basit bir kalite analizi üretir. Analiz; dosya varlığı, süre, sample rate, kanal sayısı, codec, ortalama ses seviyesi, maksimum ses seviyesi, clipping riski ve kısa/uzun kayıt uyarıları gibi bilgileri kontrol eder.

Terminal kalite raporu varsayılan olarak şu dosyaya yazılır:

```text
outputs/reports/reference_audio_report.json
```

Gradio akışı her üretim denemesinde ham referans ve ön işlenmiş referans raporunu arayüzde gösterir. Aynı raporlar JSON olarak şu klasöre kaydedilir:

```text
outputs/reports/gradio_quality_reports/
```

Kalite sonucu `GOOD`, `WARNING` veya `BAD` olabilir. `BAD` sonuç, her zaman ses üretiminin teknik olarak imkansız olduğu anlamına gelmez; ancak çıktının mutlaka dinlenerek kontrol edilmesi gerektiğini gösterir.

## 15. Etik kullanım notu

VoxForge yalnızca kullanıcının kendi sesiyle veya açık izinli seslerle denenmelidir. Başka bir kişinin sesini izinsiz kopyalamak, taklit etmek, yayınlamak veya ticari amaçla kullanmak etik değildir ve hukuki risk oluşturabilir.

Gradio arayüzündeki izin checkbox'ı bu sınırı kullanıcıya açık şekilde hatırlatmak için vardır. Kullanıcı, ses üretmeden önce referans ses üzerinde hakkı veya açık izni olduğunu onaylamalıdır.

## 16. Bilinen sınırlamalar

- Proje şu anda Windows odaklıdır.
- Proje sadece local çalışma için tasarlanmıştır.
- Gerçek ses dosyaları GitHub'a yüklenmemelidir.
- Yerel voice profile klasörleri GitHub'a yüklenmemelidir.
- Yerel fine-tuning dataset, experiment ve model klasörleri GitHub'a yüklenmemelidir.
- `outputs/` klasörü üretilen sesleri ve raporları yerelde tutar; GitHub'a yüklenmez.
- `.venv/`, model dosyaları, cache dosyaları ve büyük çıktılar repoya dahil edilmez.
- Ses benzerliği garanti değildir; model, kayıt kalitesi, referans süresi ve metin içeriğine bağlıdır.
- Bu aşama zero-shot/reference-based voice cloning MVP'sidir.
- Fine-tuning ürünleşmiş bir kullanıcı özelliği değildir; yalnızca deneysel local XTTS training altyapısı vardır.
- Fine-tuning dataset exportu eğitim başlatmaz; training yalnızca kullanıcı ilgili komutu çalıştırırsa başlar.
- Yerel voice profile sistemi fine-tuning değildir; sadece referans ses seçimini ve tekrar kullanımını düzenler.
- Gradio demo local arayüzdür; ürünleşmiş bir web uygulaması değildir.
- Kalite raporu teknik sinyaller verir, nihai ses kalitesini tek başına garanti etmez.

## 17. Sonraki adımlar

Planlanan geliştirme yönleri:

- Windows kurulum deneyimini daha net hale getirmek
- Referans ses hazırlama notlarını güçlendirmek
- Voice profile kalite geçmişi akışını eklemek
- Demo anlatımı ve ekran görüntüleriyle portfolyo sunumunu desteklemek
- Daha ayrıntılı kalite metrikleri eklemek
- Farklı referans süreleriyle karşılaştırma yapmak
- Deneysel fine-tuning sonuçlarını base XTTS çıktılarıyla karşılaştırmak

## 18. Portfolyo değeri

VoxForge, local yapay zeka modeli kullanımı, Python tabanlı ses işleme, Gradio ile hızlı demo arayüzü, Windows PowerShell otomasyonu, FFmpeg tabanlı ses ön işleme ve hassas dosya yönetimi gibi alanlarda somut bir MVP örneği sunar.

Portfolyo açısından proje şu noktaları gösterir:

- Yerel AI model entegrasyonu
- Referans ses tabanlı Türkçe TTS denemesi
- Kullanıcı izni ve etik sınır kontrolü
- Ses ön işleme ve kalite raporu akışı
- Yerel voice profile yönetimi
- Fine-tuning öncesi local dataset hazırlığı ve doğrulama yaklaşımı
- Deneysel XTTS fine-tuning export, manifest ve training başlatma altyapısı
- Üretilen dosyaları GitHub dışında tutan repo hijyeni
- MVP kapsamında sade ama çalışan bir local demo yapısı
