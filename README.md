# VoxForge

## 1. VoxForge

VoxForge, Windows üzerinde yerel çalışan Python tabanlı bir Türkçe TTS ve voice profile deney aracıdır. Proje, Coqui XTTS-v2 modeliyle reference-based ses üretimi, yerel profil yönetimi, referans ses ön işleme, kalite raporu, fine-tuning hazırlığı ve deneysel XTTS GPT fine-tuning değerlendirme akışlarını aynı local-first çalışma düzeninde toplar.

Sistem public servis, uzak API veya bulut depolama katmanı içermez. Ses kayıtları, profiller, datasetler, checkpointler, model çıktıları ve raporlar kullanıcının makinesinde kalır.

## 2. Kısa açıklama

VoxForge iki ana kullanım alanını ayırır:

- **Reference-based XTTS üretimi:** Kişiye özel model eğitmeden, verilen referans sesi `speaker_wav` olarak kullanarak ses üretir.
- **Deneysel fine-tuning akışı:** Yerel dataset hazırlama, doğrulama, export, training runner, checkpoint kontrolü ve değerlendirme raporlarıyla XTTS GPT fine-tuning denemelerini izlenebilir hale getirir.

Reference-based üretim ve fine-tuning aynı şey değildir. Voice profile sistemi de model eğitmez; referans sesi local olarak düzenler, ön işler ve tekrar kullanılabilir hale getirir.

## 3. Mevcut durum

Mevcut sürüm teknik prototip seviyesinde local olarak çalışır:

- Local Gradio demo çalışır.
- XTTS-v2 ile reference-based / zero-shot Türkçe ses üretimi yapılır.
- Yerel voice profile oluşturma, seçme, yenileme ve silme akışları vardır.
- Referans sesler FFmpeg ile mono, 24000 Hz WAV formatına ön işlenir.
- Ham ve ön işlenmiş referanslar için kalite raporu üretilir.
- Gradio tarafında inference presetleri, uzun metin chunking, çıktı normalize ve A/B karşılaştırma kontrolleri bulunur.
- Fine-tuning dataset iskeleti, kayıt planı, metadata üretimi, validation ve readiness report akışları vardır.
- Gradio içinde experimental fine-tuning hazırlık paneli bulunur; gerçek training başlatmaz.
- Deneysel XTTS GPT fine-tuning pipeline çalıştırılmış, checkpoint üretilmiş ve checkpoint inference denenmiştir.
- Matrix evaluation, human evaluation scorecard, inference parameter sweep ve blind experiment comparison akışları vardır.

Deneysel fine-tuning sonucu kalite garantisi vermez. Mevcut yaklaşık 7.45 dakika / 80 örneklik dataset ile kalite artışı sınırlıdır; daha güçlü sonuçlar için daha fazla veri, daha tutarlı kayıt, checkpoint seçimi ve inference ayarı denemeleri gerekir.

## 4. Temel özellikler

- Türkçe metin için XTTS-v2 tabanlı ses üretimi
- Seçili profil, yüklenen referans ses ve varsayılan referans ses arasında net öncelik sırası
- Gradio üzerinden yerel voice profile oluşturma
- Terminalden profil oluşturma, listeleme, yenileme ve silme
- Referans ses için güvenli ön işleme
- `GOOD`, `WARNING`, `BAD` durumlarıyla teknik kalite raporu
- Inference presetleriyle kontrollü üretim denemeleri
- 220 karakter üstü metinlerde chunking ve final WAV birleştirme
- Final WAV için opsiyonel hafif normalize
- Aynı metinle A/B üretim karşılaştırması
- Fine-tuning dataset hazırlığı ve doğrulama
- Readiness report ile dataset hazırlık seviyesi
- Deneysel XTTS GPT fine-tuning export ve training runner
- Artifact temelli checkpoint başarı kontrolü
- Fine-tuned checkpoint inference
- Matrix evaluation ve manuel human scorecard
- Exp01 / Exp02 gibi deneyler için blind A/B karşılaştırma
- Yerel ses, model, dataset ve çıktı dosyalarını Git dışında tutan `.gitignore`

## 5. Sistem akışı

Reference-based kullanım akışı:

1. Proje Windows makinede açılır.
2. Python sanal ortamı hazırlanır ve bağımlılıklar kurulur.
3. Kullanıcı kendi sesi veya açık izinli bir referans ses sağlar.
4. Gradio demo başlatılır.
5. Kullanıcı mevcut voice profile seçer veya yeni referans ses yükler.
6. Kullanıcı üretim modunu, normalize seçeneğini ve gerekirse A/B karşılaştırmayı seçer.
7. Referans ses ön işlenir ve kalite raporu oluşturulur.
8. Türkçe metin girilir ve izin checkbox'ı onaylanır.
9. XTTS-v2 inference çalışır.
10. Uzun metinlerde chunking uygulanır ve final WAV birleştirilir.
11. Üretilen WAV ve raporlar `outputs/` altında local olarak saklanır.

Fine-tuning deney akışı:

1. Dataset klasörü oluşturulur.
2. Kayıt planı hazırlanır.
3. Tamamlanan kayıtlar `metadata.csv` dosyasına aktarılır.
4. Dataset validation ve readiness report çalıştırılır.
5. Dataset deney klasörüne export edilir.
6. Training için dry-run yapılır.
7. Gerçek training yalnızca kullanıcı komutuyla başlar.
8. Training sonunda checkpoint artifact aranır.
9. Checkpoint ile inference, matrix evaluation ve human evaluation yapılır.
10. Deneyler gerekirse blind A/B karşılaştırma ile dinlenir.

## 6. Yerel voice profile yönetimi

Voice profile sistemi, sık kullanılan referans sesleri local bir profil klasörü olarak saklar. Bu sistem yeni model eğitmez; referans ses seçimini, ön işlemeyi ve tekrar kullanımı düzenler.

Profil klasörü yapısı:

```text
profiles/<profile_slug>/
|-- original_reference.wav
|-- preprocessed_reference.wav
`-- profile.json
```

Gradio arayüzünden profil oluşturmak için kullanıcı profil adı girer, referans ses yükler ve ses üzerinde hakkı veya açık izni olduğunu checkbox ile onaylar. Profil oluşturulduktan sonra dropdown güncellenir. Aynı profil adı tekrar kullanılırsa mevcut profilin üzerine yazılmaz.

Bir profil seçildiğinde `profiles/<slug>/preprocessed_reference.wav` doğrudan XTTS `speaker_wav` olarak kullanılır. Bu durumda ayrıca yüklenen referans ses bilinçli olarak yok sayılır.

Profil yenileme işlemi `original_reference.wav` dosyasını korur, `preprocessed_reference.wav` dosyasını yeniden üretir ve `profile.json` içindeki kalite bilgisini günceller. Profil silme işlemi yalnızca ilgili profil klasörünü kaldırır; `profiles/.gitkeep` korunur.

## 7. Reference-based XTTS üretimi

Reference-based / zero-shot üretimde kişiye özel yeni bir model eğitilmez. XTTS-v2, verilen referans sesin konuşma karakteristiğini inference sırasında kullanmaya çalışır.

Referans seçim sırası:

1. Seçili yerel voice profile
2. Gradio üzerinden yüklenen referans ses
3. Varsayılan local dosya: `samples/my_voice.wav`

Referans ses için pratik koşullar:

- Tek kişinin konuştuğu temiz kayıt kullanılmalıdır.
- Arka plan müziği, ortam gürültüsü ve clipping olmamalıdır.
- 30-90 saniye arası doğal konuşma reference-based kullanım için daha tutarlı sonuç verebilir.
- Kalite raporu teknik sinyal verir; nihai ses kalitesi dinlenerek değerlendirilmelidir.

## 8. Gradio üretim kontrolleri

Gradio arayüzünde reference-based üretim için şu kontroller bulunur:

- `Üretim modu`, XTTS inference çağrısında kullanılan preset parametrelerini seçer.
- `Çıktı sesini normalize et`, final WAV için hafif FFmpeg normalize uygular; başarısız olursa ham çıktı korunur.
- `A/B karşılaştırma üret`, aynı metni seçili preset ve `Daha stabil` preset ile iki ayrı WAV olarak üretir.
- 220 karakter üstü metinlerde chunking otomatik uygulanır ve parçalar final WAV olarak birleştirilir.

Bu kontroller otomatik kalite ölçümü değildir. Ama farklı üretim ayarlarını kontrollü şekilde karşılaştırmayı ve uzun metinlerde sessiz kırpılma riskini azaltmayı sağlar.

## 9. Fine-tuning hazırlığı

Fine-tuning hazırlığı, yerel datasetin teknik olarak kullanılabilir olup olmadığını kontrol eder. Bu aşama tek başına training başlatmaz.

Hazırlık bileşenleri:

- Dataset klasörü ve `wavs/` alt klasörü
- `metadata.csv` dosyası
- Kayıt planı üretimi
- DONE kayıtlarından metadata oluşturma
- WAV dosyası, süre, kanal, sample rate ve clipping kontrolleri
- Toplam süre ve örnek sayısına göre readiness report

10 dakikadan kısa ama hatasız datasetler teknik olarak geçerli olabilir; ancak kalite beklentisi açısından küçük kabul edilir. Bu durumda readiness seviyesi `DATASET_VALID_BUT_SMALL` olarak raporlanabilir.

Gradio arayüzünde ayrıca `Experimental Fine-tuning Hazırlığı` paneli bulunur. Bu panel geçerli datasetleri listeler, readiness raporu çalıştırır, experiment export eder ve training scriptini yalnızca dry-run modunda çağırır. Panel gerçek training başlatmaz, checkpoint üretmez ve fine-tuned model seçimi yapmaz.

## 10. Deneysel fine-tuning ve değerlendirme

VoxForge içinde XTTS-v2 GPT encoder fine-tuning deneyi için export, dry-run, training, checkpoint doğrulama ve değerlendirme akışları bulunur.

Mevcut deney özeti:

- Dataset: yaklaşık 7.45 dakika / 80 örnek
- Training pipeline çalıştı.
- Checkpoint üretildi.
- Fine-tuned checkpoint inference çalıştı.
- Matrix evaluation yapıldı.
- Human evaluation scorecard oluşturuldu.
- Kalite artışı sınırlı kaldı.
- Uzun metinlerde chunking ve WAV birleştirme eklendi.

Training başarısı yalnızca trainer loglarına göre değerlendirilmez. Gerçek training sonunda `training_output/` altında `.pth`, `.pt`, `.ckpt`, `.safetensors` veya checkpoint benzeri artifact bulunmalıdır. Checkpoint yoksa işlem başarısız kabul edilir.

Blind experiment comparison, exp_a ve exp_b çıktılarını rastgele A/B dosya adlarıyla kopyalar ve kullanıcıya doldurulacak scorecard CSV üretir. Bu otomatik kalite ölçümü değildir; objektif metrikler yalnızca yardımcı sinyaldir ve nihai karar blind dinleme puanlarıyla verilmelidir.

## 11. Proje yapısı

```text
VoxForge/
|-- app/
|   `-- gradio_xtts_demo.py
|-- scripts/
|   |-- first_xtts_test.py
|   |-- create_voice_profile.py
|   |-- list_voice_profiles.py
|   |-- delete_voice_profile.py
|   |-- recreate_voice_profile.py
|   |-- voice_profile_utils.py
|   |-- init_finetune_dataset.py
|   |-- validate_finetune_dataset.py
|   |-- finetune_readiness_report.py
|   |-- export_xtts_finetune_dataset.py
|   |-- train_xtts_gpt_experiment.py
|   |-- evaluate_xtts_finetuned_checkpoint.py
|   |-- evaluate_xtts_checkpoint_matrix.py
|   |-- create_human_eval_report.py
|   |-- evaluate_xtts_inference_params.py
|   |-- compare_finetune_experiments.py
|   |-- text_chunking_utils.py
|   |-- audio_concat_utils.py
|   |-- prepare_reference_audio.py
|   `-- audio_quality_utils.py
|-- docs/
|   |-- DEMO_SCRIPT.md
|   |-- FINE_TUNING_PREP.md
|   |-- PROJECT_STATUS.md
|   |-- RECORDING_TEXT_SET_TR.md
|   |-- SETUP_WINDOWS.md
|   |-- TEST_CHECKLIST.md
|   |-- VOICE_PROFILES.md
|   |-- VOICE_REFERENCE_GUIDE.md
|   `-- XTTS_FINETUNING_EXPERIMENT.md
|-- samples/
|-- profiles/
|-- datasets/
|-- experiments/
|-- fine_tuned_models/
|-- outputs/
|-- requirements.txt
|-- run_*.ps1
|-- .gitignore
`-- README.md
```

`samples/`, `profiles/`, `datasets/`, `experiments/`, `fine_tuned_models/` ve `outputs/` altındaki gerçek çalışma dosyaları GitHub'a eklenmemelidir. Bu klasörlerde yalnızca `.gitkeep` gibi iskelet dosyaları repoda tutulur.

## 12. Kurulum

Proje Windows ve PowerShell odaklıdır.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

XTTS ilk çalıştırmada model dosyalarını indirebilir. Model cache dosyaları büyük olabilir ve repoya dahil edilmez.

FFmpeg/FFprobe, referans ses ön işleme ve kalite analizi için gereklidir. PowerShell runner dosyaları `Gyan.FFmpeg.Shared` kurulumunu yaygın WinGet dizinlerinde bulmaya çalışır; bulunursa ilgili klasör geçici olarak `PATH` başına eklenir.

## 13. Çalıştırma komutları

Temel kontroller:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_check.ps1
powershell -ExecutionPolicy Bypass -File .\run_first_xtts_test.ps1
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Voice profile yönetimi:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_voice_profile.ps1 -Name baglare -InputPath .\samples\my_voice.wav
powershell -ExecutionPolicy Bypass -File .\run_list_voice_profiles.ps1
powershell -ExecutionPolicy Bypass -File .\run_recreate_voice_profile.ps1 -Slug baglare
powershell -ExecutionPolicy Bypass -File .\run_delete_voice_profile.ps1 -Slug baglare -Yes
```

Referans ses kalite kontrolleri:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_audio_quality_report.ps1
powershell -ExecutionPolicy Bypass -File .\run_compare_reference_quality.ps1
```

Fine-tuning dataset hazırlığı:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_init_finetune_dataset.ps1 -Name baglare_finetune_v1
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80
powershell -ExecutionPolicy Bypass -File .\run_build_metadata.ps1 -Dataset .\datasets\baglare-finetune-v1
powershell -ExecutionPolicy Bypass -File .\run_validate_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1
powershell -ExecutionPolicy Bypass -File .\run_finetune_readiness_report.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Deneysel XTTS fine-tuning:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_export_xtts_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1 -RunName baglare_xtts_exp01
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1 -DryRun
powershell -ExecutionPolicy Bypass -File .\run_train_xtts_experiment.ps1 -Experiment .\experiments\baglare-xtts-exp01 -MaxSteps 300 -Epochs 1 -BatchSize 1 -GradAccum 16 -SaveStep 1
```

Fine-tuned checkpoint değerlendirme:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_finetuned.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Text "Merhaba, bu ilk fine-tuned testidir."
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_matrix.ps1 -Experiment .\experiments\baglare-xtts-exp01
powershell -ExecutionPolicy Bypass -File .\run_create_human_eval_report.ps1 -MatrixRoot .\outputs\finetuned_eval\matrix\<timestamp> -UseDefaultScores
powershell -ExecutionPolicy Bypass -File .\run_evaluate_xtts_inference_params.ps1 -Experiment .\experiments\baglare-xtts-exp01 -Variant checkpoint_71
powershell -ExecutionPolicy Bypass -File .\run_compare_finetune_experiments.ps1 -ExpA .\experiments\baglare-xtts-exp01 -ExpB .\experiments\baglare-xtts-exp02
```

Gradio üretim raporları `outputs/reports/gradio_quality_reports/` altında tutulur. Raporlarda kullanılan inference preset, A/B durumu, post-processing seçimi, chunking bilgisi ve çıktı yolları yer alır.

## 14. Yerel veri ve gizlilik

VoxForge yalnızca kullanıcının kendi sesiyle veya açık izinli seslerle kullanılmalıdır. Başka bir kişinin sesini izinsiz kopyalamak, taklit etmek, yayınlamak veya ticari amaçla kullanmak etik değildir ve hukuki risk oluşturabilir.

GitHub'a eklenmemesi gereken klasörler:

```text
samples/
profiles/
datasets/
outputs/
experiments/
fine_tuned_models/
```

Bu klasörlerde kişisel ses kayıtları, profile metadata'sı, dataset dosyaları, üretilen WAV dosyaları, kalite raporları, checkpointler ve model çıktıları bulunabilir. `.gitignore` bu klasörlerdeki gerçek dosyaları dışarıda bırakır; `.gitkeep` dosyaları ise boş klasör niyetini korur.

## 15. Bilinen sınırlamalar

- Proje Windows odaklıdır.
- Gradio demo local arayüzdür; ürünleşmiş web uygulaması değildir.
- Ses benzerliği referans kayda ve model davranışına bağlıdır.
- Kalite raporu nihai ses kalitesini otomatik ölçmez.
- Gradio presetleri, normalize ve A/B karşılaştırma üretim davranışını incelemek için kullanılır.
- Yerel voice profile sistemi fine-tuning değildir.
- Fine-tuning akışı deneysel ve local runner düzeyindedir.
- Küçük dataset ile üretilen checkpointlerde kalite artışı sınırlı olabilir.
- `best_model.pth` tek başına başarılı ses sonucu anlamına gelmez.
- Matrix ve human evaluation çıktıları insan dinlemesiyle yorumlanmalıdır.
- Uzun metin chunking, kırpılma riskini azaltır; parça geçişleri ayrıca dinlenerek kontrol edilmelidir.

## 16. Yol haritası

Sonraki teknik adımlar:

- Windows kurulum ve hata ayıklama notlarını sadeleştirmek
- Referans kayıt kalite yönergelerini güçlendirmek
- Voice profile kalite geçmişini daha izlenebilir hale getirmek
- Dataset büyütme ve kayıt çeşitliliği stratejisini netleştirmek
- Fine-tuning deneylerinde farklı checkpoint ve inference ayarlarını sistematik karşılaştırmak
- Matrix ve human evaluation raporlarını daha okunabilir hale getirmek
- Uzun metin chunking davranışını daha fazla örnekle test etmek

Güncel proje durumu için `docs/PROJECT_STATUS.md`, fine-tuning ayrıntıları için `docs/XTTS_FINETUNING_EXPERIMENT.md`, manuel test adımları için `docs/TEST_CHECKLIST.md` dosyalarına bakın.
