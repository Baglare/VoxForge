# VoxForge

## 1. VoxForge nedir?

VoxForge, Windows üzerinde yerel çalışan Python tabanlı bir Türkçe TTS deney aracıdır. Proje, Coqui XTTS-v2 modeliyle referans sese dayalı ses üretimi, yerel voice profile yönetimi, referans ses ön işleme, kalite raporlama, Gradio üretim kontrolleri ve deneysel XTTS GPT fine-tuning akışlarını tek bir local çalışma düzeninde toplar.

Ana kullanım modeli local-first olarak tasarlanmıştır. Ses kayıtları, profiller, datasetler, deney çıktıları, checkpointler ve raporlar kullanıcının makinesinde kalır. Public hosting, hesap sistemi, uzak API servisi veya bulut tabanlı ses depolama bu kapsamda yoktur.

VoxForge kalite garantisi veren bir ses klonlama ürünü değildir. Model çıktısı; referans kayıt kalitesi, veri miktarı, metin içeriği, inference ayarları ve XTTS model davranışına bağlıdır.

## 2. Mevcut durum

Proje şu anda yerel kullanım için çalışan bir teknik prototip seviyesindedir:

- Local Gradio demo çalışır.
- XTTS-v2 ile reference-based / zero-shot Türkçe ses üretimi yapılır.
- Yerel voice profile oluşturma, seçme, yenileme ve silme akışları vardır.
- Referans sesler FFmpeg ile mono, 24000 Hz WAV formatına ön işlenir.
- Ham ve ön işlenmiş referanslar için kalite raporu üretilir.
- Gradio üretiminde inference preset seçimi, uzun metin chunking, çıktı normalize ve A/B karşılaştırma kontrolleri vardır.
- Fine-tuning dataset iskeleti oluşturma, metadata üretme, dataset doğrulama ve readiness report akışları vardır.
- Deneysel XTTS GPT fine-tuning için dataset export ve kontrollü training runner bulunur.
- Mevcut deneyde yaklaşık 7.45 dakika / 80 örnek ile training pipeline çalıştırılmış, checkpoint üretilmiş ve fine-tuned checkpoint inference denenmiştir.
- Matrix evaluation, human evaluation scorecard ve inference parameter sweep akışları eklenmiştir.
- Uzun Türkçe metinlerde chunking ve WAV birleştirme desteği vardır.

Deneysel fine-tuning sonucu teknik olarak çalışır durumdadır; ancak kalite artışı sınırlıdır. Daha güçlü sonuçlar için daha fazla veri, daha dengeli kayıt seti, checkpoint seçimi ve inference ayarı denemeleri gerekir.

## 3. Mühendislik kapsamı

VoxForge aşağıdaki teknik alanları kapsar:

- Windows PowerShell runner yapısı
- Python sanal ortamı ve local dosya sözleşmeleri
- Coqui XTTS-v2 model entegrasyonu
- Gradio tabanlı yerel kullanım arayüzü
- Referans ses ön işleme ve kalite analizi
- Kalıcı yerel voice profile yönetimi
- Fine-tuning dataset hazırlığı ve doğrulama
- Artifact temelli training başarı kontrolü
- Base / fine-tuned checkpoint karşılaştırması
- Yerel veri gizliliği ve GitHub hijyeni

## 4. Temel özellikler

- Türkçe metin için XTTS-v2 tabanlı ses üretimi
- Seçili profil > yüklenen referans ses > `samples/my_voice.wav` önceliği
- Gradio üzerinden yerel voice profile oluşturma
- Terminalden profil oluşturma, listeleme, yenileme ve silme
- Referans ses için safe preprocessing
- `GOOD`, `WARNING`, `BAD` durumlarıyla kalite raporu
- Gradio üretim modu presetleri: `Dengeli`, `Daha stabil`, `Daha doğal deneme`, `Daha uzun çıktı denemesi`
- Reference-based üretimde 220 karakter üstü metinler için chunking ve WAV birleştirme
- Final WAV için opsiyonel hafif normalize
- Aynı metinle seçili preset ve `Daha stabil` preset arasında A/B karşılaştırma
- Fine-tuning dataset klasörü ve kayıt planı üretimi
- Metadata üretimi ve dataset validation
- Readiness report ile dataset hazırlık seviyesi
- Deneysel XTTS GPT fine-tuning export ve training akışı
- Checkpoint üretimi zorunlu training başarı kriteri
- Fine-tuned checkpoint inference
- Matrix evaluation ve manuel human scorecard
- Inference parameter sweep
- Uzun metinlerde chunking ve final WAV birleştirme
- Yerel ses, model, dataset ve çıktı dosyalarını Git dışında tutan `.gitignore`

## 5. Sistem akışı

Temel reference-based kullanım akışı:

1. Proje Windows makinede açılır.
2. Python sanal ortamı hazırlanır ve bağımlılıklar kurulur.
3. Kullanıcının kendi sesi veya açık izinli bir referans ses local klasöre eklenir.
4. Gradio demo başlatılır.
5. Kullanıcı mevcut voice profile seçer veya yeni referans ses yükler.
6. Kullanıcı üretim modunu, çıktı normalize seçeneğini ve gerekirse A/B karşılaştırmayı seçer.
7. Referans ses ön işlenir ve kalite raporu üretilir.
8. Kullanıcı Türkçe metni girer ve izin checkbox'ını onaylar.
9. XTTS-v2 inference çalışır; uzun metinlerde metin parçalara bölünüp final WAV olarak birleştirilir.
10. Üretilen WAV ve raporlar `outputs/` altında local olarak saklanır.

Fine-tuning deney akışı:

1. Dataset klasörü oluşturulur.
2. Kayıt planı hazırlanır.
3. Tamamlanan kayıtlar metadata dosyasına aktarılır.
4. Dataset validation ve readiness report çalıştırılır.
5. Dataset deney klasörüne export edilir.
6. Training için dry-run yapılır.
7. Gerçek training yalnızca kullanıcı komutu çalıştırırsa başlar.
8. Training başarı kriteri olarak checkpoint artifact aranır.
9. Checkpoint ile inference, matrix evaluation ve human evaluation yapılır.

## 6. Yerel voice profile yönetimi

Voice profile sistemi, sık kullanılan referans sesleri local bir profil klasörü olarak saklar. Bu sistem yeni model eğitmez; sadece referans ses seçimini, ön işlemeyi ve tekrar kullanımı düzenler.

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

## 7. Reference-based ses üretimi

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

### Gradio üretim kontrolleri

Gradio arayüzünde reference-based üretim için güvenli kullanıcı kontrolleri bulunur:

- `Üretim modu`, XTTS inference çağrısında kullanılan preset parametrelerini seçer.
- `Çıktı sesini normalize et`, final WAV için hafif FFmpeg normalize uygular; başarısız olursa ham çıktı korunur.
- `A/B karşılaştırma üret`, aynı metni seçili preset ve `Daha stabil` preset ile iki ayrı WAV olarak üretir.
- 220 karakter üstü metinlerde chunking otomatik uygulanır ve parçalar final WAV olarak birleştirilir.

Bu kontroller kalite garantisi değildir. Ama farklı üretim ayarlarını kontrollü şekilde karşılaştırmayı ve uzun metinlerde sessiz kırpılma riskini azaltmayı sağlar.

## 8. Fine-tuning hazırlığı

Fine-tuning hazırlığı, yerel datasetin teknik olarak kullanılabilir olup olmadığını kontrol eder. Bu aşama tek başına training başlatmaz.

Hazırlık bileşenleri:

- Dataset klasörü ve `wavs/` alt klasörü
- `metadata.csv` dosyası
- Kayıt planı üretimi
- DONE kayıtlarından metadata oluşturma
- WAV dosyası, süre, kanal, sample rate ve clipping kontrolleri
- Toplam süre ve örnek sayısına göre readiness report

10 dakikadan kısa ama hatasız datasetler teknik olarak geçerli olabilir; ancak gerçek kalite beklentisi açısından küçük kabul edilir. Bu durumda readiness seviyesi `DATASET_VALID_BUT_SMALL` olarak raporlanabilir.

## 9. Deneysel fine-tuning ve değerlendirme

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

Bu deney kalite garantisi vermez. Küçük dataset, checkpoint seçimi, inference parametreleri ve kayıt çeşitliliği çıktıyı doğrudan etkiler.

## 10. Proje yapısı

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

## 11. Kurulum

Proje Windows ve PowerShell odaklıdır.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

XTTS ilk çalıştırmada model dosyalarını indirebilir. Model cache dosyaları büyük olabilir ve repoya dahil edilmez.

FFmpeg/FFprobe, referans ses ön işleme ve kalite analizi için gereklidir. PowerShell runner dosyaları `Gyan.FFmpeg.Shared` kurulumunu yaygın WinGet dizinlerinde bulmaya çalışır; bulunursa ilgili klasör geçici olarak `PATH` başına eklenir.

## 12. Çalıştırma komutları

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
```

Gradio üretim raporları `outputs/reports/gradio_quality_reports/` altında tutulur. Raporlarda kullanılan inference preset, A/B durumu, post-processing seçimi, chunking bilgisi ve çıktı yolları yer alır.

## 13. Yerel veri, gizlilik ve etik kullanım

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

## 14. Bilinen sınırlamalar

- Proje Windows odaklıdır.
- Gradio demo local arayüzdür; ürünleşmiş web uygulaması değildir.
- Ses benzerliği garanti değildir.
- Kalite raporu nihai ses kalitesini otomatik ölçmez.
- Gradio üretim presetleri, normalize ve A/B karşılaştırma kalite garantisi vermez; yalnızca üretim davranışını daha kontrollü incelemeyi sağlar.
- Yerel voice profile sistemi fine-tuning değildir.
- Fine-tuning akışı deneysel ve local runner düzeyindedir.
- Küçük dataset ile üretilen checkpointlerde kalite artışı sınırlı olabilir.
- `best_model.pth` tek başına kalite garantisi değildir.
- Matrix ve human evaluation çıktıları insan dinlemesiyle yorumlanmalıdır.
- Uzun metin chunking, kırpılma riskini azaltır; ses geçişlerinin tamamen doğal olacağını garanti etmez.

## 15. Yol haritası

Sonraki teknik adımlar:

- Windows kurulum ve hata ayıklama notlarını sadeleştirmek
- Referans kayıt kalite yönergelerini güçlendirmek
- Voice profile kalite geçmişini daha izlenebilir hale getirmek
- Dataset büyütme ve kayıt çeşitliliği stratejisini netleştirmek
- Fine-tuning deneylerinde farklı checkpoint ve inference ayarlarını sistematik karşılaştırmak
- Matrix ve human evaluation raporlarını daha okunabilir hale getirmek
- Uzun metin chunking davranışını daha fazla örnekle test etmek

Güncel proje durumu için `docs/PROJECT_STATUS.md`, fine-tuning ayrıntıları için `docs/XTTS_FINETUNING_EXPERIMENT.md`, manuel test adımları için `docs/TEST_CHECKLIST.md` dosyalarına bakın.
