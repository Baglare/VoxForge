# VoxForge Proje Durumu

## 1. Mevcut sürüm özeti

VoxForge, Windows üzerinde yerel çalışan Python tabanlı bir Türkçe TTS deney aracıdır. Mevcut sürüm; XTTS-v2 ile reference-based ses üretimi, local Gradio demo, yerel voice profile yönetimi, referans ses ön işleme, kalite raporlama, Gradio üretim kontrolleri ve deneysel fine-tuning değerlendirme akışlarını içerir.

Proje local-first çalışır. Ses kayıtları, profiller, datasetler, deney klasörleri, checkpointler ve üretilen çıktılar kullanıcının makinesinde kalır. GitHub üzerinde yalnızca kaynak kod, runner dosyaları, dokümantasyon ve boş klasör niyetini koruyan `.gitkeep` dosyaları tutulmalıdır.

## 2. Tamamlanan bileşenler

- Local Gradio demo
- Reference-based / zero-shot XTTS-v2 ses üretimi
- Yerel voice profile oluşturma
- Voice profile seçme, yenileme ve silme
- Referans ses safe preprocessing
- Ham ve ön işlenmiş referans kalite raporu
- Gradio inference preset seçimi
- Gradio uzun metin chunking ve WAV birleştirme
- Final çıktı için opsiyonel normalize
- A/B üretim modu
- Fine-tuning dataset iskeleti oluşturma
- Kayıt planı üretimi
- DONE kayıtlarından metadata oluşturma
- Dataset validation
- Readiness report
- Deneysel XTTS GPT fine-tuning dataset export
- Training dry-run ve gerçek training runner
- Training sonrası checkpoint artifact kontrolü
- Fine-tuned checkpoint inference
- Matrix evaluation
- Human evaluation scorecard
- Inference parameter sweep
- Uzun metinler için chunking ve WAV birleştirme

## 3. Çalışan uçtan uca akışlar

Reference-based demo akışı:

1. Gradio demo başlatılır.
2. Kullanıcı yerel profil seçer veya referans ses yükler.
3. Referans ses ön işlenir.
4. Kullanıcı üretim presetini, normalize seçeneğini ve A/B karşılaştırmayı seçebilir.
5. Kalite raporu üretilir.
6. Türkçe metin XTTS-v2 ile seslendirilir.
7. Uzun metinlerde chunking uygulanır ve final WAV birleştirilir.
8. Üretilen WAV ve raporlar `outputs/` altında local kalır.

Voice profile akışı:

1. Profil adı ve açık izinli referans ses girilir.
2. Profil `profiles/<profile_slug>/` altında oluşturulur.
3. `original_reference.wav`, `preprocessed_reference.wav` ve `profile.json` saklanır.
4. Profil sonraki üretimlerde doğrudan referans olarak seçilebilir.

Fine-tuning hazırlık akışı:

1. Dataset klasörü oluşturulur.
2. Kayıt planı hazırlanır.
3. Tamamlanan kayıtlar metadata dosyasına aktarılır.
4. Dataset validation çalıştırılır.
5. Readiness report ile veri hazırlık seviyesi raporlanır.

Deneysel fine-tuning değerlendirme akışı:

1. Dataset deney formatına export edilir.
2. Training dry-run ile ortam ve config kontrol edilir.
3. Gerçek training kullanıcı komutuyla başlatılır.
4. Training sonunda checkpoint artifact aranır.
5. Fine-tuned checkpoint ile inference denenir.
6. Base ve fine-tuned çıktılar matrix evaluation ile karşılaştırılır.
7. Human evaluation scorecard ile manuel dinleme sonuçları kaydedilir.
8. Inference parameter sweep ile erken kesilme ve ayar etkisi incelenir.

## 4. Deneysel fine-tuning durumu

Mevcut deneyde kullanılan dataset yaklaşık 7.45 dakika / 80 örnek düzeyindedir. Dataset teknik olarak geçerli kabul edilmiştir; ancak gerçek fine-tuning kalitesi için küçük bir veri miktarıdır.

Deney durumu:

- Training pipeline çalıştı.
- Checkpoint üretildi.
- Fine-tuned inference çalıştı.
- Matrix evaluation yapıldı.
- Human evaluation scorecard oluşturuldu.
- Kalite artışı sınırlı kaldı.
- Uzun metinlerde chunking eklendi.
- Daha güçlü sonuçlar için daha fazla veri ve ayar denemesi gerekli.

Training başarısı trainer loglarına, dry-run sonucuna veya yalnızca eval çıktısına göre kabul edilmez. Gerçek training başarı kriteri, `training_output/` altında checkpoint artifact oluşmasıdır. Checkpoint yoksa training başarılı sayılmaz.

## 5. Değerlendirme sonuçları

Değerlendirme akışları base XTTS ve fine-tuned checkpoint çıktılarının karşılaştırmalı dinlenmesini hedefler. Matrix evaluation farklı checkpoint varyantlarını ve Türkçe test cümlelerini üretir; human evaluation scorecard ise manuel dinleme puanlarını kaydeder.

Mevcut yorum:

- Fine-tuned inference teknik olarak çalışmıştır.
- Base ve fine-tuned çıktılar karşılaştırılabilir hale gelmiştir.
- Bazı checkpointlerde sınırlı iyileşme gözlenebilir.
- Bazı varyantlarda robotiklik veya erken kesilme riski devam eder.
- `best_model.pth` kalite garantisi değildir.
- Human evaluation sonuçları küçük dataset bağlamında yorumlanmalıdır.

## 6. Bilinen sınırlamalar

- Proje Windows odaklıdır.
- Gradio demo yalnızca local çalışma içindir.
- Reference-based ses üretiminde ses benzerliği garanti değildir.
- Kalite raporu nihai ses kalitesini otomatik ölçmez.
- Gradio preset, normalize ve A/B kontrolleri kalite garantisi vermez; üretim davranışını daha kontrollü karşılaştırmak için kullanılır.
- Voice profile sistemi fine-tuning yapmaz.
- Fine-tuning akışı deneysel düzeydedir.
- Yaklaşık 7.45 dakikalık dataset kalite için sınırlıdır.
- Checkpoint üretimi teknik başarıdır; iyi ses kalitesi anlamına gelmez.
- Uzun metin chunking kırpılma riskini azaltır, doğal geçiş garantisi vermez.

## 7. Yerel dosya / gizlilik politikası

Aşağıdaki klasörlerdeki gerçek çalışma dosyaları GitHub'a eklenmemelidir:

```text
samples/
profiles/
datasets/
outputs/
experiments/
fine_tuned_models/
```

Bu klasörlerde kişisel ses kayıtları, ön işlenmiş referanslar, profile metadata'sı, dataset WAV dosyaları, trainer çıktıları, checkpointler, model dosyaları ve raporlar bulunabilir. `.gitignore` bu dosyaları dışarıda bırakacak şekilde düzenlenmiştir.

VoxForge yalnızca kullanıcının kendi sesiyle veya açık izinli seslerle kullanılmalıdır. Başka bir kişinin sesini izinsiz kullanmak etik değildir ve hukuki risk oluşturabilir.

## 8. Sonraki teknik adımlar

- Dataset süresini ve kayıt çeşitliliğini artırmak
- Referans kayıt kalitesi için daha net kabul kriterleri belirlemek
- Fine-tuning denemelerinde checkpoint, epoch ve inference ayarlarını sistematik karşılaştırmak
- Human evaluation raporlarını daha okunabilir özetlerle desteklemek
- Uzun metin chunking davranışını daha geniş metin setiyle test etmek
- Windows kurulum ve hata giderme dokümantasyonunu güncel tutmak
- Yerel veri klasörlerinin GitHub Desktop değişiklik listesine girmediğini düzenli kontrol etmek
