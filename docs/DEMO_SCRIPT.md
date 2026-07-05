# VoxForge Demo Anlatımı

Bu doküman, VoxForge'un teknik demo sırasında hangi sırayla anlatılacağını ve hangi davranışların gösterileceğini özetler. Anlatım, mevcut sistemi abartmadan ve kalite garantisi vermeden açıklamaya odaklanır.

## 1. Kısa tanıtım

VoxForge, Windows üzerinde yerel çalışan Python tabanlı bir Türkçe TTS deney aracıdır. Coqui XTTS-v2 modeliyle referans sese dayalı ses üretimi yapar, yerel voice profile yönetimi sağlar ve fine-tuning hazırlık/değerlendirme akışlarını local dosya sistemi üzerinde tutar.

Sistem public servis değildir. Ses kayıtları, profiller, datasetler, checkpointler ve üretilen çıktılar kullanıcının makinesinde kalır.

## 2. Demo kapsamı

Bu demo şu akışları gösterebilir:

1. Local Gradio arayüzünü başlatma
2. Yerel voice profile oluşturma
3. Var olan profili seçerek reference-based ses üretme
4. Referans ses kalite raporunu yorumlama
5. Üretilen dosyaların local klasörlerde kaldığını gösterme
6. İsteğe bağlı olarak fine-tuning deney durumunu doküman üzerinden özetleme

## 3. Gradio demosunu başlatma

Proje kök dizininde:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Beklenen davranış:

- Arayüz local olarak açılır.
- Public paylaşım veya uzak depolama kullanılmaz.
- Kullanıcı ses üretmeden önce izin checkbox'ını onaylamalıdır.

## 4. Yeni voice profile oluşturma

Gradio arayüzünde:

1. Profil adı girilir.
2. Referans ses dosyası yüklenir.
3. Ses üzerinde hak veya açık izin olduğunu belirten checkbox işaretlenir.
4. `Profil oluştur` butonuna basılır.

Oluşan yapı:

```text
profiles/<profile_slug>/
|-- original_reference.wav
|-- preprocessed_reference.wav
`-- profile.json
```

Bu profil local kalır. Gradio kapatılsa bile silinmez ve sonraki çalıştırmada tekrar listelenir.

## 5. Profil seçerek ses üretme

Referans seçim önceliği:

1. Seçili profil
2. Yüklenen referans ses
3. Varsayılan dosya: `samples/my_voice.wav`

Bir profil seçiliyse ayrıca yüklenen ses dosyası kullanılmaz. Çünkü profil daha önce hazırlanmış ve kalite raporu alınmış yerel referansı temsil eder.

Demo sırasında kısa ve anlaşılır bir Türkçe metin kullanılmalıdır. Çıktı Gradio arayüzünde dinlenir ve local çıktı klasörüne yazılır.

## 6. Kalite raporunu yorumlama

VoxForge, referans ses için teknik kalite raporu üretir. Rapor şu sinyalleri içerir:

- Dosya varlığı ve süre
- Sample rate
- Kanal sayısı
- Ortalama ve maksimum ses seviyesi
- Clipping riski
- Kayıt uzunluğu uyarıları

Sonuçlar:

- `GOOD`: Teknik olarak daha temiz görünen referans
- `WARNING`: Kullanılabilir, ancak dikkat edilmesi gereken durum var
- `BAD`: Daha temiz kayıtla tekrar denemek daha doğru olabilir

Kalite raporu nihai ses benzerliğini garanti etmez. Üretilen ses dinlenerek değerlendirilmelidir.

## 7. Yerel çıktı klasörleri

Demo sırasında oluşan dosyalar local klasörlerde tutulur:

```text
profiles/<profile_slug>/
outputs/gradio_outputs/
outputs/preprocessed_references/
outputs/reports/gradio_quality_reports/
```

Bu klasörlerdeki gerçek ses dosyaları, profil içerikleri ve rapor çıktıları GitHub'a eklenmemelidir.

## 8. Fine-tuning durumunu kısa anlatma

Fine-tuning anlatılacaksa demo sırasında eğitim başlatmak yerine mevcut durum dokümanından özet verilmelidir:

- Dataset yaklaşık 7.45 dakika / 80 örnektir.
- Training pipeline çalışmıştır.
- Checkpoint üretilmiştir.
- Fine-tuned checkpoint inference çalışmıştır.
- Matrix ve human evaluation yapılmıştır.
- Kalite artışı sınırlıdır.
- Daha iyi sonuç için daha fazla veri ve ayar denemesi gerekir.

Bu bölümde kalite garantisi verilmemelidir. Fine-tuning çıktısı teknik olarak değerlendirilmiş deney sonucu olarak anlatılmalıdır.

## 9. Etik kullanım notu

VoxForge yalnızca kullanıcının kendi sesiyle veya açık izinli seslerle kullanılmalıdır. Başka bir kişinin sesini izinsiz kopyalamak, taklit etmek, yayınlamak veya ticari amaçla kullanmak etik değildir ve hukuki risk oluşturabilir.

Gradio içindeki izin checkbox'ı bu sınırı kullanıcıya açık şekilde hatırlatır.

## 10. Kısa konuşma metni

VoxForge, Windows üzerinde local çalışan bir Türkçe TTS deney aracıdır. Sistem Coqui XTTS-v2 modelini kullanır ve referans sese dayalı ses üretimi yapar.

Bu bölümde kişiye özel yeni bir model eğitilmiyor. Seçilen referans ses, üretim sırasında XTTS'e `speaker_wav` olarak veriliyor. Kullanıcı isterse referans sesi tek seferlik yükleyebilir veya kalıcı bir yerel voice profile oluşturabilir.

Voice profile oluşturulduğunda sistem referansı local olarak saklar, ön işler ve kalite raporunu kaydeder. Sonraki üretimlerde kullanıcı dropdown üzerinden bu profili seçebilir. Profil seçiliyse sistem doğrudan profilin ön işlenmiş referansını kullanır.

Referans ses kalite raporu kayıt süresi, sample rate, kanal sayısı, ses seviyesi ve clipping riski gibi teknik sinyalleri gösterir. Bu rapor kalite garantisi değildir; üretilen ses yine dinlenerek değerlendirilir.

Fine-tuning tarafında deneysel bir pipeline bulunur. Yaklaşık 7.45 dakikalık 80 örnekli dataset ile training pipeline çalışmış, checkpoint üretilmiş ve fine-tuned inference denenmiştir. Sonuç teknik olarak doğrulanmıştır, ancak kalite artışı sınırlıdır ve daha fazla veri/ayar çalışması gerektirir.

Tüm kişisel sesler, profiller, datasetler, checkpointler ve üretilen çıktılar local klasörlerde kalır. GitHub'a gerçek ses dosyası, model checkpointi veya üretim çıktısı eklenmez.
