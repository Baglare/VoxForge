# VoxForge Demo Anlatımı

Bu doküman, VoxForge'u kısa bir portfolyo demosunda anlatmak için kullanılabilir. Amaç, projeyi teknik olarak abartmadan, mevcut MVP akışını net şekilde göstermektir.

## 1. Proje kısa tanıtımı

VoxForge, Windows üzerinde local çalışan bir Python ses üretim MVP'sidir. Coqui XTTS-v2 modeliyle referans ses kullanarak Türkçe metin seslendirme denemesi yapar.

Bu aşama fine-tuning değildir. Proje, reference-based / zero-shot voice cloning yaklaşımıyla çalışır; yani model, seçilen referans sesin konuşma karakteristiğini üretim sırasında kullanmaya çalışır.

## 2. Gradio demosunu başlatma

Proje kök dizininde Gradio demosu şu komutla başlatılır:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Demo local çalışır. Public hosting, hesap sistemi veya bulut tabanlı ses depolama bu MVP kapsamında yoktur.

## 3. Yeni profil oluşturma

Gradio arayüzünde yeni bir yerel voice profile oluşturmak için:

1. Profil adı girilir.
2. Referans ses dosyası yüklenir.
3. Ses üzerinde hak veya açık izin olduğunu belirten checkbox işaretlenir.
4. `Profil oluştur` butonuna basılır.

Oluşturulan profil `profiles/<profile_slug>/` altında saklanır. Profil klasöründe `original_reference.wav`, `preprocessed_reference.wav` ve `profile.json` bulunur.

## 4. Profil seçme

Profil oluşturulduktan sonra Gradio profil dropdown'ı güncellenir ve mümkünse yeni profil seçili hale gelir.

Demo sırasında referans seçimi şu öncelikle çalışır:

1. Seçili profil
2. Yüklenen referans ses
3. Varsayılan ses: `samples/my_voice.wav`

Bir profil seçiliyse yüklenen ses dosyası yok sayılır. Çünkü profil, daha önce hazırlanmış ve kalite raporu alınmış yerel referansı temsil eder.

## 5. Metin seslendirme

Seslendirmek istenen Türkçe metin Gradio arayüzündeki metin alanına yazılır. İzin checkbox'ı işaretlendikten sonra üretim başlatılır.

Üretilen ses Gradio arayüzünde dinlenebilir ve local çıktı klasöründe saklanır.

## 6. Kalite raporunu yorumlama

VoxForge, referans ses için kalite raporu üretir. Rapor; süre, sample rate, kanal sayısı, ses seviyesi ve clipping riski gibi teknik sinyalleri gösterir.

Sonuç `GOOD`, `WARNING` veya `BAD` olabilir:

- `GOOD`: Referans teknik olarak daha temiz görünüyor.
- `WARNING`: Kullanılabilir, ama dikkat edilmesi gereken bir durum var.
- `BAD`: Daha temiz bir kayıtla tekrar denemek daha doğru olabilir.

Kalite raporu nihai ses benzerliğini garanti etmez. Üretilen ses her zaman dinlenerek kontrol edilmelidir.

## 7. Çıktılar yerelde nereye kaydedilir?

Demo sırasında oluşan dosyalar local klasörlerde tutulur:

```text
profiles/<profile_slug>/
outputs/gradio_outputs/
outputs/preprocessed_references/
outputs/reports/gradio_quality_reports/
```

Bu klasörlerdeki gerçek ses dosyaları, profiller ve üretim çıktıları GitHub'a yüklenmemelidir.

## 8. Etik kullanım notu

VoxForge yalnızca kullanıcının kendi sesiyle veya açık izinli seslerle denenmelidir. Başka bir kişinin sesini izinsiz kopyalamak, taklit etmek, yayınlamak veya ticari amaçla kullanmak etik değildir ve hukuki risk oluşturabilir.

Gradio içindeki izin checkbox'ı bu sınırı kullanıcıya açık şekilde hatırlatmak için vardır.

## 9. Portfolyo videosu için kısa konuşma metni

Bu projede VoxForge adlı local bir ses üretim MVP'si geliştiriyorum. Proje Windows üzerinde Python ile çalışıyor ve Coqui XTTS-v2 modelini kullanarak Türkçe metni referans sese göre seslendirmeyi deniyor.

Buradaki önemli nokta, sistemin fine-tuning yapmaması. Yani kişiye özel yeni bir model eğitilmiyor. Bunun yerine reference-based, yani zero-shot voice cloning yaklaşımı kullanılıyor. Kullanıcı bir referans ses veriyor, model de bu referansı üretim sırasında kullanıyor.

Gradio arayüzünde artık local voice profile oluşturma akışı var. Kullanıcı profil adını giriyor, referans sesini yüklüyor, ses üzerinde hakkı veya açık izni olduğunu checkbox ile onaylıyor ve profil oluşturuyor. Bu profil `profiles` klasörü altında local olarak saklanıyor. Gradio kapatılsa bile profil silinmiyor.

Sonraki üretimlerde kullanıcı dropdown'dan bu profili seçebiliyor. Sistem referans seçiminde önce seçili profili, sonra yüklenen sesi, en son da varsayılan `samples/my_voice.wav` dosyasını kullanıyor. Böylece demo sırasında aynı referans sesi tekrar tekrar yüklemek gerekmiyor.

Proje ayrıca referans ses için kalite raporu üretiyor. Bu rapor, kaydın süresi, sample rate değeri, ses seviyesi ve clipping riski gibi teknik sinyalleri gösteriyor. Bu rapor nihai ses kalitesini garanti etmiyor, ama demo sırasında kaydın neden iyi veya sorunlu olabileceğini yorumlamayı kolaylaştırıyor.

Tüm profiller, referans sesler ve üretilen çıktılar local kalıyor. GitHub'a gerçek ses dosyası veya üretim çıktısı yüklenmiyor. Bu da projeyi hem portfolyo için daha temiz hem de kişisel ses verisi açısından daha kontrollü hale getiriyor.
