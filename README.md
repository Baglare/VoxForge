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
- Yerel voice profile sistemi ile sık kullanılan referans sesler `profiles/` altında saklanabilir.
- Gradio arayüzünde yerel ses profili dropdown'ı vardır.
- Ses üretimi için izin checkbox kontrolü vardır.
- Boş metin girişi kontrol edilir.
- FFmpeg ile referans ses ön işleme yapılır.
- Ön işlenmiş referanslar `outputs/preprocessed_references/` altına kaydedilir.
- Gradio ile üretilen sesler `outputs/gradio_outputs/` altına kaydedilir.
- Referans ses kalite analizi yapılır.
- Ham ve ön işlenmiş referans kalite raporları Gradio'da gösterilir.
- Kalite raporları `outputs/reports/gradio_quality_reports/` altına JSON olarak kaydedilir.

Proje sadece local çalışacak şekilde tasarlanmıştır. Public hosting, hesap sistemi, uzak API servisi veya bulut tabanlı ses depolama bu MVP kapsamında yoktur.

## 3. Temel özellikler

- Windows odaklı PowerShell çalıştırma dosyaları
- Coqui XTTS-v2 modeli ile Türkçe ses üretimi
- Referans ses ile zero-shot ses klonlama denemesi
- Local Gradio demo arayüzü
- Varsayılan referans ses kullanımı: `samples/my_voice.wav`
- Harici referans ses yükleme desteği
- Yerel voice profile oluşturma ve Gradio'da profil seçme desteği
- İzin checkbox kontrolü
- Boş metin kontrolü
- FFmpeg ile mono, 24000 Hz WAV referans hazırlama
- Ham ve ön işlenmiş referans kalite raporu
- Üretilen sesleri ve raporları local `outputs/` klasöründe tutma
- Hassas ses dosyalarını, voice profile dosyalarını ve çıktıları GitHub dışında bırakmaya uygun `.gitignore` yapısı

## 4. Kullanıcı akışı

1. Projeyi Windows makinede local olarak açın.
2. Python sanal ortamını oluşturun ve bağımlılıkları kurun.
3. Kendi sesinizi veya açık izinli bir referans sesi `samples/my_voice.wav` olarak ekleyin.
4. İlk terminal denemesini çalıştırarak XTTS akışını doğrulayın.
5. İsterseniz `profiles/` altında yerel bir voice profile oluşturun.
6. Gradio demosunu başlatın.
7. Türkçe metni girin.
8. Gradio'da bir yerel profil seçin veya profil seçmeden harici referans ses yükleyin.
9. Sesin size ait olduğunu veya kullanma izniniz olduğunu checkbox ile onaylayın.
10. Ses üretimini başlatın.
11. Üretilen sesi Gradio arayüzünde dinleyin.
12. Ham ve ön işlenmiş referans kalite raporlarını kontrol edin.
13. Local çıktı dosyalarını `outputs/` altında inceleyin.

## 5. Proje yapısı

```text
VoxForge/
|-- app/
|   `-- gradio_xtts_demo.py
|-- scripts/
|   |-- first_xtts_test.py
|   |-- create_voice_profile.py
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
|-- run_audio_quality_report.ps1
|-- run_compare_reference_quality.ps1
|-- docs/
|   |-- SETUP_WINDOWS.md
|   |-- VOICE_REFERENCE_GUIDE.md
|   `-- VOICE_PROFILES.md
|-- .gitignore
`-- README.md
```

Not: `samples/`, `profiles/` ve `outputs/` altındaki gerçek ses dosyaları GitHub'a yüklenmemelidir. Bu klasörler local çalışma verileri içindir. `profiles/.gitkeep`, boş `profiles/` klasörünün GitHub'da görünür kalması için tutulur.

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

Yerel voice profile oluşturma:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_voice_profile.ps1 -Name baglare -InputPath .\samples\my_voice.wav
```

Referans ses kalite raporu:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_audio_quality_report.ps1
```

Ham ve ön işlenmiş referans ses A/B karşılaştırması:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_compare_reference_quality.ps1
```

## 9. Referans ses hazırlama

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

## 10. Voice profile sistemi

Voice profile sistemi, sık kullanılan bir referans sesi yerel bir profil klasörü olarak saklama akışıdır. Bu sistem yeni bir model eğitmez ve fine-tuning yapmaz; mevcut aşama hâlâ reference-based / zero-shot voice cloning MVP'sidir.

Profil oluşturma komutu:

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

1. Seçili yerel profil
2. Yüklenen referans ses
3. Varsayılan `samples/my_voice.wav`

`profiles/` altındaki gerçek dosyalar GitHub'a yüklenmez. Bu klasörde kişisel ses kayıtları, ön işlenmiş sesler ve kalite metadata'sı bulunabilir; bunlar hem mahremiyet hem de repo boyutu nedeniyle local kalmalıdır. `.gitignore` içinde `profiles/*` kuralı gerçek profil içeriklerini dışarıda bırakır, `!profiles/.gitkeep` kuralı ise boş klasör niyetinin repoda görünmesini sağlar.

Güvenli ön işleme varsayılan olarak `safe_normalized` yaklaşımını kullanır. Bu yaklaşım sesi mono, 24000 Hz, `pcm_s16le` WAV formatına getirir ve ses seviyesini dengeler. Agresif sessizlik kırpma varsayılan akışta kullanılmaz; çünkü bazı referans sesleri fazla kısaltıp konuşmacı karakterini zayıflatabilir.

Daha ayrıntılı profil dokümanı için `docs/VOICE_PROFILES.md` dosyasına bakın.

## 11. Kalite raporu sistemi

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

## 12. Etik kullanım notu

VoxForge yalnızca kullanıcının kendi sesiyle veya açık izinli seslerle denenmelidir. Başka bir kişinin sesini izinsiz kopyalamak, taklit etmek, yayınlamak veya ticari amaçla kullanmak etik değildir ve hukuki risk oluşturabilir.

Gradio arayüzündeki izin checkbox'ı bu sınırı kullanıcıya açık şekilde hatırlatmak için vardır. Kullanıcı, ses üretmeden önce referans ses üzerinde hakkı veya açık izni olduğunu onaylamalıdır.

## 13. Bilinen sınırlamalar

- Proje şu anda Windows odaklıdır.
- Proje sadece local çalışma için tasarlanmıştır.
- Gerçek ses dosyaları GitHub'a yüklenmemelidir.
- `outputs/` klasörü üretilen sesleri ve raporları yerelde tutar; GitHub'a yüklenmez.
- `.venv/`, model dosyaları, cache dosyaları ve büyük çıktılar repoya dahil edilmez.
- Ses benzerliği garanti değildir; model, kayıt kalitesi, referans süresi ve metin içeriğine bağlıdır.
- Bu aşama zero-shot/reference-based voice cloning MVP'sidir.
- Fine-tuning henüz uygulanmamıştır; daha sonraki gelişmiş aşama olarak planlanmıştır.
- Yerel voice profile sistemi fine-tuning değildir; sadece referans ses seçimini ve tekrar kullanımını düzenler.
- Gradio demo local arayüzdür; ürünleşmiş bir web uygulaması değildir.
- Kalite raporu teknik sinyaller verir, nihai ses kalitesini tek başına garanti etmez.

## 14. Sonraki adımlar

Planlanan geliştirme yönleri:

- Windows kurulum deneyimini daha net hale getirmek
- Referans ses hazırlama notlarını güçlendirmek
- Voice profile silme, yenileme ve kalite geçmişi akışlarını eklemek
- Demo ekran görüntüleriyle portfolyo sunumunu desteklemek
- Daha ayrıntılı kalite metrikleri eklemek
- Farklı referans süreleriyle karşılaştırma yapmak
- Fine-tuning aşamasını ayrı ve daha gelişmiş bir hedef olarak değerlendirmek

## 15. Portfolyo değeri

VoxForge, local yapay zeka modeli kullanımı, Python tabanlı ses işleme, Gradio ile hızlı demo arayüzü, Windows PowerShell otomasyonu, FFmpeg tabanlı ses ön işleme ve hassas dosya yönetimi gibi alanlarda somut bir MVP örneği sunar.

Portfolyo açısından proje şu noktaları gösterir:

- Yerel AI model entegrasyonu
- Referans ses tabanlı Türkçe TTS denemesi
- Kullanıcı izni ve etik sınır kontrolü
- Ses ön işleme ve kalite raporu akışı
- Yerel voice profile yönetimi
- Üretilen dosyaları GitHub dışında tutan repo hijyeni
- MVP kapsamında sade ama çalışan bir local demo yapısı
