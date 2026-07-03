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
- İzin checkbox kontrolü
- Boş metin kontrolü
- FFmpeg ile mono, 24000 Hz WAV referans hazırlama
- Ham ve ön işlenmiş referans kalite raporu
- Üretilen sesleri ve raporları local `outputs/` klasöründe tutma
- Hassas ses dosyalarını GitHub dışında bırakmaya uygun `.gitignore` yapısı

## 4. Kullanıcı akışı

1. Projeyi Windows makinede local olarak açın.
2. Python sanal ortamını oluşturun ve bağımlılıkları kurun.
3. Kendi sesinizi veya açık izinli bir referans sesi `samples/my_voice.wav` olarak ekleyin.
4. İlk terminal denemesini çalıştırarak XTTS akışını doğrulayın.
5. Gradio demosunu başlatın.
6. Türkçe metni girin.
7. İsterseniz varsayılan `samples/my_voice.wav` yerine harici referans ses yükleyin.
8. Sesin size ait olduğunu veya kullanma izniniz olduğunu checkbox ile onaylayın.
9. Ses üretimini başlatın.
10. Üretilen sesi Gradio arayüzünde dinleyin.
11. Ham ve ön işlenmiş referans kalite raporlarını kontrol edin.
12. Local çıktı dosyalarını `outputs/` altında inceleyin.

## 5. Proje yapısı

```text
VoxForge/
|-- app/
|   `-- gradio_xtts_demo.py
|-- scripts/
|   |-- first_xtts_test.py
|   |-- prepare_reference_audio.py
|   |-- compare_reference_quality.py
|   |-- analyze_reference_audio.py
|   `-- audio_quality_utils.py
|-- samples/
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
|-- run_audio_quality_report.ps1
|-- run_compare_reference_quality.ps1
|-- .gitignore
`-- README.md
```

Not: `samples/` ve `outputs/` altındaki gerçek ses dosyaları GitHub'a yüklenmemelidir. Bu klasörler local çalışma verileri içindir.

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

## 10. Kalite raporu sistemi

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

## 11. Etik kullanım notu

VoxForge yalnızca kullanıcının kendi sesiyle veya açık izinli seslerle denenmelidir. Başka bir kişinin sesini izinsiz kopyalamak, taklit etmek, yayınlamak veya ticari amaçla kullanmak etik değildir ve hukuki risk oluşturabilir.

Gradio arayüzündeki izin checkbox'ı bu sınırı kullanıcıya açık şekilde hatırlatmak için vardır. Kullanıcı, ses üretmeden önce referans ses üzerinde hakkı veya açık izni olduğunu onaylamalıdır.

## 12. Bilinen sınırlamalar

- Proje şu anda Windows odaklıdır.
- Proje sadece local çalışma için tasarlanmıştır.
- Gerçek ses dosyaları GitHub'a yüklenmemelidir.
- `outputs/` klasörü üretilen sesleri ve raporları yerelde tutar; GitHub'a yüklenmez.
- `.venv/`, model dosyaları, cache dosyaları ve büyük çıktılar repoya dahil edilmez.
- Ses benzerliği garanti değildir; model, kayıt kalitesi, referans süresi ve metin içeriğine bağlıdır.
- Bu aşama zero-shot/reference-based voice cloning MVP'sidir.
- Fine-tuning henüz uygulanmamıştır; daha sonraki gelişmiş aşama olarak planlanmıştır.
- Gradio demo local arayüzdür; ürünleşmiş bir web uygulaması değildir.
- Kalite raporu teknik sinyaller verir, nihai ses kalitesini tek başına garanti etmez.

## 13. Sonraki adımlar

Planlanan geliştirme yönleri:

- Windows kurulum deneyimini daha net hale getirmek
- Referans ses hazırlama notlarını güçlendirmek
- Demo ekran görüntüleriyle portfolyo sunumunu desteklemek
- Daha ayrıntılı kalite metrikleri eklemek
- Farklı referans süreleriyle karşılaştırma yapmak
- Fine-tuning aşamasını ayrı ve daha gelişmiş bir hedef olarak değerlendirmek

## 14. Portfolyo değeri

VoxForge, local yapay zeka modeli kullanımı, Python tabanlı ses işleme, Gradio ile hızlı demo arayüzü, Windows PowerShell otomasyonu, FFmpeg tabanlı ses ön işleme ve hassas dosya yönetimi gibi alanlarda somut bir MVP örneği sunar.

Portfolyo açısından proje şu noktaları gösterir:

- Yerel AI model entegrasyonu
- Referans ses tabanlı Türkçe TTS denemesi
- Kullanıcı izni ve etik sınır kontrolü
- Ses ön işleme ve kalite raporu akışı
- Üretilen dosyaları GitHub dışında tutan repo hijyeni
- MVP kapsamında sade ama çalışan bir local demo yapısı
