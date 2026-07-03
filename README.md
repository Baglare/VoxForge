# VoxForge

## VoxForge nedir?

VoxForge, Windows üzerinde çalışan yerel bir Python ses üretim denemesidir. Proje, Coqui XTTS-v2 modeliyle Türkçe metinden ses üretmeyi ve bunu hem terminal testi hem de basit bir Gradio arayüzü üzerinden göstermeyi hedefler.

## Mevcut durum

Proje şu anda yerel ortamda çalışan iki ana akışa sahiptir:

- İlk minimum XTTS denemesi: `scripts/first_xtts_test.py`
- Yerel Gradio demosu: `app/gradio_xtts_demo.py`

Kişisel referans sesleri `samples/` klasörü altında, üretilen sesler ise `outputs/` klasörü altında tutulur. Gerçek ses dosyaları portfolyo reposuna dahil edilmez.

## Özellikler

- Windows odaklı PowerShell çalıştırma dosyaları
- Türkçe metinden ses üretimi denemesi
- Referans ses dosyasıyla XTTS-v2 kullanımı
- Gradio üzerinden yerel demo arayüzü
- Üretilen seslerin `outputs/` altında saklanması
- Kişisel ses ve çıktı dosyalarını GitHub dışında tutan `.gitignore` yapısı

## Proje yapısı

```text
VoxForge/
|-- app/
|   `-- gradio_xtts_demo.py
|-- scripts/
|   `-- first_xtts_test.py
|-- samples/
|   `-- .gitkeep
|-- outputs/
|   |-- .gitkeep
|   `-- gradio_outputs/
|       `-- .gitkeep
|-- requirements.txt
|-- run_first_xtts_test.ps1
|-- run_gradio_demo.ps1
|-- .gitignore
`-- README.md
```

## Kurulum notları

Bu proje Windows üzerinde yerel Python ortamı için hazırlanmıştır. `.venv/` gibi sanal ortam klasörleri GitHub'a yüklenmez.

Genel kurulum akışı:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

XTTS çalıştırmadan önce `samples/` klasörüne kendi sesinizi veya açık izinli bir referans ses dosyasını eklemeniz gerekir. Kişisel ses dosyaları repoya dahil edilmemelidir.

## Çalıştırma

İlk terminal testini çalıştırmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_first_xtts_test.ps1
```

Gradio demosunu çalıştırmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

## Kullanım akışı

1. Sanal ortamı oluşturun ve bağımlılıkları kurun.
2. `samples/` klasörüne kullanma hakkınız olan bir referans ses dosyası ekleyin.
3. İlk terminal testiyle modelin yerel ortamda çalıştığını doğrulayın.
4. Gradio demosunu başlatın.
5. Türkçe metni girin, referans sesi seçin ve izin onayını işaretleyin.
6. Üretilen ses çıktısını `outputs/` altında kontrol edin.

Sonuç sesleri `outputs/` ve Gradio akışında `outputs/gradio_outputs/` altına kaydedilir. Bu gerçek çıktı ses dosyaları GitHub'a yüklenmez.

## Etik kullanım notu

Bu proje yalnızca size ait olan seslerle veya açıkça kullanım izni verilmiş seslerle denenmelidir. Başka kişilerin sesini izinsiz şekilde kopyalamak, taklit etmek veya yayınlamak etik değildir ve hukuki risk oluşturabilir.

## Bilinen sınırlamalar

- Proje şu aşamada Windows odaklıdır.
- Model dosyaları, önbellekler ve büyük çıktı dosyaları repoya dahil edilmez.
- Kalite; referans sesin temizliğine, metne, donanıma ve modelin yerel çalışma koşullarına bağlıdır.
- Bu repo bir ürünleştirilmiş ses platformu değil, portfolyo amaçlı yerel bir demo projesidir.

## Sonraki adımlar

- Kurulum adımlarını farklı Windows makinelerinde doğrulamak
- Daha net hata mesajları ve kullanım notları eklemek
- Örnek ekran görüntüsü veya kısa demo çıktısı ile portfolyo sunumunu güçlendirmek
- Etik kullanım uyarılarını proje içinde daha görünür hale getirmek

## Portfolyo değeri

VoxForge; yerel yapay zeka modeli kullanımı, Python tabanlı demo geliştirme, Gradio arayüzü, dosya organizasyonu ve hassas veri yönetimi gibi alanlarda uygulanabilir bir örnek sunar. Ses dosyalarının repoya alınmaması, projenin portfolyo için daha temiz ve güvenli paylaşılmasını sağlar.
