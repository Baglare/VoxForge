# VoxForge Windows Kurulum Notları

Bu doküman VoxForge'u Windows üzerinde local çalıştırmak için gereken temel ortam notlarını açıklar. Proje şu aşamada local MVP olarak tasarlanmıştır; `.venv`, model cache dosyaları, kişisel sesler ve üretilen çıktılar GitHub'a eklenmemelidir.

## 1. Python sanal ortam mantığı

Python sanal ortamı, bu projeye ait paketleri sistem Python kurulumundan ayırır. VoxForge gibi ses, model ve arayüz kütüphaneleri kullanan projelerde bu ayrım önemlidir; çünkü PyTorch, Coqui TTS, Gradio, Transformers ve TorchCodec gibi paketler sürüm uyumu ister.

Önerilen akış:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Bu işlemden sonra proje scriptleri doğrudan `.venv\Scripts\python.exe` üzerinden çalıştırılır.

## 2. `.venv` neden GitHub'a eklenmez?

`.venv` klasörü makineye özel ve büyüktür. İçinde işletim sistemi, Python sürümü, CPU/GPU seçimi ve yerel wheel dosyalarına bağlı paketler bulunur.

Bu yüzden:

- `.venv/` GitHub'a yüklenmez.
- Her geliştirici kendi makinesinde yeniden oluşturur.
- Paylaşılması gereken kaynak `requirements.txt` dosyasıdır.
- Model cache dosyaları ve büyük binary paketler repo dışında kalır.

## 3. PyTorch / CUDA notu

`requirements.txt` içinde mevcut ortam için PyTorch paketleri sabitlenmiştir:

```text
torch==2.12.1+cu126
torchaudio==2.11.0+cu126
torchvision==0.27.1+cu126
```

`+cu126` ifadesi CUDA 12.6 uyumlu wheel kullanıldığını gösterir. Bu, bilgisayarda NVIDIA GPU ve uygun sürücü varsa GPU kullanılabileceği anlamına gelir. GPU yoksa veya CUDA tarafı uygun değilse proje CPU ile çalışabilir; ancak XTTS üretimi daha yavaş olur.

Kontrol için:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`True` sonucu GPU'nun PyTorch tarafından görüldüğünü, `False` sonucu CPU akışına düşüleceğini gösterir.

## 4. Coqui TTS kurulumu

VoxForge, XTTS-v2 için `coqui-tts` paketini kullanır. Mevcut sabit sürüm:

```text
coqui-tts==0.27.5
```

Normal kurulumda tek tek paket kurmak yerine şu komut tercih edilir:

```powershell
pip install -r requirements.txt
```

Sorun giderme sırasında sadece Coqui TTS kontrolü yapmak için:

```powershell
.\.venv\Scripts\python.exe -c "from TTS.api import TTS; print('coqui-tts import OK')"
```

## 5. Transformers sürüm sabitleme notu

XTTS ve Coqui TTS tarafında `transformers` sürümü önemlidir. Bu projede mevcut sürüm sabitlenmiştir:

```text
transformers==4.57.6
```

`transformers` paketini tek başına yükseltmek veya düşürmek, Coqui TTS import hatalarına yol açabilir. Bu yüzden sürüm değişikliği yapılacaksa aynı anda `coqui-tts`, `torch`, `torchaudio`, `torchcodec` ve `huggingface_hub` uyumu da kontrol edilmelidir.

## 6. TorchCodec notu

Mevcut ortamda `torchcodec==0.14.0` kullanılır. TorchCodec, ses okuma/yazma tarafında PyTorch ve FFmpeg DLL'leriyle birlikte çalışır.

Windows üzerinde TorchCodec sorunları genellikle üç nedenden çıkar:

- PyTorch ve TorchCodec sürümleri uyumsuzdur.
- FFmpeg DLL'leri PATH içinde bulunamıyordur.
- PATH içinde yanlış FFmpeg dağıtımı daha önce geliyordur.

Kontrol için:

```powershell
.\.venv\Scripts\python.exe -c "import torchcodec; print('torchcodec import OK')"
```

## 7. FFmpeg Shared gereksinimi

Bu projede kalite analizi, referans ses ön işleme ve bazı ses kütüphaneleri için FFmpeg/FFprobe gerekir. Windows tarafında özellikle `Gyan.FFmpeg.Shared` dağıtımı tercih edilir.

Sebep:

- `ffmpeg.exe` ve `ffprobe.exe` komutlarını sağlar.
- TorchCodec gibi paketlerin ihtiyaç duyabileceği paylaşımlı DLL dosyalarını içerir.
- `ffmpeg essentials` kurulumu bazı durumlarda komutları sağlasa bile gerekli shared DLL yapısını sağlamayabilir.

Kontrol için:

```powershell
where.exe ffmpeg
where.exe ffprobe
```

## 8. PATH sırası problemi

Windows'ta aynı anda birden fazla FFmpeg kurulumu olabilir. Örneğin biri `ffmpeg essentials`, diğeri `Gyan.FFmpeg.Shared` olabilir.

`PATH` sırası önemlidir; çünkü Windows ilk bulduğu `ffmpeg.exe` dosyasını kullanır. Yanlış FFmpeg önce gelirse:

- `ffmpeg` komutu çalışıyor gibi görünür.
- TorchCodec yine de DLL hatası verebilir.
- Ses ön işleme farklı veya eksik davranabilir.

Bu yüzden sadece `where.exe ffmpeg` çıktısında bir FFmpeg görmek yeterli değildir. Hangi dağıtımın önce geldiği de kontrol edilmelidir.

## 9. PowerShell scriptleri neden FFmpeg Shared yolunu PATH başına ekliyor?

Projede yer alan PowerShell dosyaları yaygın WinGet dizinlerinde `Gyan.FFmpeg.Shared*` klasörlerini arar. Bulursa ilgili `ffmpeg.exe` dosyasının klasörünü geçici olarak `PATH` başına ekler.

Bu davranışın amacı:

- Ortamdaki yanlış FFmpeg kurulumunun önce seçilmesini engellemek
- TorchCodec ve FFmpeg kullanan scriptlerin daha tutarlı çalışmasını sağlamak
- Çalıştırma sırasında kullanılan Python ve FFmpeg yolunu terminalde görünür yapmak

Bu değişiklik kalıcı sistem PATH ayarı yapmaz; sadece o PowerShell çalıştırma oturumu için geçerlidir.

## 10. Kurulum sonrası kontrol komutları

Sanal ortam Python yolunu kontrol edin:

```powershell
.\.venv\Scripts\python.exe --version
```

Temel paket import kontrolü:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from TTS.api import TTS; import gradio; import transformers; print('imports OK')"
```

CUDA kontrolü:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

FFmpeg/FFprobe kontrolü:

```powershell
where.exe ffmpeg
where.exe ffprobe
```

Referans ses kalite raporu:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_audio_quality_report.ps1
```

İlk XTTS testi:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_first_xtts_test.ps1
```

Gradio demo:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

## 11. Sık görülen hatalar

### `isin_mps_friendly` import hatası

Bu hata genellikle PyTorch, Transformers veya Coqui TTS sürüm uyumsuzluğuna işaret eder. Tek bir paketi rastgele yükseltmek sorunu büyütebilir.

Adım adım kontrol:

1. Sanal ortamın aktif olduğundan emin olun.
2. `pip install -r requirements.txt` ile sabit sürümlere dönün.
3. `transformers==4.57.6` satırının değişmediğini kontrol edin.
4. Hata sürerse temiz `.venv` oluşturup bağımlılıkları yeniden kurun.

### `torchcodec` / `libtorchcodec` DLL hatası

Bu hata TorchCodec'in gerekli native DLL'leri bulamadığını veya PyTorch/TorchCodec uyumunun bozulduğunu gösterebilir.

Adım adım kontrol:

1. `torch`, `torchaudio`, `torchvision` ve `torchcodec` sürümlerinin `requirements.txt` ile aynı olduğunu kontrol edin.
2. `where.exe ffmpeg` çıktısında hangi FFmpeg'in önce geldiğine bakın.
3. `Gyan.FFmpeg.Shared` kurulumunun mevcut olduğundan emin olun.
4. Proje PowerShell scriptlerini kullanın; bu scriptler Shared FFmpeg yolunu geçici olarak PATH başına ekler.

### FFmpeg essentials vs shared karışıklığı

`ffmpeg essentials` bazı temel komutlar için yeterli görünebilir, fakat TorchCodec gibi kütüphaneler için gereken shared DLL yapısı eksik olabilir.

Adım adım kontrol:

1. `where.exe ffmpeg` komutunu çalıştırın.
2. İlk sıradaki yolun hangi FFmpeg dağıtımına ait olduğunu kontrol edin.
3. `Gyan.FFmpeg.Shared` yolu daha aşağıdaysa PowerShell scriptlerini kullanarak doğru yolu öne alın.
4. Gerekirse sistem PATH sırasını manuel olarak düzeltin.

### Gradio ve `huggingface-hub` sürüm uyumsuzluğu

Gradio, `gradio_client` ve `huggingface_hub` paketleri birlikte değişebilen paketlerdir. Mevcut sabit sürümler:

```text
gradio==5.50.0
gradio_client==1.14.0
huggingface_hub==0.36.2
```

Uyumsuzluk belirtisi olarak import hataları, eksik fonksiyon hataları veya demo başlatma sırasında beklenmeyen Python hataları görülebilir.

Adım adım kontrol:

1. Bu paketleri tek tek yükseltmeyin.
2. `pip install -r requirements.txt` ile sabit sürümlere dönün.
3. Hata devam ederse temiz `.venv` kurulumuyla tekrar deneyin.

## 12. Temiz kurulum için kısa kontrol listesi

1. Eski `.venv` klasörünü silin.
2. `python -m venv .venv` ile yeni ortam oluşturun.
3. `.\.venv\Scripts\Activate.ps1` ile ortamı aktif edin.
4. `python -m pip install --upgrade pip` komutunu çalıştırın.
5. `pip install -r requirements.txt` ile bağımlılıkları kurun.
6. `where.exe ffmpeg` ve `where.exe ffprobe` ile FFmpeg görünürlüğünü kontrol edin.
7. `samples/my_voice.wav` dosyasını local olarak ekleyin.
8. `run_audio_quality_report.ps1` ile referans sesi kontrol edin.
9. `run_first_xtts_test.ps1` ile ilk ses üretimini deneyin.
10. `run_gradio_demo.ps1` ile local web demosunu açın.
