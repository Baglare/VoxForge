# VF-001 — Mock TTS ve otomatik test altyapısı

Mock TTS **model eğitmez**, ses kaydı istemez ve gerçek insan sesi üretmez. 24 kHz mono, 16-bit PCM test tonları oluşturur. Dolayısıyla `PASS` sonucu telaffuz, doğallık, ses benzerliği veya XTTS/GPU uyumu için kanıt değildir.

## Kapsam

VF-001 depoya uygulanmıştır; ayrı kurulum adımı gerekmez. İlgili dosyalar:

- `scripts/mock_tts_engine.py`, `scripts/tts_engine_factory.py`, `scripts/mock_tts_smoke.py`
- `app/gradio_xtts_demo.py` (motor seçimi, model tembel yükleme, Mock modu)
- `scripts/audio_concat_utils.py` (eksik WAV parçası varsa kısaltılmış çıktı üretmek yerine hata)
- `tests/`

## Testler

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/mock_tts_smoke.py
```

Testlerde yalnızca Python standart kütüphanesi gerekir. Mock smoke sonucu `outputs/mock_smoke/sample.wav` altında oluşturulur.

## Gradio Mock modu

Mevcut Gradio ortamı zaten kuruluysa:

```powershell
$env:VOXFORGE_TTS_ENGINE = 'mock'
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Mock modunda **referans ses yüklemek veya izin onayı vermek gerekmez**. Metni girip `Ses üret` düğmesine basmak yeterlidir. Elde edilen WAV **insan konuşması değildir**. Profil işlemleri normal yerel profil araçları olarak durur; Mock üretim onları kullanmaz. Çok parçalı ses üretimi, birleştirme, raporlama ve normalize etme yolları sınanabilir.

Normal XTTS-v2 için önce ortam değişkenini kaldırın:

```powershell
Remove-Item Env:VOXFORGE_TTS_ENGINE -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File .\run_gradio_demo.ps1
```

Varsayılan motor **daima XTTS**. Model indirme, gerçek XTTS inference, CUDA kullanımı ve insan dinleme kabulü VF-001'in kapsamı dışındadır.

## Güvenlik notları

- `scripts/mock_tts_engine.py` referans ses dosyasını açmaz.
- Mock motoru ve testler kişisel kayıt, model veya checkpoint kullanmaz; `samples/`, `profiles/`, `datasets/`, `experiments/` ve `fine_tuned_models/` dokunulmaz.
- Uygulamadan kalıcı model seçimi yapılmaz. Mock modu ortam değişkeniyle açıkça seçilir.
- İlk deneme sonrası Windows üzerindeki gerçek Gradio açılışı henüz ayrıca doğrulanmalıdır.
