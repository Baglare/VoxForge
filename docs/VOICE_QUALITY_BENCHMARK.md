# VF-Q1 — Ses kalitesi araştırması ve karşılaştırma

Amaç: Mevcut XTTS-v2 sesini ölçülebilir bir referansa dönüştürmek ve Türkçe ses klonlama için **Chatterbox Multilingual V3** (öncelikli) ile **VoxCPM2**'nin uygunluğunu aynı koşullarda ölçmek. Hedef donanım: RTX 4060 Laptop, 8 GB VRAM, Windows.

Bu aşama **araştırma altyapısıdır**:
- Yeni motorlar demo, Studio veya üretim kuyruğuna bağlanmaz. Entegrasyon kararı bu raporun sonucuna göre VF-009'da verilir.
- Mevcut modeller, profiller, kayıtlar ve Studio projeleri değiştirilmez.
- Referans ses, koşu klasörüne **kopyalanır**.
- Her motor kendi sanal ortamında çalışır. Ana `.venv` (XTTS/Gradio) değişmez.

## 1. Adaylar (2026-11-04 itibarıyla kaynaklardan)

| | XTTS-v2 (mevcut) | Chatterbox Multilingual V3 | VoxCPM2 |
|---|---|---|---|
| Kaynak | Coqui | Resemble AI, 10 Haziran 2026 | OpenBMB, Nisan 2026 |
| Boyut | — | 0.5B (Llama tabanı, V2 ile aynı omurga) | 2B, BF16 |
| Türkçe | var | var (`tr`); Resemble blogu Türkçe CER'ini %1–5 bandında, "üretime hazır" olarak veriyor | var (30 dil listesinde Turkish) |
| Klonlama | referans ses | referans ses (`audio_prompt_path`) | yalnız referans ("controllable") veya referans + transkript ("ultimate") |
| Çıkış | 24 kHz | `model.sr` | 48 kHz |
| VRAM | 8 GB'a sığıyor (mevcut) | belirtilmemiş; 0.5B olduğu için sığması beklenir | **~8 GB** (README). 8 GB laptop GPU'da sınırda; OOM olasılığı var |
| Hız notu | — | H100'de optimize edilmemiş PyTorch ile RTF ≈ 5 (blog) | RTX 4090'da RTF ≈ 0.30 (README) |
| Python | mevcut ortam 3.12 | 3.11 ile geliştirilmiş/test edilmiş | ≥3.10, <3.13 |
| Lisans | Coqui Public Model License | MIT | Apache-2.0 |
| Not | — | Her çıktıya **PerTh filigranı** gömülü (kapatma yolu belgelenmemiş). Resemble'a göre Chatterbox ticari olarak emekliye ayrıldı (Ağustos 2026); açık kaynak araştırma olarak duruyor, destek ve güncelleme yok. | Referans transkripti verilirse benzerlik artar ("ultimate"). |

Kaynaklar:
- Chatterbox: [GitHub README](https://github.com/resemble-ai/chatterbox), [HF model kartı](https://huggingface.co/ResembleAI/chatterbox), [V3 duyurusu](https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages).
- VoxCPM2: [GitHub README](https://github.com/OpenBMB/VoxCPM), [HF model kartı](https://huggingface.co/openbmb/VoxCPM2).

Kaynaklar arasında çelişkiler:
- Chatterbox'ın dil sayısı kaynağa göre 21, 23 veya 25 olarak geçiyor. Türkçe hepsinde var.
- VoxCPM2 model kartındaki hızlı örnek 16 kHz ile kaydediyor; README `model.tts_model.sample_rate` kullanıyor. Çalıştırıcı README'deki yolu kullanır.

API sürüm farklarına karşı çalıştırıcı, kurulu kütüphanenin kabul etmediği argümanları sessizce atmaz:
- Chatterbox'ta `t3_model="v3"` desteklenmiyorsa koşu V2'ye düşmek yerine hata verir.
- Kurulu paket sürümleri her `results.json` dosyasına yazılır.

## 2. Deney tasarımı

**Sabitler.**
- Aynı referans ses (koşu klasöründeki kopya).
- Aynı 12 maddelik Türkçe test seti (`voxforge/benchmark/testset.py`, sürüm `tr-v1`).
- Her motora aynı **okunuş metni** verilir (VF-004 normalizasyonu). Böylece sayı/tarih işleme motor farkına karışmaz.
- Sabit tohum (`seed`, varsayılan 1234).
- Motor ayarları kütüphane varsayılanı. XTTS ürün adaptörüyle ve "Dengeli" preset ile çalışır.
- VoxCPM2 için `cfg_value=2.0` ve `inference_timesteps=10` (README örneği).

**Test seti.**
- Çok kısa ifadeler ("Evet."): uydurma ve uzatma riski.
- ğ/ı/ş/ç yoğun cümleler.
- Soru ve ünlem tonlaması.
- Saat, yüzde ve tarih okunuşu.
- Bilerek dönüştürülmeyen ekli TL tutarı.
- Türkçe özel isimler.
- İngilizce kelime karışımı.
- Uzun cümle.
- Tekerleme.

| Ölçüt | Nasıl | Araç |
|---|---|---|
| Türkçe telaffuz / anlaşılırlık | Çıktının ASR transkripti ile okunuş metni arasındaki **CER** (Türkçe küçük harf, rakamlar yazıya) | faster-whisper `large-v3`, `language=tr` + `voxforge/benchmark/metrics.py` |
| Metin dışı ses | CER içindeki **ekleme oranı** (fazladan kelime, mırıldanma, tekrar). Oran %15'i aşarsa "şüpheli". Ayrıca karakter başına süre aralık dışıysa (kırpılma veya kaçak üretim) işaretlenir. Kör dinlemede ayrı bir "metin dışı ses" kutusu var. | metrics + dinleme |
| Ses benzerliği | Referans ile çıktı arasında ECAPA-TDNN konuşmacı gömmesi kosinüs benzerliği + kör dinleme puanı (1–5) | SpeechBrain `spkrec-ecapa-voxceleb` |
| Doğallık | Kör dinleme puanı (1–5). Otomatik MOS tahmincisi bilerek eklenmedi: Türkçe için güvenilirliği belirsiz, ek bağımlılık. | `listening/blind_sheet.csv` |
| Üretim süresi | Model yükleme süresi ayrı; madde başına süre ve **RTF** (üretim süresi / ses süresi); tepe VRAM | `run_engine.py` |

**Kör dinleme.**
- Her madde için başarılı tüm çıktılar karıştırılıp A/B/C… etiketleriyle `listening/` klasörüne kopyalanır.
- Hangi etiketin hangi motor olduğu `blind_key.json` dosyasındadır; puanlama bitmeden açılmaz.
- Form (`blind_sheet.csv`, `;` ayraçlı) sütunları:
  - `naturalness`, `similarity`, `pronunciation`: 1–5;
  - `extra_sound`: 1 = metin dışı ses var;
  - `notes`.
- Dolu form hiçbir zaman üzerine yazılmaz.

**Karar ölçütleri (sonuçlardan önce sabitlendi; `experiment.py` içinde aynı değerler).** Bir aday, XTTS'e göre şu koşulların hepsini sağlarsa **VF-009 entegrasyonu için aday** sayılır:
1. Kör dinlemede doğallık **ve** benzerlik ortalaması en az **+0.3**.
2. Ortalama CER en fazla **+0.02** kötü.
3. RTX 4060 Laptop'ta ortalama **RTF ≤ 1.0**.
4. Dinleyicinin metin dışı ses işaretlediği çıktı oranı **≤ %10**.
5. Başarısız çıktı yok (OOM dahil).

Eksik kanıt "eksik" olarak raporlanır, varsayılmaz.

**Sınırlar.**
- Tek referans ve tek dinleyici küçük bir örneklemdir; sonuç yön göstericidir, kesin değildir. Fark küçükse ikinci bir referansla tekrar önerilir (yeni `prepare` koşusu).
- ASR ve ECAPA da modeldir ve kendi hataları vardır. Tüm motorları aynı modeller puanladığı için karşılaştırma adildir, ancak mutlak değerler yorumlanmamalıdır.
- Chatterbox çıktılarındaki filigran algısal değildir; puanlamayı etkilemesi beklenmez.

## 3. Çalıştırma (sizin adımlarınız)

Model indirmeleri ve GPU işleri yalnız bu adımlarda olur. Varsayılan konumlar:
- Ortamlar: `C:\AvenoxData\Environments\VoxForge-bench-<ad>`
- İndirilen modeller: `C:\AvenoxData\Models\HF` (`HF_HOME`)
- Çıktılar: `outputs/voice_benchmark/<run_id>/` (Git dışı)

1. **Ortamları kur** (bir kez; her biri ayrı, ana `.venv`'e dokunmaz):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine chatterbox
   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine voxcpm
   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine eval
   ```
   - Chatterbox Python 3.11 ister; bu makinede 3.11 yok. Kurmak için `winget install Python.Python.3.11`. Ya da `-PythonVersion 3.10` ile deneyin (resmî olarak test edilmemiş).
   - Betik, CUDA'sız torch gelirse aynı sürümü CUDA indeksinden yeniden kurar ve `requirements-lock.txt` yazar.
2. **Koşuyu hazırla** (profilin ön işlenmiş referansı kopyalanır). Referansın birebir metnini biliyorsanız `-ReferenceText` ekleyin; VoxCPM2 "ultimate" modu bunu ister.
   ```powershell
   .\run_voice_benchmark.ps1 -Step prepare -ProfileSlug <profil_slug>
   ```
3. **Üret.** Her motor ayrı süreçte çalışır, böylece GPU belleği boşalır. Önce `-Only t01,t03` ile kısa bir deneme önerilir.
   ```powershell
   .\run_voice_benchmark.ps1 -Step generate -Run <koşu> -Engine xtts
   .\run_voice_benchmark.ps1 -Step generate -Run <koşu> -Engine chatterbox-v3
   .\run_voice_benchmark.ps1 -Step generate -Run <koşu> -Engine voxcpm2
   .\run_voice_benchmark.ps1 -Step generate -Run <koşu> -Engine voxcpm2-ultimate   # yalnız -ReferenceText verildiyse
   ```
   - VoxCPM2 8 GB'ta OOM verirse ilgili madde "HATA" olarak kaydedilir ve koşu devam eder. Bu da bir sonuçtur: bu donanımda uygun değil.
   - Tarayıcı ve diğer GPU uygulamalarını kapatmak belleği artırır.
4. **Otomatik değerlendirme** (ASR + benzerlik). İlk çalıştırmada Whisper large-v3 (~3 GB) ve ECAPA modeli indirilir.
   ```powershell
   .\run_voice_benchmark.ps1 -Step evaluate -Run <koşu>
   ```
5. **Kör dinleme formunu oluştur.** `listening/` klasörünü dinleyip `blind_sheet.csv` dosyasını doldurun. `blind_key.json` dosyasını puanlama bitmeden açmayın.
   ```powershell
   .\run_voice_benchmark.ps1 -Step blind -Run <koşu>
   ```
6. **Rapor.** `report.md` dosyasında karşılaştırma tablosu ve önceden sabitlenmiş ölçütlere göre karar yer alır.
   ```powershell
   .\run_voice_benchmark.ps1 -Step report -Run <koşu>
   ```

Model indirmeden boru hattını denemek için `-Engine mock` kullanılabilir; ana ortamla çalışır.

## 4. Bilinen riskler

- **VoxCPM2 VRAM:** README ~8 GB diyor. Laptop GPU'sunda ekran da bellek kullanır; OOM olasılığı yüksek. Sonuç "uygun değil" olabilir ya da daha düşük ayarlarla tekrar denenebilir.
- **Chatterbox:** Python 3.11 gerekiyor. Çıktılar filigranlı. Resemble tarafında ticari destek bitmiş; uzun vadeli bakım riski var.
- **faster-whisper GPU:** CUDA kütüphanelerini bulamazsa (cuDNN/cuBLAS DLL) ASR hata verir. Değerlendirme betiği torch'u önce yükleyerek torch'un CUDA DLL'lerini kullanır. Sorun sürerse ortamda `nvidia-cublas-cu12 nvidia-cudnn-cu12` paketleri gerekir.
- **SpeechBrain (Windows):** model dosyaları sembolik link yerine kopyalanır (`LocalStrategy.COPY`).
- **Lisans:**
  - XTTS-v2'nin Coqui Public Model License'ı ticari kullanımı kısıtlar.
  - Chatterbox (MIT) ve VoxCPM2 (Apache-2.0) bu açıdan daha esnektir.
  - Ürün kararında kalite kadar bu da tartılmalıdır.
