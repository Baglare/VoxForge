# VF-018: Quality Director + Voice Takes

Ana takip: [Issue #18](https://github.com/Baglare/VoxForge/issues/18). Amaç: üretilen bir cümlenin beklenen
metne ve sağlıklı sese uygunluğunu **açıklanabilir** biçimde denetlemek, şüpheli üretimler için farklı tohumlarla
alternatif sesler (take) üretmek, kullanıcının seçmesine ve önceki seçime dönmesine izin vermek. Kalite denetimi
ve ses üretimi birbirinden bağımsızdır; Playground bugün, Studio (VF-019) sonra aynı sözleşmeyi kullanır.

## Mimari

| Katman | Dosya | Görev |
|---|---|---|
| Kurallar | `voxforge/quality/checks.py` | Saf fonksiyonlar: ses ölçümleri + ASR kelime hizalaması → bulgular, güven, karar |
| ASR | `voxforge/quality/asr.py`, `scripts/quality/asr_worker.py` | faster-whisper large-v3 (+ ECAPA benzerliği), mevcut `VoxForge-bench-eval` ortamında kalıcı alt süreç |
| Denetçi | `voxforge/quality/director.py` | `QualityDirector.check(wav, spoken_text, intended_pauses, reference)` → rapor |
| Take'ler | `voxforge/quality/takes.py` | `TakeStore` (grup JSON'u, seçim geçmişi, geri dönüş), `TakeDirector` (yeni tohumlar, denetim, sıralama) |
| Playground adaptörü | `voxforge/playground/quality.py` | Kayıt → tarif, rapor önbelleği, cümle bazlı grup, üretim sonrası kanca |
| Arayüz | `app/playground_quality_tab.py`, `app/voice_playground.py` | "Kalite ve take'ler" sekmesi; üretim sonrası otomatik denetim |

`voxforge/quality` Gradio'yu, Playground servisini veya motorları bilmez. Studio için gereken tek şey bir
`Synthesizer` çağrılabilirdir: `(recipe: dict, seed: int) -> {"id", "wav", "settings": {"seed", ...}, ...}`.
`TakeDirector(TakeStore(kök), QualityDirector(AsrClient(...)), synthesizer)` kurulur; take grubu kimliği Studio'da
segment kimliğinden türetilebilir.

## Kanıt türleri ve güven

- **Ses ölçümü** (dalga biçiminden, güvenilir): sessiz/boş çıktı, karakter başına süre (VF-Q1 eşikleri
  0,035-0,16 sn), hece/sn hızı (Türkçede ünlü sayısı = hece; istenen duraklamalar düşülür; > 9,5 veya < 3,5),
  beklenmeyen iç sessizlik (duraklama istenmediyse > 1,0 sn; istendiyse en uzun istenen + 0,6 sn), baş/son
  sessizlik, ortalama seviye (< -32 dBFS), kırpılma (tam ölçek örnek oranı > %0,1), sessizliksiz ani bitiş.
- **ASR görüşü** (kesin TTS hatası sayılmaz): beklenen okunuş ile Whisper transkripti, VF-Q1 normalizasyonuyla
  (Türkçe küçük harf, sayılar yazıyla) kelime düzeyinde hizalanır. Hata aralıkları sınıflanır: eksik kelime,
  sonda eksik (kesilme), tekrar (eklenen kelimeler komşu kelimeleri tekrarlıyor), metin dışı kelime, farklı
  kelime, yakın kelime (tek kelime ve benzerlik ≥ 0,75 → telaffuz/ASR farkı, küçük; yalnız not). Yalnız boşluk
  farkı yok sayılır. Emin bir ASR'da başka büyük bulgu yokken CER > 0,15 "yüksek metin hatası" olur.
- **ASR + ses birlikte**: ASR'ın hiçbir kelimeye bağlamadığı sesli süre (kelime zaman damgası ± 0,1 sn dışı).
  Whisper tekrarlanan ifadeyi transkriptten sık sık atar (CER 0 kalır); ses ise oradadır. > 0,6 sn büyük bulgu,
  ≥ 1,0 sn ve ASR genel olarak eminse yüksek güven ("Tanınmayan konuşma").
- **Konuşmacı benzerliği** (ECAPA, VF-Q1 ile aynı model): yalnız alternatifleri orijinal take ile kıyaslar;
  0,05'ten fazla düşüş "kimlik kayması şüphesi" (0,10 üstü orta güven).

Güven kuralları: ilgili kelimelerde Whisper olasılığı < 0,6 ise bulgu **düşük** güvenlidir; transkript ortalaması
< 0,65 ise tüm ASR bulguları düşüğe iner. Tek kısa işlev kelimesinin (≤ 3 harf: "ve", "de", "bir") düşmesi küçük
ve düşük güvenlidir. ASR bulgusu **yüksek** güvene yalnız ses ölçümüyle birlikte çıkar: sonda eksik kelime +
(aşırı kısa veya ani bitiş) → kesilme; tekrar + (aşırı uzun veya ≥ 2 kelimelik, olasılığı ≥ 0,8 tekrar).

Karar ayrı tutulur:

| Karar | Anlamı | Otomatik işlem |
|---|---|---|
| Başarılı (`ok`) | Büyük bulgu yok; küçük ASR farkları ve düşük güvenli notlar gösterilir | Yok |
| Şüpheli (`suspect`) | Dinleyip karar verin | **Yok** (öneri gösterilir) |
| Sorunlu (`problem`) | En az bir büyük bulgu yüksek güvenle | Ayar açıksa 1-3 alternatif üretilir |

Seçim hiçbir zaman kendiliğinden değişmez; en iyi take yalnız **önerilir** (sıralama: karar, güvenli büyük/küçük
bulgu sayısı, CER, benzerlik).

## Voice Takes

- Grup: `outputs/playground/takes/pg-<kayıt>.json` (cümle için `pg-<kayıt>-s<n>.json`). Tarif (varyant, profil,
  referans, kontroller, preset, normalizasyon, cihaz, temel tohum), take'ler (tohum, ayar etiketi/uygulanan ayar,
  WAV, süre/RTF, VRAM tepe, kalite raporu, benzerlik) ve seçim geçmişi atomik JSON ile saklanır.
- Orijinal üretim `t1` olarak eklenir ve başlangıçta seçilidir. Alternatifler aynı tarifle, temel tohum ve mevcut
  take sayısından türetilen **tekrar üretilebilir, birbirinden farklı** tohumlarla üretilir. Bir alternatifin
  hatası diğerlerini durdurmaz.
- Take ve WAV silinmez. Seçim geçmişi yalnız eklenir; "Önceki seçime dön" bir yığın gibi geri alır ve geri
  almayı da kaydeder.
- Çok cümleli metinde rapor bulguları cümleye bağlar; yalnız sorunlu cümle ayrı bir grupta yeniden üretilebilir.
  Seçilen cümle take'inin uzun anlatıma geri eklenmesi (ses kurgusu) VF-019/VF-023 Studio segment akışına bırakıldı;
  Playground'da cümle take'leri ayrı dinlenir.

## 8 GB VRAM ve worker yaşam döngüsü

- Whisper worker'ı **CPU'da** (int8, ≤ 16 iş parçacığı) çalışır; yönetici süreçten `CUDA_VISIBLE_DEVICES=-1` ile
  GPU'yu gizler. VoxCPM2 sıcakken ~5,8 GB + masaüstü uygulamaları ~0,6-1,8 GB tutarken large-v3'ün GPU'da yeri yok;
  GPU'da çalıştırmak her denetimde VoxCPM2'yi kapatıp yeniden yüklemeyi gerektirirdi.
- Ölçüm (bu makine, 6,8 sn ses): yükleme ~8,5 sn, beam 5 ile ~14 sn, beam 1 ile ~11 sn (≈ 2x gerçek zaman).
  Varsayılan beam 5 (VF-Q1 ile aynı); `VOXFORGE_QUALITY_ASR_BEAM`, `VOXFORGE_QUALITY_ASR_THREADS` ile değişir.
- Worker ilk denetimde başlar, sonra açık kalır; "Motoru ve kalite denetleyicisini kapat" ve uygulama çıkışı
  her iki süreci kapatır. TTS tek-motor kuralı değişmedi.

## VoxCPM2 model revizyonu sabitlendi (doğrulama sırasında bulundu)

Hub'daki `openbmb/VoxCPM2` `main` dalı 2026-11-14'te yeni revizyona (`7936080b`) geçti: config, AudioVAE ve
tokenizer aynı, `model.safetensors` farklı. Playground ve benchmark `from_pretrained("openbmb/VoxCPM2")` ile
sabitlenmemiş yüklüyordu; yerel `refs/main` yeni revizyonu gösterdiği için her açılışta ~4,6 GB indirmeye
başlıyor (önceki denemelerden ~7,4 GB yarım `.incomplete` dosya kalmış) ve tamamlansaydı sesi sessizce
değiştirecekti. Artık `scripts/benchmark/engine_pins.json` içindeki `32279eff` revizyonu (VF-015–017 ölçümleri)
**yalnız yerel önbellekten** klasör yolu olarak yüklenir; eksikse açık hata verilir, indirme yapılmaz. Başka
revizyon yalnız `VOXFORGE_VOXCPM_REVISION` ile ve önbellekteyse kullanılır; kayıtlarda `model` alanı
`openbmb/VoxCPM2@32279eff` olur. Yarım dosyalar silinmedi (kullanıcı kararı; `HF_HOME\hub\models--openbmb--VoxCPM2\blobs\*.incomplete`).

## Gerçek GPU doğrulaması

`scripts/quality/vf018_validate.py` (mimo profili, VoxCPM2 varsayılan cfg 2.0 / 10 adım, tohum 1234). Sonuçlar
aşağıdaki bölümde; ham kanıt `outputs/vf018/validation.json` (git dışı).

Ortam: RTX 4060 Laptop 8 GB, torch 2.15.0+cu134, VoxCPM2 `32279eff`. Ölçüm sırasında kullanıcının oyun istemcisi
ve oyunu açıktı (boşta 1,87 GB VRAM); kapatılmadı.

| Kontrol | Sonuç |
|---|---|
| Kısa metin regresyonu (yeni süreç, ilk üretim, tohum 1234) | 7 Kasım varsayılan çıktısıyla **bayt bayt aynı** (sha256 `3492a3ce…`) |
| VoxCPM2 VRAM | üretim tepe 6,3 GB (`max_memory_allocated`); GPU toplam 7,43 GB (oyun dahil) |
| Whisper denetimi sırasında GPU | 7432 → en çok 7581 MB (TTS dalgalanması; ASR süreci GPU'yu görmez) |
| ASR süresi (CPU) | yükleme 15,5 sn (ilk), 6-7 sn ses için 15-18 sn |
| Kısa metin (`Merhaba. Ben Mucan Arı…`) | Başarılı, CER 0,034 (yalnız not: "benle" → "benimle") |
| İki cümlelik anlatım | Başarılı, CER 0,000, 7,65 hece/sn |
| Gerçek sesi %55'te kesme | **Sorunlu (yüksek)**: aşırı kısa + kesilme (sonda 9 kelime yok, sessizliksiz bitiş) |
| Gerçek sesin orta bölümünü tekrarlatma (+1,4 sn) | **Sorunlu (yüksek)**: tanınmayan konuşma 1,28 sn (temiz take 0,06 sn); Whisper tekrarı yazmadı, CER 0 |
| 2 alternatif (tohum 918579185 ve bir diğeri) | 77 sn; ikisi de özel ismi bozdu ("Mucanları", "Burcu Yanar") → Şüpheli; öneri orijinal t1; seçim/geri dönüş doğru |
| Worker yaşam döngüsü | kapatmada TTS ve ASR süreçleri sonlandı |

Kurallar bu doğrulamada iki kez düzeltildi (`vf018-1` → `vf018-3`): tekrar yalnız ASR ile yakalanamadığı için
"tanınmayan konuşma" eklendi; çok kelimeli ad değişimi "yakın kelime" sayılıyordu, yakın kelime tek kelimeye ve
0,75 benzerliğe daraltıldı; küçük ASR farkları tek başına "şüpheli" yapmıyor. Not: oyun açıkken VRAM ~7,6/8,2 GB
olduğu için RTF 3,5-5,5 ölçüldü (VF-015'te boş GPU'da ~1,2); bu VF-018 değişikliğinden değil.

## Sınırlar

- ASR bir modelin görüşüdür; özel isim ve nadir kelimelerde Whisper'ın yanlış duyması sık görülür. Bu yüzden ASR
  bulguları tek başına otomatik işlem başlatmaz.
- Telaffuz yalnız kelime düzeyinde kelime/okunuş farkı olarak yakalanır; vurgu ve tonlama denetlenmez.
- Hız eşikleri mimo + VoxCPM2 ölçümlerine göredir; başka ses/motor için ayarlanması gerekebilir.
- Yeni referans ses kaydı gerektiren işler ertelendi (mimo 16 kHz Opus kaynağı aynen kullanıldı).

## Test

`tests/test_quality_director.py`: kurallar (temiz take, eksik/kısa işlev kelimesi, kesilme, tekrar, ASR
belirsizliği, yakın kelime, sessiz çıktı, istenen duraklama, seviye/kırpılma), take deposu (farklı tohumlar,
seçim/geri dönüş yığını, hata dayanıklılığı, kimlik kayması, grup kimliği doğrulaması), gerçek protokolle sahte
ASR worker'ı (CPU, GPU gizli, kapatma) ve Mock motorla Playground köprüsü (tarif, önbellek, cümle grubu).
