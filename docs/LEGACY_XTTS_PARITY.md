# Eski XTTS demosu → ana arayüz eşdeğerliği

Eski demo: `app/gradio_xtts_demo.py` (`run_gradio_demo.ps1`). Ana arayüz: `app/voice_playground.py`
(`run_voice_playground.ps1`). Eski demo **kaldırılmadı**; aşağıdaki eşdeğerlik kullanıcı tarafından gerçek
XTTS/GPU ile kabul edilene kadar geri dönüş olarak kalır.

| Eski demodaki özellik | Ana arayüzdeki yeri | Not |
|---|---|---|
| Profil seçme ve bilgi paneli | Hızlı ses → *Ses*; Ses kütüphanesi → *Seçili ses* | Kalite raporu (ham + ön işlenmiş), varyant, uyarılar |
| Profilleri yenile | Hızlı ses → *Sesleri yenile*; sekme açılışında otomatik | Kütüphane değişince tüm menüler aynı olayda güncellenir |
| Yeni profil oluştur (izin onayı) | Ses kütüphanesi → *Yeni ses profili* | Aynı `create_voice_profile` |
| Seçili profili yenile | Ses kütüphanesi → *Mevcut kaydı yeniden işle* | Aynı `recreate_voice_profile` |
| Seçili profili sil (onay kutusu) | Ses kütüphanesi → *Sil* | Kimlik yazarak onay; kalıcı silme yerine çöp kutusuna taşıma |
| — (yoktu) | Yeniden adlandırma, referans değiştirme | Yeni; slug/kimlik değişmez |
| Referans ses yükleme (profilsiz) | Hızlı ses → *Ses kaynağı: Tek seferlik referans yükle* | İzin onayı + aynı güvenli ön işleme |
| Varsayılan referans (`samples/my_voice.wav`) | Yalnız Studio'nun eski öncelik zincirinde | Hızlı ses açık bir ses ister (PR #25 kararı) |
| Üretim modu (Dengeli, Daha stabil, …) | Hızlı ses → XTTS seçilince *XTTS üretim modu* | Aynı `INFERENCE_PRESET_KWARGS`; yalnız XTTS'e geçer |
| Çıktı sesini normalize et | Hızlı ses → Üretim ayarları | Aynı `normalize_wav` (loudnorm); ham WAV korunur |
| A/B karşılaştırma (Daha stabil) | Hızlı ses → *A/B karşılaştırma* | Genişletildi: XTTS modları, VoxCPM2 preset'leri, Chatterbox ayarı |
| Kalite raporu metni | Son üretim bilgisi + Kalite denetimi | Tek seferlik referansın ön işleme/kalite özeti mesajda |
| Uzun metin parçalama | Playground parça planı | Değişmedi |
| Experimental Fine-tuning Hazırlığı | Geliştirici sekmesi | Aynı işleyiciler; eğitim başlatmaz |
| Sesli kitap (Studio) sekmesi | Projeler → Sesli kitap | Zaten taşınmıştı (VF-019+) |

## Kabul için açık kalan
- Gerçek XTTS-v2 üretimi (modlar + A/B) ve VoxCPM2 tasarım sesiyle Hızlı ses üretimi bu değişiklikte
  çalıştırılmadı; testler modelsiz stub ile yapıldı.
- Readiness/export/dry-run işleyicileri eski demoyla aynı fonksiyonlar; gerçek dataset üzerinde bu
  değişiklikte yeniden çalıştırılmadı (yalnız bağlantı ve seçim doğrulaması yapıldı).
- Bu kabullerden sonra eski demo kaldırılabilir veya yalnız geliştirici giriş noktası olarak bırakılabilir.
