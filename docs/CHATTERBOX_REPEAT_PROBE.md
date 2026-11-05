# Chatterbox V3 kısa ifade tekrarı (VF-Q1 yan deneyi)

Soru: Chatterbox V3 `Evet.` cümlesini `Evet, evet.` diye tekrarlıyor; ne sıklıkta ve azaltılabilir mi?
Mevcut VF-Q1 koşusuna ve motor varsayılanlarına dokunulmadı. Çıktılar `outputs/chatterbox_repeat/` altında (gitignore).

## Yöntem
- `scripts/benchmark/chatterbox_repeat_probe.py` (Chatterbox venv): ifade × tohum üretimi, `torch.manual_seed` ile tekrarlanabilir. Aynı tohum+metin birebir aynı çıktıyı verir.
- `scripts/benchmark/chatterbox_repeat_score.py` (eval venv): faster-whisper large-v3 + ECAPA, evaluate.py ile aynı modeller.
  `repeat` = beklenen ifade transkriptte 2+ kez geçiyor; `exact` = transkript ifadeye tam eşit.
- Referans: VF-Q1 koşusunun `reference.wav` dosyası. GPU'da yalnız Chatterbox çalıştı.
- Akustik "patlama sayısı" vekili tek kelimede doğal heceleri (Eve-t) tekrar saydığı için kullanılmadı; yalnız ASR sonuçları geçerli.

## Bulgular
1. **Tekrar gerçek ve yaygın, kısa ifadelerde.** Varsayılan ayar, 8 tohum: `Evet.` 4/8, `Tamam.` 7/8, `Hayır.` 3/8, `Merhaba.` 4/8, `Peki.` 2/8, `Teşekkürler.` 0/8; `Evet, tamam.` ve uzun cümle 0/8. `Evet`, `Evet.`, `Evet!` aynı davranır.
2. **Tek parametre taraması** (3 ifade × 8 tohum): `repetition_penalty` yükseltmek işe yaramıyor (1.5: CER 1.13, 2.0: CER 5.02, metin bozuluyor); `cfg_weight`, `temperature` etkisiz (gürültü). `min_p=0.02` ve `exaggeration=0.3` aday.
3. **Doğrulama** (3 ifade × 24 tohum = 72 örnek/ayar):

| Ayar | Tekrar | Tam doğru | CER | ECAPA benzerliği (ort.) |
|---|---|---|---|---|
| varsayılan | 51/72 | 19 | 0.96 | 0.193 |
| `min_p=0.02` | 28/72 | 38 | 0.74 | 0.176 |
| `exaggeration=0.3` | 44/72 | 25 | 0.93 | 0.193 |
| `min_p=0.02` + `exaggeration=0.3` | 31/72 | 37 | 0.89 | 0.202 |

`min_p=0.02` tekrarı yaklaşık %71'den %39'a, tam doğru oranını 19'dan 38'e çıkarıyor (örnekler bağımsız tohumlar; fark gürültüden büyük). Tekrar yine de sürüyor: `Tamam.` 13/24. `exaggeration=0.3` tek başına anlamlı katkı vermiyor; birleşim `min_p` tek başına olandan iyi değil.

## Sınırlar
- Doğallık dinlenmedi (kör dinleme yok); ECAPA kısa kliplerde (<2 sn) düşük ve gürültülü, yalnız aynı ifadede ayarlar arası göreli bakılabilir (min_p=0.02'de 0.176 vs 0.193, anlamlı bir kayıp ya da kazanç iddia edilmez).
- Tek referans ses, tek model commit'i (engine_pins.json). Başka ses/ifadede sonuç değişebilir.
- Tekrarı sıfırlayan bir ayar bulunamadı.

## Öneri (varsayılan değiştirilmedi)
- `min_p=0.02` Chatterbox için umut verici aday; benimsemeden önce kör dinleme ile doğallık/benzerlik kontrolü ve ikinci bir referansla tekrar gerekir.
- Kısa tek kelimelik girdiler için ürün tarafında ASR doğrulamalı yeniden deneme (aynı tohum dışı) veya cümleyi bağlamla uzatma, sonraki deneyde ölçülebilir.
- Tekrar ürün kalitesini etkiliyorsa VF-009 motor kararında kısa-ifade riski olarak not edilmeli.

## Yeniden üretme
```
<chatterbox-venv>\python.exe scripts/benchmark/chatterbox_repeat_probe.py --reference <reference.wav> --phase combo --combo "min_p=0.02" --seeds 24 --name probe2
<eval-venv>\python.exe scripts/benchmark/chatterbox_repeat_score.py --probe outputs/chatterbox_repeat/probe2 --reference <reference.wav>
```
`setup_env.ps1 -Engine eval` artık `av<17` sabitler (`av 19`, faster-whisper 1.2.1 ile `metadata_errors` hatası veriyordu).
