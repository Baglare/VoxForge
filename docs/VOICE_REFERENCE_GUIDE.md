# VoxForge Referans Ses Rehberi

Bu doküman, VoxForge içinde XTTS-v2 için kullanılacak referans sesin nasıl hazırlanması gerektiğini açıklar. Referans sesin kalitesi, üretilen sesin anlaşılabilirliğini ve ses benzerliğini doğrudan etkiler.

## 1. İyi referans ses nasıl alınır?

İyi bir referans ses, modelin konuşma karakterini daha net yakalamasına yardım eder. Amaç yüksek sesle bağırmak değil; temiz, dengeli ve doğal bir konuşma kaydı almaktır.

Önerilen kayıt özellikleri:

- Tek kişi konuşmalı.
- Konuşma doğal tempoda olmalı.
- Ses çok kısık olmamalı.
- Ses patlayacak kadar yüksek olmamalı.
- Ortam mümkün olduğunca sessiz olmalı.
- Kayıtta müzik, fan, klavye veya arka plan konuşması olmamalı.

## 2. 30-90 saniye doğal konuşma önerisi

XTTS referans sesi için pratik aralık 30-90 saniyedir.

- 30 saniyeden kısa kayıtlar ses karakterini yeterince temsil etmeyebilir.
- 90 saniyeden uzun kayıtlar gereksiz gürültü, nefes, sessizlik veya kalite farkı taşıyabilir.
- En iyi sonuç için 30-90 saniye arası, kesintisiz ve doğal bir konuşma tercih edin.

Okuma metni günlük konuşmaya yakın olmalıdır. Çok monoton, fısıltı gibi veya aşırı teatral kayıtlar üretim kalitesini düşürebilir.

## 3. Sessiz ortam

Kayıt alırken arka plandaki sabit gürültüler modelin referans sesi yanlış yorumlamasına neden olabilir.

Dikkat edilmesi gerekenler:

- Fan veya klima sesi olmasın.
- Klavye ve mouse sesi olmasın.
- Arka planda müzik olmasın.
- Başka biri konuşmasın.
- Oda yankısı mümkün olduğunca az olsun.

Sessiz bir oda, perde/halı gibi yankıyı azaltan yüzeyler ve sabit mikrofon konumu daha iyi sonuç verir.

## 4. Tek kişi konuşması

Referans seste yalnızca hedef kişinin sesi olmalıdır. Birden fazla kişi konuşursa model hangi ses karakterini takip edeceğini karıştırabilir.

Bu yüzden:

- Diyalog kaydı kullanmayın.
- Arka planda başka konuşma olmasın.
- Video, yayın veya toplantı kaydından kesilmiş karışık sesleri tercih etmeyin.

## 5. Mikrofona çok yakın konuşmama

Mikrofona çok yakın konuşmak patlama seslerine, nefes gürültüsüne ve clipping riskine yol açabilir. Çok uzak konuşmak ise sesin kısık ve odalı duyulmasına neden olur.

Pratik öneri:

- Mikrofonu sabit tutun.
- Ağız ile mikrofon arasında makul mesafe bırakın.
- Patlayan `p`, `b`, `t` seslerinde mikrofona doğrudan üflememeye çalışın.
- Kayıt boyunca aynı mesafeyi koruyun.

## 6. Arka plan müziği, fan ve klavye sesi olmaması

XTTS referans sesten yalnızca konuşma karakterini almak ister. Arka plan müziği, fan uğultusu, klavye sesi veya oda gürültüsü modele yanlış sinyal verir.

Bu tür gürültüler:

- Üretilen seste yapay titreşim oluşturabilir.
- Konuşma netliğini düşürebilir.
- Ses benzerliğini zayıflatabilir.
- Kalite raporunda uyarı oluşmasına neden olabilir.

## 7. Çok kısık ses kaliteyi neden düşürür?

Çok kısık kayıtlar modelin konuşma detaylarını ayırt etmesini zorlaştırır. Ses seviyesi düşük olduğunda nefes, oda sesi ve mikrofon dip gürültüsü konuşmaya göre daha baskın hale gelebilir.

İyi kayıt için:

- Normal konuşma sesinizle konuşun.
- Mikrofon kazancını çok düşürmeyin.
- Kayıt sonrası sesi aşırı yükseltmek yerine baştan dengeli kayıt alın.

## 8. Clipping riski nedir?

Clipping, sesin çok yüksek kaydedilmesi nedeniyle dalga formunun tepe noktalarının kesilmesidir. Bu durum seste çatlama, patlama veya bozulma gibi duyulur.

Clipping olduğunda:

- Ses teknik olarak bozulur.
- Model bozuk sesi referans alabilir.
- Üretilen ses doğal olmayan sertlikler taşıyabilir.
- Kalite raporu `BAD` veya `WARNING` verebilir.

Mikrofona çok yakın konuşmamak ve kayıt seviyesini makul tutmak clipping riskini azaltır.

## 9. `GOOD`, `WARNING`, `BAD` kalite sonuçları

VoxForge kalite raporu referans ses için sade bir sonuç üretir:

- `GOOD`: Teknik olarak belirgin bir sorun görülmedi. Yine de sonucu dinleyerek kontrol etmek gerekir.
- `WARNING`: Kayıt kullanılabilir olabilir, ancak süre, sample rate, kanal sayısı, ses seviyesi veya clipping gibi konularda uyarı vardır.
- `BAD`: Referans dosya eksik olabilir veya teknik kalite ciddi risk taşıyor olabilir. Üretim engellenmeyebilir, ama sonuç mutlaka dikkatle dinlenmelidir.

Bu sınıflandırma yardımcı bir teknik kontroldür. Nihai kalite kararı için üretilen sesi dinlemek gerekir.

## 10. Ham referans ve ön işlenmiş referans farkı

Ham referans, kullanıcının verdiği orijinal ses dosyasıdır. Bu dosya stereo, farklı sample rate değerinde, uzun sessizlikler içeren veya dengesiz ses seviyesine sahip olabilir.

Ön işlenmiş referans ise FFmpeg ile XTTS için daha tutarlı hale getirilmiş sürümdür. Gradio akışında bu işlem şunları hedefler:

- Sesi mono WAV formatına çevirmek
- Sample rate değerini 24000 Hz yapmak
- Ses seviyesini daha dengeli hale getirmek
- Referans süresini gereksiz kısaltmadan güvenli bir normalize edilmiş WAV üretmek

Varsayılan akış artık `safe_normalized` yaklaşımını kullanır. Bu yaklaşım agresif sessizlik kırpma yapmadan formatı ve ses seviyesini düzenler. Amaç, XTTS'e daha tutarlı bir referans vermek ama konuşmacı karakterini taşıyan bölümleri yanlışlıkla kesmemektir.

Agresif sessizlik kırpma bazı sesleri fazla kısaltabilir. Özellikle düşük sesli konuşma, doğal konuşma araları veya nefesli kayıtlar sessizlik gibi algılanırsa referans süresi gereğinden fazla düşebilir. Bu yüzden agresif kırpma varsayılan akışta kullanılmaz; güvenli normalize edilmiş referans tercih edilir.

Ön işlenmiş dosyalar local olarak şu klasöre kaydedilir:

```text
outputs/preprocessed_references/
```

Ham ve ön işlenmiş referans kalite raporları Gradio arayüzünde ayrı ayrı gösterilir. Bu sayede orijinal dosyanın ve XTTS'e verilen hazırlanmış dosyanın teknik farkı görülebilir.

## 11. Ses benzerliği neden her zaman birebir olmaz?

Zero-shot/reference-based voice cloning yaklaşımında model, kısa bir referanstan konuşma karakterini tahmin eder. Bu yüzden sonuç her zaman birebir kopya olmaz.

Benzerliği etkileyen başlıca faktörler:

- Referans sesin temizliği
- Referans ses süresi
- Konuşma tonu ve tempo
- Mikrofon kalitesi
- Arka plan gürültüsü
- Metnin uzunluğu ve telaffuz zorluğu
- Modelin Türkçe ve konuşmacı karakterini yorumlama biçimi
- CPU/GPU çalışma koşulları ve kullanılan model sürümü

Bu aşama fine-tuning yapılmış özel bir ses modeli değildir. Fine-tuning daha ileri bir aşama olarak değerlendirilebilir, fakat mevcut MVP'nin amacı local zero-shot referans ses denemesini güvenli ve anlaşılır şekilde göstermektir.

## 12. Kayıt öncesi kısa kontrol listesi

1. Sessiz bir oda seçin.
2. Fan, müzik, TV ve klavye sesini kapatın.
3. Tek kişinin konuştuğundan emin olun.
4. Mikrofonu sabit tutun.
5. Mikrofona çok yakın konuşmayın.
6. 30-90 saniye arası doğal konuşma kaydı alın.
7. Çok kısık veya çok yüksek kayıt yapmayın.
8. Dosyayı local olarak `samples/my_voice.wav` yoluna koyun.
9. Kalite raporu scriptini çalıştırın.
10. `GOOD`, `WARNING` veya `BAD` sonucunu ve önerileri okuyun.
