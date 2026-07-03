# VoxForge Voice Profile Rehberi

Bu doküman, VoxForge içindeki yerel voice profile sistemini açıklar. Voice profile, referans ses dosyasını tekrar kullanılabilir bir yerel profil haline getirir. Bu aşama fine-tuning değildir; mevcut akış hâlâ reference-based / zero-shot voice cloning yaklaşımıdır.

## 1. Voice Profile nedir?

Voice profile, bir referans sesin `profiles/` altında yerel olarak saklanan hazırlanmış sürümüdür. Amaç, Gradio demosunda her denemede aynı ses dosyasını yeniden seçmek yerine daha önce kontrol edilmiş ve ön işlenmiş bir referansı kullanmaktır.

Bir profil şunları sağlar:

- Orijinal referans sesin yerel kopyası
- XTTS için hazırlanmış ön işlenmiş referans ses
- Profil adı ve slug bilgisi
- Ham ve ön işlenmiş kalite raporları
- Hangi ön işleme varyantının seçildiği
- Ön işleme sırasında oluşan uyarılar

## 2. Profil nasıl oluşturulur?

Profil oluşturmak için proje kök dizininde şu komut çalıştırılır:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_voice_profile.ps1 -Name baglare -InputPath .\samples\my_voice.wav
```

Bu komutta:

- `-Name baglare`, profil adını belirtir.
- `-InputPath .\samples\my_voice.wav`, profil için kullanılacak kaynak referans sesi belirtir.

Script profil adını güvenli bir klasör adına çevirir. Örneğin `baglare` adı için profil klasörü `profiles/baglare/` olur. Aynı slug ile profil zaten varsa üzerine yazılmaz.

## 3. Profil klasör yapısı

Örnek profil yapısı:

```text
profiles/
|-- .gitkeep
`-- baglare/
    |-- original_reference.wav
    |-- preprocessed_reference.wav
    `-- profile.json
```

`profiles/.gitkeep`, `profiles/` klasörünün GitHub üzerinde boş halde de görünür kalmasını sağlar. Gerçek profil klasörleri ve ses dosyaları GitHub'a yüklenmez.

## 4. profile.json içinde ne tutulur?

`profile.json`, profil hakkında taşınabilir metadata tutar. İçerik genel olarak şu alanlardan oluşur:

- `profile_name`: Kullanıcının verdiği profil adı
- `profile_slug`: Güvenli klasör adı
- `created_at`: Profilin oluşturulma zamanı
- `original_reference_path`: Orijinal referans sesin proje içi yolu
- `preprocessed_reference_path`: Ön işlenmiş referans sesin proje içi yolu
- `original_quality`: Orijinal referans kalite raporu
- `preprocessed_quality`: Ön işlenmiş referans kalite raporu
- `selected_preprocessing_variant`: Seçilen ön işleme varyantı
- `preprocessing_warning`: Ön işleme uyarısı
- `preprocessing_candidate_reports`: Denenen ön işleme adaylarının raporları
- `notes`: Profilin local kullanımına dair kısa notlar

Bu dosya, Gradio arayüzünün profil adını göstermesine ve kalite raporunu yeniden üretmeden mevcut profil bilgisini kullanmasına yardım eder.

## 5. Orijinal referans ve ön işlenmiş referans farkı

`original_reference.wav`, kullanıcıdan alınan giriş sesinin profil klasöründeki kopyasıdır. Bu dosya kaynak kayıt olarak saklanır.

`preprocessed_reference.wav`, XTTS'e verilecek hazırlanmış referanstır. Varsayılan güvenli ön işleme akışı sesi mono, 24000 Hz, `pcm_s16le` WAV formatına getirir ve ses seviyesini dengeler.

Gradio profil seçildiğinde doğrudan `preprocessed_reference.wav` dosyasını kullanır. Böylece her üretimde aynı hazırlanmış referansla tutarlı deneme yapılır.

## 6. Gradio profil seçimi nasıl çalışır?

Gradio arayüzü açıldığında `profiles/` klasörünü tarar. Bir klasörün dropdown'da görünmesi için içinde şu dosyalar olmalıdır:

- `profile.json`
- `preprocessed_reference.wav`

Kullanıcı bir profil seçerse Gradio:

1. Profil metadata dosyasını okur.
2. `profiles/<slug>/preprocessed_reference.wav` dosyasını seçer.
3. Bu dosyayı XTTS `speaker_wav` girdisi olarak kullanır.
4. Profilin kalite raporlarını Gradio kalite raporu alanında gösterir.

Profil seçildiğinde yüklenen ses dosyası yok sayılır. Bu davranış kasıtlıdır; çünkü seçili profil, kullanıcının daha önce hazırlanmış ve kalite raporu alınmış yerel referansı kullanmak istediği anlamına gelir.

## 7. Öncelik sırası: profil > yüklenen ses > varsayılan ses

Gradio referans ses seçimini şu sırayla yapar:

1. Seçili yerel profil varsa profilin `preprocessed_reference.wav` dosyası kullanılır.
2. Profil seçilmemişse ve kullanıcı ses yüklediyse yüklenen referans ses kullanılır.
3. Profil seçilmemişse ve kullanıcı ses yüklememişse `samples/my_voice.wav` kullanılır.

Bu sıra, profil davranışını açık ve tahmin edilebilir tutar.

## 8. Profiller neden GitHub'a yüklenmez?

Voice profile dosyaları kişisel veya izinli ses kayıtları içerebilir. Bu dosyalar hem mahremiyet hem de repo boyutu nedeniyle GitHub'a yüklenmez.

`.gitignore` içinde bu amaçla şu kurallar bulunur:

```text
profiles/*
!profiles/.gitkeep
```

Bu kurallar gerçek profil içeriklerini dışarıda bırakır, fakat klasörün projede var olması gerektiğini gösteren `.gitkeep` dosyasını korur.

## 9. Kalite raporları profil oluşturma sürecinde nasıl kullanılır?

Profil oluşturulurken hem orijinal referans hem de ön işlenmiş referans için kalite raporu alınır. Raporlar `profile.json` içine yazılır.

Kalite raporları şu konularda sinyal verir:

- Dosya var mı?
- Süre uygun mu?
- Sample rate ve kanal sayısı nedir?
- Ses seviyesi çok düşük veya çok yüksek mi?
- Clipping riski var mı?
- Sonuç `GOOD`, `WARNING` veya `BAD` mı?

Kalite raporu teknik bir yardımcıdır. Son karar için üretilen ses yine dinlenerek kontrol edilmelidir.

## 10. Güvenli ön işleme neden tercih edildi?

Varsayılan profil oluşturma akışı `safe_normalized` yaklaşımını kullanır. Bu yaklaşımın amacı referans sesi XTTS için tutarlı formata getirmek ve ses seviyesini dengelemektir.

Agresif sessizlik kırpma varsayılan akışta kullanılmaz. Çünkü bazı kayıtlarda konuşma araları, nefesler veya düşük seviyeli konuşma bölümleri yanlışlıkla sessizlik gibi algılanabilir. Bu durumda referans ses fazla kısalabilir ve konuşmacı karakteri zayıflayabilir.

Güvenli yaklaşım şu yüzden tercih edilir:

- Referans süresini korumaya çalışır.
- Ses karakterini gereksiz kırpmayla zayıflatmaz.
- Mono, 24000 Hz WAV çıktısı üretir.
- Kalite raporuyla birlikte daha kontrollü bir profil oluşturur.

## 11. Bilinen sınırlamalar

- Voice profile sistemi fine-tuning değildir.
- Profil seçimi yeni bir model oluşturmaz.
- Profil kalitesi kaynak sesin kalitesine bağlıdır.
- `profile.json` kalite sinyali verir, nihai ses benzerliğini garanti etmez.
- Profil silme ve yenileme için ayrı otomasyon henüz yoktur.
- Profil kalite geçmişi henüz tutulmaz.
- Gradio dropdown'ı demo açılırken mevcut profilleri listeler; yeni profil oluşturulduysa demoyu yeniden başlatmak gerekebilir.
- Gerçek profil dosyaları local kalmalıdır ve GitHub'a eklenmemelidir.

## 12. Sonraki adımlar

Planlanan geliştirme yönleri:

- Profil silme
- Profil yenileme
- Profil kalite geçmişi
- Fine-tuning hazırlığı

Fine-tuning sonraki ve daha gelişmiş bir aşamadır. Mevcut doküman ve kod akışı sadece yerel referans profil yönetimini anlatır.
