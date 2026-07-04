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

## 2. Web arayüzünden profil oluşturma

Gradio demosu içinde yeni bir yerel voice profile oluşturulabilir. Bunun için kullanıcı şu alanları doldurur:

1. Profil adı
2. Referans ses dosyası
3. Ses üzerinde hakkı veya açık izni olduğunu onaylayan izin checkbox'ı

Bu bilgiler girildikten sonra `Profil oluştur` butonuna basılır. Başarılı oluşturma sonrasında profil yerelde `profiles/<profile_slug>/` klasörüne yazılır.

Gradio, profil oluşturulduktan sonra profil dropdown'ını günceller ve mümkünse yeni profili seçili hale getirir. Böylece kullanıcı aynı demo oturumu içinde yeni oluşturduğu profili seçip metin seslendirmeye devam edebilir.

## 3. Terminalden profil oluşturma

Web arayüzü dışında terminalden profil oluşturma hâlâ alternatif yöntem olarak durur. Proje kök dizininde şu komut çalıştırılır:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_create_voice_profile.ps1 -Name baglare -InputPath .\samples\my_voice.wav
```

Bu komutta:

- `-Name baglare`, profil adını belirtir.
- `-InputPath .\samples\my_voice.wav`, profil için kullanılacak kaynak referans sesi belirtir.

Script profil adını güvenli bir klasör adına çevirir. Örneğin `baglare` adı için profil klasörü `profiles/baglare/` olur.

## 4. Profil klasör yapısı

Örnek profil yapısı:

```text
profiles/
|-- .gitkeep
`-- baglare/
    |-- original_reference.wav
    |-- preprocessed_reference.wav
    `-- profile.json
```

`original_reference.wav`, kullanıcıdan alınan giriş sesinin profil klasöründeki kopyasıdır. `preprocessed_reference.wav`, XTTS'e verilecek hazırlanmış referanstır. `profile.json`, profil adı, slug, dosya yolları, kalite raporları ve ön işleme bilgilerini tutar.

`profiles/.gitkeep`, `profiles/` klasörünün GitHub üzerinde boş halde de görünür kalmasını sağlar. Gerçek profil klasörleri ve ses dosyaları GitHub'a yüklenmez.

## 5. Gradio'da oluşturulan profilin kalıcı olması

Gradio arayüzünde oluşturulan profil geçici bir oturum verisi değildir. Profil dosyaları `profiles/<profile_slug>/` altına yazıldığı için Gradio kapatılsa bile profil silinmez.

Sonraki çalıştırmada Gradio tekrar `profiles/` klasörünü tarar. Geçerli bir profil klasöründe en az şu dosyalar bulunmalıdır:

- `profile.json`
- `preprocessed_reference.wav`

Bu dosyalar varsa profil dropdown'da tekrar listelenebilir.

## 6. Aynı profil adıyla tekrar oluşturma davranışı

Aynı profil adı tekrar kullanılırsa mevcut profilin üzerine yazılmaz. Bu davranış bilinçlidir; çünkü profil klasörleri kişisel veya izinli referans sesleri içerir ve yanlışlıkla değiştirilmeleri istenmez.

Aynı kişi veya aynı referans için yeni bir deneme yapmak gerekiyorsa farklı bir profil adı seçilmelidir. Örneğin `baglare`, `baglare_clean` veya `baglare_test_02` gibi ayrı adlar kullanılabilir.

## 7. Gradio profil seçimi nasıl çalışır?

Gradio arayüzü açıldığında `profiles/` klasörünü tarar. Kullanıcı bir profil seçerse Gradio:

1. Profil metadata dosyasını okur.
2. `profiles/<slug>/preprocessed_reference.wav` dosyasını seçer.
3. Bu dosyayı XTTS `speaker_wav` girdisi olarak kullanır.
4. Profilin kalite raporlarını Gradio kalite raporu alanında gösterir.

Referans ses seçimi şu öncelikle yapılır:

1. Seçili profil
2. Yüklenen referans ses
3. Varsayılan ses: `samples/my_voice.wav`

## 8. Profil seçiliyken yüklenen ses dosyası neden yok sayılır?

Profil seçildiğinde yüklenen ses dosyası bilinçli olarak yok sayılır. Çünkü seçili profil, kullanıcının daha önce hazırlanmış, kalite raporu alınmış ve güvenli şekilde ön işlenmiş referansı kullanmak istediği anlamına gelir.

Bu kural önceliği net tutar. Eğer kullanıcı yüklediği yeni sesi kullanmak istiyorsa profil seçimini boş bırakmalıdır. Eğer kullanıcı hazırlanmış yerel profili kullanmak istiyorsa dropdown'dan profil seçmelidir.

## 9. Yerel veri gizliliği

Voice profile dosyaları kişisel veya açık izinli ses kayıtları içerebilir. Bu nedenle gerçek profil klasörleri GitHub'a yüklenmez ve local çalışma verisi olarak kalır.

`.gitignore` içinde bu amaçla şu kurallar bulunur:

```text
profiles/*
!profiles/.gitkeep
```

Bu kurallar gerçek profil içeriklerini dışarıda bırakır, fakat klasörün projede var olması gerektiğini gösteren `.gitkeep` dosyasını korur.

Profil klasörleri dışında `samples/` altındaki gerçek referans sesler ve `outputs/` altındaki üretilen sesler de GitHub'a eklenmemelidir.

## 10. Profil oluşturma sırasında kalite raporu nasıl kullanılır?

Profil oluşturulurken hem orijinal referans hem de ön işlenmiş referans için kalite raporu alınır. Raporlar `profile.json` içine yazılır ve Gradio tarafında kullanıcıya gösterilebilir.

Kalite raporları şu konularda sinyal verir:

- Dosya var mı?
- Süre uygun mu?
- Sample rate ve kanal sayısı nedir?
- Ses seviyesi çok düşük veya çok yüksek mi?
- Clipping riski var mı?
- Sonuç `GOOD`, `WARNING` veya `BAD` mı?

`GOOD`, referansın teknik olarak daha temiz göründüğünü gösterir. `WARNING`, kullanılabilir ama dikkat edilmesi gereken bir durum olduğunu anlatır. `BAD`, kaydın yeniden alınmasının veya daha temiz bir referansla denenmesinin daha doğru olabileceğini gösterir.

Kalite raporu tek başına nihai ses benzerliğini garanti etmez. Son karar için üretilen ses mutlaka dinlenerek kontrol edilmelidir.

## 11. Profil oluşturma ile fine-tuning arasındaki fark

Voice profile oluşturma, yeni bir model eğitmek değildir. Bu akışta XTTS-v2 modeli aynı kalır; sadece seçilen referans ses yerel bir klasörde düzenli şekilde saklanır ve üretim sırasında `speaker_wav` olarak kullanılır.

Fine-tuning ise modelin ek veriyle yeniden eğitildiği veya uyumlandığı daha gelişmiş bir süreçtir. Bu projedeki mevcut MVP fine-tuning yapmaz, yeni ağırlık dosyası üretmez ve kişiye özel model eğitmez.

Kısaca:

- Voice profile: Referans sesi saklar, ön işler ve tekrar kullanır.
- Fine-tuning: Modelin davranışını eğitimle değiştirir.

VoxForge'un bu aşaması reference-based / zero-shot voice cloning MVP'sidir.

## 12. Güvenli ön işleme neden tercih edildi?

Varsayılan profil oluşturma akışı `safe_normalized` yaklaşımını kullanır. Bu yaklaşımın amacı referans sesi XTTS için tutarlı formata getirmek ve ses seviyesini dengelemektir.

Agresif sessizlik kırpma varsayılan akışta kullanılmaz. Çünkü bazı kayıtlarda konuşma araları, nefesler veya düşük seviyeli konuşma bölümleri yanlışlıkla sessizlik gibi algılanabilir. Bu durumda referans ses fazla kısalabilir ve konuşmacı karakteri zayıflayabilir.

Güvenli yaklaşım şu yüzden tercih edilir:

- Referans süresini korumaya çalışır.
- Ses karakterini gereksiz kırpmayla zayıflatmaz.
- Mono, 24000 Hz WAV çıktısı üretir.
- Kalite raporuyla birlikte daha kontrollü bir profil oluşturur.

## 13. Bilinen sınırlamalar

- Voice profile sistemi fine-tuning değildir.
- Profil seçimi yeni bir model oluşturmaz.
- Profil kalitesi kaynak sesin kalitesine bağlıdır.
- `profile.json` kalite sinyali verir, nihai ses benzerliğini garanti etmez.
- Profil silme ve yenileme için ayrı otomasyon henüz yoktur.
- Profil kalite geçmişi henüz tutulmaz.
- Gerçek profil dosyaları local kalmalıdır ve GitHub'a eklenmemelidir.

## 14. Sonraki adımlar

Planlanan geliştirme yönleri:

- Profil silme
- Profil yenileme
- Profil kalite geçmişi
- Fine-tuning hazırlığı

Fine-tuning sonraki ve daha gelişmiş bir aşamadır. Mevcut doküman ve kod akışı sadece yerel referans profil yönetimini anlatır.
