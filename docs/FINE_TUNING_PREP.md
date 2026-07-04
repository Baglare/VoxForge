# VoxForge Fine-Tuning Hazırlığı

Bu doküman, VoxForge içinde ileride kullanılabilecek yerel voice fine-tuning dataset altyapısını açıklar. Bu aşama model eğitimi değildir; sadece veri klasörü, metadata formatı, kalite kontrolü ve GitHub hijyeni için hazırlıktır.

## Fine-tuning bu projede ne anlama geliyor?

Fine-tuning, bir ses modelinin ek eğitim verisiyle belirli bir sese veya konuşma biçimine daha iyi uyumlanması anlamına gelir. VoxForge'un mevcut MVP akışı ise fine-tuning yapmaz; reference-based / zero-shot voice cloning yaklaşımıyla referans ses üzerinden üretim dener.

Bu hazırlık katmanı, ileride fine-tuning aşamasına geçilecekse eğitim verisinin düzenli, doğrulanabilir ve local kalmasını sağlar.

## Şu an neden fine-tuning yapmıyoruz?

Bu aşamada model eğitimi başlatılmıyor çünkü önce veri kalitesi, transkript tutarlılığı, klasör düzeni ve gizlilik sınırları netleşmelidir.

Fine-tuning daha maliyetli ve riskli bir aşamadır. Model dosyaları, eğitim çıktıları, uzun çalışma süreleri ve daha dikkatli veri hazırlığı gerektirir. Bu yüzden bu adım yalnızca dataset hazırlığı ve doğrulama ile sınırlıdır.

## Dataset klasör yapısı

Gerçek datasetler local `datasets/` klasörü altında tutulur. Örnek yapı:

```text
datasets/<dataset_slug>/
|-- wavs/
|   |-- sample_001.wav
|   `-- sample_002.wav
|-- recording_plan.csv
|-- metadata.csv
`-- dataset_report.json
```

Bu örnek yapı dokümantasyon amaçlıdır. Repoda gerçek ses dosyası veya gerçek dataset metadata dosyası tutulmamalıdır.

## metadata.csv formatı

`metadata.csv` dosyası pipe ayracıyla iki alan içerir:

```text
audio_path|text
wavs/sample_001.wav|Merhaba, bu bir örnek eğitim cümlesidir.
wavs/sample_002.wav|Bu kayıt temiz ve doğal bir konuşma içermelidir.
```

`audio_path`, dataset klasörüne göre göreli yol olmalıdır. `text`, ses dosyasında gerçekten söylenen metinle eşleşmelidir.

## Kayıt planı üretme akışı

Kayıt planı, hangi metnin hangi dosya adıyla kaydedileceğini önceden belirler. Böylece kayıt sırasında dosya adları ve transkriptler karışmaz.

Önerilen akış:

1. Dataset iskeletini oluştur.
2. `docs/RECORDING_TEXT_SET_TR.md` içindeki özgün Türkçe metinlerden `recording_plan.csv` üret.
3. Her satırdaki `target_audio_path` değerine göre sesleri `wavs/` klasörüne kaydet.
4. Kaydı tamamlanan satırların `status` alanını `DONE` yap.
5. DONE satırlardan `metadata.csv` oluştur.
6. Dataset doğrulama scriptini çalıştır.

Dataset iskeleti:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_init_finetune_dataset.ps1 -Name baglare_finetune_v1
```

Kayıt planı üretme:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80
```

Varsayılan davranış güvenlidir: `recording_plan.csv` zaten varsa üzerine yazılmaz. Planı bilinçli olarak yeniden üretmek için `-Overwrite` eklenir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_generate_recording_plan.ps1 -Dataset .\datasets\baglare-finetune-v1 -Count 80 -Overwrite
```

`recording_plan.csv` formatı:

```text
clip_id;target_audio_path;text;status;notes
VF_TR_001;wavs/VF_TR_001.wav;Bugün hava sakin ve çalışmak için uygun görünüyor.;TODO;
```

Excel notu: `recording_plan.csv` artık `utf-8-sig` encoding ve noktalı virgül (`;`) delimiter ile yazılır. Bu, Windows'ta Excel ile çift tıklayarak açıldığında Türkçe karakterlerin bozulmamasına ve sütunların düzgün ayrılmasına yardımcı olur. Eski bir kayıt planı Excel'de bozuk görünüyor veya bozuk açılıp kaydedildiyse dosyayı `-Overwrite` ile yeniden üretin.

Kayıt tamamlandığında ilgili satırdaki `status` alanı `DONE` yapılır:

```text
VF_TR_001;wavs/VF_TR_001.wav;Bugün hava sakin ve çalışmak için uygun görünüyor.;DONE;
```

DONE satırlardan metadata üretme:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_build_metadata.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Bu komut yalnızca `DONE` satırları işler. Ses dosyası eksikse ilgili satır `metadata.csv` içine yazılmaz ve terminalde uyarı gösterilir.

## İyi eğitim kaydı nasıl alınır?

İyi bir eğitim kaydı için:

- Tek kişinin konuştuğu kayıtlar kullanın.
- Arka plan müziği ve ortam gürültüsünden kaçının.
- Mikrofona çok yakın konuşup clipping oluşturmayın.
- Cümleleri doğal, net ve sabit ses seviyesiyle okuyun.
- Farklı cümle uzunlukları ve doğal tonlama kullanın.
- Başka kişilerin sesini izinsiz kullanmayın.

## Transkript neden önemli?

Fine-tuning için ses ile metin birebir eşleşmelidir. Metadata içindeki metin, WAV dosyasında duyulan konuşmadan farklıysa eğitim kalitesi düşer.

Yanlış transkript; telaffuz sorunlarına, eksik kelimelere, garip vurguya veya modelin metin-ses eşleşmesini yanlış öğrenmesine neden olabilir.

## Tek kayıt mı, çok kayıt mı?

Tek uzun kayıt yerine birden fazla kısa ve temiz kayıt daha yönetilebilir olur. Her satır bir WAV dosyasına ve o dosyanın transkriptine karşılık gelir.

Pratik yaklaşım:

- Uzun ham kaydı küçük parçalara ayırın.
- Her parça için ayrı transkript yazın.
- Her satırın metni yalnızca o WAV dosyasında duyulan konuşmayı içersin.

## Süre ve kalite beklentisi

Dataset doğrulama scripti her örnek için süre, sample rate, kanal sayısı ve temel kalite sinyallerini kontrol eder.

Beklenen teknik hedef:

- WAV formatı
- 24000 Hz sample rate
- Mono kanal
- Çok kısa veya çok uzun olmayan cümle parçaları
- Temiz, anlaşılır, clipping riski düşük kayıt

Doğrulama uyarısı her zaman dosyanın kullanılamaz olduğu anlamına gelmez; ancak kayıt dinlenerek kontrol edilmelidir.

## Dataset doğrulama nasıl çalışır?

Önce boş dataset iskeleti oluşturulur:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_init_finetune_dataset.ps1 -Name baglare_finetune_v1
```

Sonra kayıt planı üzerinden WAV dosyaları `wavs/` altına local olarak eklenir, DONE satırlardan `metadata.csv` oluşturulur ve doğrulama şu komutla çalıştırılır:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_validate_finetune_dataset.ps1 -Dataset .\datasets\baglare-finetune-v1
```

Doğrulama; boş alanları, eksik dosyaları, WAV olmayan dosyaları, süre uyarılarını, sample rate değerini, kanal sayısını ve kalite analizini kontrol eder. JSON raporu local olarak `outputs/reports/finetune_dataset_report.json` dosyasına yazılır.

## GitHub'a neden gerçek dataset yüklenmez?

Fine-tuning datasetleri kişisel ses kayıtları ve bu kayıtların transkriptlerini içerir. Bu dosyalar mahremiyet, izin ve repo boyutu nedeniyle GitHub'a yüklenmemelidir.

`.gitignore` içinde şu kurallar bulunur:

```text
datasets/*
!datasets/.gitkeep
```

Bu yapı gerçek dataset içeriklerini dışarıda bırakır, ama `datasets/` klasörünün projede var olması gerektiğini `.gitkeep` ile gösterir.

## Sonraki aşamada neler yapılabilir?

Sonraki aşamada şu işler ayrı bir kapsam olarak ele alınabilir:

- Dataset parçalama ve transkript temizleme akışı
- Daha ayrıntılı dataset kalite raporu
- Eğitim konfigürasyonu taslağı
- Fine-tuning deneme komutları
- Eğitim çıktılarının local saklama ve GitHub dışı kalma kuralları

Bu doküman yalnızca hazırlık aşamasını kapsar; model eğitimi başlatmaz.
