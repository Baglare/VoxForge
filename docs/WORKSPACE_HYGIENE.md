# VoxForge çalışma alanı düzeni (ajanlar ve insanlar için)

VF-016 disk incelemesinden çıkan kısa kurallar. Amaç: her iş için yeni worktree/venv/model kopyası açıp
diski sessizce doldurmamak. Durumu görmek için (hiçbir şey silmez):

```
py -3 scripts/workspace_report.py
```

## Kopya açma

1. **Varsayılan tek kopya**: `C:\AvenoxBrain\Projects\VoxForge` içinde dal aç (`git switch -c vf-xxx`).
   Ek worktree yalnız aynı anda iki dalda çalışmak gerçekten gerekiyorsa açılır ve iş birleşince kaldırılır.
2. Ek worktree **asla** yeni `.venv` kurmaz: Python ortamları `C:\AvenoxData\Environments\VoxForge*`
   altındadır (ana `.venv` bir junction'dır). Worktree'de `C:\AvenoxBrain\Projects\VoxForge\.venv\Scripts\python.exe`
   ya da bench ortamının yorumlayıcısı kullanılır.
3. Model önbelleği tektir: `HF_HOME=C:\AvenoxData\Models\HF`, XTTS için `TTS_HOME=C:\AvenoxData\Models\Coqui`.
   Playground/benchmark bunları kendisi ayarlar; elle bir motoru çalıştırırken de aynı değişkenleri verin,
   yoksa model `~/.cache/huggingface` altına **ikinci kez** iner (VF-016'da 3 GB'lık Chatterbox kopyası bulundu).
4. Çıktılar `outputs/` (junction → `C:\AvenoxData\Projects\VoxForge\outputs`). Worktree'deki `outputs/`
   ayrı bir klasördür: kanıt üretildiyse iş bitince ana `outputs/` altına taşınır, sonra worktree kaldırılır.

## Kaldırma ölçütleri (worktree)

Bir worktree ancak hepsi doğruysa kaldırılır:
- `git status --porcelain` boş (izlenen değişiklik ve izlenmeyen dosya yok);
- dal ucu birleşmiş bir PR'ın head commit'i (squash merge yüzünden `git branch --merged` yetmez; `gh pr list
  --state merged --json headRefOid` kullanın);
- `git status --ignored` ile görünen yok sayılan dosyalar (ör. `outputs/…` benchmark kanıtı) ana çıktılara
  **üzerine yazmadan** kopyalanıp dosya dosya doğrulandı; farklı sürümler ayrı adla saklandı;
- başka bir oturum/ajan o klasörde çalışmıyor (kabuğun çalışma dizini orada değil);
- stash girdileri depoya aittir, worktree ile silinmez; dokunmayın.

Kaldırma: `git -C C:\AvenoxBrain\Projects\VoxForge worktree remove <yol>`; ardından `git worktree prune`.
Dal silinmez (geçmiş ve PR bağlantısı kalır).

## Ne silinmez

Profiller, ses kayıtları, Studio projeleri, `experiments/` (eğitim çıktıları), benchmark/VF-0xx kanıtları,
gerekli ortamlar ve modeller, stash kayıtları. Git'in yok saydığı bir dosya bu yüzden silinebilir sayılmaz.

## Paylaşılan önbellekler (kullanıcı kararı)

pip ve uv önbellekleri tüm projelerce paylaşılır; ajanlar bunları kendiliğinden temizlemez. İndirme
önbelleğidir, silinince yalnız yeniden indirme gerekir:
- pip: `py -3 -m pip cache purge` (torch CUDA tekerlekleri 2-2,5 GB/adet; hard link değildir, tamamı yer açar).
- uv: `uv cache prune` (ortamlara hard link olan kısım silinse de yer açmaz; prune yalnız kullanılmayanı alır).
