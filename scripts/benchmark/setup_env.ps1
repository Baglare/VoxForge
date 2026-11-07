# VF-Q1: aday motorlar ve degerlendirme icin AYRI sanal ortamlar kurar.
# Ana .venv'e (XTTS/Gradio) dokunmaz. Model indirmez; modeller ilk calistirmada indirilir.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine chatterbox
#   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine voxcpm
#   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine eval
#
# Mevcut ortamda yalniz CPU torch'u CUDA'li surumle degistirmek (paketlere dokunmaz):
#   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine voxcpm -TorchOnly
#   (-CudaIndex cu132 ile indeks zorlanabilir; varsayilan: surucunun destekledigi en yeni uygun indeks)
param(
    [Parameter(Mandatory = $true)][ValidateSet("chatterbox", "voxcpm", "eval")][string]$Engine,
    [string]$PythonVersion = "",
    [switch]$TorchOnly,
    [string]$CudaIndex = ""
)
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")
$DataRoot = if ($env:AVENOX_DATA_ROOT) { $env:AVENOX_DATA_ROOT } else { Join-Path ([IO.Path]::GetPathRoot("$ProjectRoot")) 'AvenoxData' }
$EnvDir = Join-Path $DataRoot "Environments\VoxForge-bench-$Engine"

# Kaynaklarin onerdigi Python surumleri: Chatterbox 3.11 ile gelistirilmis; VoxCPM >=3.10,<3.13.
$Defaults = @{ chatterbox = "3.11"; voxcpm = "3.12"; eval = "3.12" }
if (-not $PythonVersion) { $PythonVersion = $Defaults[$Engine] }

# Chatterbox V3: PyPI'daki chatterbox-tts 0.1.7 t3_model="v3" desteklemiyor. Resmi GitHub kaynagi
# engine_pins.json'daki commit'e sabitlenir (surum numarasi iki kaynakta da 0.1.7 oldugu icin yetmez).
$Pins = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot "engine_pins.json") | ConvertFrom-Json
$ChatterboxSpec = "$($Pins.chatterbox.package) @ git+$($Pins.chatterbox.repo)@$($Pins.chatterbox.commit)"

$Packages = @{
    chatterbox = @($ChatterboxSpec)
    voxcpm     = @("voxcpm", "soundfile")
    eval       = @("faster-whisper", "av<17", "speechbrain", "soundfile")
}

Write-Host "Ortam: $EnvDir (Python $PythonVersion)"
& py "-$PythonVersion" -c "import sys" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python $PythonVersion bulunamadi. Kurmak icin: winget install Python.Python.$PythonVersion" -ForegroundColor Yellow
    Write-Host "Ya da farkli surum deneyin: -PythonVersion 3.10 (resmi olarak test edilmemis olabilir)."
    exit 1
}

if (-not (Test-Path -LiteralPath (Join-Path $EnvDir "Scripts\python.exe"))) {
    if ($TorchOnly) { throw "-TorchOnly icin mevcut ortam gerekli: $EnvDir" }
    & py "-$PythonVersion" -m venv $EnvDir
}
$Py = Join-Path $EnvDir "Scripts\python.exe"

function Install-CudaTorch {
    # PyPI Windows'ta yalniz CPU torch verir. Ayni torch surumunu, o surumu GERCEKTEN barindiran ve
    # surucunun destekledigi CUDA indeksinden kurar (torch_cuda.py). Eskiden >=2.6 icin sabit cu126
    # seciliyordu; torch 2.15.0 orada yok, kurulum basarisiz olup ortam sessizce +cpu kaliyordu.
    $Probe = & $Py -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "torch ice aktarilamadi." }
    $TorchVersion = $Probe[0].Split("+")[0]
    if ($Probe[1] -eq "True") { Write-Host "torch $($Probe[0]) CUDA'li; degisiklik yok."; return }
    $ChooseArgs = @((Join-Path $PSScriptRoot "torch_cuda.py"), "--torch-version", $TorchVersion)
    if ($CudaIndex) { $ChooseArgs += @("--index", $CudaIndex) }
    $Choice = (& $Py @ChooseArgs) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $Choice.index) {
        throw "torch $TorchVersion icin uygun CUDA indeksi yok (surucu CUDA $($Choice.driver_cuda), denenen: $($Choice.checked -join ', ')). Farkli torch surumu veya -CudaIndex deneyin."
    }
    $Lock = Join-Path $EnvDir "requirements-lock.before-cuda-torch.txt"
    & $Py -m pip freeze | Out-File -Encoding utf8 $Lock
    Write-Host "torch $TorchVersion CUDA'siz; $($Choice.index) indeksinden yeniden kuruluyor (surucu CUDA $($Choice.driver_cuda)). Onceki liste: $Lock"
    # Yalniz torch: torchaudio/torchcodec bu indekslerde ayni surumle bulunmayabilir; CPU G/C icin yeterli.
    & $Py -m pip install --force-reinstall --no-deps "torch==$TorchVersion" --index-url $Choice.index_url
    if ($LASTEXITCODE -ne 0) { throw "CUDA'li torch kurulumu basarisiz ($($Choice.index_url))." }
    & $Py -c "import torch, sys; ok = torch.cuda.is_available() and (torch.ones(8, device='cuda') * 2).sum().item() == 16; sys.exit(0 if ok else 1)"
    if ($LASTEXITCODE -ne 0) { throw "torch kuruldu ama CUDA calismiyor (surucu/paket uyumsuzlugu)." }
}

if ($TorchOnly) {
    Install-CudaTorch
    & $Py -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
    Write-Host "Hazir (yalniz torch)."
    exit 0
}

& $Py -m pip install --upgrade pip

if ($Engine -eq "eval") {
    # Degerlendirme ortami: once CUDA'li torch, sonra ASR ve konusmaci benzerligi paketleri.
    & $Py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
}
if ($Engine -eq "chatterbox") {
    & git --version *> $null
    if ($LASTEXITCODE -ne 0) { throw "git bulunamadi; GitHub kaynagindan kurulum icin gerekli (winget install Git.Git)." }
    # Ayni surum numarali PyPI kurulumu (V3'suz) varsa kaldir; yalniz bu izole ortamda.
    & $Py -m pip uninstall -y chatterbox-tts *> $null
    Write-Host "Chatterbox kaynagi: $ChatterboxSpec"
}
& $Py -m pip install @($Packages[$Engine])
if ($LASTEXITCODE -ne 0) { throw "Paket kurulumu basarisiz." }

Install-CudaTorch

& $Py -c "import torch, platform; print('Python', platform.python_version(), '| torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
& $Py -m pip freeze | Out-File -Encoding utf8 (Join-Path $EnvDir "requirements-lock.txt")
Write-Host "Surum kilidi: $(Join-Path $EnvDir 'requirements-lock.txt')"

if ($Engine -eq "chatterbox") {
    # Model indirmeden: kaynak commit'i, t3_model="v3" API'si, Turkce destegi ve CUDA.
    & $Py (Join-Path $PSScriptRoot "check_engine.py") --engine chatterbox
    if ($LASTEXITCODE -ne 0) { throw "Chatterbox V3 API kontrolu basarisiz (yukaridaki JSON'a bakin)." }
}
Write-Host "Hazir."
