# VF-Q1: aday motorlar ve degerlendirme icin AYRI sanal ortamlar kurar.
# Ana .venv'e (XTTS/Gradio) dokunmaz. Model indirmez; modeller ilk calistirmada indirilir.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine chatterbox
#   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine voxcpm
#   powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine eval
param(
    [Parameter(Mandatory = $true)][ValidateSet("chatterbox", "voxcpm", "eval")][string]$Engine,
    [string]$PythonVersion = ""
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
    eval       = @("faster-whisper", "speechbrain", "soundfile")
}

Write-Host "Ortam: $EnvDir (Python $PythonVersion)"
& py "-$PythonVersion" -c "import sys" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python $PythonVersion bulunamadi. Kurmak icin: winget install Python.Python.$PythonVersion" -ForegroundColor Yellow
    Write-Host "Ya da farkli surum deneyin: -PythonVersion 3.10 (resmi olarak test edilmemis olabilir)."
    exit 1
}

if (-not (Test-Path -LiteralPath (Join-Path $EnvDir "Scripts\python.exe"))) {
    & py "-$PythonVersion" -m venv $EnvDir
}
$Py = Join-Path $EnvDir "Scripts\python.exe"
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

# Windows'ta PyPI'dan gelen torch CPU surumu olabilir; ayni surumu CUDA indeksinden yeniden kur.
$Probe = & $Py -c "import torch; print(torch.__version__.split('+')[0]); print(torch.cuda.is_available())"
$TorchVersion = $Probe[0]; $CudaOk = $Probe[1]
if ($CudaOk -ne "True") {
    $Parts = $TorchVersion.Split("."); $Minor = [int]$Parts[1]; $Major = [int]$Parts[0]
    $Index = if ($Major -ge 2 -and $Minor -ge 6) { "cu126" } elseif ($Major -ge 2 -and $Minor -ge 4) { "cu124" } else { "cu121" }
    $AudioProbe = & $Py -c "import importlib.metadata as m; print(m.version('torchaudio').split('+')[0])" 2>$null
    $Spec = @("torch==$TorchVersion")
    if ($AudioProbe) { $Spec += "torchaudio==$AudioProbe" }
    Write-Host "torch $TorchVersion CUDA'siz; $Index indeksinden yeniden kuruluyor..."
    & $Py -m pip install --force-reinstall --no-deps @Spec --index-url "https://download.pytorch.org/whl/$Index"
}

& $Py -c "import torch, platform; print('Python', platform.python_version(), '| torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
& $Py -m pip freeze | Out-File -Encoding utf8 (Join-Path $EnvDir "requirements-lock.txt")
Write-Host "Surum kilidi: $(Join-Path $EnvDir 'requirements-lock.txt')"

if ($Engine -eq "chatterbox") {
    # Model indirmeden: kaynak commit'i, t3_model="v3" API'si, Turkce destegi ve CUDA.
    & $Py (Join-Path $PSScriptRoot "check_engine.py") --engine chatterbox
    if ($LASTEXITCODE -ne 0) { throw "Chatterbox V3 API kontrolu basarisiz (yukaridaki JSON'a bakin)." }
}
Write-Host "Hazir."
