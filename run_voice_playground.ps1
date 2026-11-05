# VoxForge Voice Playground: XTTS-v2 / Chatterbox V3 / VoxCPM2 ile serbest Turkce metin uretimi.
# Arayuz ana .venv'de calisir; motorlar mevcut izole ortamlarinda alt surec olarak yuklenir
# (GPU'da ayni anda tek motor). Ayrintilar: docs/VOICE_PLAYGROUND.md
#
#   .\run_voice_playground.ps1            # http://127.0.0.1:7861
#   .\run_voice_playground.ps1 -Port 7870
param([int]$Port = 7861)
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$App = Join-Path $ProjectRoot "app\voice_playground.py"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

# HF_HOME, TTS_HOME ve FFmpeg shared PATH ayarlari Python tarafinda (voxforge/playground/workers.py)
# run_voice_benchmark.ps1 ile ayni kurallarla motor sureclerine verilir.
$env:PYTHONIOENCODING = "utf-8"
$env:VOXFORGE_PLAYGROUND_PORT = "$Port"

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $App
exit $LASTEXITCODE
