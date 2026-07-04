param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$BuildMetadataScript = Join-Path $ProjectRoot "scripts\build_metadata_from_recording_plan.py"

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Error ".venv klasoru bulunamadi: $VenvDir"
    exit 1
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $BuildMetadataScript -PathType Leaf)) {
    Write-Error "Metadata olusturma scripti bulunamadi: $BuildMetadataScript"
    exit 1
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Dataset yolu: $Dataset"
Write-Host "Calistirilacak script: $BuildMetadataScript"

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $BuildMetadataScript --dataset $Dataset
exit $LASTEXITCODE
