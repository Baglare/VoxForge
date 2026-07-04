param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [int]$Count = 80
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$GeneratePlanScript = Join-Path $ProjectRoot "scripts\generate_recording_plan.py"

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Error ".venv klasoru bulunamadi: $VenvDir"
    exit 1
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $GeneratePlanScript -PathType Leaf)) {
    Write-Error "Kayit plani scripti bulunamadi: $GeneratePlanScript"
    exit 1
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Dataset yolu: $Dataset"
Write-Host "Kayit sayisi: $Count"
Write-Host "Calistirilacak script: $GeneratePlanScript"

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $GeneratePlanScript --dataset $Dataset --count $Count
exit $LASTEXITCODE
