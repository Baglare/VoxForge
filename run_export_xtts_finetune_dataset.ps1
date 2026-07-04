param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$RunName
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$ExportScript = Join-Path $ProjectRoot "scripts\export_xtts_finetune_dataset.py"

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Error ".venv klasoru bulunamadi: $VenvDir"
    exit 1
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $ExportScript -PathType Leaf)) {
    Write-Error "XTTS dataset export scripti bulunamadi: $ExportScript"
    exit 1
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Dataset yolu: $Dataset"
Write-Host "Run name: $RunName"
Write-Host "Calistirilacak script: $ExportScript"

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $ExportScript --dataset $Dataset --run-name $RunName
exit $LASTEXITCODE
