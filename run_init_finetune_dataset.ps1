param(
    [Parameter(Mandatory = $true)]
    [string]$Name
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$InitDatasetScript = Join-Path $ProjectRoot "scripts\init_finetune_dataset.py"

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Error ".venv klasoru bulunamadi: $VenvDir"
    exit 1
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $InitDatasetScript -PathType Leaf)) {
    Write-Error "Dataset baslatma scripti bulunamadi: $InitDatasetScript"
    exit 1
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Dataset adi: $Name"
Write-Host "Calistirilacak script: $InitDatasetScript"

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $InitDatasetScript --name $Name
exit $LASTEXITCODE
