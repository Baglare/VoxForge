param(
    [Parameter(Mandatory = $true)]
    [string]$Slug,

    [switch]$Yes
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$DeleteProfileScript = Join-Path $ProjectRoot "scripts\delete_voice_profile.py"

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Error ".venv klasoru bulunamadi: $VenvDir"
    exit 1
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $DeleteProfileScript -PathType Leaf)) {
    Write-Error "Profil silme scripti bulunamadi: $DeleteProfileScript"
    exit 1
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Profil slug: $Slug"
Write-Host "Calistirilacak script: $DeleteProfileScript"

if (-not $Yes) {
    Write-Warning "Profil silinmedi. Silmek icin -Yes parametresini ekleyin."
    exit 1
}

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $DeleteProfileScript --slug $Slug --yes
exit $LASTEXITCODE
