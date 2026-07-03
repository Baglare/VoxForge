param(
    [switch]$Json
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$ListProfilesScript = Join-Path $ProjectRoot "scripts\list_voice_profiles.py"

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Error ".venv klasoru bulunamadi: $VenvDir"
    exit 1
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $ListProfilesScript -PathType Leaf)) {
    Write-Error "Voice profile listeleme scripti bulunamadi: $ListProfilesScript"
    exit 1
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Calistirilacak script: $ListProfilesScript"

$PythonArgs = @($ListProfilesScript)
if ($Json) {
    $PythonArgs += "--json"
}

Set-Location -LiteralPath $ProjectRoot
& $PythonExe @PythonArgs
exit $LASTEXITCODE
