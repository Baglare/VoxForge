param(
    [Parameter(Mandatory = $true)]
    [string]$Slug
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$RecreateProfileScript = Join-Path $ProjectRoot "scripts\recreate_voice_profile.py"

function Find-GyanSharedFfmpeg {
    # Winget genellikle Gyan.FFmpeg.Shared paketini bu koklerde tutar.
    $candidateRoots = @(
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"),
        (Join-Path $env:ProgramFiles "WinGet\Packages"),
        (Join-Path ${env:ProgramFiles(x86)} "WinGet\Packages")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }

    foreach ($root in $candidateRoots) {
        $packageDirs = Get-ChildItem `
            -LiteralPath $root `
            -Directory `
            -Filter "Gyan.FFmpeg.Shared*" `
            -ErrorAction SilentlyContinue

        foreach ($packageDir in $packageDirs) {
            $ffmpegExe = Get-ChildItem `
                -LiteralPath $packageDir.FullName `
                -Recurse `
                -File `
                -Filter "ffmpeg.exe" `
                -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1

            if ($ffmpegExe) {
                return $ffmpegExe.FullName
            }
        }
    }

    # PATH icinde varsa ve yol Gyan.FFmpeg.Shared iceriyorsa onu da kabul et.
    $pathFfmpeg = where.exe ffmpeg 2>$null |
        Where-Object { $_ -like "*Gyan.FFmpeg.Shared*" } |
        Select-Object -First 1

    if ($pathFfmpeg) {
        return $pathFfmpeg
    }

    return $null
}

function Find-PathFfmpeg {
    $pathFfmpeg = where.exe ffmpeg 2>$null | Select-Object -First 1
    if ($pathFfmpeg) {
        return $pathFfmpeg
    }

    return $null
}

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Error ".venv klasoru bulunamadi: $VenvDir"
    exit 1
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error ".venv icindeki python.exe bulunamadi: $PythonExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $RecreateProfileScript -PathType Leaf)) {
    Write-Error "Profil yenileme scripti bulunamadi: $RecreateProfileScript"
    exit 1
}

$FfmpegExe = Find-GyanSharedFfmpeg
if ($FfmpegExe) {
    $FfmpegBin = Split-Path -Parent $FfmpegExe
    $env:PATH = "$FfmpegBin;$env:PATH"
    Write-Host "FFmpeg yolu: $FfmpegExe"
    Write-Host "PATH basina eklenen klasor: $FfmpegBin"
}
else {
    Write-Warning "Gyan.FFmpeg.Shared icindeki ffmpeg.exe otomatik bulunamadi. Mevcut PATH ile devam edilecek."
    $FfmpegExe = Find-PathFfmpeg
    if (-not $FfmpegExe) {
        $FfmpegExe = "bulunamadi"
    }
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Kullanilacak FFmpeg yolu: $FfmpegExe"
Write-Host "Profil slug: $Slug"
Write-Host "Calistirilacak script: $RecreateProfileScript"

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $RecreateProfileScript --slug $Slug
exit $LASTEXITCODE
