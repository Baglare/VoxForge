param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment,

    [string]$Text = "",

    [string]$SpeakerWav = ""
)

$ErrorActionPreference = "Stop"

# Proje kok dizini bu PowerShell dosyasinin bulundugu klasordur.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$EvalScript = Join-Path $ProjectRoot "scripts\evaluate_xtts_finetuned_checkpoint.py"

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

function Find-PathTool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ToolName
    )

    $toolPath = where.exe $ToolName 2>$null | Select-Object -First 1
    if ($toolPath) {
        return $toolPath
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

if (-not (Test-Path -LiteralPath $EvalScript -PathType Leaf)) {
    Write-Error "Fine-tuned XTTS evaluation scripti bulunamadi: $EvalScript"
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
    $FfmpegExe = Find-PathTool -ToolName "ffmpeg"
    if (-not $FfmpegExe) {
        $FfmpegExe = "bulunamadi"
    }
}

$TextForLog = $Text
if ([string]::IsNullOrWhiteSpace($TextForLog)) {
    $TextForLog = "(Python varsayilan test metni)"
}

Write-Host "Python yolu: $PythonExe"
Write-Host "Kullanilacak FFmpeg yolu: $FfmpegExe"
Write-Host "Experiment yolu: $Experiment"
Write-Host "Text: $TextForLog"
if (-not [string]::IsNullOrWhiteSpace($SpeakerWav)) {
    Write-Host "Speaker wav: $SpeakerWav"
}
Write-Host "Calistirilacak script: $EvalScript"

Set-Location -LiteralPath $ProjectRoot

$PythonArgs = @(
    $EvalScript,
    "--experiment", $Experiment
)

if (-not [string]::IsNullOrWhiteSpace($Text)) {
    $PythonArgs += @("--text", $Text)
}

if (-not [string]::IsNullOrWhiteSpace($SpeakerWav)) {
    $PythonArgs += @("--speaker-wav", $SpeakerWav)
}

& $PythonExe @PythonArgs
exit $LASTEXITCODE
