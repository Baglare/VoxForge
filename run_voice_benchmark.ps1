# VF-Q1 ses kalitesi karsilastirmasi adimlari. Her adim ayri calisir; GPU adimlarini siz baslatirsiniz.
# Ayrintilar: docs/VOICE_QUALITY_BENCHMARK.md
#
#   .\run_voice_benchmark.ps1 -Step prepare  -ProfileSlug <profil_slug> [-ReferenceText "referansin birebir metni"]
#   .\run_voice_benchmark.ps1 -Step generate -Run <kosu_klasoru> -Engine xtts|chatterbox-v3|voxcpm2|voxcpm2-ultimate|mock [-Only "t01,t03"]
#        (-Only ile pilot; ayni komut -Only olmadan tekrar calisinca eksik maddeler eklenir, pilot ciktilar korunur)
#   .\run_voice_benchmark.ps1 -Step evaluate -Run <kosu_klasoru>
#   .\run_voice_benchmark.ps1 -Step blind    -Run <kosu_klasoru> [-Pilot]
#   .\run_voice_benchmark.ps1 -Step report   -Run <kosu_klasoru>
param(
    [Parameter(Mandatory = $true)][ValidateSet("prepare", "generate", "evaluate", "blind", "report")][string]$Step,
    [string]$Run = "",
    [ValidateSet("", "xtts", "chatterbox-v3", "voxcpm2", "voxcpm2-ultimate", "mock")][string]$Engine = "",
    [string]$ProfileSlug = "",
    [string]$Reference = "",
    [string]$ReferenceText = "",
    [string]$Only = "",
    [switch]$Overwrite,
    [switch]$Pilot
)
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataRoot = if ($env:AVENOX_DATA_ROOT) { $env:AVENOX_DATA_ROOT } else { Join-Path ([IO.Path]::GetPathRoot($ProjectRoot)) 'AvenoxData' }
$MainPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BenchDir = Join-Path $ProjectRoot "scripts\benchmark"

# Indirilen modeller repo disinda, ortak veri klasorunde tutulur.
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $DataRoot "Models\HF" }
$CoquiRoot = Join-Path $DataRoot 'Models\Coqui'
if (-not $env:TTS_HOME -and (Test-Path -LiteralPath (Join-Path $CoquiRoot 'tts') -PathType Container)) { $env:TTS_HOME = $CoquiRoot }
$env:PYTHONIOENCODING = "utf-8"

# TorchCodec (XTTS/Coqui ses yukleme) Windows'ta FFmpeg shared DLL'lerini PATH uzerinden arar.
# System32 veya C:/ffmpeg gibi essentials kurulumlari DLL icermez; Gyan.FFmpeg.Shared bin klasoru PATH basina eklenir.
function Add-SharedFfmpegToPath {
    $Roots = @(
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"),
        (Join-Path $env:ProgramFiles "WinGet\Packages")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }
    foreach ($Root in $Roots) {
        $Exe = Get-ChildItem -LiteralPath $Root -Directory -Filter "Gyan.FFmpeg.Shared*" -ErrorAction SilentlyContinue |
            ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Filter "avcodec-*.dll" -ErrorAction SilentlyContinue } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($Exe) {
            $env:PATH = "$($Exe.DirectoryName);$env:PATH"
            Write-Host "FFmpeg shared yolu: $($Exe.DirectoryName)"
            return
        }
    }
    Write-Warning "Gyan.FFmpeg.Shared bulunamadi; XTTS TorchCodec hatasi verebilir (docs/SETUP_WINDOWS.md)."
}
Add-SharedFfmpegToPath

function Get-BenchPython([string]$Name) {
    $Py = Join-Path $DataRoot "Environments\VoxForge-bench-$Name\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Py)) {
        throw "Ortam yok: $Py. Once: powershell -ExecutionPolicy Bypass -File .\scripts\benchmark\setup_env.ps1 -Engine $Name"
    }
    return $Py
}

function Require-Run {
    if (-not $Run) { throw "-Run <kosu_klasoru> gerekli (prepare adiminin yazdigi yol)." }
}

switch ($Step) {
    "prepare" {
        $CliArgs = @("prepare")
        if ($ProfileSlug) { $CliArgs += @("--profile", $ProfileSlug) } elseif ($Reference) { $CliArgs += @("--reference", $Reference) } else { throw "-ProfileSlug veya -Reference gerekli." }
        if ($ReferenceText) { $CliArgs += @("--reference-text", $ReferenceText) }
        & $MainPy (Join-Path $BenchDir "bench.py") @CliArgs
    }
    "generate" {
        Require-Run
        if (-not $Engine) { throw "-Engine gerekli." }
        $Py = switch ($Engine) {
            "chatterbox-v3" { Get-BenchPython "chatterbox" }
            { $_ -like "voxcpm2*" } { Get-BenchPython "voxcpm" }
            default { $MainPy }
        }
        $CliArgs = @((Join-Path $BenchDir "run_engine.py"), "--run", $Run, "--engine", $Engine)
        if ($Only) { $CliArgs += @("--only", $Only) }
        if ($Overwrite) { $CliArgs += "--overwrite" }
        & $Py @CliArgs
    }
    "evaluate" { Require-Run; & (Get-BenchPython "eval") (Join-Path $BenchDir "evaluate.py") --run $Run }
    "blind" {
        Require-Run
        if ($Pilot) { & $MainPy (Join-Path $BenchDir "bench.py") blind --run $Run --pilot }
        else { & $MainPy (Join-Path $BenchDir "bench.py") blind --run $Run }
    }
    "report" { Require-Run; & $MainPy (Join-Path $BenchDir "bench.py") report --run $Run }
}
exit $LASTEXITCODE
