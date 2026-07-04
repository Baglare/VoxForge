param(
    [Parameter(Mandatory = $true)]
    [string]$MatrixRoot,

    [switch]$UseDefaultScores
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReportScript = Join-Path $ProjectRoot "scripts\create_human_eval_report.py"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path -LiteralPath $ReportScript)) {
    throw "Human eval script not found: $ReportScript"
}

Write-Host "Python: $PythonExe"
Write-Host "Matrix root: $MatrixRoot"
Write-Host "Use default scores: $UseDefaultScores"
Write-Host "Script: $ReportScript"

$PythonArgs = @(
    $ReportScript,
    "--matrix-root",
    $MatrixRoot
)

if ($UseDefaultScores) {
    $PythonArgs += "--use-default-scores"
}

Push-Location -LiteralPath $ProjectRoot
try {
    & $PythonExe @PythonArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
