$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
$env:PYTHONUTF8 = '1'

$pythonExe = if ($env:TIANCHI_PYTHON) { $env:TIANCHI_PYTHON } else { 'python' }

Write-Host "Using Python: $pythonExe"
& $pythonExe 'run_pipeline.py' --mode all --recall multi --din --gpu 0 @args
if ($LASTEXITCODE -ne 0) {
    throw "Pipeline failed with exit code $LASTEXITCODE"
}
